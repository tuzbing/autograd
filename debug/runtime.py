import os
import sys
from pathlib import Path

import numpy as np
import functools
from math import prod
import time
from collections import defaultdict
from dataclasses import dataclass

_DEBUG_ROOT = Path(__file__).resolve().parent
os.chdir(_DEBUG_ROOT)
if str(_DEBUG_ROOT) not in sys.path:
	sys.path.insert(0, str(_DEBUG_ROOT))
if str(_DEBUG_ROOT.parent) not in sys.path:
	sys.path.insert(1, str(_DEBUG_ROOT.parent))

from storage import CPUAllocator, MetalAllocator
from backend import CPUBackend, MetalBackend
from op import MatMulOp
from tensor import Tensor
from scratch import ScratchViews, plan_node_scratch
from device import Device

VIEW_OPS = ['View', 'Reshape', 'Permute', 'Transpose']
METAL_SCRATCH_OPS = ["FlashAttentionOp", 'FusedRMSNormMLPOp', 'FusedRMSNormFlashAttentionOp', 'FusedLinearAddOp', 'FusedLinearCrossEntropyOp', 
					 "KVCacheAttentionOp", "RolloutSampleOp", "FusedRMSNormGQAOp", "FusedRMSNormLinearCrossEntropyOp"]

class MemoryPlanner:
	def __init__(self):
		# used memory block [offset, size, tensor_id]
		self.active_blocks = []
		# tensor offsets
		self.offsets = {}
		self.peak = 0

	def alloc(self, tensor, align=256):
		t_id = id(tensor)
		if t_id in self.offsets:
			return

		size = prod(tensor.shape) * np.dtype(tensor.dtype).itemsize
		size = (size + align - 1) // align * align

		# sort by offset
		self.active_blocks.sort(key=lambda x: x[0])
		curr_offset = 0
		alloc_offset = -1
		for off, sz, _ in self.active_blocks:
			if curr_offset + size <= off:
				alloc_offset = curr_offset
				break
			curr_offset = max(curr_offset, off+sz)

		if alloc_offset == -1:
			alloc_offset = curr_offset

		self.active_blocks.append((alloc_offset, size, t_id))
		self.offsets[t_id] = alloc_offset
		self.peak = max(self.peak, alloc_offset+size)

	def free(self, tensor):
		t_id = id(tensor)
		if t_id in self.offsets:
			self.active_blocks = [b for b in self.active_blocks if b[2] != t_id]

	def get_offset(self, tensor):
		return self.offsets.get(id(tensor), None)
 
class Runtime:
	_default_runtime = None

	@classmethod
	def get_default(cls):
		if cls._default_runtime is None:
			cls._default_runtime = Runtime()
		return cls._default_runtime

	def __init__(self, device="cpu", forward_only=False, owns_params=True):
		self.loss = None
		self.topo = []
		self.forward_only = forward_only
		self.owns_params = owns_params
		self.is_compiled = False
		self.unified_arena = None
		self.allocator = Device.get_allocator(device)
		self.backend = Device.get_backend(device)
		self.node_to_clear_grads = {}
		self.need_scratch = (device == "cpu")
		self.grad_scale = 1.0
		self.accumulate_parameter_grads = False
		self.use_fp16_gemm = False

		self.memory_plan = None
		self.arena_bytes = 0

	def backward(self, loss, grad_scale=1.0, accumulate_parameter_grads=False, use_fp16_gemm=False):
		self.loss = loss
		self.grad_scale = float(grad_scale)
		self.accumulate_parameter_grads = bool(accumulate_parameter_grads)
		self.backend.accumulate_parameter_grads = self.accumulate_parameter_grads
		self.backend.grad_scale = self.grad_scale
		self.use_fp16_gemm = use_fp16_gemm
		self.backend.use_fp16_gemm = self.use_fp16_gemm
		if not self.is_compiled:
			self.build_topo(loss)
			self.bind_memory()
			self.is_compiled = True
		self.execute()

	def run_forward(self, output, feeds=None, return_value=True):
		if not self.is_compiled:
			self.loss = output
			self.build_topo(output)
			self.bind_memory()
			self.is_compiled = True

		if feeds is not None:
			for tensor, host_value in feeds.items():
				self.backend.copy_from_host(tensor, host_value)

		self.execute()

		if not return_value:
			return None
		self.backend.synchronize()
		if hasattr(self.backend, "numpy_view"):
			return self.backend.numpy_view(output)

		return self.backend.copy_to_host(output)

	def build_topo(self, loss):
		visited_node = set()
		visited_tensors = set()
		leaves = []

		weight_bytes = 0
		weight_grad_bytes = 0
		def dfs(tensor):
			node = tensor.op
			if node is None:
				return
			if node in visited_node:
				return
			visited_node.add(node)
			for src in node.inputs:
				dfs(src)

			for t in node.inputs + node.outputs:
				if t not in visited_tensors:
					visited_tensors.add(t)
					if t.op is None:
						leaves.append(t)
					if not self.forward_only and t.requires_grad and t.grad is None:
						grad_dtype = t.grad_dtype if t.op is None and t.requires_grad else t.dtype
						t.grad = Tensor(shape=t.shape, dtype=t.dtype, grad_dtype=grad_dtype, requires_grad=False, device=t.device)

			# print(f"add node {id(node)}")
			self.topo.append(node)

		dfs(loss)

		metal_params = [t for t in leaves if self.owns_params and not self.forward_only and t.requires_grad and t.grad is not None 
			and hasattr(self.allocator, "ctx") and np.dtype(t.dtype) == np.dtype(np.float32)]
		metal_param_ids = {id(t) for t in metal_params}
		if metal_params:
			order = [t for t in metal_params if t.ndim > 1] + [t for t in metal_params if t.ndim <= 1]
			align = 16
			max_bytes = min(int(self.allocator.ctx.max_buffer_length()), ((2**32-1)*4)) // align * align
			plans = []
			current = []
			cursor = 0
			for t in order:
				nbytes = prod(t.shape) * np.dtype(t.dtype).itemsize
				offset = (cursor + align - 1) // align * align
				if current and offset + nbytes > max_bytes:
					plans.append((cursor, current))
					current = []
					cursor = 0
					offset = 0
				if nbytes > max_bytes:
					raise RuntimeError(f"parameter needs {nbytes} bytes, exceeding Metal maxBufferLength {max_bytes}")
				current.append((t, offset))
				cursor = offset + nbytes
			if current:
				plans.append((cursor, current))

			for nbytes, items in plans:
				parameter_storage = self.allocator.empty((nbytes//4,), np.float32)
				# gradient_storage = self.allocator.empty((nbytes//4,), np.float32)
				gradient_offsets = []
				gradient_cursor = 0
				for t, _ in items:
					grad_is_bf16 = str(t.grad_dtype).lower() in ("bf16", "bfloat16")
					grad_itemsize = 2 if grad_is_bf16 else 4
					gradient_offset = (gradient_cursor + align - 1) // align * align
					gradient_nbytes = prod(t.shape) * grad_itemsize
					gradient_offsets.append((id(t), gradient_offset, gradient_nbytes, grad_is_bf16))
					gradient_cursor = gradient_offset + gradient_nbytes
				gradient_arena_bytes = (gradient_cursor + align - 1) // align * align
				gradient_storage = self.allocator.empty((gradient_arena_bytes,), np.uint8)

				self.backend.fill_storage(parameter_storage, 0.0)
				self.backend.fill_storage2(gradient_storage, 0.0)
				gradient_offset_by_id = {tensor_id: (offset, nbytes, is_bf16) for (tensor_id, offset, nbytes, is_bf16) in gradient_offsets}
				for t, offset in items:
					t.storage = self.allocator.view(parameter_storage, offset, t.shape, t.dtype)
					# t.grad.storage = self.allocator.view(gradient_storage, offset, t.shape, t.dtype)
					gradient_offset, gradient_nbytes, grad_is_bf16 = gradient_offset_by_id[id(t)]
					gradient_storage_dtype = np.uint16 if grad_is_bf16 else np.float32
					t.grad.storage = self.allocator.view(gradient_storage, gradient_offset, t.shape, gradient_storage_dtype)
					self.backend.init(t)

					weight_bytes += t.storage.nbytes
					weight_grad_bytes += t.grad.storage.nbytes

		for t in leaves:
			if id(t) in metal_param_ids:
				continue
			if t.requires_grad and not self.owns_params:
				if t.storage is None:
					raise RuntimeError(f"shared parameter {t.name} has no storage")
				if not self.forward_only:
					if t.grad.storage is None or t.grad.storage is None:
						raise RuntimeError(f"shared parameter {t.name} has no gradient storage")
				continue
			if t.storage is None:
				t.storage = self.allocator.alloc(t.shape, t.dtype)
				self.backend.init(t)
				if t.requires_grad:
					weight_bytes += t.storage.nbytes

			if not self.forward_only and t.grad is not None and t.grad.storage is None:
				t.grad.storage = self.allocator.alloc(t.grad.shape, t.grad.dtype)
				self.backend.init(t.grad)
				if t.requires_grad:
					weight_grad_bytes += t.grad.storage.nbytes

		print(
			"[MODEL-WEIGHT-MEMORY] "
			f"weight_bytes={weight_bytes} "
			f"weight_grad_bytes={weight_grad_bytes} "
			f"weight_mib={weight_bytes / (1024 ** 2):.2f} "
			f"weight_grad_mib={weight_grad_bytes / (1024 ** 2):.2f} "
			f"weight_gib={weight_bytes / (1024 ** 3):.4f} "
			f"weight_grad_gib={weight_grad_bytes / (1024 ** 3):.4f}",
			flush=True,
		)

		for node in self.topo:
			node.forward = self.backend.get_forward_kernel(node.op)
			node.backward = self.backend.get_backward_kernel(node.op)
			fwd, bwd = plan_node_scratch(node, self.use_fp16_gemm)
			if fwd is not None: node.scratch_fwd_layout = fwd
			if bwd is not None: node.scratch_bwd_layout = bwd

			if self.need_scratch == False: # mps
				from op import MatMulOp, AddOp
				if isinstance(node.op, MatMulOp):
					node.forward = self.backend.bind_matmul_forward(node)
					node.backward = self.backend.bind_matmul_backward(node, None)
				elif isinstance(node.op, AddOp):
					node.forward = self.backend.bind_add_forward(node)
					node.backward = self.backend.bind_add_backward(node)

	def find_single_add_consumer_output(self, tensor):
		for node in self.topo:
			if node.op.op_name == 'Add' and tensor in node.inputs and len(node.outputs) == 1:
				out = node.outputs[0]
				if out.shape == tensor.shape:
					return out
		return None

	def find_single_consumer_output(self, tensor, consumer_op_name):
		for node in self.topo:
			if tensor in node.inputs and len(node.outputs) == 1:
				if node.op.op_name == consumer_op_name:
					return node.outputs[0] 
		return None

	def bind_memory(self):
		self.node_to_clear_grads = {node: [] for node in self.topo}
		execute_seqs = []
		cleared = set()
		for node in self.topo:
			execute_seqs.append((0, node)) # 0 for forward
		if not self.forward_only:
			for node in reversed(self.topo):
				execute_seqs.append((1, node)) # 1 for backward
				for x in node.inputs:
					if x.requires_grad and x.op is not None and x not in cleared:
						self.node_to_clear_grads[node].append(x)
						cleared.add(x)

		tensor_lifetimes = {}
		scratch_tensors = {}
		for step, (phase, node) in enumerate(execute_seqs):
			if phase == 0: # forward
				for y in node.outputs: # activation
					# print(f"forward: node {id(node)} output tensor {id(y)}")
					if y not in tensor_lifetimes:
						tensor_lifetimes[y] = [step, step]
					else:
						tensor_lifetimes[y][1] = max(tensor_lifetimes[y][1], step)
				for x in node.inputs: # save for backward
					# print(f"forward: node {id(node)} input tensor {id(x)}")
					if x in tensor_lifetimes:
						tensor_lifetimes[x][1] = max(tensor_lifetimes[x][1], step)

				if node.op.op_name in ("FusedLinearCrossEntropy", "FusedRMSNormLinearCrossEntropy"):
					xg = node.inputs[0].grad
					if xg is not None and node.inputs[0].op is not None:
						if xg not in tensor_lifetimes:
							tensor_lifetimes[xg] = [step, step]
						else:
							tensor_lifetimes[xg][0] = min(tensor_lifetimes[xg][0], step)

				if (self.need_scratch or type(node.op).__name__ in METAL_SCRATCH_OPS) and node.scratch_fwd_layout is not None:
					fake = Tensor(shape=(node.scratch_fwd_layout.nbytes,), dtype=np.uint8, requires_grad=False, device=self.loss.device)
					# print(f"forward: add scratch layout {id(fake)}")
					scratch_tensors[fake] = [phase, node]
					tensor_lifetimes[fake] = [step, step]
			else: # backward
				needed = node.op.save_for_backward(node.inputs, node.outputs)
				for t in needed: # activation needed by backward
					# print(f"backward: node {id(node)} need tensor {id(t)}")
					if t in tensor_lifetimes:
						tensor_lifetimes[t][1] = max(tensor_lifetimes[t][1], step)

				if node.op.op_name == "FusedRMSNormMLP" and node.inputs:
					x_after_attn = node.inputs[0]
					fla_node = x_after_attn.op 
					if fla_node is not None and fla_node.op.op_name == "FusedLinearAdd" and len(fla_node.inputs) == 4:
						attn_out = fla_node.inputs[1]
						if attn_out.op is not None and attn_out.op.op.op_name in ("FusedRMSNormFlashAttention", "FusedRMSNormGQA") and attn_out.grad is not None:
							if attn_out.grad not in tensor_lifetimes:
								tensor_lifetimes[attn_out.grad] = [step, step]
							else:
								tensor_lifetimes[attn_out.grad][0] = min(tensor_lifetimes[attn_out.grad][0], step)

				for t in node.inputs + node.outputs: # grad tensors
					if t.grad is not None:
						if t.op is None: # leaf node(model parameter grad) has own memory
							continue
						# print(f"backward: tensor {id(t)} with grad {id(t.grad)}")
						if t.grad not in tensor_lifetimes:
							tensor_lifetimes[t.grad] = [step, step]
						else:
							tensor_lifetimes[t.grad][1] = step 

				if (self.need_scratch or type(node.op).__name__ in METAL_SCRATCH_OPS) and node.scratch_bwd_layout is not None:
					fake = Tensor(shape=(node.scratch_bwd_layout.nbytes,), dtype=np.uint8, requires_grad=False, device=self.loss.device)
					# print(f"backward: add scratch layout {id(fake)}")
					scratch_tensors[fake] = [phase, node]
					tensor_lifetimes[fake] = [step, step]

		if self.loss in tensor_lifetimes:
			tensor_lifetimes[self.loss][1] = len(execute_seqs)

		active_tensors = {t: lifetime for t, lifetime in tensor_lifetimes.items() if t.numel > 0 and lifetime[1] >= lifetime[0]}
		if not active_tensors:
			self.unified_arena = None
			return

		if self.allocator.device == "mps":
			for mlp_node in self.topo:
				if mlp_node.op.op_name != "FusedRMSNormMLP":
					continue
				if not mlp_node.inputs:
					continue
				x_after_attn = mlp_node.inputs[0]
				fla_node = x_after_attn.op 
				if fla_node is None or fla_node.op.op_name != "FusedLinearAdd" or len(fla_node.inputs) != 4:
					continue
				x_in, attn_out, proj_w, proj_b = fla_node.inputs
				fa_node = attn_out.op
				if fa_node is None or fa_node.op.op_name not in ("FusedRMSNormFlashAttention", "FusedRMSNormGQA"):
					continue
				if x_after_attn.shape != x_in.shape or x_after_attn.dtype != x_in.dtype:
					continue
				has_residual_add_consumer = False
				for consumer in self.topo:
					if consumer.op.op_name != "Add":
						continue
					if x_after_attn not in consumer.inputs:
						continue
					if len(consumer.outputs) != 1:
						continue
					if consumer.outputs[0].shape != x_after_attn.shape:
						continue
					has_residual_add_consumer = True
					break

				if not has_residual_add_consumer:
					continue
				lifetime = active_tensors.get(x_after_attn)
				if lifetime is None:
					raise RuntimeError("recomputable x_after_attn has no lifetime")
				forward_last_use = lifetime[0]
				for forward_step, consumer in enumerate(self.topo):
					if x_after_attn in consumer.inputs:
						forward_last_use = max(forward_last_use, forward_step)
				lifetime[1] = forward_last_use

		# FusedLinearAdd output -> residual alias, 只有residual的原始值在FLA forward后彻底死亡时才允许output覆盖residual
		linear_add_alias_outputs = []
		# linear_add_alias_output_ids = set()
		for node in self.topo:
			if node.op.op_name != "FusedLinearAdd":
				continue
			base = node.inputs[0]
			out = node.outputs[0]
			base_lifetime = active_tensors.get(base)
			out_lifetime = active_tensors.get(out)
			if base_lifetime is None or out_lifetime is None:
				continue
			alias_write_step = out_lifetime[0]
			base_value_dead_after_fla = base_lifetime[1] <= alias_write_step
			same_layout = (base.shape == out.shape and base.dtype == out.dtype)
			if not base_value_dead_after_fla or not same_layout:
				continue
			base_lifetime[0] = min(base_lifetime[0], out_lifetime[0])
			base_lifetime[1] = max(base_lifetime[1], out_lifetime[1])
			del active_tensors[out]
			linear_add_alias_outputs.append(out)
			# linear_add_alias_output_ids.add(id(out))

		# FusedRMSNormMLP output -> residual Add output, MLP backward会重算MLP output,因此MLP output不需要forward/backward保存
		mlp_recompute_outputs = [] # FusedRMSNormMLP -> Add
		# attn_recompute_outputs = [] # FusedRMSNormFlashAttention -> FusedLinearAdd
		for node in self.topo:
			if node.op.op_name != "FusedRMSNormMLP":
				continue
			out = node.outputs[0]
			consumer_out = self.find_single_consumer_output(out, "Add")
			if consumer_out is not None:
				mlp_recompute_outputs.append(out)

		for t in mlp_recompute_outputs:
			active_tensors.pop(t, None)

		planner = MemoryPlanner()
		all_tensors_in_planner = list(active_tensors.keys())
		id_to_tensor = {id(t): t for t in all_tensors_in_planner}
		max_step = max(end for start, end in active_tensors.values()) if active_tensors else 0
		for step in range(max_step+1):
			# free
			for t in list(planner.offsets.keys()):
				tensor_obj = id_to_tensor.get(t)
				if tensor_obj and tensor_obj in active_tensors and active_tensors[tensor_obj][1] == step - 1:
					planner.free(tensor_obj)
			# alloc
			for t in all_tensors_in_planner:
				if t in active_tensors and active_tensors[t][0] == step:
					if t.op is not None and t.op.op.op_name in VIEW_OPS:
						base_tensor = t.op.inputs[0]
						base_offset = planner.get_offset(base_tensor)
						if base_offset is not None:
							planner.offsets[id(t)] = base_offset
							if t.grad is not None and base_tensor.grad is not None: # grad share
								base_grad_offset = planner.get_offset(base_tensor.grad)
								if base_grad_offset is not None:
									planner.offsets[id(t.grad)] = base_grad_offset
					else:
						planner.alloc(t)
						# print(f"allocate for tensor {id(t)} shape {t.shape}")

		print(f"Training arena need {planner.peak}")
		self.unified_arena = self.allocator.empty((planner.peak,), dtype=np.uint8) if planner.peak > 0 else None
		buffer_id = getattr(self.unified_arena.ptr, "buffer_id", None,)
		print(f"[ARENA-TRAIN-ALLOC] bytes={planner.peak} buffer_id={buffer_id}", flush=True,)

		# mlp_out与其他Add consumer output共享storage
		for t in mlp_recompute_outputs:
			consumer_out = self.find_single_consumer_output(t, 'Add')
			if consumer_out is None:
				raise RuntimeError(f"recompute output {id(t)} has no matching Add consumer")
			off = planner.get_offset(consumer_out)
			if off is None:
				raise RuntimeError(f"consumer output {id(consumer_out)} not allocated for recompute alias")
			t.storage = self.allocator.view(self.unified_arena, off, t.shape, t.dtype)
			planner.offsets[id(t)] = off 

		# 合法的FusedLinearAddd output -> residual alias
		for t in linear_add_alias_outputs:
			base = t.op.inputs[0]
			base_offset = planner.get_offset(base)
			if base_offset is None:
				raise RuntimeError(f"FusedLinearAdd output {id(t)} has no base offset")
			t.storage = self.allocator.view(self.unified_arena, base_offset, t.shape, t.dtype)
			planner.offsets[id(t)] = base_offset
		
		# bind activation
		for t in active_tensors.keys():
			offset = planner.get_offset(t)
			if offset is None:
				if t.op is not None and t.op.op.op_name in VIEW_OPS: # for View Op
					base_tensor = t.op.inputs[0]
					if base_tensor.storage is not None:
						t.storage = self.allocator.view(base_tensor.storage, 0, t.shape, t.dtype)
				continue
			storage = self.allocator.view(self.unified_arena, offset, t.shape, t.dtype)
			if t in scratch_tensors:
				phase, node = scratch_tensors[t]
				scratch_views = self.scratch_bind(storage, node.scratch_fwd_layout if phase == 0 else node.scratch_bwd_layout)
				from op import MatMulOp
				if isinstance(node.op, MatMulOp):
					node.forward = self.backend.bind_matmul_forward(node)
					node.backward = self.backend.bind_matmul_backward(node, scratch_views)
				else:
					if phase == 0:
						node.forward = functools.partial(node.forward, scratch=scratch_views)
					else:
						node.backward = functools.partial(node.backward, scratch=scratch_views)
			else:
				t.storage = storage
				if t.grad is not None and t.op is not None: # activation grad
					grad_offset = planner.get_offset(t.grad)
					if grad_offset is not None:
						t.grad.storage = self.allocator.view(self.unified_arena, grad_offset, t.grad.shape, t.grad.dtype)
				if t.op is not None and t.op.op.op_name not in VIEW_OPS:
					self.backend.init(t)

	def plan_memory(self):
		if self.memory_plan is not None:
			return self.memory_plan

		self.node_to_clear_grads = {node: [] for node in self.topo}
		execute_seqs = []
		cleared = set()
		for node in self.topo:
			execute_seqs.append((0, node)) # 0 for forward
		if not self.forward_only:
			for node in reversed(self.topo):
				execute_seqs.append((1, node)) # 1 for backward
				for x in node.inputs:
					if x.requires_grad and x.op is not None and x not in cleared:
						self.node_to_clear_grads[node].append(x)
						cleared.add(x)

		tensor_lifetimes = {}
		scratch_tensors = {}
		for step, (phase, node) in enumerate(execute_seqs):
			if phase == 0: # forward
				for y in node.outputs: # activation
					# print(f"forward: node {id(node)} output tensor {id(y)}")
					if y not in tensor_lifetimes:
						tensor_lifetimes[y] = [step, step]
					else:
						tensor_lifetimes[y][1] = max(tensor_lifetimes[y][1], step)
				for x in node.inputs: # save for backward
					# print(f"forward: node {id(node)} input tensor {id(x)}")
					if x in tensor_lifetimes:
						tensor_lifetimes[x][1] = max(tensor_lifetimes[x][1], step)

				if node.op.op_name == "FusedLinearCrossEntropy":
					xg = node.inputs[0].grad
					if xg is not None and node.inputs[0].op is not None:
						if xg not in tensor_lifetimes:
							tensor_lifetimes[xg] = [step, step]
						else:
							tensor_lifetimes[xg][0] = min(tensor_lifetimes[xg][0], step)

				if (self.need_scratch or type(node.op).__name__ in METAL_SCRATCH_OPS) and node.scratch_fwd_layout is not None:
					fake = Tensor(shape=(node.scratch_fwd_layout.nbytes,), dtype=np.uint8, requires_grad=False, device=self.loss.device)
					# print(f"forward: add scratch layout {id(fake)}")
					scratch_tensors[fake] = [phase, node]
					tensor_lifetimes[fake] = [step, step]
			else: # backward
				needed = node.op.save_for_backward(node.inputs, node.outputs)
				for t in needed: # activation needed by backward
					# print(f"backward: node {id(node)} need tensor {id(t)}")
					if t in tensor_lifetimes:
						tensor_lifetimes[t][1] = max(tensor_lifetimes[t][1], step)

				if node.op.op_name == "FusedRMSNormMLP" and node.inputs:
					x_after_attn = node.inputs[0]
					fla_node = x_after_attn.op 
					if fla_node is not None and fla_node.op.op_name == "FusedLinearAdd" and len(fla_node.inputs) == 4:
						attn_out = fla_node.inputs[1]
						if attn_out.op is not None and attn_out.op.op.op_name in ("FusedRMSNormFlashAttention", "FusedRMSNormGQA") and attn_out.grad is not None:
							if attn_out.grad not in tensor_lifetimes:
								tensor_lifetimes[attn_out.grad] = [step, step]
							else:
								tensor_lifetimes[attn_out.grad][0] = min(tensor_lifetimes[attn_out.grad][0], step)

				for t in node.inputs + node.outputs: # grad tensors
					if t.grad is not None:
						if t.op is None: # leaf node(model parameter grad) has own memory
							continue
						# print(f"backward: tensor {id(t)} with grad {id(t.grad)}")
						if t.grad not in tensor_lifetimes:
							tensor_lifetimes[t.grad] = [step, step]
						else:
							tensor_lifetimes[t.grad][1] = step 

				if (self.need_scratch or type(node.op).__name__ in METAL_SCRATCH_OPS) and node.scratch_bwd_layout is not None:
					fake = Tensor(shape=(node.scratch_bwd_layout.nbytes,), dtype=np.uint8, requires_grad=False, device=self.loss.device)
					# print(f"backward: add scratch layout {id(fake)}")
					scratch_tensors[fake] = [phase, node]
					tensor_lifetimes[fake] = [step, step]

		if self.loss in tensor_lifetimes:
			tensor_lifetimes[self.loss][1] = len(execute_seqs)

		active_tensors = {t: lifetime for t, lifetime in tensor_lifetimes.items() if t.numel > 0 and lifetime[1] >= lifetime[0]}
		if not active_tensors:
			self.unified_arena = None
			return

		if self.allocator.device == "mps":
			for mlp_node in self.topo:
				if mlp_node.op.op_name != "FusedRMSNormMLP":
					continue
				if not mlp_node.inputs:
					continue
				x_after_attn = mlp_node.inputs[0]
				fla_node = x_after_attn.op 
				if fla_node is None or fla_node.op.op_name != "FusedLinearAdd" or len(fla_node.inputs) != 4:
					continue
				x_in, attn_out, proj_w, proj_b = fla_node.inputs
				fa_node = attn_out.op
				if fa_node is None or fa_node.op.op_name not in ("FusedRMSNormFlashAttention", "FusedRMSNormGQA"):
					continue
				if x_after_attn.shape != x_in.shape or x_after_attn.dtype != x_in.dtype:
					continue
				has_residual_add_consumer = False
				for consumer in self.topo:
					if consumer.op.op_name != "Add":
						continue
					if x_after_attn not in consumer.inputs:
						continue
					if len(consumer.outputs) != 1:
						continue
					if consumer.outputs[0].shape != x_after_attn.shape:
						continue
					has_residual_add_consumer = True
					break

				if not has_residual_add_consumer:
					continue
				lifetime = active_tensors.get(x_after_attn)
				if lifetime is None:
					raise RuntimeError("recomputable x_after_attn has no lifetime")
				forward_last_use = lifetime[0]
				for forward_step, consumer in enumerate(self.topo):
					if x_after_attn in consumer.inputs:
						forward_last_use = max(forward_last_use, forward_step)
				lifetime[1] = forward_last_use

		# FusedLinearAdd output -> residual alias, 只有residual的原始值在FLA forward后彻底死亡时才允许output覆盖residual
		linear_add_alias_outputs = []
		# linear_add_alias_output_ids = set()
		for node in self.topo:
			if node.op.op_name != "FusedLinearAdd":
				continue
			base = node.inputs[0]
			out = node.outputs[0]
			base_lifetime = active_tensors.get(base)
			out_lifetime = active_tensors.get(out)
			if base_lifetime is None or out_lifetime is None:
				continue
			alias_write_step = out_lifetime[0]
			base_value_dead_after_fla = base_lifetime[1] <= alias_write_step
			same_layout = (base.shape == out.shape and base.dtype == out.dtype)
			if not base_value_dead_after_fla or not same_layout:
				continue
			base_lifetime[0] = min(base_lifetime[0], out_lifetime[0])
			base_lifetime[1] = max(base_lifetime[1], out_lifetime[1])
			del active_tensors[out]
			linear_add_alias_outputs.append(out)
			# linear_add_alias_output_ids.add(id(out))

		# FusedRMSNormMLP output -> residual Add output, MLP backward会重算MLP output,因此MLP output不需要forward/backward保存
		mlp_recompute_outputs = [] # FusedRMSNormMLP -> Add
		# attn_recompute_outputs = [] # FusedRMSNormFlashAttention -> FusedLinearAdd
		for node in self.topo:
			if node.op.op_name != "FusedRMSNormMLP":
				continue
			out = node.outputs[0]
			consumer_out = self.find_single_consumer_output(out, "Add")
			if consumer_out is not None:
				mlp_recompute_outputs.append(out)

		for t in mlp_recompute_outputs:
			active_tensors.pop(t, None)

		planner = MemoryPlanner()
		all_tensors_in_planner = list(active_tensors.keys())
		id_to_tensor = {id(t): t for t in all_tensors_in_planner}
		max_step = max(end for start, end in active_tensors.values()) if active_tensors else 0
		for step in range(max_step+1):
			# free
			for t in list(planner.offsets.keys()):
				tensor_obj = id_to_tensor.get(t)
				if tensor_obj and tensor_obj in active_tensors and active_tensors[tensor_obj][1] == step - 1:
					planner.free(tensor_obj)
			# alloc
			for t in all_tensors_in_planner:
				if t in active_tensors and active_tensors[t][0] == step:
					if t.op is not None and t.op.op.op_name in VIEW_OPS:
						base_tensor = t.op.inputs[0]
						base_offset = planner.get_offset(base_tensor)
						if base_offset is not None:
							planner.offsets[id(t)] = base_offset
							if t.grad is not None and base_tensor.grad is not None: # grad share
								base_grad_offset = planner.get_offset(base_tensor.grad)
								if base_grad_offset is not None:
									planner.offsets[id(t.grad)] = base_grad_offset
					else:
						planner.alloc(t)
						# print(f"allocate for tensor {id(t)} shape {t.shape}")

		print(f"Training arena need {planner.peak}")

		self.memory_plan = {
			"planner": planner,
			"active_tensors": active_tensors,
			"scratch_tensors": scratch_tensors,
			"linear_add_alias_outputs": linear_add_alias_outputs,
			"mlp_recompute_outputs": mlp_recompute_outputs,
		}
		self.arena_bytes = planner.peak

		return self.memory_plan

	def bind_memory_new(self, arena=None):
		if self.memory_plan is None:
			self.plan_memory()
		print(f"[ARENA-BIND] required={self.arena_bytes} provided={arena is not None}", flush=True,)
		if arena is None:
			arena = self.allocator.empty((self.arena_bytes,), dtype=np.uint8)
			buffer_id = getattr(arena.ptr, "buffer_id", None,)
			print(f"[ARENA-ALLOC] bytes={arena.nbytes} buffer_id={buffer_id}", flush=True,)
		else:
			buffer_id = getattr(arena.ptr, "buffer_id", None,)
			print(f"[ARENA-REUSE] required={self.arena_bytes} arena_bytes={arena.nbytes} buffer_id={buffer_id}", flush=True,)
		if arena.nbytes < self.arena_bytes:
			raise RuntimeError(f"arena too small: need={self.arena_bytes}, have={arena.nbytes}")
		self.unified_arena = arena
		self._bind_memory_views()

	def _bind_memory_views(self):
		plan = self.memory_plan
		planner = plan["planner"]
		active_tensors = plan["active_tensors"]
		scratch_tensors = plan["scratch_tensors"]
		mlp_recompute_outputs = plan["mlp_recompute_outputs"]
		linear_add_alias_outputs = plan["linear_add_alias_outputs"]

		# mlp_out与其他Add consumer output共享storage
		for t in mlp_recompute_outputs:
			consumer_out = self.find_single_consumer_output(t, 'Add')
			if consumer_out is None:
				raise RuntimeError(f"recompute output {id(t)} has no matching Add consumer")
			off = planner.get_offset(consumer_out)
			if off is None:
				raise RuntimeError(f"consumer output {id(consumer_out)} not allocated for recompute alias")
			t.storage = self.allocator.view(self.unified_arena, off, t.shape, t.dtype)
			planner.offsets[id(t)] = off 

		# 合法的FusedLinearAddd output -> residual alias
		for t in linear_add_alias_outputs:
			base = t.op.inputs[0]
			base_offset = planner.get_offset(base)
			if base_offset is None:
				raise RuntimeError(f"FusedLinearAdd output {id(t)} has no base offset")
			t.storage = self.allocator.view(self.unified_arena, base_offset, t.shape, t.dtype)
			planner.offsets[id(t)] = base_offset
		
		# bind activation
		for t in active_tensors.keys():
			offset = planner.get_offset(t)
			if offset is None:
				if t.op is not None and t.op.op.op_name in VIEW_OPS: # for View Op
					base_tensor = t.op.inputs[0]
					if base_tensor.storage is not None:
						t.storage = self.allocator.view(base_tensor.storage, 0, t.shape, t.dtype)
				continue
			storage = self.allocator.view(self.unified_arena, offset, t.shape, t.dtype)
			if t in scratch_tensors:
				phase, node = scratch_tensors[t]
				scratch_views = self.scratch_bind(storage, node.scratch_fwd_layout if phase == 0 else node.scratch_bwd_layout)
				from op import MatMulOp
				if isinstance(node.op, MatMulOp):
					node.forward = self.backend.bind_matmul_forward(node)
					node.backward = self.backend.bind_matmul_backward(node, scratch_views)
				else:
					if phase == 0:
						node.forward = functools.partial(node.forward, scratch=scratch_views)
					else:
						node.backward = functools.partial(node.backward, scratch=scratch_views)
			else:
				t.storage = storage
				if t.grad is not None and t.op is not None: # activation grad
					grad_offset = planner.get_offset(t.grad)
					if grad_offset is not None:
						t.grad.storage = self.allocator.view(self.unified_arena, grad_offset, t.grad.shape, t.grad.dtype)
				if t.op is not None and t.op.op.op_name not in VIEW_OPS:
					self.backend.init(t)

	def scratch_bind(self, storage, layout):
		if layout is None:
			return None 
		views = {}
		for name, (off, shape, dtype) in layout.slots.items():
			views[name] = self.allocator.view(storage, off, shape, dtype)
		return ScratchViews(views, layout.gemm_configs)

	def execute(self):
		if self.forward_only:
			self.backend.begin_pass()
			for node in self.topo:
				node.forward(node.inputs, node.outputs)
			self.backend.end_pass()
			return

		if self.allocator.device != "mps":
			if not self.accumulate_parameter_grads:
				self.zero_grad()
			self.forward()
			self.backend.fill(self.loss.grad, self.grad_scale)
			self.backward_graph()
			return

		self.backend.begin_pass()
		# Forward
		for node in self.topo:
			node.forward(node.inputs, node.outputs)
		self.backend.fill(self.loss.grad, self.grad_scale)
		# Backward
		for node in reversed(self.topo):
			node.backward(node.inputs, node.outputs)
		self.backend.end_pass()

	def zero_grad(self):
		visited = set()
		for node in self.topo:
			for t in node.inputs + node.outputs:
				if t in visited:
					continue
				visited.add(t)
				if t.grad is not None and t.op is None:
					self.backend.fill(t.grad, 0)

	def forward(self):
		self.backend.begin_pass()
		for node in self.topo:
			node.forward(node.inputs, node.outputs)
		self.backend.end_pass()

	def backward_graph(self):
		self.backend.begin_pass()
		for node in list(reversed(self.topo)):
			for t in self.node_to_clear_grads.get(node, []):
				if t.grad is not None:
					cached = getattr(self.backend, "_recomputed_attention_cache", None)
					if cached is not None and id(t) in cached:
						continue
					self.backend.fill(t.grad, 0)
			node.backward(node.inputs, node.outputs)
		self.backend.end_pass()

@dataclass
class ExecutionPlan:
	runtime: Runtime
	input_ids: Tensor
	target_ids: Tensor|None
	row_scale: Tensor|None
	output: Tensor
	seq_len: int
	mode: str

	def prepare(self):
		self.runtime.loss = self.output
		self.runtime.build_topo(self.output)
		self.runtime.plan_memory()
		self.runtime.is_compiled = True
		return self

	def bind_to(self, arena):
		self.runtime.bind_memory_new(arena)

	def set_inputs(self, values):
		self.runtime.backend.copy_from_host(self.input_ids, values)

	def set_targets(self, values):
		self.runtime.backend.copy_from_host(self.target_ids, values)

	def set_row_scale(self, values):
		self.runtime.backend.copy_from_host(self.row_scale, values)

	def forward(self, feeds=None, return_value=True):
		return self.runtime.run_forward(self.output, feeds=feeds, return_value=return_value)

	def backward(self, grad_scale=1.0, accumulate_parameter_grads=False):
		if self.runtime.forward_only:
			raise ValueError("inference plan cannot use backward()")
		self.runtime.backward(self.output, grad_scale=grad_scale, 
							  accumulate_parameter_grads=accumulate_parameter_grads)

def build_train_plan(model, batch_size, seq_len, device, owns_params):
	runtime = Runtime(device=device, forward_only=False, owns_params=owns_params)

	input_ids = Tensor(shape=(batch_size, seq_len), dtype=np.int32, device=device, requires_grad=False, init_val=None, name=f"train_input_{seq_len}")
	target_ids = Tensor(shape=(batch_size, seq_len), dtype=np.int32, device=device, requires_grad=False, init_val=None, name=f"train_target_{seq_len}")
	row_scale = Tensor(shape=(batch_size*seq_len, 1), dtype=np.float32, device=device, requires_grad=False, init_val=None, name=f"row_scale_{seq_len}")

	loss = model(input_ids, targets=target_ids, mode="train", row_scale=row_scale)

	runtime.loss = loss
	runtime.build_topo(loss)
	runtime.plan_memory()
	runtime.is_compiled = True

	return ExecutionPlan(runtime=runtime, input_ids=input_ids, target_ids=target_ids, row_scale=row_scale, output=loss, seq_len=seq_len, mode="train")

def build_prefill_plan(model, seq_len, device, kv_cache, top_k, temperature, eos_token, bos_token,
	                  rng_state, rng_raw, finished, next_tokens, generated_ids, lengths, decode_position, decode_step, last_token):
	runtime = Runtime(device=device, forward_only=True, owns_params=False)

	input_ids = Tensor(shape=(1, seq_len), dtype=np.int32, device=device, requires_grad=False, init_val=None, name=f"prefill_input_{seq_len}")

	output = model(input_ids, kv_cache=kv_cache, position=kv_cache.position, top_k=top_k, temperature=temperature, 
		           eos_token=eos_token, bos_token=bos_token, rng_state=rng_state, rng_raw=rng_raw, finished=finished, next_tokens=next_tokens,
		           generated_ids=generated_ids, lengths=lengths, decode_position=decode_position, decode_step=decode_step,
		           mode="prefill")

	runtime.loss = output
	runtime.build_topo(output)
	runtime.plan_memory()
	runtime.is_compiled = True

	return ExecutionPlan(runtime=runtime, input_ids=input_ids, target_ids=None, row_scale=None, output=output, seq_len=seq_len, mode="prefill")

def build_decode_plan(model, batch_size, device, kv_cache, top_k, temperature, eos_token, bos_token,
	                  rng_state, rng_raw, finished, next_tokens, generated_ids, lengths, decode_position, decode_step,
	                  state=None, decode_steps=8):
	runtime = Runtime(device=device, forward_only=True, owns_params=False)

	input_ids = next_tokens
	if state is None:
		output = model(input_ids, kv_cache=kv_cache, position=decode_position, top_k=top_k, temperature=temperature,
			           eos_token=eos_token, bos_token=bos_token, rng_state=rng_state, rng_raw=rng_raw, finished=finished, next_tokens=next_tokens,
			           generated_ids=generated_ids, lengths=lengths, decode_position=decode_position, decode_step=decode_step,
			           mode="decode")
	else:
		output = model.decode(input_ids, kv_cache, top_k, temperature, eos_token, bos_token, rng_state, rng_raw,
			                  finished, generated_ids, lengths, state, decode_steps)

	runtime.loss = output
	runtime.build_topo(output)
	runtime.plan_memory()
	runtime.is_compiled = True

	return ExecutionPlan(runtime=runtime, input_ids=input_ids, target_ids=None, row_scale=None, output=output, seq_len=1, mode="decode")
