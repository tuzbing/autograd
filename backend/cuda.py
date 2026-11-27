
import numpy as np
from math import prod
from .base import Backend
from op import (
	GatherOp
)
from utils import f32_bits, bf16_bits


class CudaBackend(Backend):
	def __init__(self):
		super().__init__()
		import cuda_ext
		self.ctx = cuda_ext.get_ctx()
		self._rollout_cache = None
		self._rollout_cache_dirty = True
		self.forward_kernels = {
			GatherOp: self.cuda_gather_forward,
		}
		self.backward_kernels = {
			GatherOp: self.cuda_gather_backward,
		}

	def copy_from_host(self, tensor, host_np):
		if tensor.requires_grad or (tensor.ndim >= 2 and np.dtype(tensor.dtype) == np.dtype(np.float32)):
			self._rollout_cache_dirty = True
		if np.dtype(tensor.storage.dtype) == np.dtype(np.uint16):
			bits = np.asarray(host_np, dtype=np.float32).view(np.uint32)
			rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
			special = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
			nan = special & ((bits & np.uint32(0x007FFFFF)) != np.uint32(0))
			special_bits = (bits >> np.uint32(16)) | np.where(nan, np.uint32(0x0040), np.uint32(0))
			arr = np.ascontiguousarray(np.where(special, special_bits, rounded), dtype=np.uint16)
		else:
			arr = np.ascontiguousarray(host_np, dtype=tensor.dtype)
		self.ctx.copy_bytes_from_host(tensor.storage.ptr.buffer_id, tensor.storage.ptr.offset, arr)

	def copy_to_host(self, tensor):
		st_dtype = np.dtype(tensor.storage.dtype)
		raw = self.ctx.copy_bytes_to_host(
			tensor.storage.ptr.buffer_id, tensor.storage.ptr.offset, prod(tensor.shape) * st_dtype.itemsize
		)
		arr = np.frombuffer(raw, dtype=st_dtype).reshape(tuple(tensor.shape))
		if st_dtype == np.dtype(np.uint16):
			return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)
		return arr.copy()

	def get_forward_kernel(self, op):
		return self.forward_kernels.get(type(op), None)

	def get_backward_kernel(self, op):
		return self.backward_kernels.get(type(op), None)

	def fill_storage(self, storage, value):
		if storage.nbytes == 0:
			return
		if float(value) == 0.0:
			self.ctx.memset_zero(storage.ptr.buffer_id, storage.ptr.offset, storage.nbytes)
			return
		dtype = np.dtype(storage.dtype)
		if dtype == np.dtype(np.uint16):
			bits = bf16_bits(value)
		elif dtype == np.dtype(np.float32):
			bits = f32_bits(value)
		else:
			raise TypeError(f"Unsupported fill_storage dtype: {dtype}")
		numel = prod(storage.shape)
		self.ctx.dispatch("fill_scalar_kernel2",
			[storage.ptr.buffer_id], [storage.ptr.offset], [numel, bits, dtype.itemsize], 
			(numel + 255)//256, 1, 1, 256)

	def fill_storage2(self, storage, value):
		self.fill_storage(storage, value)

	def synchronize(self):
		self.ctx.synchronize()

	def begin_pass(self):
		self.ctx.begin_pass()

	def end_pass(self):
		self.ctx.end_pass()

	def cuda_gather_forward(self, inputs, outputs, scratch=None):
		indices, weight = inputs[0], inputs[1]
		out = outputs[0]

		if out.grad is not None and weight.requires_grad:
			if not getattr(self, "accumulate_parameter_grads", False):
				self.fill_storage(weight.grad.storage, 0.0)

		rows = prod(indices.shape)
		C = weight.shape[-1]
		vec4 = int(C % 4 == 0 and weight.storage.ptr.offset % 16 == 0 and out.storage.ptr.offset % 16 == 0)
		n_work = (rows * C) // 4 if vec4 else (rows * C)
		self.ctx.dispatch("gather_forward_kernel",
			[indices.storage.ptr.buffer_id, weight.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[indices.storage.ptr.offset, weight.storage.ptr.offset, out.storage.ptr.offset],
			[rows, C, vec4],
			(n_work + 255) // 256, 1, 1, 256)

	def cuda_gather_backward(self, inputs, outputs, scratch=None):
		indices, weight = inputs[0], inputs[1]
		out = outputs[0]
		if out.grad is None or not weight.requires_grad:
			return
		rows = prod(indices.shape)
		C = weight.shape[-1]
		buf_ids = [indices.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, weight.grad.storage.ptr.buffer_id]
		offsets = [indices.storage.ptr.offset, out.grad.storage.ptr.offset, weight.grad.storage.ptr.offset]
		if np.dtype(weight.grad.storage.dtype) == np.dtype(np.uint16):
			self.ctx.dispatch("gather_backward_bf16_kernel",
				buf_ids, offsets, [rows, C], (C+127)//128, rows, 1, 32)
		else:
			total = rows * C
			self.ctx.dispatch("gather_backward_kernel",
				buf_ids, offsets, [total, C], (total+255)//256, 1, 1, 256)

	def bind_add_forward(self, node):
		x, y = node.inputs[0], node.inputs[1]
		out = node.outputs[0]
		grid_size = prod(out.shape)
		if x.shape == y.shape:
			def _fwd(inputs, outputs, scratch=None):
				xs, ys, os_ = inputs[0].storage, inputs[1].storage, outputs[0].storage
				if (grid_size % 4 == 0 and xs.ptr.offset % 16 == 0
						and ys.ptr.offset % 16 == 0 and os_.ptr.offset % 16 == 0):
					n4 = grid_size // 4
					self.ctx.dispatch("add_residual_forward_vec4",
						[xs.ptr.buffer_id, ys.ptr.buffer_id, os_.ptr.buffer_id],
						[xs.ptr.offset, ys.ptr.offset, os_.ptr.offset],
						[n4], (n4 + 255) // 256, 1, 1, 256)
				else:
					self.ctx.dispatch("add_residual_forward",
						[xs.ptr.buffer_id, ys.ptr.buffer_id, os_.ptr.buffer_id],
						[xs.ptr.offset, ys.ptr.offset, os_.ptr.offset],
						[grid_size], (grid_size + 255) // 256, 1, 1, 256)
			return _fwd
		elif y.ndim >= 1 and y.shape[-1] == out.shape[-1] and prod(y.shape[:-1]) == 1:
			c_val = out.shape[-1]
			def _fwd(inputs, outputs, scratch=None):
				xs, ys, os_ = inputs[0].storage, inputs[1].storage, outputs[0].storage
				self.ctx.dispatch("add_bias_forward",
					[xs.ptr.buffer_id, ys.ptr.buffer_id, os_.ptr.buffer_id],
					[xs.ptr.offset, ys.ptr.offset, os_.ptr.offset],
					[grid_size, c_val], (grid_size + 255) // 256, 1, 1, 256)
			return _fwd
		elif y.ndim >= 2 and y.shape[-2:] == out.shape[-2:] and prod(y.shape[:-2]) == 1:
			tc_val = out.shape[-2] * out.shape[-1]
			def _fwd(inputs, outputs, scratch=None):
				xs, ys, os_ = inputs[0].storage, inputs[1].storage, outputs[0].storage
				self.ctx.dispatch("add_pos_forward",
					[xs.ptr.buffer_id, ys.ptr.buffer_id, os_.ptr.buffer_id],
					[xs.ptr.offset, ys.ptr.offset, os_.ptr.offset],
					[grid_size, tc_val], (grid_size + 255) // 256, 1, 1, 256)
			return _fwd
		else:
			raise RuntimeError(f"Unsupported Add shape: x.shape={x.shape}, y.shape={y.shape}")

	def bind_add_backward(self, node):
		x, y = node.inputs[0], node.inputs[1]
		out = node.outputs[0]
		grid_size = prod(out.shape)
		if x.shape == y.shape:
			def _bwd(inputs, outputs, scratch=None):
				if outputs[0].grad is None:
					return
				og = outputs[0].grad.storage
				do_x = inputs[0].requires_grad
				do_y = inputs[1].requires_grad
				if do_x and inputs[0].grad.storage.ptr.buffer_id == og.ptr.buffer_id \
						and inputs[0].grad.storage.ptr.offset == og.ptr.offset:
					do_x = False
				if do_y and inputs[1].grad.storage.ptr.buffer_id == og.ptr.buffer_id \
						and inputs[1].grad.storage.ptr.offset == og.ptr.offset:
					do_y = False
				if not do_x and not do_y:
					return

				buf_dx = inputs[0].grad.storage.ptr.buffer_id if do_x else 0
				off_dx = inputs[0].grad.storage.ptr.offset if do_x else 0
				buf_dy = inputs[1].grad.storage.ptr.buffer_id if do_y else 0
				off_dy = inputs[1].grad.storage.ptr.offset if do_y else 0
				if (grid_size % 4 == 0 and og.ptr.offset % 16 == 0
						and (not do_x or off_dx % 16 == 0)
						and (not do_y or off_dy % 16 == 0)):
					n4 = grid_size // 4
					self.ctx.dispatch("add_residual_backward_vec4",
						[og.ptr.buffer_id, buf_dx, buf_dy], [og.ptr.offset, off_dx, off_dy],
						[n4], (n4 + 255) // 256, 1, 1, 256)
				else:
					self.ctx.dispatch("add_residual_backward",
						[og.ptr.buffer_id, buf_dx, buf_dy], [og.ptr.offset, off_dx, off_dy],
						[grid_size], (grid_size + 255) // 256, 1, 1, 256)
			return _bwd
		elif y.ndim >= 1 and y.shape[-1] == out.shape[-1] and prod(y.shape[:-1]) == 1:
			rows_val, c_val = prod(out.shape[:-1]), out.shape[-1]
			def _bwd(inputs, outputs, scratch=None):
				if outputs[0].grad is None:
					return
				og = outputs[0].grad.storage
				if inputs[0].requires_grad:
					xg = inputs[0].grad.storage
					if xg.ptr.buffer_id != og.ptr.buffer_id or xg.ptr.offset != og.ptr.offset:
						self.ctx.dispatch("contiguous_backward_kernel",
							[og.ptr.buffer_id, xg.ptr.buffer_id], [og.ptr.offset, xg.ptr.offset],
							[grid_size], (grid_size + 255) // 256, 1, 1, 256)
				if inputs[1].requires_grad:
					yg = inputs[1].grad.storage
					accum = int(bool(getattr(self, "accumulate_parameter_grads", False)))
					kernel = "bias_grad_reduce_bf16" if np.dtype(yg.dtype) == np.dtype(np.uint16) else "bias_grad_reduce"
					self.ctx.dispatch(kernel,
						[og.ptr.buffer_id, yg.ptr.buffer_id], [og.ptr.offset, yg.ptr.offset],
						[rows_val, c_val, accum], (c_val + 15) // 16, 1, 1, 256)
			return _bwd
		elif y.ndim >= 2 and y.shape[-2:] == out.shape[-2:] and prod(y.shape[:-2]) == 1:
			b_val, tc_val = prod(out.shape[:-2]), out.shape[-2] * out.shape[-1]
			def _bwd(inputs, outputs, scratch=None):
				if outputs[0].grad is None:
					return
				og = outputs[0].grad.storage
				if inputs[0].requires_grad:
					xg = inputs[0].grad.storage
					if xg.ptr.buffer_id != og.ptr.buffer_id or xg.ptr.offset != og.ptr.offset:
						self.ctx.dispatch("contiguous_backward_kernel",
							[og.ptr.buffer_id, xg.ptr.buffer_id], [og.ptr.offset, xg.ptr.offset],
							[grid_size], (grid_size + 255) // 256, 1, 1, 256)
				if inputs[1].requires_grad:
					yg = inputs[1].grad.storage
					accum = int(inputs[1].op is None and bool(getattr(self, "accumulate_parameter_grads", False)))
					self.ctx.dispatch("pos_emb_grad",
						[og.ptr.buffer_id, yg.ptr.buffer_id], [og.ptr.offset, yg.ptr.offset],
						[b_val, tc_val, accum], (tc_val + 255) // 256, 1, 1, 256)
			return _bwd
		else:
			raise RuntimeError(f"Unsupported Add shape: x.shape={x.shape}, y.shape={y.shape}")

