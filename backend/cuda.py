
import numpy as np
from math import prod
from .base import Backend
from op import (
	GatherOp, FusedLinearAddOp
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
			FusedLinearAddOp: self.cuda_fused_linear_add_forward,
		}
		self.backward_kernels = {
			GatherOp: self.cuda_gather_backward,
			FusedLinearAddOp: self.cuda_fused_linear_add_backward,
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
		cols_work = (C >> 2) if vec4 else C
		self.ctx.dispatch("gather_forward_kernel",
			[indices.storage.ptr.buffer_id, weight.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[indices.storage.ptr.offset, weight.storage.ptr.offset, out.storage.ptr.offset],
			[rows, C, vec4],
			(cols_work + 255) // 256, min(rows, 65535), 1, 256)

	def cuda_gather_backward(self, inputs, outputs, scratch=None):
		indices, weight = inputs[0], inputs[1]
		out = outputs[0]
		if out.grad is None or not weight.requires_grad:
			return
		rows = prod(indices.shape)
		C = weight.shape[-1]
		buf_ids = [indices.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, weight.grad.storage.ptr.buffer_id]
		offsets = [indices.storage.ptr.offset, out.grad.storage.ptr.offset, weight.grad.storage.ptr.offset]
		out_bf16 = int(np.dtype(weight.grad.storage.dtype) == np.dtype(np.uint16))
		self.ctx.dispatch("gather_backward_kernel",
			buf_ids, offsets, [rows, C, out_bf16],
			(C + 255) // 256, min(rows, 65535), 1, 256)

	def cuda_fused_linear_add_forward(self, inputs, outputs, scratch=None):
		x, act, w, b = inputs
		out = outputs[0]
		M = prod(x.shape[:-1])
		K = act.shape[-1]
		N = w.shape[-1]
		b_id = b.storage.ptr.buffer_id if b is not None else 0
		b_off = b.storage.ptr.offset if b is not None else 0
		self.ctx.dispatch("fused_linear_add_forward_kernel",
			[x.storage.ptr.buffer_id, act.storage.ptr.buffer_id, w.storage.ptr.buffer_id, b_id, out.storage.ptr.buffer_id],
			[x.storage.ptr.offset, act.storage.ptr.offset, w.storage.ptr.offset, b_off, out.storage.ptr.offset],
			[M, N, K], (N+63) // 64, (M+63)//64, 1, 256)

	def cuda_fused_linear_add_backward(self, inputs, outputs, scratch=None):
		x, act, w, b = inputs
		out = outputs[0]
		if out.grad is None:
			return
		M = prod(x.shape[:-1])
		K = act.shape[-1]
		N = w.shape[-1]
		accumulate = int(bool(getattr(self, "accumulate_parameter_grads", False)))

		# 1. dW = act^T [K, M] @ dO [M, N]
		if w.requires_grad:
			dw_bf16 = int(np.dtype(w.grad.storage.dtype) == np.dtype(np.uint16))
			self.ctx.dispatch("cuda_sgemm_tn_kernel",
				[act.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
				[act.storage.ptr.offset, out.grad.storage.ptr.offset, w.grad.storage.ptr.offset],
				[K, N, M, accumulate, dw_bf16],
				(N + 63) // 64, (K + 63) // 64, 1, 256)

		# 2. dx = dO
		if x.requires_grad:
			numel = prod(x.shape)
			accum_x = int(x.op is None and bool(accumulate))
			same_buf = (out.grad.storage.ptr.buffer_id == x.grad.storage.ptr.buffer_id and out.grad.storage.ptr.offset == x.grad.storage.ptr.offset)
			if not (x.is_contiguous and not accum_x and same_buf):
				contig = int(bool(x.is_contiguous))
				vec4 = int(contig and numel % 4 == 0 and out.grad.storage.ptr.offset % 16 == 0 and x.grad.storage.ptr.offset % 16 == 0)
				n_work = (numel >> 2) if vec4 else numel
				shape4 = ([1] * (4 - x.ndim) + list(x.shape))[-4:]
				strides4 = ([0] * (4 - x.ndim) + list(x.strides))[-4:]
				x_grad_off = x.grad.storage.ptr.offset if contig else (x.grad.storage.ptr.offset + x.offset * 4)
				self.ctx.dispatch("linear_add_dx_backward_kernel",
					[out.grad.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
					[out.grad.storage.ptr.offset, x_grad_off],
					[numel, contig, vec4, accum_x, shape4[1], shape4[2], shape4[3], strides4[0], strides4[1], strides4[2], strides4[3]],
					(n_work + 255) // 256, 1, 1, 256)

		# 3. dAct = dO[M, N] @ W^T[N, K]
		if act.requires_grad:
			accum_act = int(act.op is None and bool(accumulate))
			self.ctx.dispatch("cuda_sgemm_nt_kernel",
				[out.grad.storage.ptr.buffer_id, w.storage.ptr.buffer_id, act.grad.storage.ptr.buffer_id],
				[out.grad.storage.ptr.offset, w.storage.ptr.offset, act.grad.storage.ptr.offset],
				[M, K, N, accum_act], (K + 63) // 64, (M + 63) // 64, 1, 256)

		# 4. db = sum(dO, axis=0)
		if b is not None and b.requires_grad:
			accum_b = int(b.op is None and bool(accumulate))
			db_bf16 = int(np.dtype(b.grad.storage.dtype) == np.dtype(np.uint16))
			self.ctx.dispatch("bias_grad_reduce_kernel",
				[out.grad.storage.ptr.buffer_id, b.grad.storage.ptr.buffer_id],
				[out.grad.storage.ptr.offset, b.grad.storage.ptr.offset],
				[M, N, accum_b, db_bf16], (N + 7) // 8, 1, 1, 256)
