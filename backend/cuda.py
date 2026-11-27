
import numpy as np
from math import prod
from .base import Backend
from op import (
	GatherOp, FusedLinearAddOp, FusedRMSNormMLPOp, FusedRMSNormFlashAttentionOp
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
			FusedRMSNormMLPOp: self.cuda_fused_rmsnorm_mlp_forward,
			FusedRMSNormFlashAttentionOp: self.cuda_fused_rms_fa_forward,
		}
		self.backward_kernels = {
			GatherOp: self.cuda_gather_backward,
			FusedLinearAddOp: self.cuda_fused_linear_add_backward,
			FusedRMSNormMLPOp: self.cuda_fused_rmsnorm_mlp_backward,
			FusedRMSNormFlashAttentionOp: self.cuda_fused_rms_fa_backward,
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
			if act.op is not None and act.op.op.op_name == "FusedRMSNormFlashAttention":
				act_storage = act.grad.storage
				self.cuda_recompute_fused_rms_fa(act, scratch["fa_k"], scratch["fa_v"], act_storage)
			else:
				act_storage = act.storage
			dw_bf16 = int(np.dtype(w.grad.storage.dtype) == np.dtype(np.uint16))
			self.ctx.dispatch("cuda_sgemm_tn_kernel",
				[act_storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
				[act_storage.ptr.offset, out.grad.storage.ptr.offset, w.grad.storage.ptr.offset],
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

	def cuda_fused_rmsnorm_mlp_forward(self, inputs, outputs, scratch=None):
		if len(inputs) == 3:
			x, wgu, wdown = inputs
			norm_weight = None
		elif len(inputs) == 4:
			x, norm_weight, wgu, wdown = inputs
		out = outputs[0]

		N = prod(x.shape[:-1])
		C = x.shape[-1]
		H = wgu.shape[-1] // 2
		eps_bits = f32_bits(out.op.op.eps)
		hidden = scratch["hidden"]

		gamma_id = norm_weight.storage.ptr.buffer_id if norm_weight is not None else 0
		gamma_off = norm_weight.storage.ptr.offset if norm_weight is not None else 0

		# 1. X_norm = RMSNorm(x) -> out.storage
		self.ctx.dispatch("rmsnorm_forward_kernel",
			[x.storage.ptr.buffer_id, gamma_id, out.storage.ptr.buffer_id],
			[x.storage.ptr.offset, gamma_off, out.storage.ptr.offset],
			[N, C, eps_bits], (N+3)//4, 1, 1, 128)

		# 2. HIDDEN = SiLU(X_norm @ W_gate) * (X_norm @ W_up) -> hidden
		self.ctx.dispatch("fused_wgu_swiglu_forward_kernel",
			[out.storage.ptr.buffer_id, wgu.storage.ptr.buffer_id, hidden.ptr.buffer_id],
			[out.storage.ptr.offset, wgu.storage.ptr.offset, hidden.ptr.offset],
			[N, H, C], (H + 31) // 32, (N + 63) // 64, 1, 256)

		# 3. OUT = HIDDEN @ Wdown -> out.storage
		self.ctx.dispatch("cuda_sgemm_nn_kernel",
			[hidden.ptr.buffer_id, wdown.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[hidden.ptr.offset, wdown.storage.ptr.offset, out.storage.ptr.offset],
			[N, C, H, 0], (C + 63) // 64, (N + 63) // 64, 1, 256)

	def cuda_fused_rmsnorm_mlp_backward(self, inputs, outputs, scratch=None):
		if len(inputs) == 3:
			x, wgu, wdown = inputs
			norm_weight = None
		elif len(inputs) == 4:
			x, norm_weight, wgu, wdown = inputs
		out = outputs[0]
		if out.grad is None:
			return
		need_dgamma = norm_weight is not None and norm_weight.requires_grad
		if not (x.requires_grad or wgu.requires_grad or wdown.requires_grad or need_dgamma):
			return

		N = prod(x.shape[:-1])
		C = x.shape[-1]
		H2 = wgu.shape[-1]
		H = H2 // 2
		eps_bits = f32_bits(out.op.op.eps)
		accumulate = int(bool(getattr(self, "accumulate_parameter_grads", False)))

		gu = scratch["gu"]
		hidden = scratch["hidden"]
		gamma_id = norm_weight.storage.ptr.buffer_id if norm_weight is not None else 0
		gamma_off = norm_weight.storage.ptr.offset if norm_weight is not None else 0

		# 1. 重算 X_norm -> hidden [N, C]（不借用 x.grad，保护残差梯度）
		self.ctx.dispatch("rmsnorm_forward_kernel",
			[x.storage.ptr.buffer_id, gamma_id, hidden.ptr.buffer_id],
			[x.storage.ptr.offset, gamma_off, hidden.ptr.offset],
			[N, C, eps_bits], (N + 3) // 4, 1, 1, 128)

		# 2. 重算 GU = X_norm @ Wgu -> gu [N, 2H]
		self.ctx.dispatch("cuda_sgemm_nn_kernel",
			[hidden.ptr.buffer_id, wgu.storage.ptr.buffer_id, gu.ptr.buffer_id],
			[hidden.ptr.offset, wgu.storage.ptr.offset, gu.ptr.offset],
			[N, H2, C, 0], (H2 + 63) // 64, (N + 63) // 64, 1, 256)

		# 3. dWdown = HIDDEN^T @ dOUT（支持 FP32/BF16 直接累加写出）
		if wdown.requires_grad:
			self.ctx.dispatch("swiglu_forward_kernel",
				[gu.ptr.buffer_id, hidden.ptr.buffer_id],
				[gu.ptr.offset, hidden.ptr.offset],
				[N, H], (H + 255) // 256, min(N, 65535), 1, 256)
			dwdown_bf16 = int(np.dtype(wdown.grad.storage.dtype) == np.dtype(np.uint16))
			self.ctx.dispatch("cuda_sgemm_tn_kernel",
				[hidden.ptr.buffer_id, out.grad.storage.ptr.buffer_id, wdown.grad.storage.ptr.buffer_id],
				[hidden.ptr.offset, out.grad.storage.ptr.offset, wdown.grad.storage.ptr.offset],
				[H, C, N, accumulate, dwdown_bf16],
				(C + 63) // 64, (H + 63) // 64, 1, 256)

		if not (wgu.requires_grad or x.requires_grad or need_dgamma):
			return

		# 4. dHidden = dOUT @ Wdown^T -> hidden [N, H]
		self.ctx.dispatch("cuda_sgemm_nt_kernel",
			[out.grad.storage.ptr.buffer_id, wdown.storage.ptr.buffer_id, hidden.ptr.buffer_id],
			[out.grad.storage.ptr.offset, wdown.storage.ptr.offset, hidden.ptr.offset],
			[N, H, C, 0], (H + 63) // 64, (N + 63) // 64, 1, 256)

		# 5. 原地计算 dGU = [dGate, dUp] -> gu [N, 2H]
		self.ctx.dispatch("swiglu_backward_inplace_kernel",
			[gu.ptr.buffer_id, hidden.ptr.buffer_id],
			[gu.ptr.offset, hidden.ptr.offset],
			[N, H], (H + 255) // 256, min(N, 65535), 1, 256)

		# 6. dWgu = X_norm^T @ dGU（复用释放后的 hidden 重算 X_norm）
		if wgu.requires_grad:
			self.ctx.dispatch("rmsnorm_forward_kernel",
				[x.storage.ptr.buffer_id, gamma_id, hidden.ptr.buffer_id],
				[x.storage.ptr.offset, gamma_off, hidden.ptr.offset],
				[N, C, eps_bits], (N + 3) // 4, 1, 1, 128)
			dwgu_bf16 = int(np.dtype(wgu.grad.storage.dtype) == np.dtype(np.uint16))
			self.ctx.dispatch("cuda_sgemm_tn_kernel",
				[hidden.ptr.buffer_id, gu.ptr.buffer_id, wgu.grad.storage.ptr.buffer_id],
				[hidden.ptr.offset, gu.ptr.offset, wgu.grad.storage.ptr.offset],
				[C, H2, N, accumulate, dwgu_bf16],
				(H2 + 63) // 64, (C + 63) // 64, 1, 256)

		# 7. dX_norm = dGU @ Wgu^T -> hidden [N, C]，随后执行 RMSNorm 反向累加至 x.grad
		if x.requires_grad or need_dgamma:
			self.ctx.dispatch("cuda_sgemm_nt_kernel",
				[gu.ptr.buffer_id, wgu.storage.ptr.buffer_id, hidden.ptr.buffer_id],
				[gu.ptr.offset, wgu.storage.ptr.offset, hidden.ptr.offset],
				[N, C, H2, 0], (C + 63) // 64, (N + 63) // 64, 1, 256)

			dx_id = x.grad.storage.ptr.buffer_id if x.requires_grad else 0
			dx_off = x.grad.storage.ptr.offset if x.requires_grad else 0
			inv_id = gu.ptr.buffer_id if need_dgamma else 0
			inv_off = gu.ptr.offset if need_dgamma else 0
			accum_x = int(x.op is not None or bool(accumulate))
			self.ctx.dispatch("rmsnorm_backward_kernel",
				[x.storage.ptr.buffer_id, hidden.ptr.buffer_id, gamma_id, dx_id, inv_id],
				[x.storage.ptr.offset, hidden.ptr.offset, gamma_off, dx_off, inv_off],
				[N, C, eps_bits, accum_x], (N + 3) // 4, 1, 1, 128)

			if need_dgamma:
				accum_g = int(norm_weight.op is None and bool(accumulate))
				dg_bf16 = int(np.dtype(norm_weight.grad.storage.dtype) == np.dtype(np.uint16))
				self.ctx.dispatch("rmsnorm_dgamma_kernel",
					[x.storage.ptr.buffer_id, hidden.ptr.buffer_id, gu.ptr.buffer_id, norm_weight.grad.storage.ptr.buffer_id],
					[x.storage.ptr.offset, hidden.ptr.offset, gu.ptr.offset, norm_weight.grad.storage.ptr.offset],
					[N, C, accum_g, dg_bf16], (C + 7) // 8, 1, 1, 256)

	def cuda_recompute_fused_rms_fa(self, act, k_buf, v_buf, out_buf):
		x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = act.op.inputs
		B, T, C = x.shape
		H = act.op.op.n_head
		D = C // H
		N = B * T
		is_causal = int(bool(act.op.op.is_causal))
		eps_bits = f32_bits(act.op.op.eps)
		total_pairs = N*H*(D // 2)

		self.ctx.dispatch("fused_rms_fa_qkv_kernel",
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
			 wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
			 out_buf.ptr.buffer_id, k_buf.ptr.buffer_id, v_buf.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
			 wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
			 out_buf.ptr.offset, k_buf.ptr.offset, v_buf.ptr.offset],
			[B, T, C, H, eps_bits, 1], (C + 63) // 64, (N + 63) // 64, 3, 256)

		self.ctx.dispatch("rope_qk_inplace_kernel",
			[out_buf.ptr.buffer_id, k_buf.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
			[out_buf.ptr.offset, k_buf.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
			[B, H, T, D, 2], (total_pairs + 255) // 256, 1, 1, 256)

		self.ctx.dispatch("flash_attn_fwd_kernel",
			[out_buf.ptr.buffer_id, k_buf.ptr.buffer_id, v_buf.ptr.buffer_id, out_buf.ptr.buffer_id, 0],
			[out_buf.ptr.offset, k_buf.ptr.offset, v_buf.ptr.offset, out_buf.ptr.offset, 0],
			[B, H, T, C, is_causal, 0, 1], (T + 31) // 32, B * H, 1, 256)

	def cuda_fused_rms_fa_forward(self, inputs, outputs, scratch=None):
		x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
		out = outputs[0]
		B, T, C = x.shape
		H = out.op.op.n_head
		D = C // H
		N = B * T
		is_causal = int(bool(out.op.op.is_causal))
		eps_bits = f32_bits(out.op.op.eps)
		total_pairs = N*H*(D//2)
		q, k, v = scratch["q"], scratch["k"], scratch["v"]

		self.ctx.dispatch("fused_rms_fa_qkv_kernel",
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
			 wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
			 q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
			 wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
			 q.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, T, C, H, eps_bits, 0], (C + 63) // 64, (N + 63) // 64, 3, 256)

		self.ctx.dispatch("rope_qk_inplace_kernel",
			[q.ptr.buffer_id, k.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
			[B, H, T, D, 0], (total_pairs + 255) // 256, 1, 1, 256)

		self.ctx.dispatch("flash_attn_fwd_kernel",
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out.storage.ptr.buffer_id, 0],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, out.storage.ptr.offset, 0],
			[B, H, T, C, is_causal, 0, 0], (T + 31) // 32, B * H, 1, 256)

	def cuda_fused_rms_fa_backward(self, inputs, outputs, scratch=None):
		x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
		out = outputs[0]
		if out.grad is None:
			return
		if not any(t.requires_grad for t in (x, wq, bq, wk, bk, wv, bv)):
			return

		B, T, C = x.shape
		H = out.op.op.n_head
		D = C // H
		N = B * T
		is_causal = int(bool(out.op.op.is_causal))
		eps_bits = f32_bits(out.op.op.eps)
		accumulate = int(bool(getattr(self, "accumulate_parameter_grads", False)))
		total_pairs = N * H * (D // 2)

		q, k, v = scratch["q"], scratch["k"], scratch["v"]
		dq, dk, dv = scratch["dq"], scratch["dk"], scratch["dv"]
		lse, d_val = scratch["lse"], scratch["d_val"]

		# 1. 重算 Q, K, V 及前向 RoPE
		self.ctx.dispatch("fused_rms_fa_qkv_kernel",
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
			 wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
			 q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
			 wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
			 q.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, T, C, H, eps_bits, 0], (C + 63) // 64, (N + 63) // 64, 3, 256)

		self.ctx.dispatch("rope_qk_inplace_kernel",
			[q.ptr.buffer_id, k.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
			[B, H, T, D, 0], (total_pairs + 255) // 256, 1, 1, 256)

		# 2. 重算 O（复用 dk）与 LSE，计算 d_val = sum(O * dO)
		self.ctx.dispatch("flash_attn_fwd_kernel",
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, dk.ptr.buffer_id, lse.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, dk.ptr.offset, lse.ptr.offset],
			[B, H, T, C, is_causal, 1, 0], (T + 31) // 32, B * H, 1, 256)

		self.ctx.dispatch("flash_attn_bwd_prepare_kernel",
			[dk.ptr.buffer_id, out.grad.storage.ptr.buffer_id, d_val.ptr.buffer_id],
			[dk.ptr.offset, out.grad.storage.ptr.offset, d_val.ptr.offset],
			[B, H, T, C], (N * H + 3) // 4, 1, 1, 128)

		# 3. 注意力反向写出紧凑 [B, T, C] 布局的 dQ_rot, dK_rot, dV，并原地执行反向 RoPE
		self.ctx.dispatch("flash_attn_bwd_kernel",
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out.grad.storage.ptr.buffer_id,
			 dq.ptr.buffer_id, dk.ptr.buffer_id, dv.ptr.buffer_id, lse.ptr.buffer_id, d_val.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, out.grad.storage.ptr.offset,
			 dq.ptr.offset, dk.ptr.offset, dv.ptr.offset, lse.ptr.offset, d_val.ptr.offset],
			[B, H, T, C, is_causal], (T + 15) // 16, B * H, 2, 256)

		self.ctx.dispatch("rope_qk_inplace_kernel",
			[dq.ptr.buffer_id, dk.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
			[dq.ptr.offset, dk.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
			[B, H, T, D, 1], (total_pairs + 255) // 256, 1, 1, 256)

		# 4. 复用 q 暂存 X_norm，直接复用 cuda_sgemm_tn_kernel 计算 FP32/BF16 dWq, dWk, dWv
		if wq.requires_grad or wk.requires_grad or wv.requires_grad:
			self.ctx.dispatch("rmsnorm_forward_kernel",
				[x.storage.ptr.buffer_id, 0, q.ptr.buffer_id],
				[x.storage.ptr.offset, 0, q.ptr.offset],
				[N, C, eps_bits], (N + 3) // 4, 1, 1, 128)
			for weight, dproj in ((wq, dq), (wk, dk), (wv, dv)):
				if weight.requires_grad:
					dw_bf16 = int(np.dtype(weight.grad.storage.dtype) == np.dtype(np.uint16))
					self.ctx.dispatch("cuda_sgemm_tn_kernel",
						[q.ptr.buffer_id, dproj.ptr.buffer_id, weight.grad.storage.ptr.buffer_id],
						[q.ptr.offset, dproj.ptr.offset, weight.grad.storage.ptr.offset],
						[C, C, N, accumulate, dw_bf16], (C + 63) // 64, (C + 63) // 64, 1, 256)

		# 5. 直接复用 bias_grad_reduce_kernel 计算 FP32/BF16 dbq, dbk, dbv
		for bias, dproj in ((bq, dq), (bk, dk), (bv, dv)):
			if bias.requires_grad:
				accum_b = int(bias.op is None and bool(accumulate))
				db_bf16 = int(np.dtype(bias.grad.storage.dtype) == np.dtype(np.uint16))
				self.ctx.dispatch("bias_grad_reduce_kernel",
					[dproj.ptr.buffer_id, bias.grad.storage.ptr.buffer_id],
					[dproj.ptr.offset, bias.grad.storage.ptr.offset],
					[N, C, accum_b, db_bf16], (C + 7) // 8, 1, 1, 256)

		# 6. 复用 k 累加 dX_norm = dQ@Wq^T + dK@Wk^T + dV@Wv^T，并经 rmsnorm_backward_kernel 写入 x.grad
		if x.requires_grad:
			for accum_flag, (dproj, weight) in enumerate(((dq, wq), (dk, wk), (dv, wv))):
				self.ctx.dispatch("cuda_sgemm_nt_kernel",
					[dproj.ptr.buffer_id, weight.storage.ptr.buffer_id, k.ptr.buffer_id],
					[dproj.ptr.offset, weight.storage.ptr.offset, k.ptr.offset],
					[N, C, C, int(accum_flag > 0)], (C + 63) // 64, (N + 63) // 64, 1, 256)
			accum_x = int(x.op is not None or bool(accumulate))
			self.ctx.dispatch("rmsnorm_backward_kernel",
				[x.storage.ptr.buffer_id, k.ptr.buffer_id, 0, x.grad.storage.ptr.buffer_id, 0],
				[x.storage.ptr.offset, k.ptr.offset, 0, x.grad.storage.ptr.offset, 0],
				[N, C, eps_bits, accum_x], (N + 3) // 4, 1, 1, 128)
