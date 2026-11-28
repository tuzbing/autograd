
import numpy as np
from math import prod
from .base import Backend
from op import (
	GatherOp, FusedLinearAddOp, FusedRMSNormMLPOp, FusedRMSNormFlashAttentionOp, FusedLinearCrossEntropyOp
)
from utils import f32_bits, bf16_bits


class CudaBackend(Backend):
	def __init__(self):
		super().__init__()
		import cuda_ext
		self.ctx = cuda_ext.get_ctx()
		self._num_sms = int(self.ctx.num_sms())
		self._rollout_cache = None
		self._rollout_cache_dirty = True
		self.forward_kernels = {
			GatherOp: self.cuda_gather_forward,
			FusedLinearAddOp: self.cuda_fused_linear_add_forward,
			FusedRMSNormMLPOp: self.cuda_fused_rmsnorm_mlp_forward,
			FusedRMSNormFlashAttentionOp: self.cuda_fused_rms_fa_forward,
			FusedLinearCrossEntropyOp: self.cuda_fused_linear_ce_forward,
		}
		self.backward_kernels = {
			GatherOp: self.cuda_gather_backward,
			FusedLinearAddOp: self.cuda_fused_linear_add_backward,
			FusedRMSNormMLPOp: self.cuda_fused_rmsnorm_mlp_backward,
			FusedRMSNormFlashAttentionOp: self.cuda_fused_rms_fa_backward,
			FusedLinearCrossEntropyOp: self.cuda_fused_linear_ce_backward,
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

	@staticmethod
	def _add_broadcast_mod(x, y, out):
		if x.shape == y.shape:
			return 0
		if x.shape == out.shape:
			if len(y.shape) == 2 or (len(y.shape) == 3 and y.shape[0] == 1) or (len(y.shape) == 4 and y.shape[0] == 1 and y.shape[1] == 1):
				return out.shape[-2] * out.shape[-1]
			if len(y.shape) == 1 or (len(y.shape) == 3 and y.shape[0] == 1 and y.shape[1] == 1) or (len(y.shape) == 4 and y.shape[0] == 1 and y.shape[1] == 1 and y.shape[2] == 1):
				return out.shape[-1]
		raise RuntimeError(f"Unsupported Add shape: x.shape={x.shape}, y.shape={y.shape}")

	def bind_add_forward(self, node):
		x0, y0 = node.inputs[0], node.inputs[1]
		out0 = node.outputs[0]
		numel = prod(out0.shape)
		mod_y = self._add_broadcast_mod(x0, y0, out0)

		def _fwd(inputs, outputs, scratch=None):
			x, y, out = inputs[0], inputs[1], outputs[0]
			vec4 = int(
				numel % 4 == 0
				and (mod_y == 0 or mod_y % 4 == 0)
				and x.storage.ptr.offset % 16 == 0
				and y.storage.ptr.offset % 16 == 0
				and out.storage.ptr.offset % 16 == 0
			)
			n_work = (numel >> 2) if vec4 else numel
			self.ctx.dispatch("add_forward_kernel",
				[x.storage.ptr.buffer_id, y.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
				[x.storage.ptr.offset, y.storage.ptr.offset, out.storage.ptr.offset],
				[numel, mod_y, vec4], (n_work + 255) // 256, 1, 1, 256)
		return _fwd

	def bind_add_backward(self, node):
		x0, y0 = node.inputs[0], node.inputs[1]
		out0 = node.outputs[0]
		numel = prod(out0.shape)
		mod_y = self._add_broadcast_mod(x0, y0, out0)

		def _bwd(inputs, outputs, scratch=None):
			out = outputs[0]
			if out.grad is None:
				return
			x, y = inputs[0], inputs[1]
			accumulate = bool(getattr(self, "accumulate_parameter_grads", False))

			do_x = x.requires_grad
			accum_x = int(x.op is None and accumulate)
			dx_bf16 = int(do_x and np.dtype(x.grad.storage.dtype) == np.dtype(np.uint16))
			if (do_x and not accum_x and not dx_bf16
					and x.grad.storage.ptr.buffer_id == out.grad.storage.ptr.buffer_id
					and x.grad.storage.ptr.offset == out.grad.storage.ptr.offset):
				do_x = False

			if mod_y == 0:
				do_y = y.requires_grad
				accum_y = int(y.op is None and accumulate)
				dy_bf16 = int(do_y and np.dtype(y.grad.storage.dtype) == np.dtype(np.uint16))
				if (do_y and not accum_y and not dy_bf16
						and y.grad.storage.ptr.buffer_id == out.grad.storage.ptr.buffer_id
						and y.grad.storage.ptr.offset == out.grad.storage.ptr.offset):
					do_y = False
				if not do_x and not do_y:
					return
				buf_dx = x.grad.storage.ptr.buffer_id if do_x else 0
				off_dx = x.grad.storage.ptr.offset if do_x else 0
				buf_dy = y.grad.storage.ptr.buffer_id if do_y else 0
				off_dy = y.grad.storage.ptr.offset if do_y else 0
				vec4 = int(
					not dx_bf16 and not dy_bf16
					and numel % 4 == 0
					and out.grad.storage.ptr.offset % 16 == 0
					and (off_dx % 16 == 0 if do_x else True)
					and (off_dy % 16 == 0 if do_y else True)
				)
				n_work = (numel >> 2) if vec4 else numel
				self.ctx.dispatch("add_residual_backward_kernel",
					[out.grad.storage.ptr.buffer_id, buf_dx, buf_dy],
					[out.grad.storage.ptr.offset, off_dx, off_dy],
					[numel, vec4, accum_x, accum_y, dx_bf16, dy_bf16],
					(n_work + 255) // 256, 1, 1, 256)
				return

			if do_x:
				off_dx = x.grad.storage.ptr.offset
				vec4 = int(not dx_bf16 and numel % 4 == 0 and out.grad.storage.ptr.offset % 16 == 0 and off_dx % 16 == 0)
				n_work = (numel >> 2) if vec4 else numel
				self.ctx.dispatch("add_residual_backward_kernel",
					[out.grad.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id, 0],
					[out.grad.storage.ptr.offset, off_dx, 0],
					[numel, vec4, accum_x, 0, dx_bf16, 0],
					(n_work + 255) // 256, 1, 1, 256)

			if y.requires_grad:
				rows_y = numel // mod_y
				accum_y = int(y.op is None and accumulate)
				dy_bf16 = int(np.dtype(y.grad.storage.dtype) == np.dtype(np.uint16))
				self.ctx.dispatch("bias_grad_reduce_kernel",
					[out.grad.storage.ptr.buffer_id, y.grad.storage.ptr.buffer_id],
					[out.grad.storage.ptr.offset, y.grad.storage.ptr.offset],
					[rows_y, mod_y, accum_y, dy_bf16], (mod_y + 7) // 8, 1, 1, 256)
		return _bwd

	def cuda_fused_linear_add_forward(self, inputs, outputs, scratch=None):
		x, act, w, b = inputs
		out = outputs[0]
		M = prod(x.shape[:-1])
		K = act.shape[-1]
		N = w.shape[-1]
		b_id = b.storage.ptr.buffer_id if b is not None else 0
		b_off = b.storage.ptr.offset if b is not None else 0
		self.ctx.dispatch("fused_linear_add_128_kernel",
			[x.storage.ptr.buffer_id, act.storage.ptr.buffer_id, w.storage.ptr.buffer_id, b_id, out.storage.ptr.buffer_id],
			[x.storage.ptr.offset, act.storage.ptr.offset, w.storage.ptr.offset, b_off, out.storage.ptr.offset],
			[M, N, K], (N+127)//128, (M+127)//128, 1, 256)

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
			dw_conf = scratch.gemm_configs["dw"]
			self.ctx.dispatch(dw_conf["kernel"],
				[act_storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
				[act_storage.ptr.offset, out.grad.storage.ptr.offset, w.grad.storage.ptr.offset],
				[K, N, M, accumulate, dw_bf16],
				(N + dw_conf["bn"] - 1) // dw_conf["bn"], (K + dw_conf["bm"] - 1) // dw_conf["bm"], dw_conf["split_k"], 256)

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
			dact_conf = scratch.gemm_configs["dact"]
			if dact_conf["split_k"] > 1 and not accum_act:
				self.ctx.memset_zero(act.grad.storage.ptr.buffer_id, act.grad.storage.ptr.offset, M * K * 4)
			self.ctx.dispatch(dact_conf["kernel"],
				[out.grad.storage.ptr.buffer_id, w.storage.ptr.buffer_id, act.grad.storage.ptr.buffer_id],
				[out.grad.storage.ptr.offset, w.storage.ptr.offset, act.grad.storage.ptr.offset],
				[M, K, N, accum_act],
				(K + dact_conf["bn"] - 1) // dact_conf["bn"], (M + dact_conf["bm"] - 1) // dact_conf["bm"], dact_conf["split_k"], 256) 

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
		self.ctx.dispatch("fused_wgu_swiglu_128_kernel",
			[out.storage.ptr.buffer_id, wgu.storage.ptr.buffer_id, hidden.ptr.buffer_id],
			[out.storage.ptr.offset, wgu.storage.ptr.offset, hidden.ptr.offset],
			[N, H, C], (H+63)//64, (N + 127) // 128, 1, 256)

		# 3. OUT = HIDDEN @ Wdown -> out.storage
		out_conf = scratch.gemm_configs["out"]
		if out_conf["split_k"] > 1:
			self.ctx.memset_zero(out.storage.ptr.buffer_id, out.storage.ptr.offset, N * C * 4)
		self.ctx.dispatch(out_conf["kernel"],
			[hidden.ptr.buffer_id, wdown.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[hidden.ptr.offset, wdown.storage.ptr.offset, out.storage.ptr.offset],
			[N, C, H, 0],
			(C + out_conf["bn"] - 1) // out_conf["bn"], (N + out_conf["bm"] - 1) // out_conf["bm"], out_conf["split_k"], 256)

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

		fla_node = x.op
		recompute_fla = (
			fla_node is not None
			and fla_node.op.op_name == "FusedLinearAdd"
			and len(fla_node.inputs) == 4
			and fla_node.inputs[1].op is not None
			and fla_node.inputs[1].op.op.op_name == "FusedRMSNormFlashAttention"
		)
		if recompute_fla:
			if H2 < 3 * C or H < 2 * C:
				raise RuntimeError(f"MLP scratch too small for FA recompute: C={C}, H={H}")
			x_in, attn_out, proj_w, proj_b = fla_node.inputs
			x_nbytes = N * C * 4
			self.cuda_recompute_fused_rms_fa(
				attn_out,
				(gu.ptr.buffer_id, gu.ptr.offset + x_nbytes),
				(gu.ptr.buffer_id, gu.ptr.offset + 2 * x_nbytes),
				(gu.ptr.buffer_id, gu.ptr.offset),
			)
			b_id = proj_b.storage.ptr.buffer_id if proj_b is not None else 0
			b_off = proj_b.storage.ptr.offset if proj_b is not None else 0
			x_id, x_off = hidden.ptr.buffer_id, hidden.ptr.offset
			work_id, work_off = hidden.ptr.buffer_id, hidden.ptr.offset + x_nbytes
			work_floats = N * (H - C)
			self.ctx.dispatch("fused_linear_add_128_kernel",
				[x_in.storage.ptr.buffer_id, gu.ptr.buffer_id, proj_w.storage.ptr.buffer_id, b_id, x_id],
				[x_in.storage.ptr.offset, gu.ptr.offset, proj_w.storage.ptr.offset, b_off, x_off],
				[N, C, C], (C + 127) // 128, (N + 127) // 128, 1, 256)
		else:
			x_id, x_off = x.storage.ptr.buffer_id, x.storage.ptr.offset
			work_id, work_off = hidden.ptr.buffer_id, hidden.ptr.offset
			work_floats = N * H

		# 1. 重算 X_norm -> hidden [N, C]（不借用 x.grad，保护残差梯度）
		self.ctx.dispatch("rmsnorm_forward_kernel",
			[x_id, gamma_id, work_id],
			[x_off, gamma_off, work_off],
			[N, C, eps_bits], (N + 3) // 4, 1, 1, 128)

		# 2. 重算 GU = X_norm @ Wgu -> gu [N, 2H]
		gu_conf = scratch.gemm_configs["gu"]
		if gu_conf["split_k"] > 1:
			self.ctx.memset_zero(gu.ptr.buffer_id, gu.ptr.offset, N * H2 * 4) 
		self.ctx.dispatch(gu_conf["kernel"],
			[work_id, wgu.storage.ptr.buffer_id, gu.ptr.buffer_id],
			[work_off, wgu.storage.ptr.offset, gu.ptr.offset],
			[N, H2, C, 0],
			(H2 + gu_conf["bn"] - 1) // gu_conf["bn"], (N + gu_conf["bm"] - 1) // gu_conf["bm"], gu_conf["split_k"], 256)

		# 3. dWdown = HIDDEN^T @ dOUT（支持 FP32/BF16 直接累加写出）
		if wdown.requires_grad:
			dwdown_bf16 = int(np.dtype(wdown.grad.storage.dtype) == np.dtype(np.uint16))
			dwdown_itemsize = 2 if dwdown_bf16 else 4
			h_tile = min(H, work_floats // N)
			for h0 in range(0, H, h_tile):
				h_sub = min(h_tile, H - h0)
				self.ctx.dispatch("swiglu_forward_kernel",
					[gu.ptr.buffer_id, work_id],
					[gu.ptr.offset + h0 * 4, work_off],
					[N, H, h_sub], (h_sub + 255) // 256, min(N, 65535), 1, 256)
				dw_conf = scratch.gemm_configs[("dwdown", h_sub)]
				self.ctx.dispatch(dw_conf["kernel"],
					[work_id, out.grad.storage.ptr.buffer_id, wdown.grad.storage.ptr.buffer_id],
					[work_off, out.grad.storage.ptr.offset, wdown.grad.storage.ptr.offset + h0 * C * dwdown_itemsize],
					[h_sub, C, N, accumulate, dwdown_bf16],
					(C + dw_conf["bn"] - 1) // dw_conf["bn"], (h_sub + dw_conf["bm"] - 1) // dw_conf["bm"], dw_conf["split_k"], 256)

		if not (wgu.requires_grad or x.requires_grad or need_dgamma):
			return

		# 4 & 5. 按行分块计算 dHidden = dOUT @ Wdown^T 并原地更新 gu -> dGU [N, 2H]
		chunk_rows = max(1, work_floats // H)
		for r0 in range(0, N, chunk_rows):
			rows = min(chunk_rows, N - r0)
			dh_conf = scratch.gemm_configs[("dhidden", rows)]
			if dh_conf["split_k"] > 1:
				self.ctx.memset_zero(work_id, work_off, rows * H * 4)
			self.ctx.dispatch(dh_conf["kernel"],
				[out.grad.storage.ptr.buffer_id, wdown.storage.ptr.buffer_id, work_id],
				[out.grad.storage.ptr.offset + r0 * C * 4, wdown.storage.ptr.offset, work_off],
				[rows, H, C, 0],
				(H + dh_conf["bn"] - 1) // dh_conf["bn"], (rows + dh_conf["bm"] - 1) // dh_conf["bm"], dh_conf["split_k"], 256)
			self.ctx.dispatch("swiglu_backward_inplace_kernel",
				[gu.ptr.buffer_id, work_id],
				[gu.ptr.offset + r0 * H2 * 4, work_off],
				[rows, H], (H + 255) // 256, min(rows, 65535), 1, 256)

		# 6. dWgu = X_norm^T @ dGU（直接读取常驻的 x_id/x_off，无需二次重算 FA）
		if wgu.requires_grad:
			self.ctx.dispatch("rmsnorm_forward_kernel",
				[x_id, gamma_id, work_id],
				[x_off, gamma_off, work_off],
				[N, C, eps_bits], (N + 3) // 4, 1, 1, 128)
			dwgu_bf16 = int(np.dtype(wgu.grad.storage.dtype) == np.dtype(np.uint16))
			dwgu_conf = scratch.gemm_configs["dwgu"]
			self.ctx.dispatch(dwgu_conf["kernel"],
				[work_id, gu.ptr.buffer_id, wgu.grad.storage.ptr.buffer_id],
				[work_off, gu.ptr.offset, wgu.grad.storage.ptr.offset],
				[C, H2, N, accumulate, dwgu_bf16],
				(H2 + dwgu_conf["bn"] - 1) // dwgu_conf["bn"], (C + dwgu_conf["bm"] - 1) // dwgu_conf["bm"], dwgu_conf["split_k"], 256)

		# 7. dX_norm = dGU @ Wgu^T -> work_buf [N, C]，随后执行 RMSNorm 反向累加至 x.grad
		if x.requires_grad or need_dgamma:
			dx_conf = scratch.gemm_configs["dx_norm"]
			if dx_conf["split_k"] > 1:
				self.ctx.memset_zero(work_id, work_off, N * C * 4)
			self.ctx.dispatch(dx_conf["kernel"],
				[gu.ptr.buffer_id, wgu.storage.ptr.buffer_id, work_id],
				[gu.ptr.offset, wgu.storage.ptr.offset, work_off],
				[N, C, H2, 0],
				(C + dx_conf["bn"] - 1) // dx_conf["bn"], (N + dx_conf["bm"] - 1) // dx_conf["bm"], dx_conf["split_k"], 256)

			dx_id = x.grad.storage.ptr.buffer_id if x.requires_grad else 0
			dx_off = x.grad.storage.ptr.offset if x.requires_grad else 0
			inv_id = gu.ptr.buffer_id if need_dgamma else 0
			inv_off = gu.ptr.offset if need_dgamma else 0
			accum_x = int(x.op is not None or bool(accumulate))
			self.ctx.dispatch("rmsnorm_backward_kernel",
				[x_id, work_id, gamma_id, dx_id, inv_id],
				[x_off, work_off, gamma_off, dx_off, inv_off],
				[N, C, eps_bits, accum_x], (N + 3) // 4, 1, 1, 128)

			if need_dgamma:
				accum_g = int(norm_weight.op is None and bool(accumulate))
				dg_bf16 = int(np.dtype(norm_weight.grad.storage.dtype) == np.dtype(np.uint16))
				self.ctx.dispatch("rmsnorm_dgamma_kernel",
					[x_id, work_id, gu.ptr.buffer_id, norm_weight.grad.storage.ptr.buffer_id],
					[x_off, work_off, gu.ptr.offset, norm_weight.grad.storage.ptr.offset],
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
		out_id, out_off = out_buf if isinstance(out_buf, tuple) else (out_buf.ptr.buffer_id, out_buf.ptr.offset)
		k_id, k_off = k_buf if isinstance(k_buf, tuple) else (k_buf.ptr.buffer_id, k_buf.ptr.offset)
		v_id, v_off = v_buf if isinstance(v_buf, tuple) else (v_buf.ptr.buffer_id, v_buf.ptr.offset)

		self.ctx.dispatch("fused_rms_fa_qkv_128_kernel",
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
			 wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
			 out_id, k_id, v_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
			 wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
			 out_off, k_off, v_off],
			[B, T, C, H, eps_bits, 1], (C + 127) // 128, (N + 127) // 128, 3, 256)

		self.ctx.dispatch("rope_qk_inplace_kernel",
			[out_id, k_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
			[out_off, k_off, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
			[B, H, T, D, 2], (total_pairs + 255) // 256, 1, 1, 256)

		self.ctx.dispatch("flash_attn_fwd_kernel",
			[out_id, k_id, v_id, out_id, 0],
			[out_off, k_off, v_off, out_off, 0],
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

		self.ctx.dispatch("fused_rms_fa_qkv_128_kernel",
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
			 wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
			 q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
			 wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
			 q.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, T, C, H, eps_bits, 0], (C + 127) // 128, (N + 127) // 128, 3, 256)

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
		self.ctx.dispatch("fused_rms_fa_qkv_128_kernel",
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
			 wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
			 q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
			 wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
			 q.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, T, C, H, eps_bits, 0], (C + 127) // 128, (N + 127) // 128, 3, 256)

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
			dw_conf = scratch.gemm_configs["dw"]
			for weight, dproj in ((wq, dq), (wk, dk), (wv, dv)):
				if weight.requires_grad:
					dw_bf16 = int(np.dtype(weight.grad.storage.dtype) == np.dtype(np.uint16))
					self.ctx.dispatch(dw_conf["kernel"],
						[q.ptr.buffer_id, dproj.ptr.buffer_id, weight.grad.storage.ptr.buffer_id],
						[q.ptr.offset, dproj.ptr.offset, weight.grad.storage.ptr.offset],
						[C, C, N, accumulate, dw_bf16],
						(C + dw_conf["bn"] - 1) // dw_conf["bn"], (C + dw_conf["bm"] - 1) // dw_conf["bm"], dw_conf["split_k"], 256)

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
			dx_conf = scratch.gemm_configs["dx_norm"]
			if dx_conf["split_k"] > 1:
				self.ctx.memset_zero(k.ptr.buffer_id, k.ptr.offset, N * C * 4)
			for accum_flag, (dproj, weight) in enumerate(((dq, wq), (dk, wk), (dv, wv))):
				self.ctx.dispatch(dx_conf["kernel"],
					[dproj.ptr.buffer_id, weight.storage.ptr.buffer_id, k.ptr.buffer_id],
					[dproj.ptr.offset, weight.storage.ptr.offset, k.ptr.offset],
					[N, C, C, int(accum_flag > 0)], 
					(C + dx_conf["bn"] - 1) // dx_conf["bn"], (N + dx_conf["bm"] - 1) // dx_conf["bm"], dx_conf["split_k"], 256)
			accum_x = int(x.op is not None or bool(accumulate))
			self.ctx.dispatch("rmsnorm_backward_kernel",
				[x.storage.ptr.buffer_id, k.ptr.buffer_id, 0, x.grad.storage.ptr.buffer_id, 0],
				[x.storage.ptr.offset, k.ptr.offset, 0, x.grad.storage.ptr.offset, 0],
				[N, C, eps_bits, accum_x], (N + 3) // 4, 1, 1, 128)

	def cuda_fused_linear_ce_forward(self, inputs, outputs, scratch=None):
		x, w, targets = inputs[0], inputs[1], inputs[2]
		row_scale = inputs[3] if len(inputs) > 3 else None
		out_loss = outputs[0]

		N = prod(x.shape[:-1])
		C = x.shape[-1]
		V = w.shape[0]
		row_loss = scratch["row_loss"]
		lse = scratch["lse"]
		logits = scratch["logits"]
		chunk_rows = min(logits.shape[0], N)
		itemsize = np.dtype(x.dtype).itemsize

		accumulate = bool(getattr(self, "accumulate_parameter_grads", False))
		grad_scale = float(getattr(self, "grad_scale", 1.0))
		need_grad = (out_loss.grad is not None) and (x.requires_grad or w.requires_grad)
		dw_bf16 = need_grad and w.requires_grad and np.dtype(w.grad.storage.dtype) == np.dtype(np.uint16)
		defer_dw_bf16 = dw_bf16 and (N > chunk_rows)

		if row_scale is None:
			scale_bits = f32_bits(grad_scale / float(N))
			rs_id, rs_off = 0, 0
		else:
			scale_bits = f32_bits(grad_scale)
			rs_id, rs_off = row_scale.storage.ptr.buffer_id, row_scale.storage.ptr.offset
		ignore_bits = int(out_loss.op.op.ignore_index) & 0xFFFFFFFF

		for row_base in range(0, N, chunk_rows):
			rows = min(chunk_rows, N - row_base)
			x_off = x.storage.ptr.offset + row_base * C * itemsize

			# 1. logits = X_chunk [rows, C] @ W^T [V, C]
			logits_conf = scratch.gemm_configs[("logits", rows)]
			self.ctx.dispatch(logits_conf["kernel"],
				[x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
				[x_off, w.storage.ptr.offset, logits.ptr.offset],
				[rows, V, C, 0], (V+logits_conf["bn"]-1)//logits_conf["bn"], (rows+logits_conf["bm"]-1)//logits_conf["bm"], logits_conf["split_k"], 256)

			# 2. 单趟在线归约 row_loss / lse，并在需要梯度时原地覆写 dlogits
			self.ctx.dispatch("fused_linear_ce_logits_to_dlogits_kernel",
				[logits.ptr.buffer_id, targets.storage.ptr.buffer_id, row_loss.ptr.buffer_id, lse.ptr.buffer_id, rs_id],
				[logits.ptr.offset, targets.storage.ptr.offset, row_loss.ptr.offset, lse.ptr.offset, rs_off],
				[rows, V, row_base, N, scale_bits, int(need_grad), ignore_bits],
				rows, 1, 1, 256)

			if not need_grad:
				continue

			# 3. dX_chunk = dlogits [rows, V] @ W [V, C]
			if x.requires_grad:
				dx_off = x.grad.storage.ptr.offset + row_base * C * itemsize
				accum_x = int(x.op is None and accumulate)
				dx_conf = scratch.gemm_configs[("dx", rows)]
				if dx_conf["split_k"] > 1 and not accum_x:
					self.ctx.memset_zero(x.grad.storage.ptr.buffer_id, dx_off, rows * C * 4)
				self.ctx.dispatch(dx_conf["kernel"],
					[logits.ptr.buffer_id, w.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
					[logits.ptr.offset, w.storage.ptr.offset, dx_off],
					[rows, C, V, accum_x],
					(C + dx_conf["bn"] - 1) // dx_conf["bn"], (rows + dx_conf["bm"] - 1) // dx_conf["bm"], dx_conf["split_k"], 256)

			# 4. dW += dlogits^T [V, rows] @ X_chunk [rows, C]
			# FP32 或单分块 BF16 直接寄存器级一次写出；仅多分块 BF16 延迟至二阶段保证全 N 行 FP32 累加精度
			if w.requires_grad and not defer_dw_bf16:
				accum_w = int(accumulate or row_base != 0)
				dw_conf = scratch.gemm_configs[("dw", rows)]
				self.ctx.dispatch(dw_conf["kernel"],
					[logits.ptr.buffer_id, x.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
					[logits.ptr.offset, x_off, w.grad.storage.ptr.offset],
					[V, C, rows, accum_w, int(dw_bf16)],
					(C + dw_conf["bn"] - 1) // dw_conf["bn"], (V + dw_conf["bm"] - 1) // dw_conf["bm"], dw_conf["split_k"], 256)

		# 5. 归约所有 token 的 row_loss 输出标量 loss
		self.ctx.dispatch("fused_linear_ce_reduce_kernel",
			[row_loss.ptr.buffer_id, out_loss.storage.ptr.buffer_id],
			[row_loss.ptr.offset, out_loss.storage.ptr.offset],
			[N], 1, 1, 1, 256)

		if not defer_dw_bf16:
			return

		# 6. 多分块 BF16 权重梯度：复用 logits 缓冲区前部存 dlogits_tile、后部存 FP32 dw_tile，全 N 行累加后单次转 BF16
		tile_rows, tile_v = scratch.gemm_configs["ce_tile_shape"]
		dw_off = logits.ptr.offset + tile_rows * tile_v * 4

		for vocab_base in range(0, V, tile_v):
			cols = min(tile_v, V - vocab_base)
			w_off = w.storage.ptr.offset + vocab_base * C * 4
			for row_base in range(0, N, tile_rows):
				rows = min(tile_rows, N - row_base)
				x_off = x.storage.ptr.offset + row_base * C * 4

				lt_conf = scratch.gemm_configs[("logits_tile", rows, cols)]
				if lt_conf["split_k"] > 1:
					self.ctx.memset_zero(logits.ptr.buffer_id, logits.ptr.offset, rows * cols * 4)
				self.ctx.dispatch(lt_conf["kernel"],
					[x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
					[x_off, w_off, logits.ptr.offset],
					[rows, cols, C, 0],
					(cols + lt_conf["bn"] - 1) // lt_conf["bn"], (rows + lt_conf["bm"] - 1) // lt_conf["bm"], lt_conf["split_k"], 256)

				count = rows * cols
				self.ctx.dispatch("ce_tile_logits_to_dlogits_kernel",
					[logits.ptr.buffer_id, lse.ptr.buffer_id, targets.storage.ptr.buffer_id, rs_id],
					[logits.ptr.offset, lse.ptr.offset, targets.storage.ptr.offset, rs_off],
					[rows, cols, vocab_base, V, row_base, scale_bits, ignore_bits],
					(count + 255) // 256, 1, 1, 256)

				dt_conf = scratch.gemm_configs[("dw_tile", cols, rows)]
				self.ctx.dispatch(dt_conf["kernel"],
					[logits.ptr.buffer_id, x.storage.ptr.buffer_id, logits.ptr.buffer_id],
					[logits.ptr.offset, x_off, dw_off],
					[cols, C, rows, int(row_base != 0), 0],
					(C + dt_conf["bn"] - 1) // dt_conf["bn"], (cols + dt_conf["bm"] - 1) // dt_conf["bm"], dt_conf["split_k"], 256)

			dw_count = cols * C
			self.ctx.dispatch("fp32_to_bf16_accum_kernel",
				[logits.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
				[dw_off, w.grad.storage.ptr.offset + vocab_base * C * 2],
				[dw_count, int(accumulate)], (dw_count + 255) // 256, 1, 1, 256)

	def cuda_fused_linear_ce_backward(self, inputs, outputs, scratch=None):
		return

	def grad_norm_sq(self, g_storage, partial, numel):
		numel = int(numel)
		if numel <= 0:
			return
		grad_bf16 = int(np.dtype(g_storage.dtype) == np.dtype(np.uint16))
		n_groups = (numel + 255) // 256
		self.ctx.dispatch("grad_norm_sq_kernel",
			[g_storage.ptr.buffer_id, partial.ptr.buffer_id],
			[g_storage.ptr.offset, partial.ptr.offset],
			[numel, grad_bf16], n_groups, 1, 1, 256)

	def adamw_step(self, p_storage, g_storage, m_storage, v_storage, tmp_storage,
				   lr, lr_t, b1, b2, eps, weight_decay, numel=None, decay_numel=None,
				   grad_scale=1.0, lr_bc1=None, inv_sqrt_bc2=1.0):
		self._rollout_cache_dirty = True
		if numel is None:
			numel = prod(p_storage.shape)
		if decay_numel is None:
			decay_numel = numel if weight_decay != 0.0 else 0
		numel = int(numel)
		decay_numel = int(decay_numel)
		if numel <= 0:
			return

		decay_factor = 1.0 - float(lr) * float(weight_decay)
		update_lr = lr_t if lr_bc1 is None else lr_bc1
		grad_bf16 = int(np.dtype(g_storage.dtype) == np.dtype(np.uint16))
		g_align = 8 if grad_bf16 else 16
		vec4 = int(
			p_storage.ptr.offset % 16 == 0
			and m_storage.ptr.offset % 16 == 0
			and v_storage.ptr.offset % 16 == 0
			and g_storage.ptr.offset % g_align == 0
		)
		n_vec = (numel + 3) // 4
		self.ctx.dispatch("adamw_step_kernel",
			[p_storage.ptr.buffer_id, g_storage.ptr.buffer_id, m_storage.ptr.buffer_id, v_storage.ptr.buffer_id],
			[p_storage.ptr.offset, g_storage.ptr.offset, m_storage.ptr.offset, v_storage.ptr.offset],
			[f32_bits(decay_factor), f32_bits(update_lr), f32_bits(b1), f32_bits(b2), f32_bits(eps),
			 numel, decay_numel, f32_bits(grad_scale), f32_bits(inv_sqrt_bc2), grad_bf16, vec4],
			(n_vec + 255) // 256, 1, 1, 256)

	def adamw8bit_step(self, p, g, qm, qv, absmax_m, absmax_v, qmap_m, qmap_v,
					   lr, lr_bc1, inv_sqrt_bc2, b1, b2, eps, weight_decay,
					   numel, decay_numel, first_step, grad_scale=1.0):
		self._rollout_cache_dirty = True
		numel = int(numel)
		if numel <= 0:
			return
		grad_bf16 = int(np.dtype(g.dtype) == np.dtype(np.uint16))
		self.ctx.dispatch("adamw8bit_step_kernel",
			[p.ptr.buffer_id, g.ptr.buffer_id, qm.ptr.buffer_id, qv.ptr.buffer_id,
			 absmax_m.ptr.buffer_id, absmax_v.ptr.buffer_id, qmap_m.ptr.buffer_id, qmap_v.ptr.buffer_id],
			[p.ptr.offset, g.ptr.offset, qm.ptr.offset, qv.ptr.offset,
			 absmax_m.ptr.offset, absmax_v.ptr.offset, qmap_m.ptr.offset, qmap_v.ptr.offset],
			[f32_bits(1.0 - float(lr) * float(weight_decay)), f32_bits(lr_bc1),
			 f32_bits(b1), f32_bits(b2), f32_bits(eps),
			 numel, int(decay_numel), int(bool(first_step)),
			 f32_bits(inv_sqrt_bc2), grad_bf16, f32_bits(grad_scale)],
			(numel + 255) // 256, 1, 1, 256)

	@staticmethod
	def _mini_v_group_count(numel, block_elems, cols, head_cols):
		import operator
		try:
			numel, block_elems, cols, head_cols = (operator.index(x) for x in (numel, block_elems, cols, head_cols))
		except TypeError as exc:
			raise TypeError("AdamW-mini dimensions must be integers") from exc
		if not 0 < numel <= 0xFFFFFF00:
			raise ValueError("invalid AdamW-mini numel")
		if not 0 < block_elems <= numel:
			raise ValueError("invalid AdamW-mini block_elems")
		if not 0 < cols <= numel:
			raise ValueError("invalid AdamW-mini cols")
		if not 0 <= head_cols <= cols:
			raise ValueError("invalid AdamW-mini head_cols")
		if head_cols:
			if numel % cols or cols % head_cols or block_elems != (numel // cols) * head_cols:
				raise ValueError("invalid AdamW-mini head grouping")
			return cols // head_cols
		return (numel + block_elems - 1) // block_elems

	def mini_update_v(self, g, v, b2, numel, block_elems, *, cols, head_cols, first_step, grad_scale=1.0):
		n_v_blocks = self._mini_v_group_count(numel, block_elems, cols, head_cols)
		grad_dtype = np.dtype(g.dtype)
		grad_bf16 = int(grad_dtype == np.dtype(np.uint16))
		if g.nbytes < numel * grad_dtype.itemsize or v.nbytes < n_v_blocks * 4:
			raise ValueError("AdamW-mini gradient/v storage is too small")
		if head_cols == 0 and block_elems == 1:
			grid_x = (numel + 255) // 256
		elif head_cols == 1:
			grid_x = (cols + 7) // 8
		else:
			grid_x = n_v_blocks
		self.ctx.dispatch("mini_update_v_kernel",
			[g.ptr.buffer_id, v.ptr.buffer_id],
			[g.ptr.offset, v.ptr.offset],
			[f32_bits(b2), f32_bits(grad_scale), int(numel), int(block_elems),
			 int(cols), int(head_cols), int(bool(first_step)), grad_bf16],
			grid_x, 1, 1, 256)

	def adamw_mini8bit_step(self, p, g, m, meta, v, lr, lr_bc1, inv_sqrt_bc2, b1, eps, weight_decay,
							numel, decay_numel, block_elems, *, cols, head_cols, first_step, seed,
							stochastic_round, transpose_m, quantized, grad_scale=1.0):
		self._rollout_cache_dirty = True
		self._mini_v_group_count(numel, block_elems, cols, head_cols)
		if not 0 <= decay_numel <= numel:
			raise ValueError("invalid decay_numel")
		if transpose_m and numel % cols:
			raise ValueError("Invalid transposed matrix shape")
		if quantized and meta is None:
			raise ValueError("quantized AdamW-mini requires metadata storage")

		grad_bf16 = int(np.dtype(g.dtype) == np.dtype(np.uint16))
		meta_id = meta.ptr.buffer_id if meta is not None else 0
		meta_off = meta.ptr.offset if meta is not None else 0
		self.ctx.dispatch("adamw_mini8bit_step_kernel",
			[p.ptr.buffer_id, g.ptr.buffer_id, m.ptr.buffer_id, meta_id, v.ptr.buffer_id],
			[p.ptr.offset, g.ptr.offset, m.ptr.offset, meta_off, v.ptr.offset],
			[f32_bits(1.0 - float(lr) * float(weight_decay)), f32_bits(lr_bc1), f32_bits(inv_sqrt_bc2),
			 f32_bits(b1), f32_bits(eps), f32_bits(grad_scale),
			 int(numel), int(decay_numel), int(block_elems), int(cols), int(head_cols),
			 int(bool(first_step)), int(seed) & 0xFFFFFFFF, int(bool(stochastic_round)),
			 int(bool(transpose_m)), int(bool(quantized)), grad_bf16],
			(int(numel) + 255) // 256, 1, 1, 256)
