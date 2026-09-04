import importlib.util
import os
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from math import prod
import struct

_ROOT_BACKEND_PATH = Path(__file__).resolve().parent.parent / "backend.py"
_SPEC = importlib.util.spec_from_file_location("_autograd_root_backend", _ROOT_BACKEND_PATH)
_ROOT_BACKEND = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_ROOT_BACKEND)

Backend = _ROOT_BACKEND.Backend
CPUBackend = _ROOT_BACKEND.CPUBackend


def _host_like(tensor, data=None):
	host = SimpleNamespace()
	host.shape = tensor.shape
	host.dtype = tensor.dtype
	host.requires_grad = tensor.requires_grad
	host.op = tensor.op
	if data is None:
		data = np.empty(tensor.shape, dtype=tensor.dtype)
	host.storage = SimpleNamespace(data=data)
	host.grad = None
	return host


def _scratch_host_map(scratch):
	if scratch is None:
		return None
	views = getattr(scratch, "_views", {})
	host = {}
	for name, view in views.items():
		host[name] = SimpleNamespace(data=np.empty(view.shape, dtype=view.dtype))
	return host


class _ScratchAlias:
	def __init__(self, scratch):
		self._scratch = scratch

	def __getitem__(self, key):
		def _fake_view(buf, offset):
			return SimpleNamespace(ptr=SimpleNamespace(buffer_id=buf.ptr.buffer_id, offset=offset))

		aliases = {
			"fa_k": ("k", "fa_k"),
			"fa_v": ("v", "fa_v"),
			"fa_mean": ("mean", "mean_bwd", "fa_mean"),
			"fa_inv_std": ("inv_std", "inv_std_bwd", "fa_inv_std"),
		}
		for candidate in aliases.get(key, (key,)):
			try:
				return self._scratch[candidate]
			except KeyError:
				pass
		if key == "fa_mean" or key == "fa_inv_std":
			lse = self._scratch["lse"]
			q = self._scratch["q"]
			B, H, T, _ = q.shape
			mean_buf = lse
			if key == "fa_mean":
				return _fake_view(mean_buf, mean_buf.ptr.offset)
			return _fake_view(mean_buf, mean_buf.ptr.offset + B * T * 4)
		raise KeyError(key)


def _ensure_fa_fwd_scratch(host, B, T, C, H, D, dtype):
	def put(name, shape, dt=dtype):
		if name not in host:
			host[name] = SimpleNamespace(data=np.empty(shape, dtype=dt))
	put("inv_rms", (B, T, 1))
	put("x_norm", (B, T, C))
	put("q", (B, T, H, D))
	put("k", (B, T, H, D))
	put("v", (B, T, H, D))
	put("attn", (B, H, T, T))
	put("row_sum", (B, H, T, 1))
	put("row_max", (B, H, T, 1))
	put("mask", (T, T), bool)


def _ensure_fa_bwd_scratch(host, B, T, C, H, D, dtype):
	def put(name, shape, dt=dtype):
		if name not in host:
			host[name] = SimpleNamespace(data=np.empty(shape, dtype=dt))
	put("tmp_dq", (B, H, T, D))
	put("tmp_dk", (B, H, T, D))
	put("tmp_dv", (B, H, T, D))
	put("probs", (B, H, T, T))
	put("work", (B, H, T, T))
	put("tmp", (B, H, T, T))
	put("row_sum", (B, H, T, 1))
	put("row_max", (B, H, T, 1))
	put("tmp_dx_k", (B, T, C))
	put("tmp_dx_v", (B, T, C))
	put("mask", (T, T), bool)
	put("dx_norm", (B, T, C))
	put("inv_rms", (B, T, 1))
	put("x_hat", (B, T, C))
	put("rms_dot", (B, T, 1))


class MetalBackend(_ROOT_BACKEND.MetalBackend):
	def __init__(self):
		cwd = os.getcwd()
		try:
			os.chdir(Path(__file__).resolve().parent)
			super().__init__()
		finally:
			os.chdir(cwd)

	def _run_fa_forward_cpu(self, inputs, outputs, scratch=None):
		self.end_pass()
		self.synchronize()
		self.begin_pass()
		host_inputs = [_host_like(t, self.copy_to_host(t)) for t in inputs]
		host_out = _host_like(outputs[0], np.empty(outputs[0].shape, dtype=outputs[0].dtype))
		host_scratch = _scratch_host_map(scratch) or {}
		B, T, C = inputs[0].shape
		H = outputs[0].op.op.n_head
		D = C // H
		_ensure_fa_fwd_scratch(host_scratch, B, T, C, H, D, outputs[0].dtype)
		_ROOT_BACKEND.cpu_fused_rms_fa_forward(host_inputs, [host_out], host_scratch)
		self.copy_from_host(outputs[0], host_out.storage.data)

	def _run_fa_backward_cpu(self, inputs, outputs, scratch=None):
		self.end_pass()
		self.synchronize()
		self.begin_pass()
		host_inputs = [_host_like(t, self.copy_to_host(t)) for t in inputs]
		host_out = _host_like(outputs[0], np.empty(outputs[0].shape, dtype=outputs[0].dtype))
		host_out.op = outputs[0].op
		host_out.grad = _host_like(outputs[0], np.empty(outputs[0].shape, dtype=outputs[0].dtype))
		host_out.grad.storage.data[...] = self.copy_to_host(outputs[0].grad)
		for host_in, real_in in zip(host_inputs, inputs):
			if real_in.requires_grad:
				host_in.grad = _host_like(real_in, np.zeros(real_in.shape, dtype=real_in.dtype))
		host_scratch = _scratch_host_map(scratch) or {}
		B, T, C = inputs[0].shape
		H = outputs[0].op.op.n_head
		D = C // H
		_ensure_fa_bwd_scratch(host_scratch, B, T, C, H, D, outputs[0].dtype)
		_ROOT_BACKEND.cpu_fused_rms_fa_backward(host_inputs, [host_out], host_scratch)
		for host_in, real_in in zip(host_inputs, inputs):
			if real_in.requires_grad and real_in.grad is not None:
				self.copy_from_host(real_in.grad, host_in.grad.storage.data)

	def fa_recompute_dispatch(self, head_dim, seq_len):
		if head_dim <= 0 or head_dim > 128:
			raise ValueError(f"fused LN/FA recompute only supports 0 < head_dim <= 128, got {head_dim}")
		capacity = 64 if head_dim <= 64 else 128
		return f"fused_rms_fa_forward_block{capacity}", 256, (seq_len+31)//32, 256

	def fa_from_qkv_dispatch(self, head_dim, seq_len):
		if head_dim <= 0 or head_dim > 512:
			raise ValueError(f"fused attention requires 0 < head_dim <= 512, got {head_dim}")
		if head_dim <= 128:
			return "fused_fa_from_qkv", 128, (seq_len+7)//8, 128
		capacity = 256 if head_dim <= 256 else 512
		return f"fused_fa_from_qkv_block{capacity}", 32, seq_len, 32

	def fa_bwd_dq_dispatch(self, head_dim, seq_len):
		if head_dim <= 0 or head_dim > 128:
			raise ValueError(f"FA backward requires 0 < head_dim <= 128, got {head_dim}")
		capacity = 64 if head_dim <= 64 else 128
		d_chunks = (head_dim + 63) // 64
		return f"fused_fa_bwd_dq_block{capacity}", 128*d_chunks, (seq_len+31)//32, 128

	def fa_bwd_dkv_dispatch(self, head_dim, seq_len):
		if head_dim <= 0 or head_dim > 128:
			raise ValueError(f"FA backward dK/dV only supports 0 < head_dim <= 128, got {head_dim}")
		if head_dim <= 64:
			return "fused_fa_bwd_dkv_mma64", 128, (seq_len+31)//32, 128
		if seq_len >= 160:
			return "fused_fa_bwd_dkv_mma128_q24_o2_sg4", 128, (seq_len+31)//32, 128
		return "fused_fa_bwd_dkv_block128", 32, seq_len, 32

	def metal_fused_rms_fa_forward(self, inputs, outputs, scratch=None):
		x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
		out = outputs[0]
		B, T, C = x.shape
		H, D = out.op.op.n_head, C // out.op.op.n_head
		is_causal = out.op.op.is_causal
		eps_bits = struct.unpack('<i', struct.pack('<f', out.op.op.eps))[0]

		q = scratch['q']
		k = scratch['k']
		v = scratch['v']
		# RMSNorm + QKV Projection
		self.ctx.dispatch_with_args_3d('fused_rms_fa_qkv',
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id, wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id,
				wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id, q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset, wk.storage.ptr.offset, bk.storage.ptr.offset,
				wv.storage.ptr.offset, bv.storage.ptr.offset, q.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, T, C, H, eps_bits], 256, (B * T + 15) // 16, 1, 256)
		# RoPE
		self.ctx.dispatch_with_args_3d("rope_qk_inplace",
			[q.ptr.buffer_id, k.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
			[B, H, T, D, 0], D//2, T, B*H, D//2)

		# Flash Attention
		fa_kernel, fa_grid_x, fa_grid_y, fa_threads = self.fa_from_qkv_dispatch(D, T)
		self.ctx.dispatch_with_args_3d(fa_kernel,
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out.storage.ptr.buffer_id, q.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, out.storage.ptr.offset, q.ptr.offset],
			[B, H, T, C, int(is_causal), 0], fa_grid_x, fa_grid_y, B * H, fa_threads)

	def metal_fused_rms_fa_backward(self, inputs, outputs, scratch=None):
		x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
		out = outputs[0]
		if out.grad is None:
			return

		B, T, C = x.shape
		H = out.op.op.n_head
		D = C // H
		eps = out.op.op.eps
		eps_bits = struct.unpack('<i', struct.pack('<f', eps))[0]
		is_causal = out.op.op.is_causal
		accumulate = int(bool(getattr(self, "accumulate_parameter_grads", False)))
		q = scratch["q"]
		k = scratch["k"]
		v = scratch["v"]
		dq = scratch["dq"]
		dk = scratch["dk"]
		dv = scratch["dv"]
		lse = scratch["lse"]
		d_val = scratch["d_val"]

		# 1. recompute QKV
		self.ctx.dispatch_with_args_3d('fused_rms_fa_qkv',
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id, wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id,
				wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id, q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset, wk.storage.ptr.offset, bk.storage.ptr.offset,
				wv.storage.ptr.offset, bv.storage.ptr.offset, q.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, T, C, H, eps_bits], 256, (B * T + 15) // 16, 1, 256)

		self.ctx.dispatch_with_args_3d("rope_qk_inplace",
			[q.ptr.buffer_id, k.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
			[B, H, T, D, 0], D//2, T, B*H, D//2)

		# 2. recompute O(reuse dk)
		fa_kernel, fa_grid_x, fa_grid_y, fa_threads = self.fa_from_qkv_dispatch(D, T)
		self.ctx.dispatch_with_args_3d(fa_kernel,
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, dk.ptr.buffer_id, lse.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, dk.ptr.offset, lse.ptr.offset],
			[B, H, T, C, int(is_causal), 1], fa_grid_x, fa_grid_y, B * H, fa_threads)

		# 3. LSE & D
		self.ctx.dispatch_with_args_3d('fused_fa_bwd_prepare',
			[dk.ptr.buffer_id, out.grad.storage.ptr.buffer_id, d_val.ptr.buffer_id],
			[dk.ptr.offset, out.grad.storage.ptr.offset, d_val.ptr.offset],
			[B, H, T, C], 128, (T + 3) // 4, B * H, 128)

		# 4. dQ
		dq_kernel, dq_grid_x, dq_grid_y, dq_threads = self.fa_bwd_dq_dispatch(D, T)
		self.ctx.dispatch_with_args_3d(dq_kernel,
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out.grad.storage.ptr.buffer_id,
				dq.ptr.buffer_id, lse.ptr.buffer_id, d_val.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, out.grad.storage.ptr.offset,
				dq.ptr.offset, lse.ptr.offset, d_val.ptr.offset],
			[B, H, T, C, int(is_causal)], dq_grid_x, dq_grid_y, B*H, dq_threads)

		# 5. dK和dV
		dkv_kernel, dkv_grid_x, dkv_grid_y, dkv_threads = self.fa_bwd_dkv_dispatch(D, T)
		self.ctx.dispatch_with_args_3d(dkv_kernel,
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out.grad.storage.ptr.buffer_id,
				dk.ptr.buffer_id, dv.ptr.buffer_id, lse.ptr.buffer_id, d_val.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, out.grad.storage.ptr.offset,
				dk.ptr.offset, dv.ptr.offset, lse.ptr.offset, d_val.ptr.offset],
			[B, H, T, C, int(is_causal)], dkv_grid_x, dkv_grid_y, B*H, dkv_threads)

		self.ctx.dispatch_with_args_3d("rope_qk_inplace",
			[dq.ptr.buffer_id, dk.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
			[dq.ptr.offset, dk.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
			[B, H, T, D, 1], D//2, T, B*H, D//2)

		# 6. Z=RMSNorm(X)(reuse q), inv_rms(reuse lse)
		self.ctx.dispatch_with_args_3d('fused_rms_recompute',
			[x.storage.ptr.buffer_id, q.ptr.buffer_id, lse.ptr.buffer_id],
			[x.storage.ptr.offset, q.ptr.offset, lse.ptr.offset],
			[B, T, C, eps_bits], 128, (B * T + 3) // 4, 1, 128)

		# 7. dWq, dWk, dWv
		if wq.requires_grad:
			self.ctx.dispatch_gemm_mps(
				[q.ptr.buffer_id, dq.ptr.buffer_id, wq.grad.storage.ptr.buffer_id],
				[q.ptr.offset, dq.ptr.offset, wq.grad.storage.ptr.offset],
				[1, C, C, B*T, 1, 0, 0, 0], 1.0, float(accumulate))
		if wk.requires_grad:
			self.ctx.dispatch_gemm_mps(
				[q.ptr.buffer_id, dk.ptr.buffer_id, wk.grad.storage.ptr.buffer_id],
				[q.ptr.offset, dk.ptr.offset, wk.grad.storage.ptr.offset],
				[1, C, C, B*T, 1, 0, 0, 0], 1.0, float(accumulate))
		if wv.requires_grad:
			self.ctx.dispatch_gemm_mps(
				[q.ptr.buffer_id, dv.ptr.buffer_id, wv.grad.storage.ptr.buffer_id],
				[q.ptr.offset, dv.ptr.offset, wv.grad.storage.ptr.offset],
				[1, C, C, B*T, 1, 0, 0, 0], 1.0, float(accumulate))

		# 8. dbq/dbk/dbv
		if bq.requires_grad or bk.requires_grad or bv.requires_grad:
			self.ctx.dispatch_with_args_3d('fused_qkv_bias_grad',
				[dq.ptr.buffer_id, dk.ptr.buffer_id, dv.ptr.buffer_id, 
					bq.grad.storage.ptr.buffer_id if bq.requires_grad else 0,
					bk.grad.storage.ptr.buffer_id if bk.requires_grad else 0,
					bv.grad.storage.ptr.buffer_id if bv.requires_grad else 0],
				[dq.ptr.offset, dk.ptr.offset, dv.ptr.offset, 
					bq.grad.storage.ptr.offset if bq.requires_grad else 0,
					bk.grad.storage.ptr.offset if bk.requires_grad else 0,
					bv.grad.storage.ptr.offset if bv.requires_grad else 0],
				[B*T, C, accumulate], 128, (C + 3) // 4, 1, 128)

		# 9. dZ =  dQ*Wq + dK*Wk + dV*Wv
		if x.requires_grad:
			self.ctx.dispatch_gemm_mps(
				[dq.ptr.buffer_id, wq.storage.ptr.buffer_id, k.ptr.buffer_id],
				[dq.ptr.offset, wq.storage.ptr.offset, k.ptr.offset],
				[1, B*T, C, C, 0, 1, 0, 0], 1.0, 0.0)
			self.ctx.dispatch_gemm_mps(
				[dk.ptr.buffer_id, wk.storage.ptr.buffer_id, k.ptr.buffer_id],
				[dk.ptr.offset, wk.storage.ptr.offset, k.ptr.offset],
				[1, B*T, C, C, 0, 1, 0, 0], 1.0, 1.0)
			self.ctx.dispatch_gemm_mps(
				[dv.ptr.buffer_id, wv.storage.ptr.buffer_id, k.ptr.buffer_id],
				[dv.ptr.offset, wv.storage.ptr.offset, k.ptr.offset],
				[1, B*T, C, C, 0, 1, 0, 0], 1.0, 1.0)
			self.ctx.dispatch_with_args_3d('fused_rmsnorm_bwd_dx',
				[x.storage.ptr.buffer_id, k.ptr.buffer_id, lse.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
				[x.storage.ptr.offset, k.ptr.offset, lse.ptr.offset, x.grad.storage.ptr.offset],
				[B*T, C], 32, B*T, 1, 32)

	def metal_fused_linear_ce_forward(self, inputs, outputs, scratch=None):
		x, w, targets = inputs[0], inputs[1], inputs[2]
		out_loss = outputs[0]
		
		N = x.shape[0] * x.shape[1] if len(x.shape) == 3 else x.shape[0]
		C = x.shape[-1]
		V = w.shape[0]
		row_loss = scratch['row_loss']
		lse = scratch['lse']
		logits = scratch["logits"]
		chunk_rows = min(logits.shape[0], N)
		itemsize = np.dtype(x.dtype).itemsize

		accumulate = bool(getattr(self, "accumulate_parameter_grads", False))
		scale = float(getattr(self, "grad_scale", 1.0)) / float(N)
		scale_bits = struct.unpack("<i", struct.pack("<f", scale))[0]
		need_grad = x.requires_grad or w.requires_grad

		for row_base in range(0, N, chunk_rows):
			rows = min(chunk_rows, N - row_base)
			x_offset = x.storage.ptr.offset + row_base*C*itemsize

			# 1) logits = X_chunk @ W.T
			self.ctx.dispatch_with_args_3d("fused_linear_ce_chunk_logits",
				[x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
				[x_offset, w.storage.ptr.offset, logits.ptr.offset],
				[rows, C, V], 256, (rows+63)//64, (V+95)//96, 256)

			# 2) row_loss/lse/dlogits
			self.ctx.dispatch_with_args_3d("fused_linear_ce_logits_to_dlogits",
				[logits.ptr.buffer_id, targets.storage.ptr.buffer_id, row_loss.ptr.buffer_id, lse.ptr.buffer_id],
				[logits.ptr.offset, targets.storage.ptr.offset, row_loss.ptr.offset, lse.ptr.offset],
				[rows, V, row_base, scale_bits, int(need_grad)], 256, rows, 1, 256)

			if not need_grad:
				continue

			# 3) dX = dlogits @ W
			if x.requires_grad:
				self.ctx.dispatch_gemm_mps([logits.ptr.buffer_id, w.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
					[logits.ptr.offset, w.storage.ptr.offset, x.grad.storage.ptr.offset+row_base*C*itemsize],
					[1, rows, C, V, 0, 0, 0, 0], 1.0, 0.0)

			# 4) dW += dlogits.T@X
			if w.requires_grad:
				self.ctx.dispatch_gemm_mps([logits.ptr.buffer_id, x.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
					[logits.ptr.offset, x_offset, w.grad.storage.ptr.offset],
					[1, V, C, rows, 1, 0, 0, 0], 1.0, 1.0 if accumulate or row_base != 0 else 0.0)

		self.ctx.dispatch_with_args_3d('fused_linear_ce_forward_reduce',
			[row_loss.ptr.buffer_id, out_loss.storage.ptr.buffer_id],
			[row_loss.ptr.offset, out_loss.storage.ptr.offset],
			[N], 256, 1, 1, 256)

	def metal_fused_linear_ce_backward(self, inputs, outputs, scratch=None):
		return

	def metal_recompute_fused_rms_fa(self, act, scratch, out_buf):
		x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = act.op.inputs
		B, T, C = x.shape
		H = act.op.op.n_head
		D = C // H
		eps_bits = struct.unpack('<i', struct.pack('<f', act.op.op.eps))[0]
		M_2d = B*T
		k = scratch['fa_k']
		v = scratch['fa_v']

		# RMSNorm/K/V
		self.ctx.dispatch_with_args_3d('fused_rms_fa_project_kv',
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
				wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
				out_buf.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
				wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
				out_buf.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, T, C, H, eps_bits], 512, (B*T+31)//32, 1, 512)
		# RoPE
		self.ctx.dispatch_with_args_3d("rope_qk_inplace",
			[out_buf.ptr.buffer_id, k.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
			[out_buf.ptr.offset, k.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
			[B, H, T, D, 2], D//2, T, B*H, D//2)
		# Q / attention / output
		fa_kernel, fa_grid_x, fa_grid_y, fa_threads = self.fa_recompute_dispatch(D, T)
		self.ctx.dispatch_with_args_3d(fa_kernel,
			[out_buf.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[out_buf.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, H, T, C, int(act.op.op.is_causal)], fa_grid_x, fa_grid_y, B*H, fa_threads)

	def metal_fused_linear_add_forward(self, inputs, outputs, scratch=None):
	    x, act, w, b = inputs
	    out = outputs[0]
	    M = prod(x.shape[:-1])
	    K = act.shape[-1]
	    N = w.shape[-1]
	    BM = 32
	    BN = 64
	    grid_x = (N + BN - 1) // BN
	    grid_y = (M + BM - 1) // BM
	    blocks = grid_x * grid_y
	    alias_x_out = int(x.storage.ptr.buffer_id == out.storage.ptr.buffer_id and x.storage.ptr.offset == out.storage.ptr.offset)
	    self.ctx.dispatch_with_args_3d('fused_linear_add_forward_kernel',
	    	[x.storage.ptr.buffer_id, act.storage.ptr.buffer_id, w.storage.ptr.buffer_id, b.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
	    	[x.storage.ptr.offset, act.storage.ptr.offset, w.storage.ptr.offset, b.storage.ptr.offset, out.storage.ptr.offset],
	    	[M, K, N, alias_x_out], 256, (N + 63)//64, (M+63)//64, 256)

	def metal_fused_linear_add_backward(self, inputs, outputs, scratch=None):
	    x, act, w, b = inputs
	    out = outputs[0]
	    if out.grad is None:
	        return
	    B_dim = prod(x.shape[:-2]) if x.ndim > 2 else 1
	    M_dim = x.shape[-2]
	    K_dim = act.shape[-1]
	    N_dim = w.shape[-1]
	    accumulate = int(bool(getattr(self, "accumulate_parameter_grads", False)))
	    if act.op is not None and act.op.op.op_name == "FusedRMSNormFlashAttention":
	    	cached = getattr(self, "_recomputed_attention_cache", None)
	    	if cached is not None and id(act) in cached:
	    		cached.remove(id(act))
	    		act_buf = act.grad.storage
	    	else:
	    		act_buf = act.grad.storage
	    		self.metal_recompute_fused_rms_fa(act, scratch, act_buf)
	    else:
	        act_buf = act.storage
	    if w.requires_grad:
	        self.ctx.dispatch_gemm_mps([act_buf.ptr.buffer_id, out.grad.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
	        	[act_buf.ptr.offset, out.grad.storage.ptr.offset, w.grad.storage.ptr.offset],
	        	[1, K_dim, N_dim, B_dim*M_dim, 1, 0, 0, 0], 1.0, float(accumulate))
	    if x.requires_grad:
	        self.ctx.dispatch_strided(
	            "strided_add_backward_kernel",
	            out.grad.storage.ptr.buffer_id,
	            out.grad.storage.ptr.offset,
	            x.grad.storage.ptr.buffer_id,
	            x.grad.storage.ptr.offset,
	            list(x.shape),
	            x.strides,
	            prod(x.shape),
	        )
	    if act.requires_grad:
	        self.ctx.dispatch_gemm_mps([out.grad.storage.ptr.buffer_id, w.storage.ptr.buffer_id, act.grad.storage.ptr.buffer_id],
	        	[out.grad.storage.ptr.offset, w.storage.ptr.offset, act.grad.storage.ptr.offset],
	        	[1, B_dim*M_dim, K_dim, N_dim, 0, 1, 0, 0], 1.0, 0.0)

	    if b.requires_grad:
	        self.ctx.dispatch_with_args_3d("bias_grad_reduce",
	        	[out.grad.storage.ptr.buffer_id, b.grad.storage.ptr.buffer_id],
	        	[out.grad.storage.ptr.offset, b.grad.storage.ptr.offset], 
	        	[prod(x.shape[:-1]), N_dim, accumulate], 256, (N_dim+15)//16, 1, 256)

	def metal_fused_rmsnorm_mlp_forward(self, inputs, outputs, scratch=None):
	    x, wgu, wdown = inputs
	    out = outputs[0]

	    N = prod(x.shape[:-1])
	    C = x.shape[-1]
	    H2 = wgu.shape[-1]
	    H = H2 // 2

	    eps_bits = struct.unpack(
	        "<i",
	        struct.pack("<f", out.op.op.eps),
	    )[0]

	    gu = scratch["gu"]
	    hidden = scratch["hidden"]

	    # 使用 out 暂存 RMSNorm 输出，避免额外的 x_norm scratch
	    self.ctx.dispatch_with_args_3d(
	        "rmsnorm_forward",
	        [
	            x.storage.ptr.buffer_id,
	            out.storage.ptr.buffer_id,
	        ],
	        [
	            x.storage.ptr.offset,
	            out.storage.ptr.offset,
	        ],
	        [C, eps_bits],
	        N * 256,
	        1,
	        1,
	        256,
	    )

	    # GU = X_norm @ Wgu
	    self.ctx.dispatch_gemm_mps(
	        [
	            out.storage.ptr.buffer_id,
	            wgu.storage.ptr.buffer_id,
	            gu.ptr.buffer_id,
	        ],
	        [
	            out.storage.ptr.offset,
	            wgu.storage.ptr.offset,
	            gu.ptr.offset,
	        ],
	        [1, N, H2, C, 0, 0, 0, 0],
	        1.0,
	        0.0,
	    )

	    # HIDDEN = SiLU(gate) * up
	    self.ctx.dispatch_with_args_3d("swiglu_forward_kernel",
	        [gu.ptr.buffer_id, hidden.ptr.buffer_id],
	        [gu.ptr.offset, hidden.ptr.offset],
	        [N, H], N*(H+3)//4, 1, 1, min(N*(H+3)//4 , 256))

	    # OUT = HIDDEN @ Wdown
	    self.ctx.dispatch_gemm_mps(
	        [
	            hidden.ptr.buffer_id,
	            wdown.storage.ptr.buffer_id,
	            out.storage.ptr.buffer_id,
	        ],
	        [
	            hidden.ptr.offset,
	            wdown.storage.ptr.offset,
	            out.storage.ptr.offset,
	        ],
	        [1, N, C, H, 0, 0, 0, 0],
	        1.0,
	        0.0,
	    )

	def metal_fused_rmsnorm_mlp_backward(self, inputs, outputs, scratch=None):
		x, wgu, wdown = inputs
		out = outputs[0]
		grad_out = out.grad
		if grad_out is None: return

		N = prod(x.shape[:-1])
		C = x.shape[-1]
		H2 = wgu.shape[-1]
		H = H2 // 2
		eps = out.op.op.eps 
		eps_bits = struct.unpack('<i', struct.pack('<f', eps))[0]
		accumulate = int(bool(getattr(self, "accumulate_parameter_grads", False)))

		gu = scratch['gu']
		hidden = scratch['hidden']
		tmp_dx = x.grad.storage
		residual_grad = grad_out.storage

		fla_node = x.op 
		recompute_x = (fla_node is not None and fla_node.op.op_name == "FusedLinearAdd"
			and len(fla_node.inputs) == 4 and fla_node.inputs[1].op is not None
			and fla_node.inputs[1].op.op.op_name == "FusedRMSNormFlashAttention")
		if recompute_x:
			x_in, attn_out, proj_w, proj_b = fla_node.inputs
			fa_node = attn_out.op
			fa_x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = fa_node.inputs
			B, T, C = fa_x.shape
			n_head = fa_node.op.n_head
			D = C // n_head
			fa_eps_bits = struct.unpack("<i", struct.pack("<f", fa_node.op.eps))[0]

			x_nbytes = N * C * 4
			fa_out_storage = attn_out.grad.storage
			fa_out_offset = gu.ptr.offset
			fa_k_offset = fa_out_offset + x_nbytes
			fa_v_offset = fa_k_offset + x_nbytes

			M_fla = prod(x_in.shape[:-1])
			K_fla = attn_out.shape[-1]
			N_fla = proj_w.shape[-1]

			# 1. RMSNorm + Q/K/V projection -> reuse h1
			self.ctx.dispatch_with_args_3d('fused_rms_fa_project_kv',
				[fa_x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
					wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
					gu.ptr.buffer_id, gu.ptr.buffer_id, gu.ptr.buffer_id],
				[fa_x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
					wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
					fa_out_offset, fa_k_offset, fa_v_offset],
				[B, T, C, n_head, fa_eps_bits], 512, (B*T+31)//32, 1, 512)
			# 2. RoPE
			self.ctx.dispatch_with_args_3d("rope_qk_inplace",
				[gu.ptr.buffer_id, gu.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
				[fa_out_offset, fa_k_offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
				[B, n_head, T, D, 2], D//2, T, B*n_head, D//2)

			# 3. FlashAttention
			fa_kernel, fa_grid_x, fa_grid_y, fa_threads = self.fa_recompute_dispatch(D, T)
			self.ctx.dispatch_with_args_3d(fa_kernel,
				[gu.ptr.buffer_id, gu.ptr.buffer_id, gu.ptr.buffer_id],
				[fa_out_offset, fa_k_offset, fa_v_offset],
				[B, n_head, T, C, int(fa_node.op.is_causal)], fa_grid_x, fa_grid_y, B*n_head, fa_threads)

			# 3. x_after_attn = x_in + attention @ W + b
			self.ctx.dispatch_with_args_3d('fused_linear_add_forward_kernel',
				[x_in.storage.ptr.buffer_id, gu.ptr.buffer_id, proj_w.storage.ptr.buffer_id, proj_b.storage.ptr.buffer_id, tmp_dx.ptr.buffer_id],
				[x_in.storage.ptr.offset, fa_out_offset, proj_w.storage.ptr.offset, proj_b.storage.ptr.offset, tmp_dx.ptr.offset],
				[M_fla, K_fla, N_fla, 0], 256, (N_fla+63)//64, (M_fla+63)//64, 256)
			self.ctx.dispatch_contiguous("contiguous_forward_kernel",
				gu.ptr.buffer_id, fa_out_offset, fa_out_storage.ptr.buffer_id, fa_out_storage.ptr.offset,
				list(attn_out.shape), list(attn_out.strides), attn_out.numel)
			if not hasattr(self, "_recomputed_attention_cache"):
				self._recomputed_attention_cache = set()
			self._recomputed_attention_cache.add(id(attn_out))

			x_buffer_id = tmp_dx.ptr.buffer_id
			x_offset = tmp_dx.ptr.offset
			x_norm_storage = hidden
		else:
			x_buffer_id = x.storage.ptr.buffer_id
			x_offset = x.storage.ptr.offset
			x_norm_storage = tmp_dx

		# 5. X_norm = RMSNorm(x)
		self.ctx.dispatch_with_args_3d("rmsnorm_forward",
			[x_buffer_id, x_norm_storage.ptr.buffer_id],
			[x_offset, x_norm_storage.ptr.offset],
			[C, eps_bits], N * 256, 1, 1, 256)

		# 6. GU = X_norm @ Wgu
		self.ctx.dispatch_gemm_mps([x_norm_storage.ptr.buffer_id, wgu.storage.ptr.buffer_id, gu.ptr.buffer_id],
			[x_norm_storage.ptr.offset, wgu.storage.ptr.offset, gu.ptr.offset],
			[1, N, H2, C, 0, 0, 0, 0], 1.0, 0.0)

		# 7. hidden = SiLU(gate)*up
		self.ctx.dispatch_with_args_3d("swiglu_forward_kernel",
			[gu.ptr.buffer_id, hidden.ptr.buffer_id],
			[gu.ptr.offset, hidden.ptr.offset],
			[N, H], N*(H+3)//4, 1, 1, min(N*(H+3)//4, 256))

		# 8. dWdown = HIDDEN.T @ dOUT
		if wdown.requires_grad:
			self.ctx.dispatch_gemm_mps([hidden.ptr.buffer_id, out.grad.storage.ptr.buffer_id, wdown.grad.storage.ptr.buffer_id],
				[hidden.ptr.offset, out.grad.storage.ptr.offset, wdown.grad.storage.ptr.offset],
				[1, H, C, N, 1, 0, 0, 0], 1.0, float(accumulate))

		# 9. hidden -> dHidden
		self.ctx.dispatch_gemm_mps([out.grad.storage.ptr.buffer_id, wdown.storage.ptr.buffer_id, hidden.ptr.buffer_id],
			[out.grad.storage.ptr.offset, wdown.storage.ptr.offset, hidden.ptr.offset],
			[1, N, H, C, 0, 1, 0, 0], 1.0, 0.0)

		# 10. gu -> dGU
		self.ctx.dispatch_with_args_3d("swiglu_backward_inplace_kernel",
			[gu.ptr.buffer_id, hidden.ptr.buffer_id],
			[gu.ptr.offset, hidden.ptr.offset],
			[N, H], N*(H+3)//4, 1, 1, min(N*(H+3)//4, 256))

		if recompute_x:
			self.ctx.dispatch_with_args_3d("rmsnorm_forward",
				[tmp_dx.ptr.buffer_id, hidden.ptr.buffer_id],
				[tmp_dx.ptr.offset, hidden.ptr.offset],
				[C, eps_bits], N*256, 1, 1, 256)
			x_norm_storage = hidden
		else:
			x_norm_storage = tmp_dx

		# 11. dWgu = X_norm.T @ dGU
		if wgu.requires_grad:
			self.ctx.dispatch_gemm_mps([x_norm_storage.ptr.buffer_id, gu.ptr.buffer_id, wgu.grad.storage.ptr.buffer_id],
				[x_norm_storage.ptr.offset, gu.ptr.offset, wgu.grad.storage.ptr.offset],
				[1, C, H2, N, 1, 0, 0, 0], 1.0, float(accumulate))

		# 12. dX_norm
		if recompute_x:
			dx_norm_storage = hidden
		else:
			dx_norm_storage = tmp_dx

		self.ctx.dispatch_gemm_mps([gu.ptr.buffer_id, wgu.storage.ptr.buffer_id, dx_norm_storage.ptr.buffer_id],
			[gu.ptr.offset, wgu.storage.ptr.offset, dx_norm_storage.ptr.offset],
			[1, N, C, H2, 0, 1, 0, 0], 1.0, 0.0)

		# 13. RMSNorm backward
		if recompute_x:
			self.ctx.dispatch_with_args_3d("rmsnorm_backward",
				[x_buffer_id, dx_norm_storage.ptr.buffer_id, tmp_dx.ptr.buffer_id, residual_grad.ptr.buffer_id],
				[x_offset, dx_norm_storage.ptr.offset, tmp_dx.ptr.offset, residual_grad.ptr.offset],
				[N, C, eps_bits], 256, (N+7)//8, 1, 256)
		else:
			self.ctx.dispatch_with_args_3d("rmsnorm_backward",
				[x_buffer_id, dx_norm_storage.ptr.buffer_id, dx_norm_storage.ptr.buffer_id, residual_grad.ptr.buffer_id],
				[x_offset, dx_norm_storage.ptr.offset, dx_norm_storage.ptr.offset, residual_grad.ptr.offset],
				[N, C, eps_bits], 256, (N+7)//8, 1, 256)