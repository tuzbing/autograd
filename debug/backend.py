import importlib.util
import os
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from math import prod
import struct
from scratch import align_up
from gemm import fp16_gemm_kernel_name
from utils import next_power_of_2

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

	def _dispatch_fp32_gemm(self, cfg, with_accum, buf_ids, offsets, m, n, k):
		kernel = fp16_gemm_kernel_name(cfg, with_accum, prefix="fp32")
		# print(f"need kernel {kernel}")
		if cfg["is_splitk"] and not with_accum:
			self.ctx.fill_scalar("fill_scalar_kernel", [buf_ids[2]], [offsets[2]], 0.0, m * n)
		args = (
			[m, n, k, cfg["lda"], cfg["ldb"], cfg["ldd"], cfg["tm"], cfg["tn"], cfg["split_k_partitions"], cfg["split_k_partition_stride"], cfg["split_k_partition_size"], cfg["k_iter_aligned"]]
			if cfg["is_splitk"] else
			[m, n, k, cfg["lda"], cfg["ldb"], cfg["ldd"], cfg["tm"], cfg["tn"], cfg["swizzle_log"], cfg["k_iter_aligned"], cfg["m_aligned"], cfg["n_aligned"], cfg["k_aligned"]]
		)
		self.ctx.dispatch_with_args_3d(kernel, buf_ids, offsets, args, cfg["grid_x"], cfg["grid_y"], cfg["grid_z"], cfg["wm"] * cfg["wn"] * 32)

	def _quantize_fp16_colwise(self, src_id, src_off, q_id, q_off, s_off, rows, cols, src_ld):
		groups_c = (cols + 63) // 64
		groups_r = max(1, min(32, (rows + 15) // 16))
		self.ctx.fill_scalar("fill_scalar_kernel", [q_id], [s_off], 0.0, cols)
		self.ctx.dispatch_with_args_3d("fp16_quant_col_max",
			[src_id, q_id],
			[src_off, s_off],
			[rows, cols, src_ld], groups_c * 256, groups_r, 1, 256)
		self.ctx.dispatch_with_args_3d("fp16_bits_to_scale",
			[q_id], [s_off], [cols], cols, 1, 1, min(cols, 256))
		self.ctx.dispatch_with_args_3d("fp16_quant_colwise",
			[src_id, q_id, q_id],
			[src_off, q_off, s_off],
			[rows, cols, src_ld], groups_c * 256, groups_r, 1, 256)

	def _quantize_fp16_tensor(self, src_id, src_off, q_id, q_off, s_off, max_off, numel, scale_len,
		                      row_scale_id=None, row_scale_off=0, cols=0):
		rs_id = src_id if row_scale_id is None else row_scale_id
		rs_off = src_off if row_scale_id is None else row_scale_off
		grid_x = max(1, min(64, (numel + 1023) // 1024)) * 256
		self.ctx.fill_scalar("fill_scalar_kernel", [q_id], [max_off], 0.0, 1)
		self.ctx.dispatch_with_args_3d("fp16_quant_tensor_max",
			[src_id, rs_id, q_id], [src_off, rs_off, max_off],
			[numel, cols], grid_x, 1, 1, 256)
		self.ctx.dispatch_with_args_3d("fp16_quant_tensor_cast",
			[src_id, rs_id, q_id, q_id, q_id], [src_off, rs_off, q_off, max_off, s_off],
			[numel, scale_len, cols], grid_x, 1, 1, 256)

	def _dispatch_fp16_gemm_raw(self, cfg, with_accum, buf_ids, offsets, m, n, k):
		kernel = fp16_gemm_kernel_name(cfg, with_accum, prefix="fp16")
		# print(f"Need kernel {kernel}")
		if cfg["is_splitk"] and not with_accum:
			self.ctx.fill_scalar("fill_scalar_kernel", [buf_ids[4]], [offsets[4]], 0.0, m * n)
		args = (
			[m, n, k, cfg["lda"], cfg["ldb"], cfg["ldd"], cfg["tm"], cfg["tn"], cfg["split_k_partitions"], cfg["split_k_partition_stride"], cfg["split_k_partition_size"], cfg["k_iter_aligned"]]
			if cfg["is_splitk"] else
			[m, n, k, cfg["lda"], cfg["ldb"], cfg["ldd"], cfg["tm"], cfg["tn"], cfg["swizzle_log"], cfg["k_iter_aligned"], cfg["m_aligned"], cfg["n_aligned"], cfg["k_aligned"]]
		)
		self.ctx.dispatch_with_args_3d(kernel, buf_ids, offsets, args, cfg["grid_x"], cfg["grid_y"], cfg["grid_z"], cfg["wm"] * cfg["wn"] * 32)

	def _dispatch_fp16_gemm(self, tiled_cfg, with_accum, buf_ids, offsets, quant_buf, m, n, k, a_src_ld=None, b_src_ld=None, d_dst_ld=None):
		a_T = tiled_cfg["a_transpose"]
		b_T = tiled_cfg["b_transpose"]
		sub_cfgs = tiled_cfg["sub_configs"]
		a_ld = (m if a_T else k) if a_src_ld is None else a_src_ld
		b_ld = (k if b_T else n) if b_src_ld is None else b_src_ld
		q_id, qa_off = quant_buf if isinstance(quant_buf, tuple) else (quant_buf.ptr.buffer_id, quant_buf.ptr.offset)
		if tiled_cfg["tile_dim"] == "k":
			kt = tiled_cfg["k_tile"]
			qb_off = qa_off + align_up(m * kt * 2, 16)
			sa_off = qb_off + align_up(n * kt * 2, 16)
			sb_off = sa_off + align_up(m * 4, 16)
			for k0 in range(0, k, kt):
				k_sub = min(kt, k - k0)
				a_off = offsets[0] + (k0 * a_ld * 4 if a_T else k0 * 4)
				b_off = offsets[1] + (k0 * 4 if b_T else k0 * b_ld * 4)
				if a_T:
					self._quantize_fp16_colwise(buf_ids[0], a_off, q_id, qa_off, sa_off, k_sub, m, a_ld)
				else:
					self.ctx.dispatch_with_args_3d("fp16_quant_rowwise", [buf_ids[0], q_id, q_id],
						[a_off, qa_off, sa_off],
						[m, k_sub, a_ld], 256, m, 1, 256)
				if b_T:
					self.ctx.dispatch_with_args_3d("fp16_quant_rowwise", [buf_ids[1], q_id, q_id],
						[b_off, qb_off, sb_off],
						[n, k_sub, b_ld], 256, n, 1, 256)
				else:
					self._quantize_fp16_colwise(buf_ids[1], b_off, q_id, qb_off, sb_off, k_sub, n, b_ld)
				step_accum = bool(with_accum or k0 > 0)
				self._dispatch_fp16_gemm_raw(sub_cfgs[k_sub], step_accum,
					[q_id, q_id, q_id, q_id, buf_ids[2]],
					[qa_off, qb_off, sa_off, sb_off, offsets[2]],
					m, n, k_sub)
		else:
			mt, nt = tiled_cfg["m_tile"], tiled_cfg["n_tile"]
			d_ld = n if d_dst_ld is None else d_dst_ld
			qb_off = qa_off + align_up(mt * k * 2, 16)
			sa_off = qb_off + align_up(nt * k * 2, 16)
			sb_off = sa_off + align_up(mt * 4, 16)
			for n0 in range(0, n, nt):
				n_sub = min(nt, n - n0)
				b_off = offsets[1] + (n0 * b_ld * 4 if b_T else n0 * 4)
				if b_T:
					self.ctx.dispatch_with_args_3d("fp16_quant_rowwise", [buf_ids[1], q_id, q_id],
						[b_off, qb_off, sb_off],
						[n_sub, k, b_ld], 256, n_sub, 1, 256)
				else:
					self._quantize_fp16_colwise(buf_ids[1], b_off, q_id, qb_off, sb_off, k, n_sub, b_ld)
				for m0 in range(0, m, mt):
					m_sub = min(mt, m - m0)
					a_off = offsets[0] + m0 * a_ld * 4
					d_off = offsets[2] + (m0 * d_ld + n0) * 4
					self.ctx.dispatch_with_args_3d("fp16_quant_rowwise", [buf_ids[0], q_id, q_id],
						[a_off, qa_off, sa_off],
						[m_sub, k, a_ld], 256, m_sub, 1, 256)
					self._dispatch_fp16_gemm_raw(sub_cfgs[(m_sub, n_sub)], bool(with_accum),
						[q_id, q_id, q_id, q_id, buf_ids[2]],
						[qa_off, qb_off, sa_off, sb_off, d_off],
						m_sub, n_sub, k)

	def fa_recompute_dispatch(self, head_dim, seq_len):
		if head_dim <= 0 or head_dim > 128:
			raise ValueError(f"fused LN/FA recompute only supports 0 < head_dim <= 128, got {head_dim}")
		capacity = 64 if head_dim <= 64 else 128
		return f"fused_rms_fa_forward_block{capacity}", 256, (seq_len+31)//32, 256

	def fa_from_qkv_dispatch(self, head_dim, seq_len):
		if head_dim <= 0 or head_dim > 128:
			raise ValueError(f"fused attention requires 0 < head_dim <= 512, got {head_dim}")
		if head_dim <= 64:
			return "fused_fa_from_qkv_d64", 128, (seq_len+7)//8, 128
		return "fused_fa_from_qkv_d128", 128, (seq_len+7)//8, 128

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
		for weight, dweigt in ((wq, dq), (wk, dk), (wv, dv)):
			if not weight.requires_grad:
				continue
			dw_bf16 = np.dtype(weight.grad.storage.dtype) == np.dtype(np.uint16)
			if dw_bf16:
				dw_capacity = (v.ptr.offset + v.nbytes - k.ptr.offset) // 4
				dw_tile_rows = min(C, dw_capacity // C)

				for row0 in range(0, C, dw_tile_rows):
					tile_rows = min(dw_tile_rows, C - row0)
					# tmp[tile_rows, C] = Z[:, row0:row0+tile_rows].T @ dweight
					self.ctx.dispatch_gemm_mps([q.ptr.buffer_id, dweigt.ptr.buffer_id, k.ptr.buffer_id],
						[q.ptr.offset+row0*4, dweigt.ptr.offset, k.ptr.offset],
						[1, tile_rows, C, B*T, 1, 0, 0, 0, 0, C*4], 1.0, 0.0)
					self.ctx.dispatch_with_args_3d("fp32_to_bf16",
						[k.ptr.buffer_id, weight.grad.storage.ptr.buffer_id],
						[k.ptr.offset, weight.grad.storage.ptr.offset+row0*C*2],
						[accumulate], tile_rows*C, 1, 1, min(tile_rows*C, 256))
			else:
				self.ctx.dispatch_gemm_mps(
					[q.ptr.buffer_id, dweigt.ptr.buffer_id, weight.grad.storage.ptr.buffer_id],
					[q.ptr.offset, dweigt.ptr.offset, weight.grad.storage.ptr.offset],
					[1, C, C, B*T, 1, 0, 0, 0], 1.0, float(accumulate))

		# 8. dbq/dbk/dbv
		if bq.requires_grad or bk.requires_grad or bv.requires_grad:
			bq_bf16 = bq.requires_grad and np.dtype(bq.grad.storage.dtype) == np.dtype(np.uint16)
			bk_bf16 = bk.requires_grad and np.dtype(bk.grad.storage.dtype) == np.dtype(np.uint16)
			bv_bf16 = bv.requires_grad and np.dtype(bv.grad.storage.dtype) == np.dtype(np.uint16)
			bf16_mask = int(bq_bf16) | (int(bk_bf16) << 1) | (int(bv_bf16) << 2)
			if bf16_mask:
				bias_kernel = "fused_qkv_bias_grad_bf16"
				enable_mask = int(bq.requires_grad) | (int(bk.requires_grad) << 1) | (int(bv.requires_grad) << 2)
				bias_args = [B*T, C, accumulate, enable_mask, bf16_mask]
			else:
				bias_kernel = "fused_qkv_bias_grad"
				bias_args = [B*T, C, accumulate]

			self.ctx.dispatch_with_args_3d(bias_kernel,
				[dq.ptr.buffer_id, dk.ptr.buffer_id, dv.ptr.buffer_id, 
					bq.grad.storage.ptr.buffer_id if bq.requires_grad else 0,
					bk.grad.storage.ptr.buffer_id if bk.requires_grad else 0,
					bv.grad.storage.ptr.buffer_id if bv.requires_grad else 0],
				[dq.ptr.offset, dk.ptr.offset, dv.ptr.offset, 
					bq.grad.storage.ptr.offset if bq.requires_grad else 0,
					bk.grad.storage.ptr.offset if bk.requires_grad else 0,
					bv.grad.storage.ptr.offset if bv.requires_grad else 0],
				bias_args, 128, (C + 3) // 4, 1, 128)

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
		row_scale = inputs[3] if len(inputs) > 3 else None
		out_loss = outputs[0]
		
		N = x.shape[0] * x.shape[1] if len(x.shape) == 3 else x.shape[0]
		C = x.shape[-1]
		V = w.shape[0]
		row_loss = scratch['row_loss']
		lse = scratch['lse']
		logits = scratch["logits"]
		chunk_rows = min(logits.shape[0], N)
		itemsize = np.dtype(x.dtype).itemsize
		dw_bf16 = w.requires_grad and np.dtype(w.grad.storage.dtype) == np.dtype(np.uint16)

		accumulate = bool(getattr(self, "accumulate_parameter_grads", False))
		use_fp16_gemm = bool(getattr(self, "use_fp16_gemm", False))
		ce_fp16 = use_fp16_gemm and row_scale is None
		need_grad = x.requires_grad or w.requires_grad
		if row_scale is None:
			kernel_name = "fused_linear_ce_logits_to_dlogits"
			scale = float(getattr(self, "grad_scale", 1.0)) / float(N)
			scale_bits = struct.unpack("<I", struct.pack("<f", scale))[0]
		else:
			kernel_name = "fused_linear_ce_logits_to_dlogits_row_scaled"

		if ce_fp16:
			q_buf = scratch["gemm_quant"]
			q_id = q_buf.ptr.buffer_id
			xq_off = q_buf.ptr.offset
			wq_off = xq_off + align_up(N * C * 2, 16)
			sx_off = wq_off + align_up(V * C * 2, 16)
			sw_off = sx_off + align_up(max(chunk_rows, C) * 4, 16)
			sdl_off = sw_off + align_up(max(V, C) * 4, 16)

			self._quantize_fp16_tensor(x.storage.ptr.buffer_id, x.storage.ptr.offset,
				q_id, xq_off, sx_off, sdl_off, N * C, max(chunk_rows, C))
			self._quantize_fp16_tensor(w.storage.ptr.buffer_id, w.storage.ptr.offset,
				q_id, wq_off, sw_off, sdl_off, V * C, max(V, C))
			if need_grad:
				self.ctx.fill_scalar("fill_scalar_kernel", [q_id], [sdl_off], scale / 32768.0, max(chunk_rows, V))

		for row_base in range(0, N, chunk_rows):
			rows = min(chunk_rows, N - row_base)
			x_offset = x.storage.ptr.offset + row_base*C*itemsize

			# 1) logits = X_chunk([rows, C]) @ W.T ([C, V])
			if ce_fp16:
				self._dispatch_fp16_gemm_raw(scratch.gemm_configs[("logits", rows, False)], False,
					[q_id, q_id, q_id, q_id, logits.ptr.buffer_id],
					[xq_off + row_base * C * 2, wq_off, sx_off, sw_off, logits.ptr.offset],
					rows, V, C)
			else:
				self._dispatch_fp32_gemm(scratch.gemm_configs[("logits", rows, False)], False,
					[x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
					[x_offset, w.storage.ptr.offset, logits.ptr.offset],
					rows, V, C)

			# 2) row_loss/lse/dlogits
			if row_scale is None:
				self.ctx.dispatch_with_args_3d(kernel_name,
					[logits.ptr.buffer_id, targets.storage.ptr.buffer_id, row_loss.ptr.buffer_id, lse.ptr.buffer_id],
					[logits.ptr.offset, targets.storage.ptr.offset, row_loss.ptr.offset, lse.ptr.offset],
					[rows, V, row_base, scale_bits, int(need_grad)|(int(ce_fp16)<<1)], 256, rows, 1, 256)
			else:
				inp_vars = [logits, targets.storage, row_loss, lse, row_scale.storage]
				self.ctx.dispatch_with_args_3d(kernel_name,
					[e.ptr.buffer_id for e in inp_vars], [e.ptr.offset for e in inp_vars],
					[rows, V, row_base, N, int(need_grad)], 256, rows, 1, 256)

			if not need_grad:
				continue

			# 3) dX = dlogits([rows, V]) @ W([V, C])
			if x.requires_grad:
				x_grad_offset = x.grad.storage.ptr.offset+row_base*C*itemsize
				if ce_fp16:
					self._dispatch_fp16_gemm_raw(scratch.gemm_configs[("dx", rows, False)], False,
						[logits.ptr.buffer_id, q_id, q_id, q_id, x.grad.storage.ptr.buffer_id],
						[logits.ptr.offset, wq_off, sdl_off, sw_off, x_grad_offset],
						rows, C, V)
				else:
					self._dispatch_fp32_gemm(scratch.gemm_configs[("dx", rows, False)], False,
						[logits.ptr.buffer_id, w.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
						[logits.ptr.offset, w.storage.ptr.offset, x_grad_offset],
						rows, C, V)

			# 4) dW += dlogits.T([V, rows])@X([rows, C])
			if w.requires_grad and not dw_bf16:
				with_accum = bool(accumulate or row_base != 0)
				if ce_fp16:
					self._dispatch_fp16_gemm_raw(scratch.gemm_configs[("dw", rows, with_accum)], with_accum,
						[logits.ptr.buffer_id, q_id, q_id, q_id, w.grad.storage.ptr.buffer_id],
						[logits.ptr.offset, xq_off + row_base * C * 2, sdl_off, sx_off, w.grad.storage.ptr.offset],
						V, C, rows)
				else:
					self._dispatch_fp32_gemm(scratch.gemm_configs[("dw", rows, with_accum)], with_accum,
						[logits.ptr.buffer_id, x.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
						[logits.ptr.offset, x_offset, w.grad.storage.ptr.offset],
						V, C, rows)

		self.ctx.dispatch_with_args_3d('fused_linear_ce_forward_reduce',
			[row_loss.ptr.buffer_id, out_loss.storage.ptr.buffer_id],
			[row_loss.ptr.offset, out_loss.storage.ptr.offset],
			[N], 256, 1, 1, 256)

		if not dw_bf16:
			return

		# 前半区存放dlogits, 后半区存放dW
		capacity = logits.nbytes // 4
		tile_rows = min(chunk_rows, capacity - C)
		tile_v = min(V, capacity // (tile_rows + C))
		dw_offset = logits.ptr.offset + tile_rows * tile_v * 4
		for vocab_base in range(0, V, tile_v):
			cols = min(tile_v, V - vocab_base)
			w_offset = w.storage.ptr.offset + vocab_base * C * 4

			for row_base in range(0, N, tile_rows):
				rows = min(tile_rows, N - row_base)
				x_offset = x.storage.ptr.offset + row_base*C*4

				# 1. 重算当前token块, 词表块的logits
				self._dispatch_fp32_gemm(scratch.gemm_configs[("logits_tile", rows, cols)], False,
					[x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
					[x_offset, w_offset, logits.ptr.offset],
					rows, cols, C)

				# 2. 使用阶段一保存的全词表LSE原地生成当前块的dlogits
				count = rows * cols
				self.ctx.dispatch_with_args_3d("ce_tile_logits_to_dlogits",
					[logits.ptr.buffer_id, lse.ptr.buffer_id, targets.storage.ptr.buffer_id],
					[logits.ptr.offset, lse.ptr.offset, targets.storage.ptr.offset],
					[cols, vocab_base, V, row_base, scale_bits], count, 1, 1, min(count, 256))

				# 3. dW_tile += dlogits_tile.T @ X_chunk
				with_accum = row_base != 0
				self._dispatch_fp32_gemm(scratch.gemm_configs[("dw_tile", cols, rows, with_accum)], with_accum,
					[logits.ptr.buffer_id, x.storage.ptr.buffer_id, logits.ptr.buffer_id],
					[logits.ptr.offset, x_offset, dw_offset],
					cols, C, rows)

			# 4. 转换成BF16
			self.ctx.dispatch_with_args_3d("fp32_to_bf16",
				[logits.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
				[dw_offset, w.grad.storage.ptr.offset + vocab_base*C*2],
				[int(accumulate)], cols*C, 1, 1, min(cols*C, 256))

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

	    dw_bf16 = w.requires_grad and np.dtype(w.grad.storage.dtype) == np.dtype(np.uint16)
	    db_bf16 = b.requires_grad and np.dtype(b.grad.storage.dtype) == np.dtype(np.uint16)
	    if dw_bf16:
	    	fa_k = scratch["fa_k"]
	    	fa_v = scratch["fa_v"]
	    	dw_buffer_id = fa_k.ptr.buffer_id
	    	dw_offset = fa_k.ptr.offset
	    	dw_capacity = (fa_v.ptr.offset + fa_v.nbytes - dw_offset) // 4
	    	dw_tile_k = min(K_dim, dw_capacity // N_dim)

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
	    	if dw_bf16:
	    		for k0 in range(0, K_dim, dw_tile_k):
	    			tile_k = min(dw_tile_k, K_dim - k0)
	    			self.ctx.dispatch_gemm_mps([act_buf.ptr.buffer_id, out.grad.storage.ptr.buffer_id, dw_buffer_id],
	    				[act_buf.ptr.offset + k0*4, out.grad.storage.ptr.offset, dw_offset],
	    				[1, tile_k, N_dim, B_dim*M_dim, 1, 0, 0, 0, 0, K_dim*4], 1.0, 0.0)
	    			self.ctx.dispatch_with_args_3d("fp32_to_bf16",
	    				[dw_buffer_id, w.grad.storage.ptr.buffer_id],
	    				[dw_offset, w.grad.storage.ptr.offset + k0*N_dim*2],
	    				[accumulate], tile_k*N_dim, 1, 1, min(tile_k*N_dim, 256))
	    	else:
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
	        bias_kernel = "bias_grad_reduce_bf16" if db_bf16 else "bias_grad_reduce"
	        self.ctx.dispatch_with_args_3d(bias_kernel,
	        	[out.grad.storage.ptr.buffer_id, b.grad.storage.ptr.buffer_id],
	        	[out.grad.storage.ptr.offset, b.grad.storage.ptr.offset], 
	        	[prod(x.shape[:-1]), N_dim, accumulate], 256, (N_dim+15)//16, 1, 256)

	def metal_fused_rmsnorm_mlp_forward(self, inputs, outputs, scratch=None):
		if len(inputs) == 3:
			x, wgu, wdown = inputs
			norm_weight = None
		elif len(inputs) == 4:
			x, norm_weight, wgu, wdown = inputs

		out = outputs[0]

		N = prod(x.shape[:-1])
		C = x.shape[-1]
		H2 = wgu.shape[-1]
		H = H2 // 2

		eps_bits = struct.unpack("<i", struct.pack("<f", out.op.op.eps))[0]
		use_fp16_gemm = bool(getattr(self, "use_fp16_gemm", False))

		gu = scratch["gu"]
		hidden = scratch["hidden"]

		if use_fp16_gemm:
			h2_tile, c_tile = scratch.gemm_configs["mlp_fwd_shape"]
			s1_bytes = align_up(N*4, 16)
			if "gemm_quant" in scratch._views:
				q_buf = scratch["gemm_quant"]
				s1_id = q1_id = q2_id = q_buf.ptr.buffer_id
				s1_off = q_buf.ptr.offset
				xq_off = wdq_off = s1_off + s1_bytes
			else:
				s1_id = q1_id = hidden.ptr.buffer_id
				s1_off = hidden.ptr.offset + ((N * H * 4) & ~15) - s1_bytes
				xq_off = hidden.ptr.offset
				q2_id, wdq_off = gu.ptr.buffer_id, gu.ptr.offset

			# 1. X_norm = RMSNorm(x) -> 暂存至 gu (或兜底区)
			self.ctx.fill_scalar("fill_scalar_kernel", [s1_id], [s1_off], 1.0, N)
			if norm_weight is None:
				self.ctx.dispatch_with_args_3d("rmsnorm_forward",
					[x.storage.ptr.buffer_id, q1_id],
					[x.storage.ptr.offset, xq_off],
					[C | 0x80000000, eps_bits], N * 256, 1, 1, 256)
			else:
				self.ctx.dispatch_with_args_3d("rmsnorm_weighted_forward",
					[x.storage.ptr.buffer_id, norm_weight.storage.ptr.buffer_id, q1_id],
					[x.storage.ptr.offset, norm_weight.storage.ptr.offset, xq_off],
					[C | 0x80000000, eps_bits], N * 256, 1, 1, 256)

			# 2. 量化 Wgu 至 hidden，计算 GU = X_norm @ Wgu -> gu
			wq_off = xq_off + align_up(N * C * 2, 16)
			sw_off = wq_off + align_up(C * h2_tile * 2, 16)
			gu_cfgs = scratch.gemm_configs["gu"]
			for j0 in range(0, H2, h2_tile):
				h2_sub = min(h2_tile, H2 - j0)
				self._quantize_fp16_colwise(wgu.storage.ptr.buffer_id, wgu.storage.ptr.offset + j0 * 4,
					q1_id, wq_off, sw_off, C, h2_sub, H2)
				self._dispatch_fp16_gemm_raw(gu_cfgs[h2_sub], False,
					[q1_id, q1_id, s1_id, q1_id, gu.ptr.buffer_id],
					[xq_off, wq_off, s1_off, sw_off, gu.ptr.offset + j0 * 4],
					N, h2_sub, C)

			# 3. HIDDEN = SiLU(gate) * up 直出 FP16 -> hidden 前半区
			self.ctx.dispatch_with_args_3d("swiglu_forward_kernel",
				[gu.ptr.buffer_id, hidden.ptr.buffer_id],
				[gu.ptr.offset, hidden.ptr.offset],
				[N, H | 0x80000000], N * ((H + 3) // 4), 1, 1, min(N * ((H + 3) // 4), 256))

			# 4. 复用空闲 gu 量化 Wdown，复用 s1_off 计算 OUT = HIDDEN @ Wdown -> out
			swd_off = wdq_off + align_up(H * c_tile * 2, 16)
			out_cfgs = scratch.gemm_configs["out"]
			for c0 in range(0, C, c_tile):
				c_sub = min(c_tile, C - c0)
				self._quantize_fp16_colwise(wdown.storage.ptr.buffer_id, wdown.storage.ptr.offset + c0 * 4,
					q2_id, wdq_off, swd_off, H, c_sub, C)
				self._dispatch_fp16_gemm_raw(out_cfgs[c_sub], False,
					[hidden.ptr.buffer_id, q2_id, s1_id, q2_id, out.storage.ptr.buffer_id],
					[hidden.ptr.offset, wdq_off, s1_off, swd_off, out.storage.ptr.offset + c0 * 4],
					N, c_sub, H)
			return

		# 使用 out 暂存 RMSNorm 输出，避免额外的 x_norm scratch
		if norm_weight is None:
			self.ctx.dispatch_with_args_3d("rmsnorm_forward",
				[x.storage.ptr.buffer_id, hidden.ptr.buffer_id],
				[x.storage.ptr.offset, hidden.ptr.offset],
				[C, eps_bits], N * 256, 1, 1, 256)
		else:
			self.ctx.dispatch_with_args_3d("rmsnorm_weighted_forward",
				[x.storage.ptr.buffer_id, norm_weight.storage.ptr.buffer_id, hidden.ptr.buffer_id],
				[x.storage.ptr.offset, norm_weight.storage.ptr.offset, hidden.ptr.offset],
				[C, eps_bits], N*256, 1, 1, 256)

		# GU = X_norm @ Wgu
		self.ctx.dispatch_gemm_mps(
			[hidden.ptr.buffer_id, wgu.storage.ptr.buffer_id, gu.ptr.buffer_id],
			[hidden.ptr.offset, wgu.storage.ptr.offset, gu.ptr.offset],
			[1, N, H2, C, 0, 0, 0, 0], 1.0, 0.0)

		# HIDDEN = SiLU(gate) * up
		self.ctx.dispatch_with_args_3d("swiglu_forward_kernel",
		    [gu.ptr.buffer_id, hidden.ptr.buffer_id],
		    [gu.ptr.offset, hidden.ptr.offset],
		    [N, H], N*(H+3)//4, 1, 1, min(N*(H+3)//4 , 256))

		# OUT = HIDDEN @ Wdown
		self.ctx.dispatch_gemm_mps(
		    [hidden.ptr.buffer_id, wdown.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
		    [hidden.ptr.offset, wdown.storage.ptr.offset, out.storage.ptr.offset],
		    [1, N, C, H, 0, 0, 0, 0], 1.0, 0.0)

	def metal_fused_rmsnorm_mlp_backward(self, inputs, outputs, scratch=None):
		if len(inputs) == 3:
			x, wgu, wdown = inputs
			norm_weight = None
		elif len(inputs) == 4:
			x, norm_weight, wgu, wdown = inputs

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
		use_fp16_gemm = bool(getattr(self, "use_fp16_gemm", False))

		gu = scratch['gu']
		hidden = scratch['hidden']
		tmp_dx = x.grad.storage
		residual_grad = grad_out.storage

		fla_node = x.op 
		recompute_fla = (fla_node is not None and fla_node.op.op_name == "FusedLinearAdd"
			and len(fla_node.inputs) == 4 and fla_node.inputs[1].op is not None
			and fla_node.inputs[1].op.op.op_name == "FusedRMSNormFlashAttention")
		recompute_gqa = (fla_node is not None and fla_node.op.op_name == "FusedLinearAdd"
			and len(fla_node.inputs) == 4 and fla_node.inputs[1].op is not None
			and fla_node.inputs[1].op.op.op_name == "FusedRMSNormGQA")
		recompute_x = recompute_fla or recompute_gqa

		down_bf16 = wdown.requires_grad and np.dtype(wdown.grad.storage.dtype) == np.dtype(np.uint16)
		gu_bf16 = wgu.requires_grad and np.dtype(wgu.grad.storage.dtype) == np.dtype(np.uint16)
		capacity = hidden.nbytes // 4
		if down_bf16:
			# [N, down_tile_h] + [down_tile_h, C]
			down_tile_h = min(H, capacity // (N+C))
			down_dw_relative = N * down_tile_h * 4
		if gu_bf16:
			# recompute_x=True: 前部保留X_norm[N, C], 否则X_norm在tmp_dx, 可使用整个hidden
			gu_dw_relative = N * C * 4 if recompute_x else 0
			gu_tile_r = min(C, (capacity - gu_dw_relative // 4) // H2)

		if recompute_fla:
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
		elif recompute_gqa:
			x_in, attn_out, proj_w, proj_b = fla_node.inputs
			fa_node = attn_out.op
			fa_x, fa_norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = fa_node.inputs
			B, T, C = fa_x.shape
			Hq, Hkv = fa_node.op.n_head, fa_node.op.n_kv_head
			D = C // Hq
			CKV = Hkv * D
			fa_eps_bits = struct.unpack("<i", struct.pack("<f", fa_node.op.eps))[0]
			fa_q_offset = gu.ptr.offset
			fa_k_offset = fa_q_offset + B*T*C*4
			fa_v_offset = fa_k_offset + B*T*CKV*4
			fa_out_storage = attn_out.grad.storage

			self.ctx.dispatch_with_args_3d("gqa_rms_qkv",
				[fa_x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id, wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id,
				 wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id, gu.ptr.buffer_id, gu.ptr.buffer_id, gu.ptr.buffer_id, fa_norm_weight.storage.ptr.buffer_id],
				[fa_x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset, wk.storage.ptr.offset, bk.storage.ptr.offset,
				 wv.storage.ptr.offset, bv.storage.ptr.offset, fa_q_offset, fa_k_offset, fa_v_offset, fa_norm_weight.storage.ptr.offset],
				[B, T, C, Hq, fa_eps_bits, Hkv], 256, (B*T+15)//16, 1, 256)

			negative_sin = int(fa_node.op.rope_sign < 0)
			self.ctx.dispatch_with_args_3d("gqa_rope_inplace",
				[gu.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
				[fa_q_offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
				[B, Hq, T, D, 0, negative_sin], D//2, T, B*Hq, D//2)

			self.ctx.dispatch_with_args_3d("gqa_rope_inplace",
				[gu.ptr.buffer_id, rope_cos.storage.ptr.buffer_id,rope_sin.storage.ptr.buffer_id],
			    [fa_k_offset,rope_cos.storage.ptr.offset,rope_sin.storage.ptr.offset],
			    [B, Hkv, T, D, 0, negative_sin], D//2, T, B*Hkv, D//2)

			fa_kernel = 'gqa_from_qkv_d64' if D <= 64 else 'gqa_from_qkv_d128'
			self.ctx.dispatch_with_args_3d(fa_kernel,
			    [gu.ptr.buffer_id,gu.ptr.buffer_id,gu.ptr.buffer_id,
			     fa_out_storage.ptr.buffer_id,gu.ptr.buffer_id],
			    [fa_q_offset,fa_k_offset,fa_v_offset,fa_out_storage.ptr.offset,fa_q_offset],
			    [B,Hq,T,C,int(fa_node.op.is_causal),0,Hkv],128,(T+7)//8,B*Hq,128)
			M_fla = prod(x_in.shape[:-1])
			K_fla, N_fla = attn_out.shape[-1], proj_w.shape[-1]
			self.ctx.dispatch_with_args_3d('fused_linear_add_forward_kernel',
			    [x_in.storage.ptr.buffer_id,fa_out_storage.ptr.buffer_id,proj_w.storage.ptr.buffer_id,
			     proj_b.storage.ptr.buffer_id,tmp_dx.ptr.buffer_id],
			    [x_in.storage.ptr.offset,fa_out_storage.ptr.offset,proj_w.storage.ptr.offset,
			     proj_b.storage.ptr.offset,tmp_dx.ptr.offset],
			    [M_fla,K_fla,N_fla,0],256,(N_fla+63)//64,(M_fla+63)//64,256)
			if not hasattr(self, '_recomputed_attention_cache'):
			    self._recomputed_attention_cache = set()
			self._recomputed_attention_cache.add(id(attn_out))
			x_buffer_id, x_offset = tmp_dx.ptr.buffer_id, tmp_dx.ptr.offset
			x_norm_storage = hidden
		else:
			x_buffer_id = x.storage.ptr.buffer_id
			x_offset = x.storage.ptr.offset
			x_norm_storage = tmp_dx

		# 5. X_norm = RMSNorm(x)
		if norm_weight is None:			
			self.ctx.dispatch_with_args_3d("rmsnorm_forward",
				[x_buffer_id, x_norm_storage.ptr.buffer_id],
				[x_offset, x_norm_storage.ptr.offset],
				[C, eps_bits], N * 256, 1, 1, 256)
		else:
			self.ctx.dispatch_with_args_3d("rmsnorm_weighted_forward",
				[x_buffer_id, norm_weight.storage.ptr.buffer_id, x_norm_storage.ptr.buffer_id],
				[x_offset, norm_weight.storage.ptr.offset, x_norm_storage.ptr.offset],
				[C, eps_bits], N*256, 1, 1, 256)

		# 6. GU = X_norm @ Wgu
		if use_fp16_gemm:
			self._dispatch_fp32_gemm(scratch.gemm_configs["gu"], False,
					[x_norm_storage.ptr.buffer_id, wgu.storage.ptr.buffer_id, gu.ptr.buffer_id],
					[x_norm_storage.ptr.offset, wgu.storage.ptr.offset, gu.ptr.offset], N, H2, C)
		else:
			self.ctx.dispatch_gemm_mps([x_norm_storage.ptr.buffer_id, wgu.storage.ptr.buffer_id, gu.ptr.buffer_id],
				[x_norm_storage.ptr.offset, wgu.storage.ptr.offset, gu.ptr.offset],
				[1, N, H2, C, 0, 0, 0, 0], 1.0, 0.0)

		# 7. hidden = SiLU(gate)*up & 8. dWdown = HIDDEN.T @ dOUT
		if use_fp16_gemm:
			chunk_rows, h_tile, c_tile, gu_tile_r, head_bytes, xnorm_bytes = scratch.gemm_configs["mlp_shape"]
			scale_len = max(chunk_rows, C)
			sdgu_bytes = align_up(N * 4, 16)
			if "gemm_quant" in scratch._views:
				q_buf = scratch["gemm_quant"]
				q_id = q_buf.ptr.buffer_id
				sdgu_off = q_buf.ptr.offset
				dout_q_off = sdgu_off + sdgu_bytes
				p3_off = dout_q_off
			else:
				q_id = hidden.ptr.buffer_id
				sdgu_off = hidden.ptr.offset + ((N * H * 4) & ~15) - sdgu_bytes
				dout_q_off = hidden.ptr.offset + head_bytes
				p3_off = hidden.ptr.offset + xnorm_bytes
			s_dout_off = dout_q_off + align_up(N * C * 2, 16)
			wq_off = s_dout_off + align_up(scale_len * 4, 16)
			sw_off = wq_off + align_up(h_tile * C * 2, 16)

		if down_bf16:
			dw_offset = hidden.ptr.offset + down_dw_relative
			for h0 in range(0, H, down_tile_h):
				h = min(down_tile_h, H - h0)
				# hidden头部: FP32 [N, h]
				self.ctx.dispatch_with_args_3d("swiglu_forward_tile_kernel",
					[gu.ptr.buffer_id, hidden.ptr.buffer_id],
					[gu.ptr.offset, hidden.ptr.offset],
					[N, H, h0, h], N*((h+3)//4), 1, 1, min(N*((h+3)//4), 256))
				# hidden尾部: FP32 [h, C]
				self.ctx.dispatch_gemm_mps([hidden.ptr.buffer_id, grad_out.storage.ptr.buffer_id, hidden.ptr.buffer_id],
					[hidden.ptr.offset, grad_out.storage.ptr.offset, dw_offset],
					[1, h, C, N, 1, 0, 0, 0], 1.0, 0.0)
				self.ctx.dispatch_with_args_3d("fp32_to_bf16",
					[hidden.ptr.buffer_id, wdown.grad.storage.ptr.buffer_id], [dw_offset, wdown.grad.storage.ptr.offset+h0*C*2],
					[accumulate], h*C, 1, 1, min(h*C, 256))
			if use_fp16_gemm:
				self._quantize_fp16_tensor(out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset,
					q_id, dout_q_off, s_dout_off, wq_off, N * C, scale_len)
		elif use_fp16_gemm:
			self._quantize_fp16_tensor(out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset,
				q_id, dout_q_off, s_dout_off, wq_off, N * C, scale_len)
			if wdown.requires_grad:
				self.ctx.dispatch_with_args_3d("swiglu_forward_kernel",
					[gu.ptr.buffer_id, hidden.ptr.buffer_id],
					[gu.ptr.offset, hidden.ptr.offset],
					[N, H | 0x80000000], N * ((H + 3) // 4), 1, 1, min(N * ((H + 3) // 4), 256))
				self.ctx.fill_scalar("fill_scalar_kernel", [q_id], [wq_off], 1.0, H)
				self._dispatch_fp16_gemm_raw(scratch.gemm_configs["dwdown"], bool(accumulate),
					[hidden.ptr.buffer_id, q_id, q_id, q_id, wdown.grad.storage.ptr.buffer_id],
					[hidden.ptr.offset, dout_q_off, wq_off, s_dout_off, wdown.grad.storage.ptr.offset],
					H, C, N)
		else:
			self.ctx.dispatch_with_args_3d("swiglu_forward_kernel",
				[gu.ptr.buffer_id, hidden.ptr.buffer_id],
				[gu.ptr.offset, hidden.ptr.offset],
				[N, H], N*((H+3)//4), 1, 1, min(N*((H+3)//4), 256))
			if wdown.requires_grad:
				self.ctx.dispatch_gemm_mps([hidden.ptr.buffer_id, out.grad.storage.ptr.buffer_id, wdown.grad.storage.ptr.buffer_id],
					[hidden.ptr.offset, out.grad.storage.ptr.offset, wdown.grad.storage.ptr.offset],
					[1, H, C, N, 1, 0, 0, 0], 1.0, float(accumulate))

		# 9. hidden -> dHidden & 10. gu->dGU
		if use_fp16_gemm:
			dhidden_cfgs = scratch.gemm_configs["dhidden"]
			if h_tile == H:
				self.ctx.dispatch_with_args_3d("fp16_quant_rowwise",
					[wdown.storage.ptr.buffer_id, q_id, q_id],
					[wdown.storage.ptr.offset, wq_off, sw_off],
					[H, C, C], 256, H, 1, 256)
			for r0 in range(0, N, chunk_rows):
				rows = min(chunk_rows, N - r0)
				for h0 in range(0, H, h_tile):
					h_sub = min(h_tile, H - h0)
					if h_tile < H:
						self.ctx.dispatch_with_args_3d("fp16_quant_rowwise",
							[wdown.storage.ptr.buffer_id, q_id, q_id],
							[wdown.storage.ptr.offset + h0 * C * 4, wq_off, sw_off],
							[h_sub, C, C], 256, h_sub, 1, 256)
					self._dispatch_fp16_gemm_raw(dhidden_cfgs[(rows, h_sub)], False,
						[q_id, q_id, q_id, q_id, hidden.ptr.buffer_id],
						[dout_q_off + r0 * C * 2, wq_off, s_dout_off, sw_off, hidden.ptr.offset + h0 * 4],
						rows, h_sub, C)
				self.ctx.dispatch_with_args_3d("swiglu_backward_inplace_kernel",
					[gu.ptr.buffer_id, hidden.ptr.buffer_id, q_id],
					[gu.ptr.offset + r0 * H2 * 4, hidden.ptr.offset, sdgu_off + r0 * 4],
					[rows, H | 0x80000000], 256, rows, 1, 256)
		else:
			self.ctx.dispatch_gemm_mps([out.grad.storage.ptr.buffer_id, wdown.storage.ptr.buffer_id, hidden.ptr.buffer_id],
				[out.grad.storage.ptr.offset, wdown.storage.ptr.offset, hidden.ptr.offset],
				[1, N, H, C, 0, 1, 0, 0], 1.0, 0.0)

			# 10. gu -> dGU
			self.ctx.dispatch_with_args_3d("swiglu_backward_inplace_kernel",
				[gu.ptr.buffer_id, hidden.ptr.buffer_id, hidden.ptr.buffer_id],
				[gu.ptr.offset, hidden.ptr.offset, hidden.ptr.offset],
				[N, H], N*H, 1, 1, 256)

		if recompute_x:
			if norm_weight is None:
				self.ctx.dispatch_with_args_3d("rmsnorm_forward",
					[tmp_dx.ptr.buffer_id, hidden.ptr.buffer_id],
					[tmp_dx.ptr.offset, hidden.ptr.offset],
					[C, eps_bits], N*256, 1, 1, 256)
			else:
				self.ctx.dispatch_with_args_3d("rmsnorm_weighted_forward",
					[tmp_dx.ptr.buffer_id, norm_weight.storage.ptr.buffer_id, hidden.ptr.buffer_id],
					[tmp_dx.ptr.offset, norm_weight.storage.ptr.offset, hidden.ptr.offset],
					[C, eps_bits], N*256, 1, 1, 256)
			x_norm_storage = hidden
		else:
			x_norm_storage = tmp_dx

		# 11. dWgu = X_norm.T @ dGU
		if wgu.requires_grad:
			if use_fp16_gemm:
				zq_off = p3_off
				sz_off = zq_off + align_up(N * C * 2, 16)
				s1_off = sz_off + align_up(C * 4, 16)
				self._quantize_fp16_tensor(x_norm_storage.ptr.buffer_id, x_norm_storage.ptr.offset,
					q_id, zq_off, sz_off, s1_off, N * C, C, q_id, sdgu_off, C)
				self.ctx.fill_scalar("fill_scalar_kernel", [q_id], [s1_off], 1.0, H2)
				if gu_bf16:
					dw_tmp_off = s1_off + align_up(H2 * 4, 16)
					dwgu_bf16_cfgs = scratch.gemm_configs["dwgu_bf16"]
					for c0 in range(0, C, gu_tile_r):
						r = min(gu_tile_r, C - c0)
						self._dispatch_fp16_gemm_raw(dwgu_bf16_cfgs[r], False,
							[q_id, gu.ptr.buffer_id, q_id, q_id, q_id],
							[zq_off + c0 * 2, gu.ptr.offset, sz_off, s1_off, dw_tmp_off],
							r, H2, N)
						self.ctx.dispatch_with_args_3d("fp32_to_bf16",
							[q_id, wgu.grad.storage.ptr.buffer_id], [dw_tmp_off, wgu.grad.storage.ptr.offset + c0 * H2 * 2],
							[accumulate], r * H2, 1, 1, min(r * H2, 256))
				else:
					self._dispatch_fp16_gemm_raw(scratch.gemm_configs["dwgu"], bool(accumulate),
						[q_id, gu.ptr.buffer_id, q_id, q_id, wgu.grad.storage.ptr.buffer_id],
						[zq_off, gu.ptr.offset, sz_off, s1_off, wgu.grad.storage.ptr.offset],
						C, H2, N)
			elif gu_bf16:
				dw_offset = hidden.ptr.offset + gu_dw_relative
				for c0 in range(0, C, gu_tile_r):
					r = min(gu_tile_r, C - c0)
					self.ctx.dispatch_gemm_mps([x_norm_storage.ptr.buffer_id, gu.ptr.buffer_id, hidden.ptr.buffer_id],
						[x_norm_storage.ptr.offset+c0*4, gu.ptr.offset, dw_offset],
						[1, r, H2, N, 1, 0, 0, 0, 0, C*4], 1.0, 0.0)
					self.ctx.dispatch_with_args_3d("fp32_to_bf16",
						[hidden.ptr.buffer_id, wgu.grad.storage.ptr.buffer_id], [dw_offset, wgu.grad.storage.ptr.offset+c0*H2*2],
						[accumulate], r*H2, 1, 1, min(r*H2, 256))
			else:
				main_rows = (N // 16) * 16
				tail_rows = N - main_rows
				if main_rows > 0:
						self.ctx.dispatch_gemm_mps([x_norm_storage.ptr.buffer_id, gu.ptr.buffer_id, wgu.grad.storage.ptr.buffer_id],
							[x_norm_storage.ptr.offset, gu.ptr.offset, wgu.grad.storage.ptr.offset],
							[1, C, H2, main_rows, 1, 0, 0, 0], 1.0, float(accumulate))
				if tail_rows:
					self.ctx.dispatch_with_args_3d("fused_rmsnorm_linear_ce_dw_tail",
						[x_norm_storage.ptr.buffer_id, gu.ptr.buffer_id, wgu.grad.storage.ptr.buffer_id],
						[x_norm_storage.ptr.offset+main_rows*C*4, gu.ptr.offset+main_rows*H2*4, wgu.grad.storage.ptr.offset],
						[tail_rows, H2, C, int(accumulate or main_rows > 0)], (H2+3)//4, C, 1, min((H2+3)//4, 128))

		# 12. dX_norm
		if recompute_x:
			dx_norm_storage = hidden
		else:
			dx_norm_storage = tmp_dx

		if use_fp16_gemm:
			wgq_off = p3_off
			swg_off = wgq_off + align_up(c_tile * H2 * 2, 16)
			dx_cfgs = scratch.gemm_configs["dx_norm"]
			for c0 in range(0, C, c_tile):
				c_sub = min(c_tile, C - c0)
				self.ctx.dispatch_with_args_3d("fp16_quant_rowwise",
					[wgu.storage.ptr.buffer_id, q_id, q_id],
					[wgu.storage.ptr.offset + c0 * H2 * 4, wgq_off, swg_off],
					[c_sub, H2, H2], 256, c_sub, 1, 256)
				self._dispatch_fp16_gemm_raw(dx_cfgs[c_sub], False,
					[gu.ptr.buffer_id, q_id, q_id, q_id, dx_norm_storage.ptr.buffer_id],
					[gu.ptr.offset, wgq_off, sdgu_off, swg_off, dx_norm_storage.ptr.offset + c0 * 4],
					N, c_sub, H2)
		else:
			self.ctx.dispatch_gemm_mps([gu.ptr.buffer_id, wgu.storage.ptr.buffer_id, dx_norm_storage.ptr.buffer_id],
				[gu.ptr.offset, wgu.storage.ptr.offset, dx_norm_storage.ptr.offset],
				[1, N, C, H2, 0, 1, 0, 0], 1.0, 0.0)

		# 13. RMSNorm backward
		if norm_weight is not None and norm_weight.requires_grad:
			norm_grad_bf16 = np.dtype(norm_weight.grad.storage.dtype) == np.dtype(np.uint16)
			self.ctx.dispatch_with_args_3d("rmsnorm_weighted_dgamma",
				[x_buffer_id, dx_norm_storage.ptr.buffer_id, norm_weight.grad.storage.ptr.buffer_id],
				[x_offset, dx_norm_storage.ptr.offset, norm_weight.grad.storage.ptr.offset],
				[N, C, eps_bits, int(accumulate), int(norm_grad_bf16)], C, 1, 1, min(C, 256))

		if norm_weight is None:
			self.ctx.dispatch_with_args_3d("rmsnorm_backward",
				[x_buffer_id, dx_norm_storage.ptr.buffer_id, tmp_dx.ptr.buffer_id, residual_grad.ptr.buffer_id],
				[x_offset, dx_norm_storage.ptr.offset, tmp_dx.ptr.offset, residual_grad.ptr.offset],
				[N, C, eps_bits], 256, (N+7)//8, 1, 256)
		else:
			inp_bufs = [x_buffer_id, dx_norm_storage.ptr.buffer_id, norm_weight.storage.ptr.buffer_id,
			            tmp_dx.ptr.buffer_id if recompute_x else dx_norm_storage.ptr.buffer_id, 
			residual_grad.ptr.buffer_id]
			inp_offs = [x_offset, dx_norm_storage.ptr.offset, norm_weight.storage.ptr.offset, 
			            tmp_dx.ptr.offset if recompute_x else dx_norm_storage.ptr.offset, residual_grad.ptr.offset]
			self.ctx.dispatch_with_args_3d("rmsnorm_weighted_backward",
				[x_buffer_id, dx_norm_storage.ptr.buffer_id, norm_weight.storage.ptr.buffer_id, tmp_dx.ptr.buffer_id, residual_grad.ptr.buffer_id],
				[x_offset, dx_norm_storage.ptr.offset, norm_weight.storage.ptr.offset, tmp_dx.ptr.offset, residual_grad.ptr.offset],
				[N, C, eps_bits], 256, (N+7)//8, 1, 256)

