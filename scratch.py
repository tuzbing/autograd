
import numpy as np
from math import prod

from op import AddOp, MulOp, RsqrtOp, SoftmaxOp, GELUOp, LayerNormOp, MatMulOp, PermuteOp, TransposeOp, CrossEntropyLossOp
from op import ScaledDotProductAttenOp, FusedLinearCrossEntropyOp, FusedMLPOp, FusedQKVProjOp, FlashAttentionOp
from op import FusedRMSNormLinearCrossEntropyOp
from gemm import FP16_QUANT_BUDGET_BYTES, get_fp16_gemm_config, fp16_gemm_k_tile, build_fp16_k_tile_configs

def align_up(n, align=64):
	return (n + align - 1) // align * align

def nbytes_of(shape, dtype):
	return prod(shape) * np.dtype(dtype).itemsize

def _reduce_keepdims_shape(shape, axis):
	shape = list(shape)
	shape[axis % len(shape)] = 1 
	return tuple(shape)

def _matmul_dw_shape(x_shape, grad_shape):
	batch = np.broadcast_shapes(x_shape[:-2], grad_shape[:-2])
	return batch + (x_shape[-1], grad_shape[-1])

def _matmul_acc_shape(x_shape, out_shape):
	dw_shape = _matmul_dw_shape(x_shape, out_shape)
	return x_shape if prod(x_shape) >= prod(dw_shape) else dw_shape

def arena_view_u8(arena_u8, offset, shape, dtype):
	dt = np.dtype(dtype)
	nbytes = prod(shape) * dt.itemsize
	end = offset + nbytes
	if end > arena_u8.nbytes:
		raise RuntimeError(f"scratch overflow: need {end}, have {arena_u8.nbytes}")
	return np.ndarray(shape, dtype=dt, buffer=arena_u8.data, offset=int(offset))

class ScratchLayout:
	__slots__ = ("slots", "nbytes", "gemm_configs")
	def __init__(self, slots, nbytes, gemm_configs=None):
		self.slots = slots 
		self.nbytes = nbytes
		self.gemm_configs = {} if gemm_configs is None else gemm_configs

class ScratchViews:
	__slots__ = ('_views', "gemm_configs")
	def __init__(self, views, gemm_configs=None):
		self._views = views
		self.gemm_configs = {} if gemm_configs is None else gemm_configs
	def __getitem__(self, key):
		return self._views[key]

class ScratchArena:
	__slots__ = ("nbytes", "storage")
	def __init__(self, nbytes, allocator):
		self.nbytes = int(nbytes)
		self.storage = allocator.alloc((self.nbytes,), np.uint8)
	def bind(self, layout, allocator):
		if layout is None:
			return None 
		views = {}
		for name, (off, shape, dtype) in layout.slots.items():
			views[name] = allocator.view(self.storage, off, shape, dtype)
		return ScratchViews(views)

def pack_slots(slot_specs, align=64):
	offset = 0
	slots = {}
	for name, shape, dtype in slot_specs:
		offset = align_up(offset, align)
		slots[name] = (offset, tuple(shape), np.dtype(dtype))
		offset += nbytes_of(shape, dtype)
	return ScratchLayout(slots, align_up(offset, align))

def pack_slots_reuse(groups, align=64):
	offset = 0  
	slots = {}
	for group in groups:
		offset = align_up(offset, align)
		max_n = 0
		for name, shape, dtype in group:
			n = nbytes_of(shape, dtype)
			max_n = max(max_n, n)
			slots[name] = (offset, tuple(shape), np.dtype(dtype))
		offset += max_n  
	return ScratchLayout(slots, align_up(offset, align))

def fp16_gemm_quant_size(*tiled_cfgs, reuse_bytes=0):
	need = max((cfg["quant_bytes"] for cfg in tiled_cfgs), default=0)
	return 0 if need <= reuse_bytes else align_up(need, 64)

def fp16_quant_bytes(rows, cols, scale_len):
	return align_up(rows * cols * 2, 16) + align_up(scale_len * 4, 16)

def plan_node_scratch(node, use_f16_gemm=False):
	op = node.op 
	inputs, outputs = node.inputs, node.outputs
	fwd, bwd = None, None 

	if isinstance(op, AddOp):
		x, y = inputs[0], inputs[1]
		bwd = pack_slots_reuse([
			[('tmp_x', x.shape, x.dtype)],
			[('tmp_y', y.shape, y.dtype)],
		])
	elif isinstance(op, MulOp):
		x, y, out = inputs[0], inputs[1], outputs[0]
		bwd = pack_slots_reuse([
			[('full', out.shape, out.dtype)],
			[('red_x', x.shape, x.dtype), ('red_y', y.shape, y.dtype)],
		])
	elif isinstance(op, RsqrtOp):
		x = inputs[0]
		fwd = pack_slots([("tmp", x.shape, x.dtype)])
	elif isinstance(op, SoftmaxOp):
		out = outputs[0]
		red = _reduce_keepdims_shape(out.shape, op.dim)
		fwd = pack_slots([("max", red, out.dtype), ("sum", red, out.dtype)])
		bwd = pack_slots_reuse([
			[('dot', red, out.dtype)],
			[('t', out.shape, out.dtype)],
		])
	elif isinstance(op, GELUOp):
		x = inputs[0]
		fwd = pack_slots_reuse([[("buf", x.shape, x.dtype)],
		])
		bwd = pack_slots_reuse([
			[('buf', x.shape, x.dtype)],
			[('acc', x.shape, x.dtype)],
		])
	elif isinstance(op, LayerNormOp):
		x, gamma, beta = inputs[0], inputs[1], inputs[2]
		axis = op.normalized_dim 
		red = _reduce_keepdims_shape(x.shape, axis)
		fwd = pack_slots_reuse([
			[("red", red, x.dtype)],
			[("full", x.shape, x.dtype)],
		])
		bwd = pack_slots_reuse([
			[('red', red, x.dtype)],
			[('full', x.shape, x.dtype)],
			[('red2', red, x.dtype)],
			[('full2', x.shape, x.dtype)],
			[('full3', x.shape, x.dtype)],
			[('gamma_tmp', gamma.shape, gamma.dtype), ('beta_tmp', beta.shape, beta.dtype)],
		])
	elif isinstance(op, CrossEntropyLossOp):
		logits = inputs[0]
		red = _reduce_keepdims_shape(logits.shape, -1)
		fwd = pack_slots_reuse([
			[("red", red, logits.dtype)],
			[('full', logits.shape, logits.dtype)],
		])
		bwd = pack_slots_reuse([
			[('red', red, logits.dtype)],
			[('full', logits.shape, logits.dtype)],
		])
	elif isinstance(op, MatMulOp):
		x, out = inputs[0], outputs[0]
		if x.requires_grad or inputs[1].requires_grad or out.requires_grad:
			dw_shape = _matmul_dw_shape(x.shape, out.shape)
			acc_shape = _matmul_acc_shape(x.shape, out.shape)
			bwd = pack_slots_reuse([
				[('acc', acc_shape, out.dtype)],
				[('tmp', dw_shape, out.dtype)],
			])
	elif isinstance(op, PermuteOp):
		x = inputs[0]
		bwd = pack_slots([('t', x.shape, x.dtype)])
	elif isinstance(op, TransposeOp):
		x = inputs[0]
		bwd = pack_slots([('t', x.shape, x.dtype)])
	elif isinstance(op, ScaledDotProductAttenOp):
		q = inputs[0]
		B, H, T, D = q.shape 
		fwd = pack_slots_reuse([
			[('attn', (B, H, T, T), q.dtype)],
			[('row_sum', (B, H, T, 1), q.dtype), ('row_max', (B, H, T, 1), q.dtype)],
			[('mask', (T, T), bool)],
		])
		bwd = pack_slots_reuse([
			[('probs', (B, H, T, T), q.dtype)],
			[('work', (B, H, T, T), q.dtype)],
			[('tmp', (B, H, T, T), q.dtype)],
			[('row_sum', (B, H, T, 1), q.dtype), ('row_max', (B, H, T, 1), q.dtype)],
			[('tmp_v', (B, H, T, D), q.dtype), ('tmp_q', (B, H, T, D), q.dtype), ('tmp_k', (B, H, T, D), q.dtype)],
			[('mask', (T, T), bool)],
		])
	elif isinstance(op, FusedLinearCrossEntropyOp):
		x, w = inputs[0], inputs[1]
		V = w.shape[0]
		C = x.shape[-1]
		N = prod(x.shape[:-1])
		logits_budget_elements = max(V, 8*N*C)
		logits_rows = max(1, min(512, logits_budget_elements//V, N))

		if x.device in ("mps", "cuda"):
			vocab_blocks = (V+95)//96
			slots = [
				('row_loss', (N, 1), x.dtype),
				('lse',      (N, 1), x.dtype),
				('logits',   (logits_rows, V), x.dtype),
			]

			configs = {}
			row_sizes = [logits_rows]
			if N % logits_rows:
				row_sizes.append(N % logits_rows)
			ce_fp16 = use_f16_gemm and len(inputs) == 3
			gemm_dtype = "fp16" if ce_fp16 else "fp32"
			dl_lda = (2 * V) if ce_fp16 else None
			for rows in row_sizes:
				configs[("logits", rows, False)] = get_fp16_gemm_config(M=rows, N=V, K=C, with_accum=False, b_transpose=True, dtype=gemm_dtype)
			if x.requires_grad:
				# dx[rows, C] = dlogits[rows, V] @ W[V, C]
				for rows in row_sizes:
					configs[("dx", rows, False)] = get_fp16_gemm_config(M=rows, N=C, K=V, with_accum=False, b_transpose=False, dtype=gemm_dtype, lda=dl_lda)
			if w.requires_grad:
				# dW[V,C] += dlogits.T[V,rows] @ X[rows,C]
				for rows in row_sizes:
					for accum in (False, True):
						configs["dw", rows, accum] = get_fp16_gemm_config(M=V, N=C, K=rows, with_accum=accum, a_transpose=True, b_transpose=False, dtype=gemm_dtype, lda=dl_lda)

			capacity = logits_rows * V
			tile_rows = min(logits_rows, capacity - C)
			tile_v = min(V, capacity // (tile_rows + C))
			tile_row_sizes = [tile_rows] + ([N % tile_rows] if N % tile_rows else [])
			tile_col_sizes = [tile_v] + ([V % tile_v] if V % tile_v else [])
			for rows in tile_row_sizes:
				for cols in tile_col_sizes:
					configs[("logits_tile", rows, cols)] = get_fp16_gemm_config(M=rows, N=cols, K=C, with_accum=False, b_transpose=True, dtype="fp32")
					for accum in (False, True):
						configs[("dw_tile", cols, rows, accum)] = get_fp16_gemm_config(M=cols, N=C, K=rows, with_accum=accum, a_transpose=True, b_transpose=False, dtype="fp32")

			if ce_fp16:
				sdl_len = max(2, logits_rows, V) if (x.requires_grad or w.requires_grad) else 2
				quant_bytes = align_up(N*C*2, 16) + align_up(V*C*2, 16) + align_up(max(logits_rows, C)*4, 16) + align_up(max(V, C)*4, 16) + align_up(sdl_len*4, 16)
				slots.append(("gemm_quant", (quant_bytes,), np.uint8))

			fwd = pack_slots(slots)
			fwd.gemm_configs = configs
			bwd = None
		else:
			fwd = pack_slots_reuse([
				[('logits', (1024, V), x.dtype)],
				[('max', (1024, 1), x.dtype), ('sum', (1024, 1), x.dtype)],
			])
			bwd = pack_slots_reuse([
				[('probs', (1024,V), x.dtype)],
				[('max', (1024, 1), x.dtype), ('sum', (1024, 1), x.dtype)],
				[('tmp_dx', (1024, C), x.dtype), ('tmp_dw', (V, C), x.dtype)]
			])
	elif isinstance(op, FusedRMSNormLinearCrossEntropyOp):
		x, w, norm_weight, targets = inputs[:4]
		V = w.shape[0]
		C = x.shape[-1]
		N = prod(x.shape[:-1])

		if x.device in ("mps", "cuda"):
			mlp_node = x.op.inputs[1].op if (x.op is not None and x.op.op.op_name == "Add" and len(x.op.inputs) == 2) else None
			if mlp_node is not None and mlp_node.op.op_name == "FusedRMSNormMLP":
				H = mlp_node.inputs[-2].shape[1] // 2
				avail_bytes = align_up(N * 2 * H * 4, 64) + align_up(N * H * 4, 64) - 2 * align_up(N * 4, 64) - 512
				raw_rows = max(1, min(512, avail_bytes // ((V + C + 1) * 4), N))
				logits_rows = (1 << (raw_rows.bit_length() - 1)) if raw_rows >= 64 else ((raw_rows // 16) * 16 if raw_rows >= 16 else raw_rows)
			else:
				logits_budget_elements = max(V, 8*N*C)
				logits_rows = max(1, min(512, logits_budget_elements//V, N))

			fwd = pack_slots_reuse([
				[("row_loss", (N, 1), x.dtype)],
				[("lse", (N, 1), x.dtype)],
				[("logits", (logits_rows, V), x.dtype)],
				[("x_norm", (logits_rows, C), x.dtype), ("dx_norm", (logits_rows, C), x.dtype)],
				[("inv_rms", (logits_rows, 1), x.dtype)],
			])
			bwd = None
		else:
			fwd = pack_slots_reuse([
				[("logits", (N, V), x.dtype)],
				[("x_norm", (N, C), x.dtype)],
				[("inv_rms", (N, 1), x.dtype)],
				[("max", (N, 1), x.dtype)], 
				[("sum", (N, 1), x.dtype)], 
				[("target_logits", (N, 1), x.dtype)],
			])
			bwd = pack_slots_reuse([
				[("logits", (N, V), x.dtype)],
				[("x_norm", (N, C), x.dtype)],
				[("dx_norm", (N, C), x.dtype)],
				[("inv_rms", (N, 1), x.dtype)],
				[("max", (N, 1), x.dtype), ("sum", (N, 1), x.dtype)],
				[("tmp_x", (N, C), x.dtype)],
				[("tmp_dw", (V, C), x.dtype)],
				[("tmp_dgamma", (C,), x.dtype)],
			])
	elif isinstance(op, FusedMLPOp):
		x, w1, b1, w2, b2 = inputs
		C4 = w1.shape[-1]
		h_shape = x.shape[:-1] + (C4,)
		fwd = pack_slots([
			('h1', h_shape, x.dtype),
			('h2', h_shape, x.dtype),
			('tmp', h_shape, x.dtype),
		])
		bwd = pack_slots([
			('h1', h_shape, x.dtype),
			('h2', h_shape, x.dtype),
			('dh_buf', h_shape, x.dtype),
			('tmp1', h_shape, x.dtype),
			('tmp2', h_shape, x.dtype),
			('tmp_dx', x.shape, x.dtype),
			('tmp_dw1', w1.shape, x.dtype),
			('tmp_dw2', w2.shape, x.dtype),
			('tmp_db1', b1.shape, x.dtype),
			('tmp_db2', b2.shape, x.dtype),
		])
	elif type(op).__name__ == "FusedQKVProjOp":
		x = inputs[0]
		B, T, C = x.shape
		fwd = pack_slots([
			('tmp_q', (B, T, C), x.dtype),
			('tmp_k', (B, T, C), x.dtype),
			('tmp_v', (B, T, C), x.dtype),
		])
		bwd = pack_slots_reuse([
			[('tmp_dq', (B, T, C), x.dtype)],
			[('tmp_dk', (B, T, C), x.dtype)],
			[('tmp_dv', (B, T, C), x.dtype)],
			[('tmp_dx', (B, T, C), x.dtype)],
			[('tmp_dx_k', (B, T, C), x.dtype), ('tmp_dx_v', (B, T, C), x.dtype)],
		])
	elif type(op).__name__ == "FusedRMSNormMLPOp":
		if len(inputs) == 3:
			x, wgu, wdown = inputs
		elif len(inputs) == 4:
			x, norm_weight, wgu, wdown = inputs
		H = wgu.shape[1] // 2
		gu_shape = x.shape[:-1] + (2*H,)
		hidden_shape = x.shape[:-1] + (H,)

		if x.device in ("mps", "cuda"):
			fwd_slots = [
				('gu', gu_shape, x.dtype),
				('hidden', hidden_shape, x.dtype),
			]
			if use_f16_gemm:
				N = prod(x.shape[:-1])
				C = x.shape[-1]
				H2 = wgu.shape[-1]
				gu_cap = (N * H2 * 4) & ~15
				hid_cap = (N * H * 4) & ~15
				s1_bytes = align_up(N * 4, 16)
				xq_bytes = align_up(N * C * 2, 16)
				max_h2 = max(1, (hid_cap - s1_bytes - xq_bytes - 32) // (C * 2 + 4))
				h2_tile = min(H2, (max_h2 // 64) * 64 if max_h2 >= 64 else max(1, (max_h2 // 8) * 8))
				wgu_fwd_bytes = fp16_quant_bytes(C, h2_tile, h2_tile)

				max_cf = max(1, (gu_cap - 32) // (H * 2 + 4))
				c_fwd_tile = min(C, (max_cf // 64) * 64 if max_cf >= 64 else max(1, (max_cf // 8) * 8))
				wdown_fwd_bytes = fp16_quant_bytes(H, c_fwd_tile, c_fwd_tile)

				need_hid = max(align_up(N * H * 2, 16), xq_bytes + wgu_fwd_bytes) + s1_bytes
				if need_hid > hid_cap or wdown_fwd_bytes > gu_cap:
					fwd_slots.append(('gemm_quant', (align_up(s1_bytes + max(xq_bytes + wgu_fwd_bytes, wdown_fwd_bytes), 64),), np.uint8))

				h2_sizes = sorted({min(h2_tile, H2 - j0) for j0 in range(0, H2, h2_tile)})
				cf_sizes = sorted({min(c_fwd_tile, C - c0) for c0 in range(0, C, c_fwd_tile)})
				fwd = pack_slots(fwd_slots)
				fwd.gemm_configs = {
					"mlp_fwd_shape": (h2_tile, c_fwd_tile),
					"gu": {
						j: get_fp16_gemm_config(M=N, N=j, K=C, with_accum=False, a_transpose=False, b_transpose=False, dtype="fp16", ldd=H2)
						for j in h2_sizes
					},
					"out": {
						c: get_fp16_gemm_config(M=N, N=c, K=H, with_accum=False, a_transpose=False, b_transpose=False, dtype="fp16", ldd=C)
						for c in cf_sizes
					},
				}
			else:
				fwd = pack_slots(fwd_slots)
			bwd_slots = [
				('gu', gu_shape, x.dtype),
				('hidden', hidden_shape, x.dtype),
			]
			if use_f16_gemm:
				N = prod(x.shape[:-1])
				C = x.shape[-1]
				H2 = wgu.shape[-1]
				gu_bf16 = wgu.requires_grad and str(getattr(wgu, "grad_dtype", "")).lower() in ("bf16", "bfloat16")
				chunk_rows = (N // 128) * 64 if N >= 128 else (max(4, (N // 8) * 4) if N >= 8 else 1)
				sdgu_bytes = align_up(N * 4, 16)
				hid_cap = (N * H * 4) & ~15
				cap_12 = hid_cap - sdgu_bytes
				head_bytes = max(align_up(N * H * 2, 16), align_up(chunk_rows * H * 4, 16))
				dout_bytes = fp16_quant_bytes(N, C, max(chunk_rows, C))
				rem_12 = cap_12 - head_bytes - dout_bytes
				h_tile = min(H, max(64, ((rem_12 - 32) // (C * 2 + 4) // 64) * 64))
				wdown_bytes = max(align_up(H * 4, 16), fp16_quant_bytes(h_tile, C, h_tile))

				xnorm_bytes = align_up(N * C * 4, 16)
				avail_3 = cap_12 - xnorm_bytes
				z_bytes = fp16_quant_bytes(N, C, C) + align_up(H2 * 4, 16)
				gu_tile_r = C
				step11_bytes = z_bytes if wgu.requires_grad else 0
				max_c = max(1, (avail_3 - 32) // (H2 * 2 + 4))
				c_tile = min(C, (max_c // 64) * 64 if max_c >= 64 else max(1, (max_c // 8) * 8))
				wgu_bytes = fp16_quant_bytes(c_tile, H2, c_tile)

				need_gu = xq_bytes + wgu_fwd_bytes + sdgu_bytes
				need_a = head_bytes + dout_bytes + wdown_bytes + sdgu_bytes
				need_b = xnorm_bytes + max(step11_bytes, wgu_bytes) + sdgu_bytes
				if max(need_gu, need_a, need_b) > hid_cap:
					bwd_slots.append(('gemm_quant', (align_up(max(xq_bytes + wgu_fwd_bytes, dout_bytes + wdown_bytes, step11_bytes, wgu_bytes) + sdgu_bytes, 64),), np.uint8))

				r_sizes = sorted({min(chunk_rows, N - r0) for r0 in range(0, N, chunk_rows)})
				h_sizes = sorted({min(h_tile, H - h0) for h0 in range(0, H, h_tile)})
				c_sizes = sorted({min(c_tile, C - c0) for c0 in range(0, C, c_tile)})
				bwd_configs = {
					"gu": fwd.gemm_configs["gu"],
					"mlp_shape": (h2_tile, chunk_rows, h_tile, c_tile, gu_tile_r, head_bytes, xnorm_bytes),
					"dhidden": {
						(r, h): get_fp16_gemm_config(M=r, N=h, K=C, a_transpose=False, b_transpose=True, dtype="fp16", ldd=H)
						for r in r_sizes for h in h_sizes
					},
					"dx_norm": {
						c: get_fp16_gemm_config(M=N, N=c, K=H2, a_transpose=False, b_transpose=True, dtype="fp16", lda=2*H2, ldd=C)
						for c in c_sizes
					},
				}
				if wdown.requires_grad:
					bwd_configs["dwdown"] = get_fp16_gemm_config(M=H, N=C, K=N, a_transpose=True, b_transpose=False, dtype="fp16")
				if wgu.requires_grad:
					bwd_configs["dwgu"] = get_fp16_gemm_config(M=C, N=H2, K=N, a_transpose=True, b_transpose=False, dtype="fp16", ldb=2*H2)
				bwd = pack_slots(bwd_slots)
				bwd.gemm_configs = bwd_configs
			else:
				bwd = pack_slots(bwd_slots)
		else:
			fwd = pack_slots([
				# RMSNorm
				('inv_rms', x.shape[:-1] + (1,), x.dtype),
				('x_norm', x.shape, x.dtype),
				# SwiGLU
				("gu", gu_shape, x.dtype),
				("hidden", hidden_shape, x.dtype),
			])
			bwd = pack_slots([
				# RMSNorm
				('inv_rms', x.shape[:-1]+(1,), x.dtype),
				('x_norm', x.shape, x.dtype),
				('dx_norm', x.shape, x.dtype),
				# GU
				('gu', gu_shape, x.dtype),

				('dhidden', hidden_shape, x.dtype),
				('gate', hidden_shape, x.dtype),
				('sigmoid', hidden_shape, x.dtype),
				('tmp_dwgu', wgu.shape, x.dtype),
				('tmp_dwdown', wdown.shape, x.dtype),
			])
	elif type(op).__name__ == "FlashAttentionOp":
		x = inputs[0]
		B, T, C = x.shape 
		# H, D = outputs[0].shape[1], outputs[0].shape[3]
		H = op.n_head
		D = C // H
		if x.device in ("mps", "cuda"):
			fwd = pack_slots([
				('k', (B, H, T, D), x.dtype),
				('v', (B, H, T, D), x.dtype),
			])
			bwd = pack_slots([
				('k', (B, H, T, D), x.dtype),
				('v', (B, H, T, D), x.dtype),
				('lse', (B, H, T), x.dtype),
				('d_val', (B, H, T), x.dtype),
				('q', (B, H, T, D), x.dtype),
				('dq', (B, H, T, D), x.dtype),
				('dk', (B, H, T, D), x.dtype),
				('dv', (B, H, T, D), x.dtype),
			])
		else:
			fwd = pack_slots([
				('q', (B, T, H, D), x.dtype),
				('k', (B, T, H, D), x.dtype),
				('v', (B, T, H, D), x.dtype),
				('attn', (B, H, T, T), x.dtype),
				('row_sum', (B, H, T, 1), x.dtype), 
				('row_max', (B, H, T, 1), x.dtype),
				('mask', (T, T), bool),
			])
			bwd = pack_slots([
				('tmp_dq', (B, H, T, D), x.dtype),
				('tmp_dk', (B, H, T, D), x.dtype),
				('tmp_dv', (B, H, T, D), x.dtype),
				('probs', (B, H, T, T), x.dtype),
				('work', (B, H, T, T), x.dtype),
				('tmp', (B, H, T, T), x.dtype),
				('row_sum', (B, H, T, 1), x.dtype), 
				('row_max', (B, H, T, 1), x.dtype),
				('tmp_dx', (B, T, C), x.dtype),
				('tmp_dx_k', (B, T, C), x.dtype),
				('tmp_dx_v', (B, T, C), x.dtype),
				('mask', (T, T), bool),
			])
	elif type(op).__name__ == "FusedRMSNormFlashAttentionOp":
		x = inputs[0]
		B, T, C = x.shape
		H = op.n_head
		D = C // H 
		if x.device in ("mps", "cuda"):
			fwd = pack_slots([
				('q', (B, H, T, D), x.dtype),
				('k', (B, H, T, D), x.dtype),
				('v', (B, H, T, D), x.dtype),
			])
			bwd = pack_slots([
				('k', (B, H, T, D), x.dtype),
				('v', (B, H, T, D), x.dtype),
				('lse', (B, H, T), x.dtype),
				('d_val', (B, H, T), x.dtype),
				('q', (B, H, T, D), x.dtype),
				('dq', (B, H, T, D), x.dtype),
				('dk', (B, H, T, D), x.dtype),
				('dv', (B, H, T, D), x.dtype),
			])
		else:
			fwd = pack_slots([
				('inv_rms', (B, T, 1), x.dtype),
				('x_norm', (B, T, C), x.dtype),
				('q', (B, T, H, D), x.dtype),
				('k', (B, T, H, D), x.dtype),
				('v', (B, T, H, D), x.dtype),
				('attn', (B, H, T, T), x.dtype),
				('row_sum', (B, H, T, 1), x.dtype), 
				('row_max', (B, H, T, 1), x.dtype),
				('mask', (T, T), bool),
			])
			bwd = pack_slots([
				('tmp_dq', (B, H, T, D), x.dtype),
				('tmp_dk', (B, H, T, D), x.dtype),
				('tmp_dv', (B, H, T, D), x.dtype),
				('probs', (B, H, T, T), x.dtype),
				('work', (B, H, T, T), x.dtype),
				('tmp', (B, H, T, T), x.dtype),
				('row_sum', (B, H, T, 1), x.dtype), 
				('row_max', (B, H, T, 1), x.dtype),
				('tmp_dx_k', (B, T, C), x.dtype),
				('tmp_dx_v', (B, T, C), x.dtype),
				('mask', (T, T), bool),
				('dx_norm', (B, T, C), x.dtype),
				('inv_rms', (B, T, 1), x.dtype),
				('x_hat', (B, T, C), x.dtype),
				('rms_dot', (B, T, 1), x.dtype),
			])
	elif type(op).__name__ == "FusedRMSNormGQAOp":
		x = inputs[0]
		B, T, C = x.shape
		H, HKV = op.n_head, op.n_kv_head
		D = C // H
		if x.device in ("mps", "cuda"):
			fwd = pack_slots([
				("q", (B, H, T, D), x.dtype),
				("k", (B, HKV, T, D), x.dtype),
				("v", (B, HKV, T, D), x.dtype),
			])
			bwd = pack_slots([
				("k", (B, HKV, T, D), x.dtype),
				("v", (B, HKV, T, D), x.dtype),
				("lse", (B, H, T), x.dtype),
				("d_val", (B, H, T), x.dtype),
				("q", (B, H, T, D), x.dtype),
				("dq", (B, H, T, D), x.dtype),
				("dk", (B, HKV, T, D), x.dtype),
				("dv", (B, HKV, T, D), x.dtype),
			])
		else:
			fwd = pack_slots([
				('q', (B,T,H,D), x.dtype),
				('k', (B,T,HKV,D), x.dtype),
				('v', (B,T,HKV,D), x.dtype),
				('z', x.shape, x.dtype),
				('inv', (B,T,1), x.dtype),
				('tmp_x', x.shape, x.dtype),
				('p', (T,T), x.dtype),
				('row', (T,1), x.dtype),
				('rope0', (T,D//2), x.dtype),
				('rope1', (T,D//2), x.dtype),
			])
			bwd = pack_slots([
				('q', (B,T,H,D), x.dtype),
				('k', (B,T,HKV,D), x.dtype),
				('v', (B,T,HKV,D), x.dtype),
				('z', x.shape, x.dtype),
				('inv', (B,T,1), x.dtype),
				('tmp_x', x.shape, x.dtype),
				('p', (T,T), x.dtype),
				('row', (T,1), x.dtype),
				('rope0', (T,D//2), x.dtype),
				('rope1', (T,D//2), x.dtype),
				('dq', (B,T,H,D), x.dtype),
				('dk', (B,T,HKV,D), x.dtype),
				('dv', (B,T,HKV,D), x.dtype),
				('dz', x.shape, x.dtype),
				('dp', (T,T), x.dtype),
				('ds', (T,T), x.dtype),
				('tmp_kv', (T,D), x.dtype),
				('tmp_w', (C,C), x.dtype),
				('tmp_b', (C,), x.dtype),
				('dot', (B,T,1), x.dtype),
			])
	elif type(op).__name__ == "FusedLinearAddOp":
		x, act, w, b = inputs
		out = outputs[0]
		if x.device in ("mps", "cuda"):
			if act.op is not None and type(act.op.op).__name__ == "FusedRMSNormFlashAttentionOp":
				B, T, C = x.shape
				H = act.op.op.n_head
				D = C // H

				bwd = pack_slots([
					# FlashAttention recompute
					('fa_k', (B, H, T, D), x.dtype),
					('fa_v', (B, H, T, D), x.dtype),
				])
			elif act.op is not None and type(act.op.op).__name__ == "FusedRMSNormGQAOp":
				B, T, C = x.shape
				H, HKV = act.op.op.n_head, act.op.op.n_kv_head
				D = C // H 
				bwd = pack_slots([
					('fa_q', (B, H, T, D), x.dtype),
					('fa_k', (B, HKV, T ,D), x.dtype),
					('fa_v', (B, HKV, T, D), x.dtype),
					('fa_out', act.shape, act.dtype),
				])
		else:
			if act.op is not None and type(act.op.op).__name__ == "FusedRMSNormFlashAttentionOp":
				B, T, C = x.shape 
				H = act.op.op.n_head
				D = C // H 

				bwd = pack_slots([
					# FlashAttention recompute
					('fa_q', (B, H, T, D), x.dtype),
					('fa_k', (B, H, T, D), x.dtype),
					('fa_v', (B, H, T, D), x.dtype),
					('fa_inv_rms', (B, T, 1), x.dtype),
					('fa_x_norm', (B, T, C), x.dtype),
					('fa_attn', (B, H, T, T), x.dtype),
					('fa_row_max', (B, H, T, 1), x.dtype),
					('fa_row_sum', (B, H, T, 1), x.dtype),
					('fa_mask', (T, T), bool),
					('fa_attn_out', (B, T, C), x.dtype),

					# backward
					('tmp_dact', act.shape, act.dtype),
					('tmp_dw', w.shape, w.dtype),
					('tmp_db', b.shape, b.dtype),
				])
			elif act.op is not None and type(act.op.op).__name__ == "FusedRMSNormGQAOp":
				B, T, C = x.shape
				H, HKV = act.op.op.n_head, act.op.op.n_kv_head
				D = C // H 
				bwd = pack_slots([
					("q", (B, T, H, D), x.dtype),
					("k", (B, T, HKV, D), x.dtype),
					("v", (B, T, HKV, D), x.dtype),
					('z', x.shape, x.dtype),
					('inv', (B, T, 1), x.dtype),
					('tmp_x', x.shape, x.dtype),
					('p', (T, T), x.dtype),
					('row', (T, 1), x.dtype),
					('rope0', (T, D//2), x.dtype),
					('rope1', (T, D//2), x.dtype),
					('fa_out', act.shape, act.dtype),
				])
			else:
				bwd = pack_slots([
					('tmp_act', act.shape, act.dtype),
					('tmp_dw', w.shape, w.dtype),
					('tmp_db', b.shape, b.dtype),
					('tmp_dact', act.shape, act.dtype),
				])
	elif type(op).__name__ == "KVCacheAttentionOp":
		x = inputs[0]
		B, T, C = x.shape
		H = op.n_head
		D = C // H
		S = inputs[7].shape[1]

		if x.device in ("mps", "cuda"):
			fwd = pack_slots([
				("q", (B, H, T, D), x.dtype),
				("k", (B, H, T, D), x.dtype),
				("v", (B, H, T, D), x.dtype),
			])
		else:
			R = min(T, 32)
			fwd = pack_slots([
				("inv_rms", (B, T, 1), x.dtype),
				("x_norm", (B, T, C), x.dtype),
				("q", (B, T, H, D), x.dtype),
				("k", (B, T, H, D), x.dtype),
				("v", (B, T, H, D), x.dtype),
				("scores", (R, S), x.dtype),
				("row_stat", (R, 1), x.dtype),
				("rope_lo", (B, T, H, D // 2), x.dtype),
				("rope_hi", (B, T, H, D // 2), x.dtype),
				("rope_q0", (B, T, H, D // 2), x.dtype),
				("rope_q1", (B, T, H, D // 2), x.dtype),
				("rope_k0", (B, T, H, D // 2), x.dtype),
				("rope_k1", (B, T, H, D // 2), x.dtype),
			])
	elif type(op).__name__ == "RolloutSampleOp":
		x = inputs[0]
		if x.device in ("mps", "cuda"):
			B, V = inputs[7].shape[0], inputs[12].shape[-1]
			P = 1 << (V-1).bit_length()
			fwd = pack_slots([
				("sort_values", (B, P), x.dtype),
				("sort_indices", (B, P), np.int32),
			])

	return fwd, bwd 



