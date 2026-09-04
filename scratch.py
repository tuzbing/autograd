
import numpy as np
from math import prod

from op import AddOp, MulOp, RsqrtOp, SoftmaxOp, GELUOp, LayerNormOp, MatMulOp, PermuteOp, TransposeOp, CrossEntropyLossOp
from op import ScaledDotProductAttenOp, FusedLinearCrossEntropyOp, FusedMLPOp, FusedQKVProjOp, FlashAttentionOp

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
	__slots__ = ("slots", "nbytes")
	def __init__(self, slots, nbytes):
		self.slots = slots 
		self.nbytes = nbytes

class ScratchViews:
	__slots__ = ('_views',)
	def __init__(self, views):
		self._views = views
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

def plan_node_scratch(node):
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
		if x.device == 'mps':
			vocab_blocks = (V+95)//96
			fwd = pack_slots([
				('row_loss', (N, 1), x.dtype),
				('lse',      (N, 1), x.dtype),
				('logits',   (logits_rows, V), x.dtype),
			])
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
		x, wgu, wdown = inputs
		H = wgu.shape[1] // 2
		gu_shape = x.shape[:-1] + (2*H,)
		hidden_shape = x.shape[:-1] + (H,)

		if x.device == 'mps':
			fwd = pack_slots([
				('gu', gu_shape, x.dtype),
				('hidden', hidden_shape, x.dtype),
			])
			bwd = pack_slots([
				('gu', gu_shape, x.dtype),
				('hidden', hidden_shape, x.dtype),
			])
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
		if x.device == 'mps':
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
		if x.device == 'mps':
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
	elif type(op).__name__ == "FusedLinearAddOp":
		x, act, w, b = inputs
		out = outputs[0]
		if x.device == 'mps':
			if act.op is not None and type(act.op.op).__name__ == "FusedRMSNormFlashAttentionOp":
				B, T, C = x.shape
				H = act.op.op.n_head
				D = C // H

				bwd = pack_slots([
					# FlashAttention recompute
					('fa_k', (B, H, T, D), x.dtype),
					('fa_v', (B, H, T, D), x.dtype),
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
			else:
				bwd = pack_slots([
					('tmp_act', act.shape, act.dtype),
					('tmp_dw', w.shape, w.dtype),
					('tmp_db', b.shape, b.dtype),
					('tmp_dact', act.shape, act.dtype),
				])

	return fwd, bwd 



