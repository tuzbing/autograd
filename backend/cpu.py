import functools
import numpy as np

from .base import Backend
from op import (
	AddOp, MulOp, ViewOp, ReshapeOp, PermuteOp, TransposeOp, ContiguousOp,
	ExpOp, LogOp, RsqrtOp, SoftmaxOp, GELUOp, LayerNormOp, GatherOp,
	CrossEntropyLossOp, ScaledDotProductAttenOp, FusedLinearCrossEntropyOp,
	FusedRMSNormLinearCrossEntropyOp, FusedMLPOp, FusedQKVProjOp,
	AttnOutContiguousOp, FusedRMSNormMLPOp, FlashAttentionOp,
	FusedRMSNormFlashAttentionOp, FusedRMSNormGQAOp, FusedLinearAddOp,
	KVCacheAttentionOp, LMHeadOp, LastPositionLMHeadOp,
	RolloutTopkOp, RolloutSampleOp, DecodeFinishedOp,
)
from utils import tensor_to_numpy, is_materialized, mul_bcast_kind

GELU_C = 0.7978845608028654
GELU_A = 0.044715
GELU_A3 = 0.134145


def cpu_add_forward(inputs, outputs):
	x, y, out = inputs[0], inputs[1], outputs[0]
	np.add(x.storage.data, y.storage.data, out=out.storage.data)

def cpu_add_backward(inputs, outputs, scratch):
	x, y, out = inputs[0], inputs[1], outputs[0]
	if out.grad is None:
		return

	g = out.grad.storage.data
	if x.requires_grad:
		if g.shape == x.shape:
			np.add(x.grad.storage.data, g, out=x.grad.storage.data)
		else:
			axes = tuple(i for i, (gi, ti) in enumerate(zip(g.shape, (1,)*(g.ndim-x.ndim)+x.shape)) if ti == 1 and gi != 1)
			if axes:
				out_shape = tuple(s for i, s in enumerate(g.shape) if i not in axes)
				np.sum(g, axis=axes, out=scratch['tmp_x'].data.reshape(out_shape))
				np.add(x.grad.storage.data, scratch['tmp_x'].data.reshape(x.shape), out=x.grad.storage.data)
			else:
				np.add(x.grad.storage.data, g.reshape(x.shape), out=x.grad.storage.data)
	if y.requires_grad:
		if g.shape == y.shape:
			np.add(y.grad.storage.data, g, out=y.grad.storage.data)
		else:
			axes = tuple(i for i, (gi, ti) in enumerate(zip(g.shape, (1,)*(g.ndim-y.ndim)+y.shape)) if ti == 1 and gi != 1)
			if axes:
				out_shape = tuple(s for i, s in enumerate(g.shape) if i not in axes)
				np.sum(g, axis=axes, out=scratch['tmp_y'].data.reshape(out_shape))
				np.add(y.grad.storage.data, scratch['tmp_y'].data.reshape(y.shape), out=y.grad.storage.data)
			else:
				np.add(y.grad.storage.data, g.reshape(y.shape), out=y.grad.storage.data)

def cpu_mul_forward(inputs, outputs): 
	x = inputs[0] 
	y = inputs[1] 
	out = outputs[0] 
	np.multiply(x.storage.data, y.storage.data, out=out.storage.data)

def cpu_mul_backward(inputs, outputs, scratch): 
	x = inputs[0] 
	y = inputs[1] 
	out = outputs[0] 
	if out.grad is None:
		return
	g = out.grad.storage.data 
	xs, ys = x.storage.data, y.storage.data
	kind = mul_bcast_kind(g.shape, ys.shape)
	if x.requires_grad:
		xg = x.grad.storage.data
		if kind == "scalar":
			s = ys.reshape(())
			if s == 1: np.add(xg, g, out=xg)
			else: np.add(xg, np.multiply(g, s, dtype=g.dtype), out=xg)
		else:
			np.multiply(g, ys, dtype=g.dtype, out=scratch['full'].data)
			np.add(xg, scratch['full'].data, out=xg)

	if not y.requires_grad:
		return
	yg = y.grad.storage.data
	if kind == 'scalar':
		np.multiply(g, xs, dtype=yg.dtype, out=scratch['full'].data)
		# np.sum(scratch['full'].data, out=scratch['red_y'].data)
		total = np.sum(scratch['full'].data, dtype=yg.dtype)
		scratch['red_y'].data[...] = total
		np.add(yg, scratch['red_y'].data, out=yg)
	elif kind == 'row':
		np.multiply(g, xs, dtype=yg.dtype, out=scratch['full'].data)
		np.sum(scratch['full'].data, axis=tuple(range(g.ndim-1)), out=scratch['red_y'].data)
		np.add(yg, scratch['red_y'].data, out=yg)
	elif kind == 'col':
		np.multiply(g, xs, dtype=yg.dtype, out=scratch['full'].data)
		np.sum(scratch['full'].data, axis=-1, keepdims=True, out=scratch['red_y'].data)
		np.add(yg, scratch['red_y'].data, out=yg)
	elif kind == 'same':
		np.multiply(g, xs, out=scratch['full'].data)
		np.add(yg, scratch['full'].data, out=yg)

def cpu_matmul_fwd_dd(inputs, outputs):
	x, w, out = inputs[0], inputs[1], outputs[0]
	np.matmul(x.storage.data, w.storage.data, out=out.storage.data)

def cpu_matmul_fwd_dv(inputs, outputs):
	x, w, out = inputs[0], inputs[1], outputs[0]
	np.matmul(x.storage.data, tensor_to_numpy(w), out=out.storage.data)

def cpu_matmul_fwd_vd(inputs, outputs):
	x, w, out = inputs[0], inputs[1], outputs[0]
	np.matmul(tensor_to_numpy(x), w.storage.data, out=out.storage.data)

def cpu_matmul_fwd_vv(inputs, outputs):
	x, w, out = inputs[0], inputs[1], outputs[0]
	np.matmul(tensor_to_numpy(x), tensor_to_numpy(w), out=out.storage.data)

_MATMUL_FORWARD_TABLE = {
	(True, True): cpu_matmul_fwd_dd,
	(True, False): cpu_matmul_fwd_dv,
	(False, True): cpu_matmul_fwd_vd,
	(False, False): cpu_matmul_fwd_vv,
}

def cpu_reduce_dw(dw, w, tmp):
	wg = w.grad.storage.data
	buf = dw
	if buf.ndim > wg.ndim:
		tmp_view = tmp.ravel()[:wg.size].reshape(wg.shape)
		np.sum(buf, axis=tuple(range(buf.ndim - wg.ndim)), out=tmp_view)
		buf = tmp_view 
	np.add(wg, buf.reshape(wg.shape), out=wg)

def cpu_matmul_bwd_dd(inputs, outputs, scratch):
	x, w, out = inputs[0], inputs[1], outputs[0]
	if out.grad is None:
		return
	g = out.grad.storage.data
	acc, tmp = scratch['acc'].data, scratch['tmp'].data
	wd = w.storage.data
	if x.requires_grad:
		dx_view = acc.ravel()[:x.storage.data.size].reshape(x.shape)
		np.matmul(g, np.swapaxes(wd, -1, -2), out=dx_view)
		np.add(x.grad.storage.data, dx_view, out=x.grad.storage.data)
	if w.requires_grad:
		dw_shape = tuple(np.broadcast_shapes(x.shape[:-2], g.shape[:-2])) + (x.shape[-1], g.shape[-1])
		dw_view = acc.ravel()[:np.prod(dw_shape)].reshape(dw_shape)
		np.matmul(np.swapaxes(x.storage.data, -1, -2), g, out=dw_view)
		cpu_reduce_dw(dw_view, w, tmp)

def cpu_matmul_bwd_dv(inputs, outputs, scratch):
	x, w, out = inputs[0], inputs[1], outputs[0]
	if out.grad is None:
		return
	g = out.grad.storage.data
	acc, tmp = scratch['acc'].data, scratch['tmp'].data
	xd = x.storage.data
	if x.requires_grad:
		wa = tensor_to_numpy(w)
		dx_view = acc.ravel()[:x.storage.data.size].reshape(x.shape)
		np.matmul(g, np.swapaxes(wa, -1, -2), out=dx_view)
		np.add(x.grad.storage.data, dx_view, out=x.grad.storage.data)
	if w.requires_grad:
		dw_shape = tuple(np.broadcast_shapes(x.shape[:-2], g.shape[:-2])) + (x.shape[-1], g.shape[-1])
		dw_view = acc.ravel()[:np.prod(dw_shape)].reshape(dw_shape)
		np.matmul(np.swapaxes(xd, -1, -2), g, out=dw_view)
		cpu_reduce_dw(dw_view, w, tmp)

def cpu_matmul_bwd_vd(inputs, outputs, scratch):
	x, w, out = inputs[0], inputs[1], outputs[0]
	if out.grad is None:
		return
	g = out.grad.storage.data
	acc, tmp = scratch['acc'].data, scratch['tmp'].data
	wd = w.storage.data
	if x.requires_grad:
		dx_view = acc.ravel()[:x.storage.data.size].reshape(x.shape)
		np.matmul(g, np.swapaxes(wd, -1, -2), out=dx_view)
		np.add(x.grad.storage.data, dx_view, out=x.grad.storage.data)
	if w.requires_grad:
		xa = tensor_to_numpy(x)
		dw_shape = tuple(np.broadcast_shapes(x.shape[:-2], g.shape[:-2])) + (x.shape[-1], g.shape[-1])
		dw_view = acc.ravel()[:np.prod(dw_shape)].reshape(dw_shape)
		np.matmul(np.swapaxes(xa, -1, -2), g, out=dw_view)
		cpu_reduce_dw(dw_view, w, tmp)

def cpu_matmul_bwd_vv(inputs, outputs, scratch):
	x, w, out = inputs[0], inputs[1], outputs[0]
	if out.grad is None:
		return
	g = out.grad.storage.data
	acc, tmp = scratch['acc'].data, scratch['tmp'].data
	if x.requires_grad:
		wa = tensor_to_numpy(w)
		dx_view = acc.ravel()[:x.storage.data.size].reshape(x.shape)
		np.matmul(g, np.swapaxes(wa, -1, -2), out=dx_view)
		np.add(x.grad.storage.data, dx_view, out=x.grad.storage.data)
	if w.requires_grad:
		xa = tensor_to_numpy(x)
		dw_shape = tuple(np.broadcast_shapes(x.shape[:-2], g.shape[:-2])) + (x.shape[-1], g.shape[-1])
		dw_view = acc.ravel()[:np.prod(dw_shape)].reshape(dw_shape)
		np.matmul(np.swapaxes(xa, -1, -2), g, out=dw_view)
		cpu_reduce_dw(dw_view, w, tmp)

_MATMUL_BACKWARD_TABLE = {
	(True, True): cpu_matmul_bwd_dd,
	(True, False): cpu_matmul_bwd_dv,
	(False, True): cpu_matmul_bwd_vd,
	(False, False): cpu_matmul_bwd_vv,
}

def cpu_view_forward(inputs, outputs):
	pass

def cpu_view_backward(inputs, outputs):
	x = inputs[0]
	out = outputs[0]
	if out.grad is None:
		return
	g = out.grad.storage.data.reshape(-1)[:out.grad.numel].reshape(out.grad.shape)
	xg = x.grad.storage.data.reshape(-1)[: x.grad.numel].reshape(x.shape)
	np.add(xg, g.reshape(x.shape), out=xg)

def cpu_reshape_forward(inputs, outputs):
	pass

def cpu_reshape_backward(inputs, outputs):
	x = inputs[0]
	out = outputs[0]
	if out.grad is None:
		return
	g = out.grad.storage.data.reshape(out.grad.shape)
	np.add(x.grad.storage.data, g.reshape(x.shape), out=x.grad.storage.data)

def cpu_permute_forward(inputs, outputs):
	pass

def cpu_permute_backward(inputs, outputs, scratch):
	x, out = inputs[0], outputs[0]
	if out.grad is None:
		return
	dims = out.op.op.dims
	inv = [0] * len(dims)
	for i, d in enumerate(dims):
		inv[d] = i
	np.copyto(scratch["t"].data, np.transpose(out.grad.storage.data, inv))
	np.add(x.grad.storage.data, scratch["t"].data.reshape(x.shape), out=x.grad.storage.data)

def cpu_transpose_forward(inputs, outputs):
	pass

def cpu_transpose_backward(inputs, outputs, scratch):
	x, out = inputs[0], outputs[0]
	if out.grad is None:
		return
	d0, d1 = out.op.op.dim0, out.op.op.dim1
	np.copyto(scratch["t"].data, np.swapaxes(out.grad.storage.data, d0, d1))
	np.add(x.grad.storage.data, scratch["t"].data.reshape(x.shape), out=x.grad.storage.data)

def cpu_contiguous_forward(inputs, outputs):
	x, out = inputs[0], outputs[0]
	np.copyto(out.storage.data, tensor_to_numpy(x))

def cpu_contiguous_backward(inputs, outputs):
	x = inputs[0]
	out = outputs[0]
	if out.grad is None:
		return
	np.add(x.grad.storage.data, out.grad.storage.data.reshape(x.shape), out=x.grad.storage.data)

def cpu_exp_forward(inputs, outputs):
	x, out = inputs[0], outputs[0]
	np.exp(x.storage.data, out=out.storage.data)

def cpu_exp_backward(inputs, outputs):
	x, out = inputs[0], outputs[0]
	if out.grad is None or not x.requires_grad:
		return
	np.add(x.grad.storage.data, out.grad.storage.data * out.storage.data, out=x.grad.storage.data)

def cpu_log_forward(inputs, outputs):
	x, out = inputs[0], outputs[0]
	np.log(x.storage.data, out=out.storage.data)

def cpu_log_backward(inputs, outputs):
	x, out = inputs[0], outputs[0]
	if out.grad is None or not x.requires_grad:
		return
	np.add(x.grad.storage.data, out.grad.storage.data / x.storage.data, out=x.grad.storage.data)

def cpu_rsqrt_forward(inputs, outputs, scratch):
	x, out = inputs[0], outputs[0]
	np.sqrt(x.storage.data, out=scratch['tmp'].data)
	np.reciprocal(scratch['tmp'].data, out=out.storage.data)

def cpu_rsqrt_backward(inputs, outputs):
	x, out = inputs[0], outputs[0]
	if out.grad is None or not x.requires_grad:
		return
	y = out.storage.data
	np.add(x.grad.storage.data, out.grad.storage.data * (-0.5) * y * y * y, out=x.grad.storage.data) 

def cpu_softmax_forward(inputs, outputs, scratch):
	x, out = inputs[0], outputs[0]
	dim = out.op.op.dim 
	xd = x.storage.data
	np.max(xd, axis=dim, keepdims=True, out=scratch['max'].data)
	np.subtract(xd, scratch['max'].data, out=out.storage.data)
	np.exp(out.storage.data, out=out.storage.data)
	np.sum(out.storage.data, axis=dim, keepdims=True, out=scratch['sum'].data)
	np.divide(out.storage.data, scratch['sum'].data, out=out.storage.data)

def cpu_softmax_backward(inputs, outputs, scratch):
	x, out = inputs[0], outputs[0]
	if out.grad is None or not x.requires_grad:
		return
	dim = out.op.op.dim 
	y, g = out.storage.data, out.grad.storage.data
	np.multiply(g, y, out=scratch['t'].data)
	np.sum(scratch['t'].data, axis=dim, keepdims=True, out=scratch['dot'].data)
	np.subtract(g, scratch['dot'].data, out=scratch['t'].data)
	np.multiply(y, scratch['t'].data, out=scratch['t'].data)
	np.add(x.grad.storage.data, scratch['t'].data, out=x.grad.storage.data)

def cpu_gelu_forward(inputs, outputs, scratch):
	x, out = inputs[0], outputs[0]
	buf = scratch['buf'].data
	xd = x.storage.data
	np.multiply(xd, xd, out=buf)
	np.multiply(buf, xd, out=buf)
	np.multiply(buf, GELU_A, out=buf)
	np.add(xd, buf, out=buf)
	np.multiply(buf, GELU_C, out=buf)
	np.tanh(buf, out=buf)
	np.add(buf, 1.0, out=buf)
	np.multiply(xd, buf, out=buf)
	np.multiply(buf, 0.5, out=out.storage.data)

def cpu_gelu_backward(inputs, outputs, scratch):
	x, out = inputs[0], outputs[0]
	if out.grad is None or not x.requires_grad:
		return
	xd = x.storage.data
	buf, acc = scratch['buf'].data, scratch['acc'].data
	np.multiply(xd, xd, out=buf)
	np.multiply(buf, xd, out=buf)
	np.multiply(buf, GELU_A, out=buf)
	np.add(xd, buf, out=buf)
	np.multiply(buf, GELU_C, out=buf)
	np.tanh(buf, out=buf)
	
	np.multiply(buf, buf, out=acc)
	np.subtract(1.0, acc, out=acc)
	
	d_inner = GELU_C * (1.0 + GELU_A3 * xd * xd)
	np.multiply(acc, 0.5 * xd * d_inner, out=acc)
	
	np.add(buf, 1.0, out=buf)
	np.multiply(buf, 0.5, out=buf)
	np.add(buf, acc, out=buf)
	
	np.multiply(out.grad.storage.data, buf, out=acc)
	np.add(x.grad.storage.data, acc, out=x.grad.storage.data)

def cpu_layernorm_forward(inputs, outputs, scratch):
	x, gamma, beta = inputs[0], inputs[1], inputs[2]
	out = outputs[0]
	axis, eps = out.op.op.normalized_dim, out.op.op.eps
	xd, g, b = x.storage.data, gamma.storage.data, beta.storage.data
	r, f = scratch['red'].data, scratch['full'].data

	np.mean(xd, axis=axis, keepdims=True, out=r)
	np.subtract(xd, r, out=f)
	np.multiply(f, f, out=f)
	np.mean(f, axis=axis, keepdims=True, out=r)
	np.add(r, eps, out=r)
	np.sqrt(r, out=r)
	np.reciprocal(r, out=r)
	np.subtract(xd, np.mean(xd, axis=axis, keepdims=True), out=f)
	np.multiply(f, r, out=f)
	np.multiply(f, g, out=out.storage.data)
	np.add(out.storage.data, b, out=out.storage.data)

def cpu_layernorm_backward(inputs, outputs, scratch):
	x, gamma, beta = inputs[0], inputs[1], inputs[2]
	out = outputs[0]
	if out.grad is None:
		return
	axis = out.op.op.normalized_dim
	eps = out.op.op.eps
	xd = x.storage.data
	dy = out.grad.storage.data
	g = gamma.storage.data
	N = xd.shape[axis]
	r, f = scratch["red"].data, scratch["full"].data
	r2, f2 = scratch['red2'].data, scratch['full2'].data
	np.mean(xd, axis=axis, keepdims=True, out=r)
	np.subtract(xd, r, out=f)
	np.multiply(f, f, out=f)
	np.mean(f, axis=axis, keepdims=True, out=r)
	np.add(r, eps, out=r)
	np.sqrt(r, out=r)
	np.reciprocal(r, out=r)
	np.subtract(xd, np.mean(xd, axis=axis, keepdims=True), out=f)	
	np.multiply(f, r, out=f)                         # x_hat

	if gamma.requires_grad:
		axes = tuple(range(xd.ndim-1))
		np.multiply(dy, f, out=f2)
		np.sum(f2, axis=axes, out=scratch['gamma_tmp'].data)
		np.add(gamma.grad.storage.data, scratch['gamma_tmp'].data.reshape(gamma.shape), out=gamma.grad.storage.data)

	if beta.requires_grad:
		axes = tuple(range(xd.ndim-1))
		np.sum(dy, axis=axes, out=scratch['beta_tmp'].data)
		np.add(beta.grad.storage.data, scratch['beta_tmp'].data.reshape(beta.shape), out=beta.grad.storage.data)

	if x.requires_grad:
		inv_std = r 
		x_hat = f 
		xg = x.grad.storage.data
		f3 = scratch['full3'].data
		
		np.multiply(dy, g, out=f2) 
		np.multiply(f2, inv_std, out=f3)
		np.add(xg, f3, out=xg) 
		
		np.mean(f2, axis=axis, keepdims=True, out=r2)
		np.multiply(r2, inv_std, out=r2) 
		np.subtract(xg, r2, out=xg) 
		
		np.multiply(f2, x_hat, out=f2) 
		np.mean(f2, axis=axis, keepdims=True, out=r2)
		np.multiply(r2, inv_std, out=r2) 
		np.multiply(x_hat, r2, out=f3) 
		np.subtract(xg, f3, out=xg) 

def cpu_crossentropy_forward(inputs, outputs, scratch):
	logits, labels, out = inputs[0], inputs[1], outputs[0]
	dim = -1
	x = logits.storage.data
	r, f = scratch["red"].data, scratch["full"].data
	np.max(x, axis=dim, keepdims=True, out=r)
	np.subtract(x, r, out=f)
	exp_f = np.exp(f)
	np.sum(exp_f, axis=dim, keepdims=True, out=r)
	np.log(r, out=r)
	np.subtract(f, r, out=f)
	idx = labels.storage.data.astype(np.int64)
	B, T, _ = x.shape
	gathered = f[np.arange(B)[:, None], np.arange(T)[None, :], idx]
	np.negative(gathered, out=out.storage.data)

def cpu_crossentropy_backward(inputs, outputs, scratch):
	logits, labels, out = inputs[0], inputs[1], outputs[0]
	if out.grad is None or not logits.requires_grad:
		return
	dim = -1
	x = logits.storage.data
	r, f = scratch["red"].data, scratch["full"].data
	np.max(x, axis=dim, keepdims=True, out=r)
	np.subtract(x, r, out=f)
	np.exp(f, out=f)
	np.sum(f, axis=dim, keepdims=True, out=r)
	np.divide(f, r, out=f)           
	acc = scratch["full"].data
	B, T, V = x.shape
	acc_flat = acc.reshape(-1, V)
	idx_flat = labels.storage.data.astype(np.int64).ravel()
	acc_flat[np.arange(B*T), idx_flat] -= 1.0
	np.multiply(acc, out.grad.storage.data[..., None], out=acc)
	np.add(logits.grad.storage.data, acc, out=logits.grad.storage.data)
	
def cpu_gather_forward(inputs, outputs):
	indices, weight = inputs[0], inputs[1]
	out = outputs[0]
	idx = indices.storage.data.astype(np.int64)
	out.storage.data[:] = weight.storage.data[idx]

def cpu_gather_backward(inputs, outputs):
	indices, weight = inputs[0], inputs[1]
	out = outputs[0]
	if out.grad is None or not weight.requires_grad:
		return
	idx = indices.storage.data.astype(np.int64).ravel()
	grad_out = out.grad.storage.data.reshape(-1, weight.shape[-1])
	np.add.at(weight.grad.storage.data, idx, grad_out)

def cpu_sdpa_forward(inputs, outputs, scratch=None):
	q, k, v = inputs[0], inputs[1], inputs[2]
	out = outputs[0]
	is_causal = out.op.op.is_causal

	Q = q.storage.data 
	K = k.storage.data 
	V = v.storage.data 
	attn = scratch['attn'].data
	row_max = scratch['row_max'].data 
	row_sum = scratch['row_sum'].data

	_, _, T, D = Q.shape
	scale = 1.0 / np.sqrt(D)

	np.matmul(Q, K.transpose(0, 1, 3, 2), out=attn)
	attn *= scale

	if out.op.op.is_causal:
		mask = scratch['mask'].data
		mask.fill(1)
		mask[...] = np.triu(mask, 1)
		attn[..., mask] = -np.inf

	row_max[...] = np.max(attn, axis=-1, keepdims=True)
	np.subtract(attn, row_max, out=attn)
	np.exp(attn, out=attn)
	row_sum[...] = np.sum(attn, axis=-1, keepdims=True)
	np.divide(attn, row_sum, out=attn)
	np.matmul(attn, V, out=out.storage.data)

def cpu_sdpa_backward(inputs, outputs, scratch=None):
	q, k, v = inputs[0], inputs[1], inputs[2]
	out = outputs[0]
	is_causal = out.op.op.is_causal

	if out.grad is None: return

	Q = q.storage.data 
	K = k.storage.data 
	V = v.storage.data 
	dO = out.grad.storage.data

	probs = scratch['probs'].data 
	work = scratch['work'].data 
	row_max = scratch['row_max'].data 
	row_sum = scratch['row_sum'].data
	tmp = scratch['tmp'].data
	tmp_v = scratch['tmp_v'].data
	tmp_q = scratch['tmp_q'].data
	tmp_k = scratch['tmp_k'].data

	_, _, T, D = Q.shape 
	scale = 1.0 / np.sqrt(D)

	np.matmul(Q, K.transpose(0, 1, 3, 2), out=probs)
	np.multiply(probs, scale, out=probs)

	if is_causal:
		mask = scratch['mask'].data
		mask.fill(1)
		mask[...] = np.triu(mask, 1)
		probs[..., mask] = -np.inf
	row_max[...] = np.max(probs, axis=-1, keepdims=True)
	np.subtract(probs, row_max, out=probs)
	np.exp(probs, out=probs)
	row_sum[...] = np.sum(probs, axis=-1, keepdims=True)
	np.divide(probs, row_sum, out=probs)

	if v.requires_grad:
		np.matmul(probs.transpose(0, 1, 3, 2), dO, out=tmp_v)
		np.add(v.grad.storage.data, tmp_v, out=v.grad.storage.data)

	np.matmul(dO, V.transpose(0, 1, 3, 2), out=work)
	np.multiply(probs, work, out=tmp)
	row_sum[...] = np.sum(tmp, axis=-1, keepdims=True)
	np.subtract(work, row_sum, out=work)
	np.multiply(work, probs, out=work)
	np.multiply(work, scale, out=work)

	if k.requires_grad:
		np.matmul(work.transpose(0, 1, 3, 2), Q, out=tmp_k)
		np.add(k.grad.storage.data, tmp_k, out=k.grad.storage.data)

	if q.requires_grad:
		np.matmul(work, K, out=tmp_q)
		np.add(q.grad.storage.data, tmp_q, out=q.grad.storage.data)

def cpu_fused_linear_ce_forward(inputs, outputs, scratch=None):
	x, w, targets = inputs[0], inputs[1], inputs[2]
	out_loss = outputs[0]

	X = x.storage.data.reshape(-1, x.shape[-1])
	W = w.storage.data
	Y = targets.storage.data.reshape(-1)
	N, C = X.shape

	logits_chunk = scratch['logits'].data
	max_chunk = scratch['max'].data 
	sum_chunk = scratch['sum'].data 
	loss_sum = 0.0
	chunk_size = logits_chunk.shape[0]

	for i in range(0, N, chunk_size):
		end = min(i+chunk_size, N)
		actual_chunk = end - i  

		X_slice = X[i:end]
		Y_slice = Y[i:end]
		L_slice = logits_chunk[:actual_chunk]
		M_slice = max_chunk[:actual_chunk]
		S_slice = sum_chunk[:actual_chunk]

		np.matmul(X_slice, W.T, out=L_slice)
		np.max(L_slice, axis=-1, keepdims=True, out=M_slice)
		np.subtract(L_slice, M_slice, out=L_slice)
		np.exp(L_slice, out=L_slice)
		np.sum(L_slice, axis=-1, keepdims=True, out=S_slice)
		np.divide(L_slice, S_slice, out=L_slice)

		for j in range(actual_chunk):
			if Y_slice[j] != -1:
				loss_sum -= np.log(L_slice[j, Y_slice[j]] + 1e-10)

	out_loss.storage.data[0] = loss_sum / N 

def cpu_fused_linear_ce_backward(inputs, outputs, scratch=None):
	x, w, targets = inputs[0], inputs[1], inputs[2]
	out_loss = outputs[0]
	if out_loss.grad is None: return 

	X = x.storage.data.reshape(-1, x.shape[-1])
	W = w.storage.data
	Y = targets.storage.data.reshape(-1)
	N, C = X.shape
	V = W.shape[0]

	grad_loss = out_loss.grad.storage.data[0] / N 
	probs_chunk = scratch['probs'].data 
	max_chunk = scratch['max'].data 
	sum_chunk = scratch['sum'].data 
	tmp_dx = scratch['tmp_dx'].data 
	tmp_dw = scratch['tmp_dw'].data 

	chunk_size = probs_chunk.shape[0]

	for i in range(0, N, chunk_size):
		end = min(i + chunk_size, N)
		actual_chunk = end - i 

		X_slice = X[i:end]
		Y_slice = Y[i:end]
		P_slice = probs_chunk[:actual_chunk]

		np.matmul(X_slice, W.T, out=P_slice)
		np.max(P_slice, axis=-1, keepdims=True, out=max_chunk[:actual_chunk])
		np.subtract(P_slice, max_chunk[:actual_chunk], out=P_slice)
		np.exp(P_slice, out=P_slice)
		np.sum(P_slice, axis=-1, keepdims=True, out=sum_chunk[:actual_chunk])
		np.divide(P_slice, sum_chunk[:actual_chunk], out=P_slice)

		for j in range(actual_chunk):
			if Y_slice[j] != -1:
				P_slice[j, Y_slice[j]] -= 1.0
				
		np.multiply(P_slice, grad_loss, out=P_slice)
		
		if x.requires_grad:
			dX_slice = tmp_dx[:actual_chunk]
			np.matmul(P_slice, W, out=dX_slice)
			np.add(x.grad.storage.data.reshape(-1, C)[i:end], dX_slice, out=x.grad.storage.data.reshape(-1, C)[i:end])
			
		if w.requires_grad:
			np.matmul(P_slice.T, X_slice, out=tmp_dw)
			np.add(w.grad.storage.data, tmp_dw, out=w.grad.storage.data)

def cpu_fused_rmsnorm_linear_ce_forward(inputs, outputs, scratch=None):
	x, w, norm_weight, targets = inputs[:4]
	row_scale = inputs[4] if len(inputs) == 5 else None
	out_loss = outputs[0]
	op = out_loss.op.op

	X = x.storage.data.reshape(-1, x.shape[-1])
	W = w.storage.data
	G = norm_weight.storage.data
	Y = targets.storage.data.reshape(-1)

	N, C = X.shape
	V = W.shape[0]

	L = scratch["logits"].data.reshape(N, V)
	Z = scratch["x_norm"].data.reshape(N, C)
	R = scratch["inv_rms"].data.reshape(N, 1)
	M = scratch["max"].data.reshape(N, 1)
	S = scratch["sum"].data.reshape(N, 1)
	TL = scratch["target_logits"].data.reshape(N, 1)

	scales = None
	if row_scale is not None:
		scales = row_scale.storage.data.reshape(-1)

	# RMSNorm: z = x * rsqrt(mean(x^2) + eps) * gamma
	np.multiply(X, X, out=Z)
	np.mean(Z, axis=1, keepdims=True, out=R)
	np.add(R, np.float32(op.eps), out=R)
	np.sqrt(R, out=R)
	np.reciprocal(R, out=R)

	np.multiply(X, R, out=Z)
	np.multiply(Z, G, out=Z)

	# logits = z @ W.T
	np.matmul(Z, W.T, out=L)

	# 先保存 max 和 sum，L 保持原始 logits，
	# 使 CPU 参考路径直接依据数学定义计算 CE。
	TL.fill(np.float32(0.0))
	valid = (Y >= 0) & (Y < V)
	valid_rows = np.flatnonzero(valid)
	if valid_rows.size:
		TL[valid_rows, 0] = L[valid_rows, Y[valid_rows]]

	np.max(L, axis=1, keepdims=True, out=M)
	np.subtract(L, M, out=L)
	np.exp(L, out=L)
	np.sum(L, axis=1, keepdims=True, out=S)

	row_loss = M[:, 0] + np.log(S[:, 0]) - TL[:, 0]
	row_loss[~valid] = np.float32(0.0)

	if scales is None:
		out_loss.storage.data[0] = (
			np.sum(row_loss, dtype=np.float32)
			/ np.float32(N)
		)
	else:
		# 当前 Metal row_scaled kernel:
		# row_loss_kernel = CE * row_scale * N
		# forward_reduce = sum(row_loss_kernel) / N
		out_loss.storage.data[0] = np.sum(
			row_loss * scales,
			dtype=np.float32,
		)

def cpu_fused_rmsnorm_linear_ce_backward(inputs, outputs, scratch=None):
	x, w, norm_weight, targets = inputs[:4]
	row_scale = inputs[4] if len(inputs) == 5 else None
	out_loss = outputs[0]
	op = out_loss.op.op

	if out_loss.grad is None:
		return

	X = x.storage.data.reshape(-1, x.shape[-1])
	W = w.storage.data
	G = norm_weight.storage.data
	Y = targets.storage.data.reshape(-1)

	N, C = X.shape
	V = W.shape[0]
	grad_out = float(out_loss.grad.storage.data[0])

	L = scratch["logits"].data.reshape(N, V)
	Z = scratch["x_norm"].data.reshape(N, C)
	DZ = scratch["dx_norm"].data.reshape(N, C)
	R = scratch["inv_rms"].data.reshape(N, 1)
	M = scratch["max"].data.reshape(N, 1)
	S = scratch["sum"].data.reshape(N, 1)
	TX = scratch["tmp_x"].data.reshape(N, C)
	TDW = scratch["tmp_dw"].data.reshape(V, C)
	TDG = scratch["tmp_dgamma"].data.reshape(C)

	scales = None
	if row_scale is not None:
		scales = row_scale.storage.data.reshape(-1)

	# 1. 完整 RMSNorm forward。
	np.multiply(X, X, out=Z)
	np.mean(Z, axis=1, keepdims=True, out=R)
	np.add(R, np.float32(op.eps), out=R)
	np.sqrt(R, out=R)
	np.reciprocal(R, out=R)

	np.multiply(X, R, out=Z)
	np.multiply(Z, G, out=Z)

	# 2. 完整 logits 和 softmax。
	np.matmul(Z, W.T, out=L)
	np.max(L, axis=1, keepdims=True, out=M)
	np.subtract(L, M, out=L)
	np.exp(L, out=L)
	np.sum(L, axis=1, keepdims=True, out=S)
	np.divide(L, S, out=L)

	# 3. dlogits。
	for row in range(N):
		target = int(Y[row])

		if target < 0 or target >= V:
			L[row, :].fill(0.0)
			continue

		L[row, target] -= 1.0

		if scales is None:
			scale = grad_out / float(N)
		else:
			scale = grad_out * float(scales[row])

		np.multiply(L[row, :], scale, out=L[row, :])

	# 4. dW = dlogits.T @ z。
	if w.requires_grad:
		np.matmul(L.T, Z, out=TDW)
		np.add(w.grad.storage.data, TDW, out=w.grad.storage.data)

	if not (x.requires_grad or norm_weight.requires_grad):
		return

	# 5. dZ = dlogits @ W。
	np.matmul(L, W, out=DZ)

	# x_hat = X * inv_rms
	np.multiply(X, R, out=TX)

	# 6. dgamma = sum(dZ * x_hat)。
	if norm_weight.requires_grad:
		np.multiply(TX, DZ, out=TX)
		np.sum(TX, axis=0, out=TDG)
		np.add(norm_weight.grad.storage.data, TDG, out=norm_weight.grad.storage.data)

	# 7. dX = inv_rms * (dZ * gamma - x_hat * mean(dZ * gamma * x_hat))。
	if x.requires_grad:
		np.multiply(DZ, G, out=DZ)

		np.multiply(X, R, out=TX)

		np.multiply(TX, DZ, out=TX)
		np.mean(TX, axis=1, keepdims=True, out=M)

		np.multiply(X, R, out=TX)
		np.multiply(TX, M, out=TX)

		np.subtract(DZ, TX, out=DZ)
		np.multiply(DZ, R, out=DZ)

		np.add(x.grad.storage.data.reshape(N, C), DZ, out=x.grad.storage.data.reshape(N, C))

def cpu_fused_mlp_forward(inputs, outputs, scratch=None):
    x, w1, b1, w2, b2 = inputs
    out = outputs[0]
    
    X = x.storage.data.reshape(-1, x.shape[-1])
    W1 = w1.storage.data
    B1 = b1.storage.data
    W2 = w2.storage.data
    B2 = b2.storage.data
    
    H1 = scratch['h1'].data.reshape(-1, w1.shape[-1])
    H2 = scratch['h2'].data.reshape(-1, w1.shape[-1])
    OUT = out.storage.data.reshape(-1, w2.shape[-1])
    TMP = scratch['tmp'].data.reshape(-1, w1.shape[-1])
    
    np.matmul(X, W1, out=H1)
    np.add(H1, B1, out=H1)
    
    np.multiply(H1, H1, out=TMP) 
    np.multiply(TMP, H1, out=TMP) 
    np.multiply(TMP, 0.044715, out=TMP) 
    np.add(TMP, H1, out=TMP) 
    np.multiply(TMP, np.sqrt(2.0 / np.pi), out=TMP) 
    np.tanh(TMP, out=TMP)
    np.add(TMP, 1.0, out=TMP)
    np.multiply(H1, 0.5, out=H2)
    np.multiply(H2, TMP, out=H2)
    
    np.matmul(H2, W2, out=OUT)
    np.add(OUT, B2, out=OUT)

def cpu_fused_mlp_backward(inputs, outputs, scratch=None):
    x, w1, b1, w2, b2 = inputs
    out = outputs[0]
    if out.grad is None: return
    X = x.storage.data.reshape(-1, x.shape[-1])
    W1 = w1.storage.data
    B1 = b1.storage.data
    W2 = w2.storage.data
    dO = out.grad.storage.data.reshape(-1, w2.shape[-1])
    H1 = scratch['h1'].data.reshape(-1, w1.shape[-1])
    H2 = scratch['h2'].data.reshape(-1, w1.shape[-1])
    DH = scratch['dh_buf'].data.reshape(-1, w1.shape[-1])
    TMP1 = scratch['tmp1'].data.reshape(-1, w1.shape[-1])
    TMP2 = scratch['tmp2'].data.reshape(-1, w1.shape[-1])

    np.matmul(X, W1, out=H1)
    np.add(H1, B1, out=H1)
    
    np.multiply(H1, H1, out=TMP1) 
    np.multiply(TMP1, H1, out=TMP2) 
    np.multiply(TMP2, 0.044715, out=TMP2) 
    np.add(TMP2, H1, out=TMP2) 
    np.multiply(TMP2, np.sqrt(2.0 / np.pi), out=TMP2) 
    np.tanh(TMP2, out=TMP2)
    np.add(TMP2, 1.0, out=TMP2)
    np.multiply(H1, 0.5, out=H2)
    np.multiply(H2, TMP2, out=H2)
    
    np.matmul(dO, W2.T, out=DH)
    
    if w2.requires_grad:
        tmp_dw2 = scratch['tmp_dw2'].data 
        np.matmul(H2.T, dO, out=tmp_dw2)
        np.add(w2.grad.storage.data, tmp_dw2, out=w2.grad.storage.data)
        
    if b2.requires_grad:
        tmp_db2 = scratch['tmp_db2'].data 
        np.sum(dO, axis=0, out=tmp_db2)
        np.add(b2.grad.storage.data, tmp_db2, out=b2.grad.storage.data)
        
    np.multiply(TMP1, H1, out=TMP2) 
    np.multiply(TMP2, 0.044715, out=TMP2)
    np.add(TMP2, H1, out=TMP2)
    np.multiply(TMP2, np.sqrt(2.0 / np.pi), out=TMP2) 
    np.tanh(TMP2, out=TMP2) 
    
    left_deriv = scratch['h2'].data.reshape(-1, w1.shape[-1]) 
    np.add(TMP2, 1.0, out=left_deriv)
    np.multiply(left_deriv, 0.5, out=left_deriv)
    
    tanh_deriv = scratch['tmp2'].data.reshape(-1, w1.shape[-1]) 
    np.multiply(TMP2, TMP2, out=tanh_deriv)
    np.subtract(1.0, tanh_deriv, out=tanh_deriv)
    
    inner_deriv = TMP1 
    np.multiply(TMP1, 3.0 * 0.044715, out=inner_deriv)
    np.add(inner_deriv, 1.0, out=inner_deriv)
    np.multiply(inner_deriv, np.sqrt(2.0 / np.pi), out=inner_deriv)
    
    right_deriv = inner_deriv
    np.multiply(right_deriv, tanh_deriv, out=right_deriv)
    np.multiply(right_deriv, H1, out=right_deriv)
    np.multiply(right_deriv, 0.5, out=right_deriv)
    
    np.add(left_deriv, right_deriv, out=left_deriv)
    np.multiply(DH, left_deriv, out=DH)
    
    if x.requires_grad:
        tmp_dx = scratch['tmp_dx'].data.reshape(-1, x.shape[-1])
        np.matmul(DH, W1.T, out=tmp_dx)
        dx_view = x.grad.storage.data.reshape(-1, x.shape[-1])
        np.add(dx_view, tmp_dx, out=dx_view)
        
    if w1.requires_grad:
        tmp_dw1 = scratch['tmp_dw1'].data 
        np.matmul(X.T, DH, out=tmp_dw1)
        np.add(w1.grad.storage.data, tmp_dw1, out=w1.grad.storage.data)
        
    if b1.requires_grad:
        tmp_db1 = scratch['tmp_db1'].data 
        np.sum(DH, axis=0, out=tmp_db1)
        np.add(b1.grad.storage.data, tmp_db1, out=b1.grad.storage.data)

def cpu_fused_qkv_proj_forward(inputs, outputs, scratch=None):
	x, wq, bq, wk, bk, wv, bv = inputs
	q_out, k_out, v_out = outputs
	B, T, C = x.shape 
	H, D = q_out.shape[1], q_out.shape[3]

	tmp_q, tmp_k, tmp_v = scratch['tmp_q'].data, scratch['tmp_k'].data, scratch['tmp_v'].data

	# Q
	np.dot(x.storage.data, wq.storage.data, out=tmp_q)
	np.add(tmp_q, bq.storage.data, out=tmp_q)
	np.copyto(q_out.storage.data, tmp_q.reshape(B, T, H, D).transpose(0, 2, 1, 3))

	# K
	np.dot(x.storage.data, wk.storage.data, out=tmp_k)
	np.add(tmp_k, bk.storage.data, out=tmp_k)
	np.copyto(k_out.storage.data, tmp_k.reshape(B, T, H, D).transpose(0, 2, 1, 3))

	# V
	np.dot(x.storage.data, wv.storage.data, out=tmp_v)
	np.add(tmp_v, bv.storage.data, out=tmp_v)
	np.copyto(v_out.storage.data, tmp_v.reshape(B, T, H, D).transpose(0, 2, 1, 3))

def cpu_fused_qkv_proj_backward(inputs, outputs, scratch=None):
	x, wq, bq, wk, bk, wv, bv = inputs
	q_out, k_out, v_out = outputs
	dq, dk, dv = q_out.grad, k_out.grad, v_out.grad
	B, T, C = x.shape

	tmp_dq, tmp_dk, tmp_dv = scratch['tmp_dq'].data, scratch['tmp_dk'].data, scratch['tmp_dv'].data
	np.copyto(tmp_dq, dq.storage.data.transpose(0, 2, 1, 3).reshape(B, T, C))
	np.copyto(tmp_dk, dk.storage.data.transpose(0, 2, 1, 3).reshape(B, T, C))
	np.copyto(tmp_dv, dv.storage.data.transpose(0, 2, 1, 3).reshape(B, T, C))

	x_flat = x.storage.data.reshape(B*T, C)

	if wq.requires_grad:
		np.dot(x_flat.T, tmp_dq.reshape(B*T, C), out=wq.grad.storage.data)
		np.sum(tmp_dq.reshape(B*T, C), axis=0, out=bq.grad.storage.data)
		np.dot(x_flat.T, tmp_dk.reshape(B*T, C), out=wk.grad.storage.data)
		np.sum(tmp_dk.reshape(B*T, C), axis=0, out=bk.grad.storage.data)
		np.dot(x_flat.T, tmp_dv.reshape(B*T, C), out=wv.grad.storage.data)
		np.sum(tmp_dv.reshape(B*T, C), axis=0, out=bv.grad.storage.data)

	if x.requires_grad:
		tmp_dx = scratch['tmp_dx'].data
		np.dot(tmp_dq.reshape(B*T, C), wq.storage.data.T, out=tmp_dx.reshape(B*T, C))

		tmp_dx_k = scratch['tmp_dx_k'].data
		np.dot(tmp_dk.reshape(B*T, C), wk.storage.data.T, out=tmp_dx_k.reshape(B*T, C))
		np.add(tmp_dx, tmp_dx_k, out=tmp_dx)

		tmp_dx_v = scratch['tmp_dx_v'].data
		np.dot(tmp_dv.reshape(B*T, C), wv.storage.data.T, out=tmp_dx_v.reshape(B*T, C))
		np.add(tmp_dx, tmp_dx_v, out=tmp_dx)

		np.add(x.grad.storage.data, tmp_dx.reshape(B, T, C), out=x.grad.storage.data)

def cpu_attn_out_contiguous_forward(inputs, outputs, scratch=None):
	x, out = inputs[0], outputs[0]
	B, H, T, D = x.shape
	C = H * D
	x_view = x.storage.data.reshape(B, H, T, D).transpose(0, 2, 1, 3).reshape(B, T, C)
	np.copyto(out.storage.data.reshape(B, T, C), x_view)

def cpu_attn_out_contiguous_backward(inputs, outputs, scratch=None):
	x, out = inputs[0], outputs[0]
	if out.grad is None or not x.requires_grad: return
	B, H, T, D = x.shape 
	grad_out_view = out.grad.storage.data.reshape(B, T, H, D).transpose(0, 2, 1, 3).reshape(B, H, T, D)
	np.add(x.grad.storage.data.reshape(B, H, T, D), grad_out_view, out=x.grad.storage.data.reshape(B, H, T, D))

def cpu_fused_rmsnorm_mlp_forward(inputs, outputs, scratch=None):
    if len(inputs) == 3:
    	x, wgu, wdown = inputs
    	norm_weight = None
    elif len(inputs) == 4:
    	x, norm_weight, wgu, wdown = inputs

    out = outputs[0]
    eps = out.op.op.eps

    X = x.storage.data.reshape(-1, x.shape[-1])
    Wgu = wgu.storage.data
    Wdown = wdown.storage.data
    OUT = out.storage.data.reshape(-1, x.shape[-1])

    N, C = X.shape
    H = Wgu.shape[1] // 2

    inv_rms = scratch["inv_rms"].data.reshape(N, 1)
    X_norm = scratch["x_norm"].data.reshape(N, C)
    GU = scratch["gu"].data.reshape(N, 2 * H)
    HIDDEN = scratch["hidden"].data.reshape(N, H)

    # inv_rms = 1 / sqrt(mean(X * X) + eps)
    np.multiply(X, X, out=X_norm)
    np.mean(X_norm, axis=-1, keepdims=True, out=inv_rms)
    np.add(inv_rms, eps, out=inv_rms)
    np.sqrt(inv_rms, out=inv_rms)
    np.reciprocal(inv_rms, out=inv_rms)

    # X_norm = X * inv_rms
    np.multiply(X, inv_rms, out=X_norm)

    if norm_weight is not None:
    	np.multiply(X_norm, norm_weight.storage.data.reshape(1, C), out=X_norm)

    # GU = X_norm @ Wgu
    np.matmul(X_norm, Wgu, out=GU)

    G = GU[:, :H]
    U = GU[:, H:]

    # HIDDEN = SiLU(G) * U
    np.negative(G, out=HIDDEN)
    np.exp(HIDDEN, out=HIDDEN)
    np.add(HIDDEN, 1.0, out=HIDDEN)
    np.reciprocal(HIDDEN, out=HIDDEN)

    # HIDDEN = G * sigmoid(G)
    np.multiply(G, HIDDEN, out=HIDDEN)

    # HIDDEN = SiLU(G) * U
    np.multiply(HIDDEN, U, out=HIDDEN)

    # OUT = HIDDEN @ Wdown
    np.matmul(HIDDEN, Wdown, out=OUT)

def cpu_fused_rmsnorm_mlp_backward(inputs, outputs, scratch=None,):
    if len(inputs) == 3:
    	x, wgu, wdown = inputs
    	norm_weight = None
    elif len(inputs) == 4:
    	x, norm_weight, wgu, wdown = inputs

    out = outputs[0]

    if out.grad is None:
        return

    eps = out.op.op.eps

    X = x.storage.data.reshape(-1, x.shape[-1])
    Wgu = wgu.storage.data
    Wdown = wdown.storage.data
    DOUT = out.grad.storage.data.reshape(-1, x.shape[-1])

    N, C = X.shape
    H = Wgu.shape[1] // 2

    inv_rms = scratch["inv_rms"].data.reshape(N, 1)
    X_norm = scratch["x_norm"].data.reshape(N, C)
    DX_NORM = scratch["dx_norm"].data.reshape(N, C)

    GU = scratch["gu"].data.reshape(N, 2 * H)
    D_HIDDEN = scratch["dhidden"].data.reshape(N, H)
    GATE = scratch["gate"].data.reshape(N, H)
    SIGMOID = scratch["sigmoid"].data.reshape(N, H)

    # inv_rms = 1 / sqrt(mean(X*X)+eps)
    np.multiply(X, X, out=X_norm)
    np.mean(X_norm, axis=-1, keepdims=True, out=inv_rms)
    np.add(inv_rms, eps, out=inv_rms)
    np.sqrt(inv_rms, out=inv_rms)
    np.reciprocal(inv_rms, out=inv_rms)

    # X_norm = x_hat * gamma; 无gamma时为x_hat
    np.multiply(X, inv_rms, out=X_norm)

    if norm_weight is not None:
    	np.multiply(X_norm, norm_weight.storage.data.reshape(1, C), out=X_norm)

    # GU = X_norm @ Wgu
    np.matmul(X_norm, Wgu, out=GU)

    G = GU[:, :H]
    U = GU[:, H:]
    np.copyto(GATE, G)

    # SIGMOID = sigmoid(GATE)
    np.negative(GATE, out=SIGMOID)
    np.exp(SIGMOID, out=SIGMOID)
    np.add(SIGMOID, 1.0, out=SIGMOID)
    np.reciprocal(SIGMOID, out=SIGMOID)

    # hidden = SiLU(GATE)*U
    np.multiply(GATE, SIGMOID, out=D_HIDDEN)
    np.multiply(D_HIDDEN, U, out=D_HIDDEN)

    # dWdown = hidden.T @ DOUT
    if wdown.requires_grad:
        TMP_DWDOWN = scratch["tmp_dwdown"].data
        np.matmul(D_HIDDEN.T, DOUT, out=TMP_DWDOWN)
        np.add(wdown.grad.storage.data, TMP_DWDOWN, out=wdown.grad.storage.data)

    # D_HIDDEN = DOUT @ Wdown.T
    np.matmul(DOUT, Wdown.T, out=D_HIDDEN)

    # GU 将被原地改写为 DGU。
    D_GATE = GU[:, :H]
    D_UP = GU[:, H:]

    # dGate D_GATE = sigmoid(g) * (1 + g*(1-sigmoid(g)))
    np.subtract(1.0, SIGMOID, out=D_GATE)
    np.multiply(D_GATE, GATE, out=D_GATE)
    np.add(D_GATE, 1.0, out=D_GATE)
    np.multiply(D_GATE, SIGMOID, out=D_GATE)

    # D_GATE *= U
    np.multiply(D_GATE, U, out=D_GATE)
    np.multiply(D_GATE, D_HIDDEN, out=D_GATE)

    # 再计算 dUp。原始 gate 保存在 GATE，SIGMOID 也仍然有效。
    np.multiply(GATE, SIGMOID, out=D_UP)
    np.multiply(D_UP, D_HIDDEN, out=D_UP)

    # dWgu = X_norm.T @ dGU
    if wgu.requires_grad:
        TMP_DWGU = scratch["tmp_dwgu"].data
        np.matmul(X_norm.T, GU, out=TMP_DWGU)
        np.add(wgu.grad.storage.data, TMP_DWGU, out=wgu.grad.storage.data)

    # dX_norm = dGU @ Wgu.T
    np.matmul(GU, Wgu.T, out=DX_NORM)

    if norm_weight is not None:
    	np.multiply(X, inv_rms, out=X_norm)

    	if norm_weight.requires_grad:
    		np.multiply(DX_NORM, X_norm, out=X_norm)

    		tmp_dgamma = scratch["tmp_dwgu"].data.reshape(-1)[:C]
    		np.sum(X_norm, axis=0, out=tmp_dgamma)
    		np.add(norm_weight.grad.storage.data, tmp_dgamma, out=norm_weight.grad.storage.data)

    	# dX_hat = dX_norm * gamma
    	np.multiply(DX_NORM, norm_weight.storage.data.reshape(1, C), out=DX_NORM)

    mean_dy_xhat = GU.reshape(-1)[:N].reshape(N, 1)
    np.multiply(X, inv_rms, out=X_norm)
    np.multiply(DX_NORM, X_norm, out=X_norm)
    np.mean(X_norm, axis=-1, keepdims=True, out=mean_dy_xhat)
    np.multiply(X, inv_rms, out=X_norm)
    np.multiply(X_norm, mean_dy_xhat, out=X_norm)
    np.subtract(DX_NORM, X_norm, out=DX_NORM)
    np.multiply(DX_NORM, inv_rms, out=DX_NORM)

    if x.requires_grad:
    	np.add(x.grad.storage.data.reshape(N, C), DX_NORM, out=x.grad.storage.data.reshape(N, C))

def cpu_flash_attention_forward(inputs, outputs, scratch=None):
    x, wq, bq, wk, bk, wv, bv = inputs
    out = outputs[0]
    B, T, C = x.shape
    # H, D = out.shape[1], out.shape[3]
    H = out.op.op.n_head
    D = C // H
    
    Q = scratch['q'].data.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    K = scratch['k'].data.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    V = scratch['v'].data.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    
    X_flat = x.storage.data.reshape(-1, C)
    np.dot(X_flat, wq.storage.data, out=scratch['q'].data.reshape(-1, C))
    np.add(scratch['q'].data.reshape(-1, C), bq.storage.data, out=scratch['q'].data.reshape(-1, C))
    
    np.dot(X_flat, wk.storage.data, out=scratch['k'].data.reshape(-1, C))
    np.add(scratch['k'].data.reshape(-1, C), bk.storage.data, out=scratch['k'].data.reshape(-1, C))
    
    np.dot(X_flat, wv.storage.data, out=scratch['v'].data.reshape(-1, C))
    np.add(scratch['v'].data.reshape(-1, C), bv.storage.data, out=scratch['v'].data.reshape(-1, C))
    
    attn = scratch['attn'].data
    row_max = scratch['row_max'].data
    row_sum = scratch['row_sum'].data
    scale = 1.0 / np.sqrt(D)
    
    np.matmul(Q, K.transpose(0, 1, 3, 2), out=attn)
    np.multiply(attn, scale, out=attn)
    
    if out.op.op.is_causal:
        mask = scratch['mask'].data
        mask.fill(1)
        mask[...] = np.triu(mask, 1)
        attn[..., mask] = -np.inf
        
    np.max(attn, axis=-1, keepdims=True, out=row_max)
    np.subtract(attn, row_max, out=attn)
    np.exp(attn, out=attn)
    np.sum(attn, axis=-1, keepdims=True, out=row_sum)
    np.divide(attn, row_sum, out=attn)
    out_view = out.storage.data.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    np.matmul(attn, V, out=out_view)

def cpu_flash_attention_backward(inputs, outputs, scratch=None):
    x, wq, bq, wk, bk, wv, bv = inputs
    out = outputs[0]
    dO = out.grad.storage.data
    if dO is None: return
    B, T, C = x.shape
    # H, D = out.shape[1], out.shape[3]
    H = out.op.op.n_head
    D = C // H
    dO_view = dO.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    
    Q = scratch['tmp_dq'].data 
    K = scratch['tmp_dk'].data
    V = scratch['tmp_dv'].data
    
    X_flat = x.storage.data.reshape(-1, C)
    np.dot(X_flat, wq.storage.data, out=Q.reshape(-1, C))
    np.add(Q.reshape(-1, C), bq.storage.data, out=Q.reshape(-1, C))
    Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3).copy() 
    
    np.dot(X_flat, wk.storage.data, out=K.reshape(-1, C))
    np.add(K.reshape(-1, C), bk.storage.data, out=K.reshape(-1, C))
    K = K.reshape(B, T, H, D).transpose(0, 2, 1, 3).copy()
    
    np.dot(X_flat, wv.storage.data, out=V.reshape(-1, C))
    np.add(V.reshape(-1, C), bv.storage.data, out=V.reshape(-1, C))
    V = V.reshape(B, T, H, D).transpose(0, 2, 1, 3).copy()
    
    probs = scratch['probs'].data
    row_max = scratch['row_max'].data
    row_sum = scratch['row_sum'].data
    scale = 1.0 / np.sqrt(D)
    
    np.matmul(Q, K.transpose(0, 1, 3, 2), out=probs)
    np.multiply(probs, scale, out=probs)
    if out.op.op.is_causal:
        mask_np = np.ones((T, T), dtype=bool)
        mask_np = np.triu(mask_np, 1)
        probs[..., mask_np] = -np.inf
    np.max(probs, axis=-1, keepdims=True, out=row_max)
    np.subtract(probs, row_max, out=probs)
    np.exp(probs, out=probs)
    np.sum(probs, axis=-1, keepdims=True, out=row_sum)
    np.divide(probs, row_sum, out=probs)
    
    tmp_dv = scratch['tmp_dv'].data
    np.matmul(probs.transpose(0, 1, 3, 2), dO_view, out=tmp_dv)
    
    work = scratch['work'].data
    tmp = scratch['tmp'].data
    np.matmul(dO_view, V.transpose(0, 1, 3, 2), out=work)
    np.multiply(probs, work, out=tmp)
    np.sum(tmp, axis=-1, keepdims=True, out=row_sum)
    np.subtract(work, row_sum, out=work)
    np.multiply(work, probs, out=work)
    np.multiply(work, scale, out=work)
    
    tmp_dk = scratch['tmp_dk'].data
    np.matmul(work.transpose(0, 1, 3, 2), Q, out=tmp_dk)
    tmp_dq = scratch['tmp_dq'].data
    np.matmul(work, K, out=tmp_dq)
    
    dQ = tmp_dq.transpose(0, 2, 1, 3).reshape(B*T, C) 
    dK = tmp_dk.transpose(0, 2, 1, 3).reshape(B*T, C)
    dV = tmp_dv.transpose(0, 2, 1, 3).reshape(B*T, C)
    
    if wq.requires_grad:
        np.dot(X_flat.T, dQ, out=wq.grad.storage.data)
        np.sum(dQ, axis=0, out=bq.grad.storage.data)
        np.dot(X_flat.T, dK, out=wk.grad.storage.data)
        np.sum(dK, axis=0, out=bk.grad.storage.data)
        np.dot(X_flat.T, dV, out=wv.grad.storage.data)
        np.sum(dV, axis=0, out=bv.grad.storage.data)
        
    if x.requires_grad:
        tmp_dx = scratch['tmp_dx'].data
        np.dot(dQ, wq.storage.data.T, out=tmp_dx.reshape(B*T, C))
        tmp_dx_k = scratch['tmp_dx_k'].data
        np.dot(dK, wk.storage.data.T, out=tmp_dx_k.reshape(B*T, C))
        np.add(tmp_dx, tmp_dx_k, out=tmp_dx)
        tmp_dx_v = scratch['tmp_dx_v'].data
        np.dot(dV, wv.storage.data.T, out=tmp_dx_v.reshape(B*T, C))
        np.add(tmp_dx, tmp_dx_v, out=tmp_dx)
        np.add(x.grad.storage.data, tmp_dx.reshape(B, T, C), out=x.grad.storage.data)

def apply_rope_np(q, k, cos, sin):
    # q/k: (B, T, H, D)
    # cos/sin: (T, D // 2)
    d2 = q.shape[-1] // 2

    q1 = q[..., :d2].copy()
    q2 = q[..., d2:].copy()
    k1 = k[..., :d2].copy()
    k2 = k[..., d2:].copy()

    c = cos[None, :, None, :]
    s = sin[None, :, None, :]

    q[..., :d2] = q1 * c + q2 * s
    q[..., d2:] = -q1 * s + q2 * c

    k[..., :d2] = k1 * c + k2 * s
    k[..., d2:] = -k1 * s + k2 * c

def apply_rope_backward_np(dq, dk, cos, sin):
    d2 = dq.shape[-1] // 2

    dq1 = dq[..., :d2].copy()
    dq2 = dq[..., d2:].copy()
    dk1 = dk[..., :d2].copy()
    dk2 = dk[..., d2:].copy()

    c = cos[None, :, None, :]
    s = sin[None, :, None, :]

    # R^T，即 forward RoPE 使用 -sin
    dq[..., :d2] = dq1 * c - dq2 * s
    dq[..., d2:] = dq1 * s + dq2 * c

    dk[..., :d2] = dk1 * c - dk2 * s
    dk[..., d2:] = dk1 * s + dk2 * c

def cpu_fused_rms_fa_forward(inputs, outputs, scratch=None):
    x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
    out = outputs[0]

    B, T, C = x.shape
    H = out.op.op.n_head
    D = C // H
    eps = np.float32(out.op.op.eps)

    X = x.storage.data.reshape(B, T, C)

    inv_rms = (
        np.float32(1.0)
        / np.sqrt(
            np.mean(X * X, axis=-1, keepdims=True)
            + eps
        )
    ).astype(np.float32)

    X_norm = (X * inv_rms).astype(np.float32)
    X_flat = X_norm.reshape(B * T, C)

    Q = (
        X_flat @ wq.storage.data
        + bq.storage.data
    ).reshape(B, T, H, D).astype(np.float32)

    K = (
        X_flat @ wk.storage.data
        + bk.storage.data
    ).reshape(B, T, H, D).astype(np.float32)

    V = (
        X_flat @ wv.storage.data
        + bv.storage.data
    ).reshape(B, T, H, D).astype(np.float32)

    apply_rope_np(
        Q,
        K,
        rope_cos.storage.data,
        rope_sin.storage.data,
    )

    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    scale = np.float32(
        1.0 / np.sqrt(np.float32(D))
    )

    scores = (
        np.matmul(
            Q,
            K.transpose(0, 1, 3, 2),
        )
        * scale
    ).astype(np.float32)

    if out.op.op.is_causal:
        mask = np.triu(
            np.ones((T, T), dtype=bool),
            k=1,
        )
        scores[..., mask] = -np.inf

    scores -= np.max(
        scores,
        axis=-1,
        keepdims=True,
    )

    probs = np.exp(scores).astype(np.float32)
    probs /= np.sum(
        probs,
        axis=-1,
        keepdims=True,
    )

    context = np.matmul(
        probs,
        V,
    ).astype(np.float32)

    out.storage.data[...] = (
        context
        .transpose(0, 2, 1, 3)
        .reshape(B, T, C)
    )

def cpu_fused_rms_fa_backward(inputs, outputs, scratch=None):
    del scratch

    x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
    out = outputs[0]

    if out.grad is None:
        return

    B, T, C = x.shape
    H = out.op.op.n_head
    D = C // H
    eps = np.float32(out.op.op.eps)

    X = x.storage.data.reshape(B, T, C)

    inv_rms = (
        np.float32(1.0)
        / np.sqrt(
            np.mean(X * X, axis=-1, keepdims=True)
            + eps
        )
    ).astype(np.float32)

    X_norm = (X * inv_rms).astype(np.float32)
    X_flat = X_norm.reshape(B * T, C)

    Q_bthd = (
        X_flat @ wq.storage.data
        + bq.storage.data
    ).reshape(B, T, H, D).astype(np.float32)

    K_bthd = (
        X_flat @ wk.storage.data
        + bk.storage.data
    ).reshape(B, T, H, D).astype(np.float32)

    V_bthd = (
        X_flat @ wv.storage.data
        + bv.storage.data
    ).reshape(B, T, H, D).astype(np.float32)

    apply_rope_np(
        Q_bthd,
        K_bthd,
        rope_cos.storage.data,
        rope_sin.storage.data,
    )

    Q = Q_bthd.transpose(0, 2, 1, 3)
    K = K_bthd.transpose(0, 2, 1, 3)
    V = V_bthd.transpose(0, 2, 1, 3)

    scale = np.float32(
        1.0 / np.sqrt(np.float32(D))
    )

    scores = (
        np.matmul(
            Q,
            K.transpose(0, 1, 3, 2),
        )
        * scale
    ).astype(np.float32)

    if out.op.op.is_causal:
        mask = np.triu(
            np.ones((T, T), dtype=bool),
            k=1,
        )
        scores[..., mask] = -np.inf

    scores -= np.max(
        scores,
        axis=-1,
        keepdims=True,
    )

    probs = np.exp(scores).astype(np.float32)
    probs /= np.sum(
        probs,
        axis=-1,
        keepdims=True,
    )

    dO = (
        out.grad.storage.data
        .reshape(B, T, H, D)
        .transpose(0, 2, 1, 3)
    )

    dV = np.matmul(
        probs.transpose(0, 1, 3, 2),
        dO,
    ).astype(np.float32)

    dProbs = np.matmul(
        dO,
        V.transpose(0, 1, 3, 2),
    ).astype(np.float32)

    dScores = (
        probs
        * (
            dProbs
            - np.sum(
                probs * dProbs,
                axis=-1,
                keepdims=True,
            )
        )
        * scale
    ).astype(np.float32)

    dQ_rot = np.matmul(
        dScores,
        K,
    ).astype(np.float32)

    dK_rot = np.matmul(
        dScores.transpose(0, 1, 3, 2),
        Q,
    ).astype(np.float32)

    dQ_bthd = (
        dQ_rot
        .transpose(0, 2, 1, 3)
        .copy()
    )

    dK_bthd = (
        dK_rot
        .transpose(0, 2, 1, 3)
        .copy()
    )

    apply_rope_backward_np(
        dQ_bthd,
        dK_bthd,
        rope_cos.storage.data,
        rope_sin.storage.data,
    )

    dQ = dQ_bthd.reshape(B * T, C)
    dK = dK_bthd.reshape(B * T, C)
    dV = (
        dV
        .transpose(0, 2, 1, 3)
        .reshape(B * T, C)
    )

    if wq.requires_grad:
        wq.grad.storage.data[...] += (
            X_flat.T @ dQ
        ).astype(np.float32)

    if bq.requires_grad:
        bq.grad.storage.data[...] += np.sum(
            dQ,
            axis=0,
        ).astype(np.float32)

    if wk.requires_grad:
        wk.grad.storage.data[...] += (
            X_flat.T @ dK
        ).astype(np.float32)

    if bk.requires_grad:
        bk.grad.storage.data[...] += np.sum(
            dK,
            axis=0,
        ).astype(np.float32)

    if wv.requires_grad:
        wv.grad.storage.data[...] += (
            X_flat.T @ dV
        ).astype(np.float32)

    if bv.requires_grad:
        bv.grad.storage.data[...] += np.sum(
            dV,
            axis=0,
        ).astype(np.float32)

    if x.requires_grad:
        dX_norm = (
            dQ @ wq.storage.data.T
            + dK @ wk.storage.data.T
            + dV @ wv.storage.data.T
        ).reshape(B, T, C).astype(np.float32)

        rms_dot = np.mean(
            dX_norm * X_norm,
            axis=-1,
            keepdims=True,
        )

        dX = (
            inv_rms
            * (
                dX_norm
                - X_norm * rms_dot
            )
        ).astype(np.float32)

        x.grad.storage.data[...] += dX

def cpu_fused_rms_gqa_forward(inputs, outputs, scratch=None):
    x, norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
    out = outputs[0]
    op = out.op.op

    B, T, C = x.shape
    H = op.n_head
    HKV = op.n_kv_head
    D = C // H
    group_size = H // HKV
    half_D = D // 2

    X = x.storage.data.reshape(B, T, C)
    z = scratch["z"].data
    inv = scratch["inv"].data
    tmp_x = scratch["tmp_x"].data
    q = scratch["q"].data
    k = scratch["k"].data
    v = scratch["v"].data

    # 1. 带权 RMSNorm。
    np.multiply(X, X, out=tmp_x)
    np.mean(tmp_x, axis=-1, keepdims=True, out=inv)
    np.add(inv, np.float32(op.eps), out=inv)
    np.sqrt(inv, out=inv)
    np.reciprocal(inv, out=inv)

    np.multiply(X, inv, out=z)
    np.multiply(z, norm_weight.storage.data, out=z)

    # 2. QKV 投影：Q 宽度 C，K/V 宽度 HKV*D。
    z_flat = z.reshape(B * T, C)

    for dest, weight, bias in ((q, wq, bq), (k, wk, bk), (v, wv, bv)):
        dest_flat = dest.reshape(B * T, -1)
        np.matmul(z_flat, weight.storage.data, out=dest_flat)
        np.add(dest_flat, bias.storage.data, out=dest_flat)

    # 3. split-half RoPE，Q/K 分别按各自头数处理。
    cos = rope_cos.storage.data[:T]
    sin = rope_sin.storage.data[:T]
    rope0 = scratch["rope0"].data
    rope1 = scratch["rope1"].data

    for data, heads in ((q, H), (k, HKV)):
        for batch in range(B):
            for head in range(heads):
                lo = data[batch, :, head, :half_D]
                hi = data[batch, :, head, half_D:]

                np.multiply(lo, cos, out=rope0)
                np.multiply(hi, sin, out=rope1)
                if op.rope_sign == 1:
                    np.subtract(rope0, rope1, out=rope0)
                else:
                    np.add(rope0, rope1, out=rope0)

                # lo 尚未覆盖，可以继续计算第二半。
                np.multiply(lo, sin, out=rope1)
                np.multiply(hi, cos, out=hi)
                if op.rope_sign == 1:
                    np.add(hi, rope1, out=hi)
                else:
                    np.subtract(hi, rope1, out=hi)

                np.copyto(lo, rope0)

    # 4. GQA：直接读取所属 KV 头，不复制 K/V。
    probs = scratch["p"].data
    row = scratch["row"].data
    scale = np.float32(D ** -0.5)
    context = out.storage.data.reshape(B, T, H, D)

    for batch in range(B):
        for head in range(H):
            kv_head = head // group_size
            Q = q[batch, :, head, :]
            K = k[batch, :, kv_head, :]
            V = v[batch, :, kv_head, :]

            np.matmul(Q, K.T, out=probs)
            np.multiply(probs, scale, out=probs)

            if op.is_causal:
                for token in range(T):
                    probs[token, token + 1:] = -np.inf

            np.max(probs, axis=-1, keepdims=True, out=row)
            np.subtract(probs, row, out=probs)
            np.exp(probs, out=probs)
            np.sum(probs, axis=-1, keepdims=True, out=row)
            np.divide(probs, row, out=probs)

            np.matmul(probs, V, out=context[batch, :, head, :])

def cpu_fused_rms_gqa_backward(inputs, outputs, scratch=None):
    out = outputs[0]
    if out.grad is None:
        return

    x, norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
    op = out.op.op

    B, T, C = x.shape
    H = op.n_head
    HKV = op.n_kv_head
    D = C // H
    group_size = H // HKV
    half_D = D // 2
    scale = np.float32(D ** -0.5)

    X = x.storage.data.reshape(B, T, C)
    z = scratch["z"].data
    inv = scratch["inv"].data
    tmp_x = scratch["tmp_x"].data
    q = scratch["q"].data
    k = scratch["k"].data
    v = scratch["v"].data

    dq = scratch["dq"].data
    dk = scratch["dk"].data
    dv = scratch["dv"].data
    dz = scratch["dz"].data

    probs = scratch["p"].data
    dp = scratch["dp"].data
    ds = scratch["ds"].data
    row = scratch["row"].data
    tmp_kv = scratch["tmp_kv"].data

    cos = rope_cos.storage.data[:T]
    sin = rope_sin.storage.data[:T]
    rope0 = scratch["rope0"].data
    rope1 = scratch["rope1"].data

    # 1. 重算带权 RMSNorm。
    np.multiply(X, X, out=tmp_x)
    np.mean(tmp_x, axis=-1, keepdims=True, out=inv)
    np.add(inv, np.float32(op.eps), out=inv)
    np.sqrt(inv, out=inv)
    np.reciprocal(inv, out=inv)

    np.multiply(X, inv, out=z)
    np.multiply(z, norm_weight.storage.data, out=z)

    # 2. 重算 QKV。
    z_flat = z.reshape(B * T, C)

    for dest, weight, bias in ((q, wq, bq), (k, wk, bk), (v, wv, bv)):
        dest_flat = dest.reshape(B * T, -1)
        np.matmul(z_flat, weight.storage.data, out=dest_flat)
        np.add(dest_flat, bias.storage.data, out=dest_flat)

    # 3. 重算前向 RoPE。
    for data, heads in ((q, H), (k, HKV)):
        for batch in range(B):
            for head in range(heads):
                lo = data[batch, :, head, :half_D]
                hi = data[batch, :, head, half_D:]

                np.multiply(lo, cos, out=rope0)
                np.multiply(hi, sin, out=rope1)
                if op.rope_sign == 1:
                    np.subtract(rope0, rope1, out=rope0)
                else:
                    np.add(rope0, rope1, out=rope0)

                np.multiply(lo, sin, out=rope1)
                np.multiply(hi, cos, out=hi)
                if op.rope_sign == 1:
                    np.add(hi, rope1, out=hi)
                else:
                    np.subtract(hi, rope1, out=hi)

                np.copyto(lo, rope0)

    # 4. 重算概率，计算 attention 梯度。
    d_out = out.grad.storage.data.reshape(B, T, H, D)
    dk.fill(0)
    dv.fill(0)

    for batch in range(B):
        for head in range(H):
            kv_head = head // group_size
            Q = q[batch, :, head, :]
            K = k[batch, :, kv_head, :]
            V = v[batch, :, kv_head, :]
            dO = d_out[batch, :, head, :]

            np.matmul(Q, K.T, out=probs)
            np.multiply(probs, scale, out=probs)

            if op.is_causal:
                for token in range(T):
                    probs[token, token + 1:] = -np.inf

            np.max(probs, axis=-1, keepdims=True, out=row)
            np.subtract(probs, row, out=probs)
            np.exp(probs, out=probs)
            np.sum(probs, axis=-1, keepdims=True, out=row)
            np.divide(probs, row, out=probs)

            # dV 累加同组 Q 头的贡献。
            np.matmul(probs.T, dO, out=tmp_kv)
            np.add(dv[batch, :, kv_head, :], tmp_kv, out=dv[batch, :, kv_head, :])

            # dS = scale * P * (dP - sum(P*dP))。
            np.matmul(dO, V.T, out=dp)
            np.multiply(probs, dp, out=ds)
            np.sum(ds, axis=-1, keepdims=True, out=row)
            np.subtract(dp, row, out=ds)
            np.multiply(ds, probs, out=ds)
            np.multiply(ds, scale, out=ds)

            np.matmul(ds, K, out=dq[batch, :, head, :])

            # dK 累加同组 Q 头的贡献。
            np.matmul(ds.T, Q, out=tmp_kv)
            np.add(dk[batch, :, kv_head, :], tmp_kv, out=dk[batch, :, kv_head, :])

    # 5. RoPE backward：使用前向旋转的转置。
    for data, heads in ((dq, H), (dk, HKV)):
        for batch in range(B):
            for head in range(heads):
                lo = data[batch, :, head, :half_D]
                hi = data[batch, :, head, half_D:]

                np.multiply(lo, cos, out=rope0)
                np.multiply(hi, sin, out=rope1)
                if op.rope_sign == 1:
                    np.add(rope0, rope1, out=rope0)
                else:
                    np.subtract(rope0, rope1, out=rope0)

                np.multiply(lo, sin, out=rope1)
                np.multiply(hi, cos, out=hi)
                if op.rope_sign == 1:
                    np.subtract(hi, rope1, out=hi)
                else:
                    np.add(hi, rope1, out=hi)

                np.copyto(lo, rope0)

    # 6. 投影参数梯度，以及带权归一化输出的梯度 dZ。
    tmp_w = scratch["tmp_w"].data.reshape(-1)
    tmp_b = scratch["tmp_b"].data
    tmp_x_flat = tmp_x.reshape(B * T, C)
    dz_flat = dz.reshape(B * T, C)

    need_dz = x.requires_grad or norm_weight.requires_grad
    if need_dz:
        dz.fill(0)

    for grad, weight, bias in ((dq, wq, bq), (dk, wk, bk), (dv, wv, bv)):
        grad_flat = grad.reshape(B * T, -1)
        width = grad_flat.shape[1]

        if weight.requires_grad:
            work = tmp_w[:C * width].reshape(C, width)
            np.matmul(z_flat.T, grad_flat, out=work)
            np.add(weight.grad.storage.data, work, out=weight.grad.storage.data)

        if bias.requires_grad:
            np.sum(grad_flat, axis=0, out=tmp_b[:width])
            np.add(bias.grad.storage.data, tmp_b[:width], out=bias.grad.storage.data)

        if need_dz:
            np.matmul(grad_flat, weight.storage.data.T, out=tmp_x_flat)
            np.add(dz_flat, tmp_x_flat, out=dz_flat)

    if not need_dz:
        return

    # 7. 所有 dW 已完成，z 复用为不带权的 x_hat。
    # 直接由 X 和 inv 计算，不除以 norm_weight。
    np.multiply(X, inv, out=z)

    if norm_weight.requires_grad:
        np.multiply(dz, z, out=tmp_x)
        np.sum(tmp_x, axis=(0, 1), out=tmp_b)
        np.add(norm_weight.grad.storage.data, tmp_b, out=norm_weight.grad.storage.data)

    # 8. RMSNorm 输入梯度，累加到已有残差梯度。
    if x.requires_grad:
        np.multiply(dz, norm_weight.storage.data, out=dz)
        np.multiply(dz, z, out=tmp_x)

        rms_dot = scratch["dot"].data
        np.mean(tmp_x, axis=-1, keepdims=True, out=rms_dot)

        np.multiply(z, rms_dot, out=tmp_x)
        np.subtract(dz, tmp_x, out=tmp_x)
        np.multiply(tmp_x, inv, out=tmp_x)

        np.add(x.grad.storage.data, tmp_x, out=x.grad.storage.data)

def cpu_adamw_step(p_storage, g_storage, m_storage, v_storage, tmp_view,
	lr, lr_t, b1, b2, eps, weight_decay):
	p_data = p_storage.data
	g = g_storage.data
	m = m_storage.data
	v = v_storage.data
	tmp = tmp_view

	if weight_decay > 0.0:
		np.multiply(p_data, lr*weight_decay, out=tmp)
		np.subtract(p_data, tmp, out=p_data)

	np.multiply(m, b1, out=m)
	np.multiply(g, 1.0-b1, out=tmp)
	np.add(m, tmp, out=m)

	np.multiply(v, b2, out=v)
	np.multiply(g, g, out=tmp)
	np.multiply(tmp, 1.0-b2, out=tmp)
	np.add(v, tmp, out=v)

	np.sqrt(v, out=tmp)
	np.add(tmp, eps, out=tmp)
	np.divide(m, tmp, out=tmp)
	np.multiply(tmp, lr_t, out=tmp)
	np.subtract(p_data, tmp, out=p_data)

def cpu_fused_linear_add_forward(inputs, outputs, scratch=None):
    x, act, w, b = inputs
    out = outputs[0]
    
    X = x.storage.data.reshape(-1, x.shape[-1])
    ACT = act.storage.data.reshape(-1, act.shape[-1])
    W = w.storage.data
    B = b.storage.data
    OUT = out.storage.data.reshape(-1, out.shape[-1])

    alias_x_out = np.shares_memory(x.storage.data, out.storage.data)
    alias_act_out = np.shares_memory(act.storage.data, out.storage.data)

    X_saved = np.array(X, copy=True) if alias_x_out else X
    ACT_saved = np.array(ACT, copy=True) if alias_act_out else ACT

    np.matmul(ACT_saved, W, out=OUT)
    np.add(OUT, B, out=OUT)
    np.add(OUT, X_saved, out=OUT)

def cpu_recompute_fused_lin_add(act, scratch):
	x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = act.op.inputs

	B, T, C = x.shape
	H = act.op.op.n_head
	D = C // H
	eps = act.op.op.eps

	X = x.storage.data.reshape(B, T, C)

	inv_rms = scratch["fa_inv_rms"].data.reshape(B, T, 1)
	x_norm = scratch["fa_x_norm"].data.reshape(B, T, C)

	np.multiply(X, X, out=x_norm)
	np.mean(
		x_norm,
		axis=-1,
		keepdims=True,
		out=inv_rms,
	)

	np.add(inv_rms, eps, out=inv_rms)
	np.sqrt(inv_rms, out=inv_rms)
	np.reciprocal(inv_rms, out=inv_rms)
	np.multiply(X, inv_rms, out=x_norm)

	X_flat = x_norm.reshape(B * T, C)

	q = scratch["fa_q"].data.reshape(B, T, C)
	k = scratch["fa_k"].data.reshape(B, T, C)
	v = scratch["fa_v"].data.reshape(B, T, C)

	np.dot(X_flat, wq.storage.data, out=q.reshape(B * T, C))
	np.add(q.reshape(B * T, C), bq.storage.data, out=q.reshape(B * T, C))

	np.dot(X_flat, wk.storage.data, out=k.reshape(B * T, C))
	np.add(k.reshape(B * T, C), bk.storage.data, out=k.reshape(B * T, C))

	np.dot(X_flat, wv.storage.data, out=v.reshape(B * T, C))
	np.add(v.reshape(B * T, C), bv.storage.data, out=v.reshape(B * T, C))

	Q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
	K = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
	V = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

	attn = scratch["fa_attn"].data
	row_max = scratch["fa_row_max"].data
	row_sum = scratch["fa_row_sum"].data

	scale = 1.0 / np.sqrt(D)

	np.matmul(
		Q,
		K.transpose(0, 1, 3, 2),
		out=attn,
	)
	np.multiply(attn, scale, out=attn)

	if act.op.op.is_causal:
		mask = scratch["fa_mask"].data
		mask.fill(1)
		mask[...] = np.triu(mask, 1)
		attn[..., mask] = -np.inf

	np.max(
		attn,
		axis=-1,
		keepdims=True,
		out=row_max,
	)
	np.subtract(attn, row_max, out=attn)
	np.exp(attn, out=attn)
	np.sum(
		attn,
		axis=-1,
		keepdims=True,
		out=row_sum,
	)
	np.divide(attn, row_sum, out=attn)

	out_view = (
		scratch["fa_attn_out"].data
		.reshape(B, T, H, D)
		.transpose(0, 2, 1, 3)
	)
	np.matmul(attn, V, out=out_view)

def cpu_fused_linear_add_backward(inputs, outputs, scratch=None):
    x, act, w, b = inputs
    out = outputs[0]

    if out.grad is None:
        return

    d_out = out.grad.storage.data.reshape(-1, out.shape[-1])

    if w.requires_grad and act.op is not None and act.op.op.op_name == "FusedRMSNormGQA":
    	fa_x, norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = act.op.inputs
    	fa_op = act.op.op
    	B, T, C = fa_x.shape
    	H = fa_op.n_head
    	HKV = fa_op.n_kv_head
    	D = C // H
    	group_size = H // HKV
    	half_D = D // 2 

    	X = fa_x.storage.data.reshape(B, T, C)
    	z = scratch["z"].data
    	inv = scratch["inv"].data
    	tmp_x = scratch["tmp_x"].data
    	q = scratch["q"].data
    	k = scratch["k"].data
    	v = scratch["v"].data

    	# 1. 重算带权RMSNorm
    	np.multiply(X, X, out=tmp_x)
    	np.mean(tmp_x, axis=-1, keepdims=True, out=inv)
    	np.add(inv, np.float32(fa_op.eps), out=inv)
    	np.sqrt(inv, out=inv)
    	np.reciprocal(inv, out=inv)

    	np.multiply(X, inv, out=z)
    	np.multiply(z, norm_weight.storage.data, out=z)

    	# 2. 重算QKV
    	z_flat = z.reshape(B*T, C)
    	for dest, weight, bias in ((q, wq, bq), (k, wk, bk), (v, wv, bv)):
    		dest_flat = dest.reshape(B*T, -1)
    		np.matmul(z_flat, weight.storage.data, out=dest_flat)
    		np.add(dest_flat, bias.storage.data, out=dest_flat)

    	# 3. 重算ROPE
    	cos = rope_cos.storage.data[:T]
    	sin = rope_sin.storage.data[:T]
    	rope0 = scratch["rope0"].data
    	rope1 = scratch["rope1"].data

    	for data, heads in ((q, H), (k, HKV)):
    		for batch in range(B):
    			for head in range(heads):
    				lo = data[batch, :, head, :half_D]
    				hi = data[batch, :, head, half_D:]

    				np.multiply(lo, cos, out=rope0)
    				np.multiply(hi, sin, out=rope1)
    				if fa_op.rope_sign == 1:
    					np.subtract(rope0, rope1, out=rope0)
    				else:
    					np.add(rope0, rope1, out=rope0)

    				np.multiply(lo, sin, out=rope1)
    				np.multiply(hi, cos, out=hi)
    				if fa_op.rope_sign == 1:
    					np.add(hi, rope1, out=hi)
    				else:
    					np.subtract(hi, rope1, out=hi)
    				np.copyto(lo, rope0)

    	# 4. 重算GQA输出到预分配的fa_out
    	probs = scratch["p"].data
    	row = scratch["row"].data
    	scale = np.float32(D ** -0.5)
    	context = scratch["fa_out"].data.reshape(B, T, H, D)

    	for batch in range(B):
    		for head in range(H):
    			kv_head = head // group_size
    			Q = q[batch, :, head, :]
    			K = k[batch, :, kv_head, :]
    			V = v[batch, :, kv_head, :]

    			np.matmul(Q, K.T, out=probs)
    			np.multiply(probs, scale, out=probs)
    			if fa_op.is_causal:
    				for token in range(T):
    					probs[token, token+1:] = -np.inf

    			np.max(probs, axis=-1, keepdims=True, out=row)
    			np.subtract(probs, row, out=probs)
    			np.exp(probs, out=probs)
    			np.sum(probs, axis=-1, keepdims=True, out=row)
    			np.divide(probs, row, out=probs)
    			np.matmul(probs, V, out=context[batch, :, head, :])

    	act_value = scratch["fa_out"].data.reshape(-1, act.shape[-1])
    else:
    	act_value = act.storage.data.reshape(-1, act.shape[-1])

    if x.requires_grad:
        x.grad.storage.data.reshape(-1, x.shape[-1])[...] += d_out

    if act.requires_grad:
        act.grad.storage.data.reshape(-1, act.shape[-1])[...] += (d_out @ w.storage.data.T).astype(np.float32)

    if w.requires_grad:
        w.grad.storage.data[...] += (act_value.T @ d_out).astype(np.float32)

    if b.requires_grad:
        b.grad.storage.data[...] += np.sum(d_out, axis=0).astype(np.float32)

def cpu_kv_cache_attention_forward(inputs, outputs, scratch=None):
	x, wq, bq, wk, bk, wv, bv, k_cache, v_cache, position, rope_cos, rope_sin = inputs

	out = outputs[0]
	op = out.op.op

	B, T, C = x.shape
	H = op.n_head
	D = C // H
	S = k_cache.shape[1]
	start = position.storage.data[0]
	end = start + T

	X = x.storage.data
	z = scratch["x_norm"].data
	inv = scratch["inv_rms"].data
	q = scratch["q"].data
	k = scratch["k"].data
	v = scratch["v"].data

	np.multiply(X, X, out=z)
	np.mean(z, axis=-1, keepdims=True, out=inv)
	np.add(inv, out.op.op.eps, out=inv)
	np.sqrt(inv, out=inv)
	np.reciprocal(inv, out=inv)
	np.multiply(X, inv, out=z)

	z2 = z.reshape(B*T, C)
	for dst, weight, bias in ((q, wq, bq), (k, wk, bk), (v, wv, bv)):
		dst2 = dst.reshape(B*T, C)
		np.matmul(z2, weight.storage.data, out=dst2)
		np.add(dst2, bias.storage.data, out=dst2)

	work = z.reshape(B, T, H, D)
	cos = rope_cos.storage.data[start:end][None, :, None, :]
	sin = rope_sin.storage.data[start:end][None, :, None, :]

	rope_lo = scratch["rope_lo"].data
	rope_hi = scratch["rope_hi"].data
	q0 = scratch["rope_q0"].data
	q1 = scratch["rope_q1"].data 
	k0 = scratch["rope_k0"].data
	k1 = scratch["rope_k1"].data 

	np.copyto(q0, q[..., : D//2])
	np.copyto(q1, q[..., D//2:])
	np.copyto(k0, k[..., : D//2])
	np.copyto(k1, k[..., D//2:])

	np.multiply(q0, cos, out=q[..., :D//2])
	np.multiply(q1, sin, out=rope_lo)
	np.add(q[..., :D//2], rope_lo, out=q[..., :D//2])

	np.multiply(q1, cos, out=q[..., D // 2:])
	np.multiply(q0, sin, out=rope_lo)
	np.subtract(q[..., D // 2:], rope_lo, out=q[..., D // 2:])

	np.multiply(k0, cos, out=k[..., :D // 2])
	np.multiply(k1, sin, out=rope_hi)
	np.add(k[..., :D // 2], rope_hi, out=k[..., :D // 2])

	np.multiply(k1, cos, out=k[..., D // 2:])
	np.multiply(k0, sin, out=rope_hi)
	np.subtract(k[..., D // 2:], rope_hi, out=k[..., D // 2:])

	K = k_cache.storage.data
	V = v_cache.storage.data
	np.copyto(K[:, start:end], k)
	np.copyto(V[:, start:end], v)

	output = out.storage.data.reshape(B, T, H, D)
	score_buffer = scratch["scores"].data
	stat_buffer = scratch["row_stat"].data
	R = score_buffer.shape[0]
	scale = np.float32(1.0 / np.sqrt(D))

	for batch in range(B):
		for head in range(H):
			for first in range(0, T, R):
				rows = min(R, T-first)
				keys = start + first + rows
				scores = score_buffer[:rows, :keys]
				stat = stat_buffer[:rows]

				np.matmul(q[batch, first:first+rows, head, :], K[batch, :keys, head, :].T, out=scores)
				np.multiply(scores, scale, out=scores)

				for row in range(rows):
					scores[row, start+first+row+1:] = -np.inf

				np.max(scores, axis=-1, keepdims=True, out=stat)
				np.subtract(scores, stat, out=scores)
				np.exp(scores, out=scores)
				np.sum(scores, axis=-1, keepdims=True, out=stat)
				np.divide(scores, stat, out=scores)
				np.matmul(scores, V[batch, :keys, head, :], out=output[batch, first:first+rows, head, :])

def cpu_kv_cache_attention_backward(inputs, outputs, scratch=None):
	return

def cpu_lm_head_forward(inputs, outputs, scratch=None):
	x, weight = inputs
	out = outputs[0]

	x_2d = x.storage.data.reshape(-1, x.shape[-1])
	w_data = weight.storage.data
	out_2d = out.storage.data.reshape(-1, out.shape[-1])
	np.matmul(x_2d, w_data.T, out=out_2d)

def cpu_last_position_lm_head_forward(inputs, outputs, scratch=None):
	x, weight, position = inputs
	out = outputs[0]

	row = int(position.storage.data[0])

	T = x.shape[1]
	C = x.shape[-1]
	V = weight.shape[0]

	# x_2d = x.storage.data.reshape(T, C)
	# out_2d = out.storage.data.reshape(T, V)
	# np.matmul(x_2d[row:row+1, :], weight.storage.data.T, out=out_2d[row:row+1,:])

	x_row = x.storage.data.reshape(T, C)[row:row+1, :]
	out_row = out.storage.data.reshape(1, V)
	np.matmul(x_row, weight.storage.data.T, out=out_row)

def cpu_rollout_topk_forward(inputs, outputs, scratch=None):
	logits, top_k, logits_position = inputs
	values_out, indices_out = outputs

	logits_data = logits.storage.data
	position_data = logits_position.storage.data

	op = values_out.op.op 

	if logits_data.ndim == 2:
		batch, vocab = logits_data.shape
		time = 1
		logits_3d = logits_data.reshape(batch, 1, vocab)
	elif logits.ndim == 3:
		batch, time, vocab = logits_data.shape
		logits_3d = logits_data
	else:
		raise ValueError(f"RolloutTopk must be [B, V] or [B, T, V], got {logits_data.shape}")

	selected_time = 0
	selected_logits = logits_3d[:, selected_time, :]

	batch = selected_logits.shape[0]
	k_max = values_out.shape[-1]
	requested_k = top_k.storage.data.reshape(-1)[0]
	k = min(requested_k, k_max, vocab)

	values_np = values_out.storage.data
	indices_np = indices_out.storage.data

	key = np.empty((batch, vocab), dtype=[('value', selected_logits.dtype), ('index', np.int64)])
	key['value'] = -selected_logits
	key['index'] = np.arange(vocab, dtype=np.int64)[None, :]

	order = np.argsort(key, axis=-1, order=['value', 'index'], kind='stable')[:, :k]

	values_np[:, :k] = np.take_along_axis(selected_logits, order, axis=-1)
	indices_np[:, :k] = order.astype(np.int32, copy=False)

def cpu_rollout_sample_forward(inputs, outputs, scratch=None):
	topk_values, topk_indices, rng_state, finished, temperature, top_k, next_tokens, generated_ids, lengths, position, step = inputs
	finished_all = outputs[0]
	op = next_tokens.op.op

	values = topk_values.storage.data
	indices = topk_indices.storage.data

	rng = rng_state.storage.data
	done = finished.storage.data.reshape(-1)
	tokens = next_tokens.storage.data.reshape(-1)
	generated = generated_ids.shape.data

	len = lengths.storage.data.reshape(-1)
	pos = position.storage.data.reshape(-1)
	step_data = step.storage.data.reshape(-1)

	B = tokens.size
	K_MAX = values.shape[1]
	capacity = generated.shape[1]
	current_step = step_data[0]
	temp = temperature.reshape.storage.data.reshape(-1)[0]
	requested_k = top_k.storage.data.reshape(-1)[0]
	K = min(requested_k, K_MAX)

	for row in range(B):
		if done[row] != 0:
			tokens[row] = op.eos_token
			continue

	temperature = temperature.storage.data.reshape(-1)[0]
	K = int(top_k.storage.data.reshape(-1)[0])
	K = min(K, values.shape[-1])

	rng = np.random.default_rng(int(state.reshape(-1)[0]) & 0xFFFFFFFF)

	for row in range(values.shape[0]):
		if done[row] != 0:
			out[row] = op.eos_token
			continue
		row_values = values[row, :K].astype(np.float64)
		row_indices = indices[row, :K]

		if temperature <= 0.0:
			out[row] = row_indices[np.argmax(row_values)]
			continue

		scaled = row_values / temperature
		scaled -= np.max(scaled)

		probs = np.exp(scaled)
		probs /= np.sum(probs)

		selected = int(rng.choice(K, p=probs))
		out[row] = row_indices[selected]

	state[0] = np.uint32((state[0] + 1) & 0xFFFFFFFF)

def cpu_decode_finished_forward(inputs, outputs, scratch=None):
	next_ids, finished, step, config = inputs
	out = outputs[0]

	finished_data = finished.storage.data.reshape(-1)
	done = np.all(finished_data != 0)
	out.storage.data.reshape(-1)[0] = 1 if done else 0

class CPUBackend(Backend):
	def __init__(self):
		self.forward_kernels = {
			AddOp: cpu_add_forward,
			MulOp: cpu_mul_forward,
			ViewOp: cpu_view_forward,
			ReshapeOp: cpu_reshape_forward,
			PermuteOp: cpu_permute_forward,
			TransposeOp: cpu_transpose_forward,
			ContiguousOp: cpu_contiguous_forward,
			ExpOp: cpu_exp_forward,
			LogOp: cpu_log_forward,
			RsqrtOp: cpu_rsqrt_forward,
			SoftmaxOp: cpu_softmax_forward,
			GELUOp: cpu_gelu_forward,
			LayerNormOp: cpu_layernorm_forward,
			GatherOp: cpu_gather_forward,
			CrossEntropyLossOp: cpu_crossentropy_forward,
			ScaledDotProductAttenOp: cpu_sdpa_forward,
			FusedLinearCrossEntropyOp: cpu_fused_linear_ce_forward,
			FusedRMSNormLinearCrossEntropyOp: cpu_fused_rmsnorm_linear_ce_forward,
			FusedMLPOp: cpu_fused_mlp_forward,
			FusedQKVProjOp: cpu_fused_qkv_proj_forward,
			AttnOutContiguousOp: cpu_attn_out_contiguous_forward,
			FusedRMSNormMLPOp: cpu_fused_rmsnorm_mlp_forward,
			FlashAttentionOp: cpu_flash_attention_forward,
			FusedRMSNormFlashAttentionOp: cpu_fused_rms_fa_forward,
			FusedRMSNormGQAOp: cpu_fused_rms_gqa_forward,
			FusedLinearAddOp: cpu_fused_linear_add_forward,
			KVCacheAttentionOp: cpu_kv_cache_attention_forward,
			LMHeadOp: cpu_lm_head_forward,
			LastPositionLMHeadOp: cpu_last_position_lm_head_forward,
			RolloutTopkOp: cpu_rollout_topk_forward,
			RolloutSampleOp: cpu_rollout_sample_forward,
			DecodeFinishedOp: cpu_decode_finished_forward,
		}
		self.backward_kernels = {
			AddOp: cpu_add_backward,
			MulOp: cpu_mul_backward,
			ViewOp: cpu_view_backward,
			ReshapeOp: cpu_reshape_backward,
			PermuteOp: cpu_permute_backward,
			TransposeOp: cpu_transpose_backward,
			ContiguousOp: cpu_contiguous_backward,
			ExpOp: cpu_exp_backward,
			LogOp: cpu_log_backward,
			RsqrtOp: cpu_rsqrt_backward,
			SoftmaxOp: cpu_softmax_backward,
			GELUOp: cpu_gelu_backward,
			LayerNormOp: cpu_layernorm_backward,
			GatherOp: cpu_gather_backward,
			CrossEntropyLossOp: cpu_crossentropy_backward,
			ScaledDotProductAttenOp: cpu_sdpa_backward,
			FusedLinearCrossEntropyOp: cpu_fused_linear_ce_backward,
			FusedRMSNormLinearCrossEntropyOp: cpu_fused_rmsnorm_linear_ce_backward,
			FusedMLPOp: cpu_fused_mlp_backward,
			FusedQKVProjOp: cpu_fused_qkv_proj_backward,
			AttnOutContiguousOp: cpu_attn_out_contiguous_backward,
			FusedRMSNormMLPOp: cpu_fused_rmsnorm_mlp_backward,
			FlashAttentionOp: cpu_flash_attention_backward,
			FusedRMSNormFlashAttentionOp: cpu_fused_rms_fa_backward,
			FusedRMSNormGQAOp: cpu_fused_rms_gqa_backward,
			FusedLinearAddOp: cpu_fused_linear_add_backward,
			KVCacheAttentionOp: cpu_kv_cache_attention_backward,
		}

	def get_forward_kernel(self, op):
		op_type = type(op)
		if op_type not in self.forward_kernels:
			return None
			# raise RuntimeError(f"forward kernel not found: {op_type}")
		return self.forward_kernels[op_type]

	def get_backward_kernel(self, op):
		op_type = type(op)
		if op_type not in self.backward_kernels:
			return None
		return self.backward_kernels[op_type]

	def bind_matmul_forward(self, node):
		x, w = node.inputs[0], node.inputs[1]
		key = (is_materialized(x), is_materialized(w))
		return _MATMUL_FORWARD_TABLE[key]

	def bind_matmul_backward(self, node, scratch_views):
		x, w = node.inputs[0], node.inputs[1]
		key = (is_materialized(x), is_materialized(w))
		base = _MATMUL_BACKWARD_TABLE[key]
		layout = node.scratch_bwd_layout
		if layout is None:
			raise RuntimeError('MatMul backward requires scratch_bwd_layout') 
		return functools.partial(base, scratch=scratch_views)

	def fill_storage(self, storage, value):
		storage.data.fill(value)

	def fill_storage2(self, storage, value):
		storage.data.fill(value)

	def copy_from_host(self, tensor, host_np):
		np.copyto(tensor.storage.data, host_np)

	def copy_to_host(self, tensor):
		return tensor.storage.data

	def adamw_step(self, p_storage, g_storage, m_storage, v_storage, tmp_storage,
		lr, lr_t, b1, b2, eps, weight_decay):
		cpu_adamw_step(p_storage, g_storage, m_storage, v_storage, tmp_storage,
			lr, lr_t, b1, b2, eps, weight_decay)