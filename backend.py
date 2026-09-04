import numpy as np
import functools
from math import prod
import struct

from op import AddOp, MulOp, MatMulOp, ViewOp, ReshapeOp, PermuteOp, TransposeOp, ContiguousOp
from op import ExpOp, LogOp, RsqrtOp, SoftmaxOp, GELUOp, CrossEntropyLossOp, LayerNormOp, GatherOp, ScaledDotProductAttenOp
from op import FusedLinearCrossEntropyOp, FusedMLPOp, FusedQKVProjOp, AttnOutContiguousOp, FusedRMSNormMLPOp, FlashAttentionOp 
from op import FusedRMSNormFlashAttentionOp, FusedLinearAddOp
from utils import xavier_uniform, reduce_broadcast_grad, tensor_to_numpy, is_materialized, grad_shape_matches, mul_bcast_kind, f32_bits

import metal_ext

GELU_C = 0.7978845608028654
GELU_A = 0.044715
GELU_A3 = 0.134145

class Backend:
	mixed_precision = False

	def get_forward_kernel(self, op):
		raise NotImplementedError()

	def get_backward_kernel(self, op):
		raise NotImplementedError()

	def fill_storage(self, storage, value):
		raise NotImplementedError()

	def fill_scalar(self, tensor, value):
		self.fill_storage(tensor.storage, float(value))

	def fill(self, tensor, value):
		self.fill_scalar(tensor, float(value))

	def init(self, tensor):
		v = tensor.init_val
		if v is None:
			return
		try:
			if isinstance(v, (int, float)):
				self.fill_scalar(tensor, float(v))
			elif isinstance(v, np.ndarray):
				self.copy_from_host(tensor, np.asarray(v, dtype=tensor.dtype))
			elif v == "xavier":
				host = xavier_uniform(tensor.shape, tensor.dtype)
				self.copy_from_host(tensor, host)
			elif v == "zeros":
				self.fill_scalar(tensor, 0.0)
			else:
				raise ValueError(f"unknown init_val: {v!r}")
		finally:
			tensor.init_val = None

	def begin_pass(self):
		pass 

	def end_pass(self):
		pass

	def synchronize(self):
		pass

# ---------------------------------------------------------------
#                           CPU
# ---------------------------------------------------------------
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
    x, wgu, wdown = inputs
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

    # GU = X_norm @ Wgu
    np.matmul(X_norm, Wgu, out=GU)

    G = GU[:, :H]
    U = GU[:, H:]

    # HIDDEN = SiLU(G) * U
    # 先在 HIDDEN 中计算 sigmoid(G)
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
    x, wgu, wdown = inputs
    out = outputs[0]

    if out.grad is None:
        return

    eps = out.op.op.eps

    X = x.storage.data.reshape(
        -1,
        x.shape[-1],
    )
    Wgu = wgu.storage.data
    Wdown = wdown.storage.data
    DOUT = out.grad.storage.data.reshape(
        -1,
        x.shape[-1],
    )

    N, C = X.shape
    H = Wgu.shape[1] // 2

    inv_rms = scratch["inv_rms"].data.reshape(N, 1)
    X_norm = scratch["x_norm"].data.reshape(N, C)
    GU = scratch["gu"].data.reshape(N, 2 * H)
    D_HIDDEN = scratch["dhidden"].data.reshape(N, H)
    GATE = scratch["gate"].data.reshape(N, H)
    SIGMOID = scratch["sigmoid"].data.reshape(N, H)
    D_X_NORM = scratch["dx_norm"].data.reshape(N, C)

    # RMSNorm forward 重算。
    np.multiply(X, X, out=X_norm)
    np.mean(
        X_norm,
        axis=-1,
        keepdims=True,
        out=inv_rms,
    )
    np.add(inv_rms, eps, out=inv_rms)
    np.sqrt(inv_rms, out=inv_rms)
    np.reciprocal(inv_rms, out=inv_rms)
    np.multiply(X, inv_rms, out=X_norm)

    # GU = X_norm @ Wgu
    np.matmul(X_norm, Wgu, out=GU)

    G = GU[:, :H]
    U = GU[:, H:]

    # 单独保存原始 gate。
    np.copyto(GATE, G)

    # SIGMOID = sigmoid(GATE)
    np.negative(GATE, out=SIGMOID)
    np.exp(SIGMOID, out=SIGMOID)
    np.add(SIGMOID, 1.0, out=SIGMOID)
    np.reciprocal(SIGMOID, out=SIGMOID)

    # D_HIDDEN 暂存 forward hidden。
    np.multiply(GATE, SIGMOID, out=D_HIDDEN)
    np.multiply(D_HIDDEN, U, out=D_HIDDEN)

    if wdown.requires_grad:
        TMP_DWDOWN = scratch["tmp_dwdown"].data
        np.matmul(
            D_HIDDEN.T,
            DOUT,
            out=TMP_DWDOWN,
        )
        np.add(
            wdown.grad.storage.data,
            TMP_DWDOWN,
            out=wdown.grad.storage.data,
        )

    # D_HIDDEN = DOUT @ Wdown.T
    np.matmul(
        DOUT,
        Wdown.T,
        out=D_HIDDEN,
    )

    # GU 将被原地改写为 DGU。
    D_GATE = GU[:, :H]
    D_UP = GU[:, H:]

    # 先计算 dGate。
    #
    # D_GATE =
    # sigmoid(g) * (1 + g*(1-sigmoid(g)))
    np.subtract(
        1.0,
        SIGMOID,
        out=D_GATE,
    )
    np.multiply(
        D_GATE,
        GATE,
        out=D_GATE,
    )
    np.add(
        D_GATE,
        1.0,
        out=D_GATE,
    )
    np.multiply(
        D_GATE,
        SIGMOID,
        out=D_GATE,
    )

    # D_GATE *= U
    # 此时 U 尚未被覆盖。
    np.multiply(
        D_GATE,
        U,
        out=D_GATE,
    )
    np.multiply(
        D_GATE,
        D_HIDDEN,
        out=D_GATE,
    )

    # 再计算 dUp。
    # 原始 gate 保存在 GATE，SIGMOID 也仍然有效。
    np.multiply(
        GATE,
        SIGMOID,
        out=D_UP,
    )
    np.multiply(
        D_UP,
        D_HIDDEN,
        out=D_UP,
    )

    DGU = GU

    if wgu.requires_grad:
        TMP_DWGU = scratch["tmp_dwgu"].data
        np.matmul(
            X_norm.T,
            DGU,
            out=TMP_DWGU,
        )
        np.add(
            wgu.grad.storage.data,
            TMP_DWGU,
            out=wgu.grad.storage.data,
        )

    np.matmul(
        DGU,
        Wgu.T,
        out=D_X_NORM,
    )

    # RMSNorm backward，无额外数组分配。
    for row in range(N):
        dot = 0.0

        for col in range(C):
            dot += (
                D_X_NORM[row, col]
                * X_norm[row, col]
            )

        mean_dot = dot / C

        for col in range(C):
            D_X_NORM[row, col] = (
                D_X_NORM[row, col]
                - X_norm[row, col] * mean_dot
            ) * inv_rms[row, 0]

    if x.requires_grad:
        np.add(
            x.grad.storage.data.reshape(N, C),
            D_X_NORM,
            out=x.grad.storage.data.reshape(N, C),
        )

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

    d_out = out.grad.storage.data.reshape(
        -1,
        out.shape[-1],
    )

    act_value = act.storage.data.reshape(
        -1,
        act.shape[-1],
    )

    if x.requires_grad:
        x.grad.storage.data.reshape(
            -1,
            x.shape[-1],
        )[...] += d_out

    if act.requires_grad:
        act.grad.storage.data.reshape(
            -1,
            act.shape[-1],
        )[...] += (
            d_out @ w.storage.data.T
        ).astype(np.float32)

    if w.requires_grad:
        w.grad.storage.data[...] += (
            act_value.T @ d_out
        ).astype(np.float32)

    if b.requires_grad:
        b.grad.storage.data[...] += np.sum(
            d_out,
            axis=0,
        ).astype(np.float32)

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
			FusedMLPOp: cpu_fused_mlp_forward,
			FusedQKVProjOp: cpu_fused_qkv_proj_forward,
			AttnOutContiguousOp: cpu_attn_out_contiguous_forward,
			FusedRMSNormMLPOp: cpu_fused_rmsnorm_mlp_forward,
			FlashAttentionOp: cpu_flash_attention_forward,
			FusedRMSNormFlashAttentionOp: cpu_fused_rms_fa_forward,
			FusedLinearAddOp: cpu_fused_linear_add_forward,
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
			FusedMLPOp: cpu_fused_mlp_backward,
			FusedQKVProjOp: cpu_fused_qkv_proj_backward,
			AttnOutContiguousOp: cpu_attn_out_contiguous_backward,
			FusedRMSNormMLPOp: cpu_fused_rmsnorm_mlp_backward,
			FlashAttentionOp: cpu_flash_attention_backward,
			FusedRMSNormFlashAttentionOp: cpu_fused_rms_fa_backward,
			FusedLinearAddOp: cpu_fused_linear_add_backward,
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
			# raise RuntimeError(f"backward kernel not found: {op_type}")
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

	def copy_from_host(self, tensor, host_np):
		np.copyto(tensor.storage.data, host_np)

	def copy_to_host(self, tensor):
		return tensor.storage.data

	def adamw_step(self, p_storage, g_storage, m_storage, v_storage, tmp_storage,
		lr, lr_t, b1, b2, eps, weight_decay):
		cpu_adamw_step(p_storage, g_storage, m_storage, v_storage, tmp_storage,
			lr, lr_t, b1, b2, eps, weight_decay)


# ---------------------------------------------------------------
#                           Metal
# ---------------------------------------------------------------
class MetalBackend(Backend):
	def __init__(self):
		self.ctx = metal_ext.MetalContext()
		self.numpy_view_funcs = {
			np.dtype(np.float32): self.ctx.numpy_view_float32,
			np.dtype(np.int32): self.ctx.numpy_view_int32,
			np.dtype(np.uint8): self.ctx.numpy_view_uint8,
		}
		self.forward_kernels = {
			# AddOp: self.metal_add_forward,
			MulOp: self.metal_mul_forward,
			ViewOp: cpu_view_forward,
			ReshapeOp: cpu_reshape_forward,
			PermuteOp: cpu_permute_forward,
			TransposeOp: cpu_transpose_forward,
			ContiguousOp: self.metal_contiguous_forward,
			ExpOp: self.make_kernel("exp_forward_kernel"),
			LogOp: self.make_kernel("log_forward_kernel"),
			RsqrtOp: self.make_kernel("rsqrt_forward_kernel"),
			SoftmaxOp: self.metal_softmax_forward,
			GELUOp: self.metal_gelu_forward,
			LayerNormOp: self.metal_layernorm_forward,
			GatherOp: self.metal_gather_forward,
			CrossEntropyLossOp: self.metal_crossentropy_forward,
			ScaledDotProductAttenOp: self.metal_sdpa_forward,
			FusedLinearCrossEntropyOp: self.metal_fused_linear_ce_forward,
			FusedMLPOp: self.metal_fused_mlp_forward,
			FusedQKVProjOp: self.metal_fused_qkv_proj_forward,
			AttnOutContiguousOp: self.metal_attn_out_contiguous_forward,
			FusedRMSNormMLPOp: self.metal_fused_rmsnorm_mlp_forward,
			FlashAttentionOp: self.metal_flash_attention_forward,
			FusedRMSNormFlashAttentionOp: self.metal_fused_rms_fa_forward,
			FusedLinearAddOp: self.metal_fused_linear_add_forward,
		}
		self.backward_kernels = {
			# AddOp: self.metal_add_backward,
			MulOp: self.metal_mul_backward,
			ViewOp: self.metal_contiguous_backward,
			ReshapeOp: self.metal_contiguous_backward,
			PermuteOp: self.metal_permute_backward,
			TransposeOp: self.metal_transpose_backward,
			ContiguousOp: self.metal_contiguous_backward,
			ExpOp: self.metal_exp_backward,
			LogOp: self.metal_log_backward,
			RsqrtOp: self.metal_rsqrt_backward,
			SoftmaxOp: self.metal_softmax_backward,
			GELUOp: self.metal_gelu_backward,
			LayerNormOp: self.metal_layernorm_backward,
			GatherOp: self.metal_gather_backward,
			CrossEntropyLossOp: self.metal_crossentropy_backward,
			ScaledDotProductAttenOp: self.metal_sdpa_backward,
			FusedLinearCrossEntropyOp: self.metal_fused_linear_ce_backward,
			FusedMLPOp: self.metal_fused_mlp_backward,
			FusedQKVProjOp: self.metal_fused_qkv_proj_backward,
			AttnOutContiguousOp: self.metal_attn_out_contiguous_backward,
			FusedRMSNormMLPOp: self.metal_fused_rmsnorm_mlp_backward,
			FlashAttentionOp: self.metal_flash_attention_backward,
			FusedRMSNormFlashAttentionOp: self.metal_fused_rms_fa_backward,
			FusedLinearAddOp: self.metal_fused_linear_add_backward,
		}

	def metal_mul_forward(self, inputs, outputs, scratch=None):
		x, y, out = inputs[0], inputs[1], outputs[0]
		
		def expand(shape, strides, target):
			pad = len(target) - len(shape)
			pshape = (1,)*pad + tuple(shape)
			pstrides = (0,)*pad + tuple(strides)
			return [s if p == t else 0 for s, p, t in zip(pstrides, pshape, target)]
			
		sx = expand(x.shape, x.strides, out.shape)
		sy = expand(y.shape, y.strides, out.shape)
		
		shape = list(out.shape)
		while len(shape) < 4:
			shape = [1] + shape
			sx = [0] + list(sx)
			sy = [0] + list(sy)
		ndim = len(shape)
		self.ctx.dispatch_with_args_3d('mul_forward_kernel',
			[x.storage.ptr.buffer_id, y.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[x.storage.ptr.offset, y.storage.ptr.offset, out.storage.ptr.offset],
			[shape[0], shape[1], shape[2], shape[3], sx[0], sx[1], sx[2], sx[3],
				sy[0], sy[1], sy[2], sy[3], ndim],
			prod(out.shape), 1, 1, min(prod(out.shape), 256))

	def metal_mul_backward(self, inputs, outputs, scratch=None):
	    x, y, out = inputs[0], inputs[1], outputs[0]
	    if out.grad is None: return

	    if y.numel == 1 and x.shape == out.shape:
	    	if x.requires_grad:
	    		self.ctx.dispatch_with_args_3d('mul_backward_scalar_dx',
	    			[out.grad.storage.ptr.buffer_id, y.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
	    			[out.grad.storage.ptr.offset, y.storage.ptr.offset, x.grad.storage.ptr.offset],
	    			[prod(out.shape)], prod(out.shape), 1, 1, min(prod(out.shape), 256))
	    	if y.requires_grad:
	    		self.ctx.dispatch_with_args_3d('mul_backward_scalar_dy',
	    			[out.grad.storage.ptr.buffer_id, x.storage.ptr.buffer_id, y.grad.storage.ptr.buffer_id],
	    			[out.grad.storage.ptr.offset, x.storage.ptr.offset, y.grad.storage.ptr.offset],
	    			[prod(out.shape)], 256, 1, 1, 256)
	    	return

	    def expand(shape, strides, target):
	    	pad = len(target) - len(shape)
	    	pshape = (1,) * pad + tuple(shape)
	    	pstrides = (0,) * pad + tuple(strides)
	    	return [s if p == t else 0 for s,p,t in zip(pstrides, pshape, target)]
	    sx = expand(x.shape, x.strides, out.shape)
	    sy = expand(y.shape, y.strides, out.shape)
	    shape = list(out.shape)
	    while len(shape) < 4:
	    	shape = [1] + shape
	    	sx = [0] + list(sx)
	    	sy = [0] + list(sy)
	    ndim = len(shape)
	    do_x = 1 if x.requires_grad else 0
	    do_y = 1 if y.requires_grad else 0
	    self.ctx.dispatch_with_args_3d('mul_backward_kernel',
	    	[out.grad.storage.ptr.buffer_id, x.storage.ptr.buffer_id, y.storage.ptr.buffer_id,
	    		x.grad.storage.ptr.buffer_id if x.requires_grad else out.grad.storage.ptr.buffer_id,
	    		y.grad.storage.ptr.buffer_id if y.requires_grad else out.grad.storage.ptr.buffer_id],
	    	[out.grad.storage.ptr.offset, x.storage.ptr.offset, y.storage.ptr.offset,
	    		x.grad.storage.ptr.offset if x.requires_grad else out.grad.storage.ptr.offset,
	    		y.grad.storage.ptr.offset if y.requires_grad else out.grad.storage.ptr.offset],
	    	[shape[0], shape[1], shape[2], shape[3], sx[0], sx[1], sx[2], sx[3],
	    		sy[0], sy[1], sy[2], sy[3], ndim, 1 if x.requires_grad else 0, 1 if y.requires_grad else 0],
	    	prod(out.shape), 1, 1, min(prod(out.shape), 256))

	def metal_contiguous_forward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		self.ctx.dispatch_contiguous("contiguous_forward_kernel", 
			x.storage.ptr.buffer_id, x.storage.ptr.offset,
			out.storage.ptr.buffer_id, out.storage.ptr.offset,
			list(x.shape), list(x.strides), prod(out.shape))

	def metal_contiguous_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None or not x.requires_grad: return
		self.ctx.dispatch_with_args_3d('contiguous_backward_kernel',
			[out.grad.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
			[out.grad.storage.ptr.offset, x.grad.storage.ptr.offset],
			[], prod(x.shape), 1, 1, min(prod(x.shape), 256))
		# ndim = len(x.shape)
		# contiguous_strides = [1] * ndim 
		# for i in range(ndim-2, -1, -1):
		# 	contiguous_strides[i] = contiguous_strides[i+1]*x.shape[i+1]
		# self.ctx.dispatch_strided('strided_add_backward_kernel',
		# 	out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset,
		# 	x.grad.storage.ptr.buffer_id, x.grad.storage.ptr.offset,
		# 	list(x.shape), list(x.strides), prod(out.shape))

	def metal_gather_forward(self, inputs, outputs, scratch=None):
		indices, weight = inputs[0], inputs[1]
		out = outputs[0]
		buffer_ids = [indices.storage.ptr.buffer_id, weight.storage.ptr.buffer_id, out.storage.ptr.buffer_id]
		offsets = [indices.storage.ptr.offset, weight.storage.ptr.offset, out.storage.ptr.offset]
		C = weight.shape[-1]
		# self.ctx.dispatch_with_args("gather_forward_kernel", buffer_ids, offsets, [C], prod(out.shape))
		self.ctx.dispatch_with_args_3d('gather_forward_kernel', buffer_ids, offsets, 
			[C], prod(out.shape), 1, 1, min(prod(out.shape), 256))

	def metal_gather_backward(self, inputs, outputs, scratch=None):
		indices, weight = inputs[0], inputs[1]
		out = outputs[0]
		if out.grad is None or not weight.requires_grad: return
		if not getattr(self, "accumulate_parameter_grads", False):
			self.ctx.dispatch_with_args_3d('fill_scalar_kernel',
				[weight.grad.storage.ptr.buffer_id], [weight.grad.storage.ptr.offset], 
				[0], prod(weight.grad.shape), 1, 1, min(prod(weight.grad.shape), 256))
		buffer_ids = [indices.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, weight.grad.storage.ptr.buffer_id]
		offsets = [indices.storage.ptr.offset, out.grad.storage.ptr.offset, weight.grad.storage.ptr.offset]
		C = weight.shape[-1]
		self.ctx.dispatch_with_args_3d('gather_backward_kernel', buffer_ids, offsets,
			[C], prod(out.shape), 1, 1, min(prod(out.shape), 256))

	def make_kernel(self, kernel_name):
		def _kernel(inputs, outputs, scratch=None):
			tensors = list(inputs) + list(outputs)
			buffer_ids = []
			offsets = []
			for t in tensors:
				buffer_ids.append(t.storage.ptr.buffer_id)
				offsets.append(t.storage.ptr.offset)
			grid_size = prod(outputs[0].shape)
			self.ctx.dispatch(kernel_name, buffer_ids, offsets, grid_size)
		return _kernel

	def metal_reshape_view_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None:
			return
		buffer_ids = [out.grad.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id]
		offsets = [out.grad.storage.ptr.offset, x.grad.storage.ptr.offset]
		grid_size = prod(x.shape)
		self.ctx.dispatch("add_backward_kernel", buffer_ids, offsets, grid_size)

	def metal_permute_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None or not x.requires_grad:
			return
		self.ctx.dispatch_strided("strided_add_backward_kernel", 
			out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset,
			x.grad.storage.ptr.buffer_id, x.grad.storage.ptr.offset,
			list(x.shape), list(x.strides), prod(x.shape))

	def metal_transpose_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None or not x.requires_grad:
			return
		ndim = len(x.shape)
		d0 = out.op.op.dim0 % ndim
		d1 = out.op.op.dim1 % ndim
		out_grad_strides = [1] * ndim 
		for i in range(ndim-2, -1, -1):
			out_grad_strides[i] = (out_grad_strides[i+1]*out.shape[i+1])
		input_to_output_strides = list(out_grad_strides)
		input_to_output_strides[d0], input_to_output_strides[d1] = input_to_output_strides[d1], input_to_output_strides[d0]
		self.ctx.dispatch_strided('strided_add_backward_kernel',
			out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset,
			x.grad.storage.ptr.buffer_id, x.grad.storage.ptr.offset,
			list(x.shape), input_to_output_strides, prod(x.shape))
		# self.ctx.dispatch_strided("strided_add_backward_kernel", 
		# 	out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset,
		# 	x.grad.storage.ptr.buffer_id, x.grad.storage.ptr.offset,
		# 	list(x.shape), list(x.strides), prod(x.shape))

	def metal_matmul_forward(self, inputs, outputs, scratch=None):
		x, w, out = inputs[0], inputs[1], outputs[0]
		if w.ndim == 2:
			M = prod(x.shape[:-1])
			K = x.shape[-1]
			N = w.shape[-1]
			self.ctx.dispatch_gemm_mps([x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
				[x.storage.ptr.offset, w.storage.ptr.offset, out.storage.ptr.offset],
				[1, M, N, K, 0, 0, 0, 0], 1.0, 0.0)
		else:
			batch_count = prod(x.shape[:-2]) if x.ndim > 2 else 1
			M = x.shape[-2]
			K = x.shape[-1]
			N = w.shape[-1]
			B, H, T, D = x.shape 
			self.ctx.dispatch_gemm_mps([x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
				[x.storage.ptr.offset, w.storage.ptr.offset, out.storage.ptr.offset],
				[batch_count, M, N, K, 0, 0, 1, 1], 1.0, 0.0)

	def metal_matmul_backward(self, inputs, outputs, scratch=None):
	    x, w, out = inputs[0], inputs[1], outputs[0]
	    if out.grad is None: return

	    M = prod(x.shape[:-1]) if w.ndim == 2 else x.shape[-2]
	    K = x.shape[-1]
	    N = w.shape[-1]
	    batch = 1 if w.ndim == 2 else (prod(x.shape[:-2]) if x.ndim > 2 else 1)
	    a_batched = 0 if w.ndim == 2 else 1
	    b_batched = 0 if w.ndim == 2 else 1
	    # dX = dO @ W^T
	    if x.requires_grad:
	    	self.ctx.dispatch_gemm_mps([out.grad.storage.ptr.buffer_id, w.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
	    		[out.grad.storage.ptr.offset, w.storage.ptr.offset, x.grad.storage.ptr.offset],
	    		[batch, M, K, N, 0, 1, a_batched, b_batched], 1.0, 0.0)
	    # dW = X^T @ dO
	    if w.requires_grad:
	    	# self.ctx.dispatch_with_args_3d('fill_scalar_kernel',
	    	# 	[w.grad.storage.ptr.buffer_id], [w.grad.storage.ptr.offset], [0], prod(w.shape), 1, 1, min(prod(w.shape), 256))
	    	self.ctx.dispatch_gemm_mps(
				[x.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
	    		[x.storage.ptr.offset, out.grad.storage.ptr.offset, w.grad.storage.ptr.offset],
	    		[batch, K, N, M, 1, 0, a_batched, b_batched], 1.0, 0.0)

	def metal_layernorm_forward(self, inputs, outputs, scratch=None):
		x, gamma, beta = inputs[0], inputs[1], inputs[2]
		out = outputs[0]
		C = x.shape[-1]
		rows = prod(x.shape[:-1])
		eps_bits = struct.unpack('<i', struct.pack('<f', out.op.op.eps))[0]
		self.ctx.dispatch_with_args_3d('layernorm_forward',
			[x.storage.ptr.buffer_id, gamma.storage.ptr.buffer_id, beta.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[x.storage.ptr.offset, gamma.storage.ptr.offset, beta.storage.ptr.offset, out.storage.ptr.offset],
			[C, eps_bits], rows*256, 1, 1, 256)

	def metal_layernorm_backward(self, inputs, outputs, scratch=None):
		x, gamma, beta = inputs[0], inputs[1], inputs[2]
		out = outputs[0]
		if out.grad is None:
			return
		
		C = x.shape[-1]
		B_T = prod(x.shape[:-1])

		self.ctx.dispatch_with_args_3d('layernorm_backward_kernel',
			[x.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, gamma.storage.ptr.buffer_id,
				x.grad.storage.ptr.buffer_id if x.requires_grad else out.grad.storage.ptr.buffer_id,
				gamma.grad.storage.ptr.buffer_id if gamma.requires_grad else out.grad.storage.ptr.buffer_id,
				beta.grad.storage.ptr.buffer_id if beta.requires_grad else out.grad.storage.ptr.buffer_id],
			[x.storage.ptr.offset, out.grad.storage.ptr.offset, gamma.storage.ptr.offset,
				x.grad.storage.ptr.offset if x.requires_grad else out.grad.storage.ptr.offset,
				gamma.grad.storage.ptr.offset if gamma.requires_grad else out.grad.storage.ptr.offset,
				beta.grad.storage.ptr.offset if beta.requires_grad else out.grad.storage.ptr.offset],
			[B_T, C], B_T*256, 1, 1, 256)

	def metal_gelu_forward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		self.ctx.dispatch_with_args_3d('gelu_forward_kernel',
			[x.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[x.storage.ptr.offset, out.storage.ptr.offset],
			[], prod(out.shape), 1, 1, min(prod(out.shape), 256))

	def metal_gelu_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None or not x.requires_grad: return
		self.ctx.dispatch_with_args_3d('gelu_backward_kernel',
			[out.grad.storage.ptr.buffer_id, x.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
			[out.grad.storage.ptr.offset, x.storage.ptr.offset, x.grad.storage.ptr.offset],
			[], prod(x.shape), 1, 1, min(prod(x.shape), 256))

	def metal_softmax_forward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		buffer_ids = [x.storage.ptr.buffer_id, out.storage.ptr.buffer_id]
		offsets = [x.storage.ptr.offset, out.storage.ptr.offset]
		C = x.shape[-1]
		self.ctx.dispatch_softmax("softmax_forward_kernel", buffer_ids, offsets, C, prod(x.shape[:-1]))

	def metal_softmax_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None or not x.requires_grad: return
		buffer_ids = [out.grad.storage.ptr.buffer_id, out.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id]
		offsets = [out.grad.storage.ptr.offset, out.storage.ptr.offset, x.grad.storage.ptr.offset]
		C = x.shape[-1]
		self.ctx.dispatch_softmax("softmax_backward_kernel", buffer_ids, offsets, C, prod(x.shape[:-1]))

	def metal_crossentropy_forward(self, inputs, outputs, scratch=None):
		logits, labels, out = inputs[0], inputs[1], outputs[0]
		buffer_ids = [logits.storage.ptr.buffer_id, labels.storage.ptr.buffer_id, out.storage.ptr.buffer_id]
		offsets = [logits.storage.ptr.offset, labels.storage.ptr.offset, out.storage.ptr.offset]
		V = logits.shape[-1]
		self.ctx.dispatch_with_args("crossentropy_forward_kernel", buffer_ids, offsets, [V], prod(labels.shape) * 1024)

	def metal_crossentropy_backward(self, inputs, outputs, scratch=None):
		logits, labels, out = inputs[0], inputs[1], outputs[0]
		if out.grad is None or not logits.requires_grad: return
		buffer_ids = [out.grad.storage.ptr.buffer_id, logits.storage.ptr.buffer_id, labels.storage.ptr.buffer_id, logits.grad.storage.ptr.buffer_id]
		offsets = [out.grad.storage.ptr.offset, logits.storage.ptr.offset, labels.storage.ptr.offset, logits.grad.storage.ptr.offset]
		V = logits.shape[-1]
		self.ctx.dispatch_with_args("crossentropy_backward_kernel", buffer_ids, offsets, [V], prod(labels.shape) * 1024)

	def metal_exp_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None or not x.requires_grad: return
		buffer_ids = [out.grad.storage.ptr.buffer_id, out.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id]
		offsets = [out.grad.storage.ptr.offset, out.storage.ptr.offset, x.grad.storage.ptr.offset]
		self.ctx.dispatch("exp_backward_kernel", buffer_ids, offsets, prod(x.shape))

	def metal_log_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None or not x.requires_grad: return
		buffer_ids = [out.grad.storage.ptr.buffer_id, x.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id]
		offsets = [out.grad.storage.ptr.offset, x.storage.ptr.offset, x.grad.storage.ptr.offset]
		self.ctx.dispatch("log_backward_kernel", buffer_ids, offsets, prod(x.shape))

	def metal_rsqrt_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None or not x.requires_grad: return
		buffer_ids = [out.grad.storage.ptr.buffer_id, out.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id]
		offsets = [out.grad.storage.ptr.offset, out.storage.ptr.offset, x.grad.storage.ptr.offset]
		self.ctx.dispatch("rsqrt_backward_kernel", buffer_ids, offsets, prod(x.shape))

	def metal_sdpa_forward(self, inputs, outputs, scratch=None):
		q, k, v = inputs[0], inputs[1], inputs[2]
		out = outputs[0]
		is_causal = out.op.op.is_causal
		B, H, T, D = q.shape
		buffer_ids = [q.storage.ptr.buffer_id, k.storage.ptr.buffer_id, v.storage.ptr.buffer_id, out.storage.ptr.buffer_id]
		offsets = [q.storage.ptr.offset, k.storage.ptr.offset, v.storage.ptr.offset, out.storage.ptr.offset]
		if D == 64:
			kernel_name = 'sdpa_forward_simd_d64_kernel'
		elif D == 128:
			kernel_name = 'sdpa_forward_simd_d128_kernel'
		else:
			kernel_name = 'sdpa_forward_simd_other_kernel'
		self.ctx.dispatch_sdpa(kernel_name, buffer_ids, offsets, [B, H, T, int(is_causal)], B*H*T)

	def metal_sdpa_backward(self, inputs, outputs, scratch=None):
		q, k, v = inputs[0], inputs[1], inputs[2]
		out = outputs[0]
		is_causal = out.op.op.is_causal
		if out.grad is None: return
		B, H, T, D = q.shape
		buffer_ids = [
			q.storage.ptr.buffer_id, 
			k.storage.ptr.buffer_id, 
			v.storage.ptr.buffer_id, 
			out.storage.ptr.buffer_id,
			out.grad.storage.ptr.buffer_id,
			q.grad.storage.ptr.buffer_id if q.requires_grad else 0,
			k.grad.storage.ptr.buffer_id if k.requires_grad else 0,
			v.grad.storage.ptr.buffer_id if v.requires_grad else 0
		]
		offsets = [
			q.storage.ptr.offset, 
			k.storage.ptr.offset, 
			v.storage.ptr.offset, 
			out.storage.ptr.offset,
			out.grad.storage.ptr.offset,
			q.grad.storage.ptr.offset if q.requires_grad else 0,
			k.grad.storage.ptr.offset if k.requires_grad else 0,
			v.grad.storage.ptr.offset if v.requires_grad else 0
		]
		self.ctx.dispatch_sdpa("sdpa_backward_simd_d64_kernel", buffer_ids, offsets, [B, H, T, D, int(is_causal)], B*H*T)

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

	def metal_fused_mlp_forward(self, inputs, outputs, scratch=None):
		x, w1, b1, w2, b2 = inputs
		out = outputs[0]

		N = prod(x.shape[:-1])
		C = x.shape[-1]
		C4 = w1.shape[-1]

		buffer_ids = [
			x.storage.ptr.buffer_id, w1.storage.ptr.buffer_id, b1.storage.ptr.buffer_id,
			w2.storage.ptr.buffer_id, b2.storage.ptr.buffer_id, out.storage.ptr.buffer_id
		]
		offsets = [
			x.storage.ptr.offset, w1.storage.ptr.offset, b1.storage.ptr.offset,
			w2.storage.ptr.offset, b2.storage.ptr.offset, out.storage.ptr.offset
		]
		grid_size = (N + 63) // 64
		self.ctx.dispatch_sdpa("fused_mlp_forward_kernel", buffer_ids, offsets, [N, C, C4], N)
		# self.ctx.dispatch_flash_attn_bwd("fused_mlp_forward_kernel", buffer_ids, offsets, [N, C, C4], grid_size)

	def metal_fused_mlp_backward(self, inputs, outputs, scratch=None):
		x, w1, b1, w2, b2 = inputs
		out = outputs[0]
		grad_out = out.grad
		if grad_out is None: return

		N = prod(x.shape[:-1])
		C = x.shape[-1]
		C4 = w1.shape[-1]

		buffer_ids = [
			x.storage.ptr.buffer_id, w1.storage.ptr.buffer_id, b1.storage.ptr.buffer_id,
			w2.storage.ptr.buffer_id, grad_out.storage.ptr.buffer_id,
			x.grad.storage.ptr.buffer_id if x.requires_grad else 0,
			w1.grad.storage.ptr.buffer_id if w1.requires_grad else 0,
			b1.grad.storage.ptr.buffer_id if b1.requires_grad else 0,
			w2.grad.storage.ptr.buffer_id if w2.requires_grad else 0,
			b2.grad.storage.ptr.buffer_id if b2.requires_grad else 0]
		offsets = [
			x.storage.ptr.offset, w1.storage.ptr.offset, b1.storage.ptr.offset,
			w2.storage.ptr.offset, grad_out.storage.ptr.offset,
			x.grad.storage.ptr.offset if x.requires_grad else 0,
			w1.grad.storage.ptr.offset if w1.requires_grad else 0,
			b1.grad.storage.ptr.offset if b1.requires_grad else 0,
			w2.grad.storage.ptr.offset if w2.requires_grad else 0,
			b2.grad.storage.ptr.offset if b2.requires_grad else 0]
		blocks = (N + 31) // 32
		self.ctx.dispatch_flash_attn_bwd("fused_mlp_backward_kernel", buffer_ids, offsets, [N, C, C4], blocks)

	def metal_fused_qkv_proj_forward(self, inputs, outputs, scratch=None):
		x, wq, bq, wk, bk, wv, bv = inputs
		q_out, k_out, v_out = outputs
		B, T, C = x.shape 
		H, D = q_out.shape[1], q_out.shape[3]
		N = B * T 

		buffer_ids = [
			x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
			wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
			q_out.storage.ptr.buffer_id, k_out.storage.ptr.buffer_id, v_out.storage.ptr.buffer_id
		]
		offsets = [
			x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
			wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
			q_out.storage.ptr.offset, k_out.storage.ptr.offset, v_out.storage.ptr.offset
		]
		grid_size = (N + 3) // 4 # BM = 4
		self.ctx.dispatch_sdpa("fused_qkv_forward_kernel", buffer_ids, offsets, [N, C, H, D, T], grid_size)

	def metal_fused_qkv_proj_backward(self, inputs, outputs, scratch=None):
		x, wq, bq, wk, bk, wv, bv = inputs
		q_out, k_out, v_out = outputs
		dq, dk, dv = q_out.grad, k_out.grad, v_out.grad
		if dq is None: return

		B, T, C = x.shape
		H, D = dq.shape[1], dq.shape[3]
		N = B * T 
		buffer_ids = [
			x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, wk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id,
			dq.storage.ptr.buffer_id, dk.storage.ptr.buffer_id, dv.storage.ptr.buffer_id,
			x.grad.storage.ptr.buffer_id if x.requires_grad else 0,
			wq.grad.storage.ptr.buffer_id if wq.requires_grad else 0,
			bq.grad.storage.ptr.buffer_id if bq.requires_grad else 0,
			wk.grad.storage.ptr.buffer_id if wk.requires_grad else 0,
			bk.grad.storage.ptr.buffer_id if bk.requires_grad else 0,
			wv.grad.storage.ptr.buffer_id if wv.requires_grad else 0,
			bv.grad.storage.ptr.buffer_id if bv.requires_grad else 0
		]
		offsets = [
			x.storage.ptr.offset, wq.storage.ptr.offset, wk.storage.ptr.offset, wv.storage.ptr.offset,
			dq.storage.ptr.offset, dk.storage.ptr.offset, dv.storage.ptr.offset, 
			x.grad.storage.ptr.offset if x.requires_grad else 0,
			wq.grad.storage.ptr.offset if wq.requires_grad else 0,
			bq.grad.storage.ptr.offset if bq.requires_grad else 0,
			wk.grad.storage.ptr.offset if wk.requires_grad else 0,
			bk.grad.storage.ptr.offset if bk.requires_grad else 0,
			wv.grad.storage.ptr.offset if wv.requires_grad else 0,
			bv.grad.storage.ptr.offset if bv.requires_grad else 0
		]
		grid_size = (N + 3) // 4
		self.ctx.dispatch_sdpa("fused_qkv_backward_kernel", buffer_ids, offsets, [N, C, H, D, T], grid_size)

	def metal_attn_out_contiguous_forward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		B, H, T, D = x.shape 
		buffer_ids = [x.storage.ptr.buffer_id, out.storage.ptr.buffer_id]
		offsets = [x.storage.ptr.offset, out.storage.ptr.offset]
		grid_size = (B * H * T * D) // 4
		self.ctx.dispatch_with_args("attn_out_contiguous_forward_kernel", buffer_ids, offsets, [B, H, T, D], grid_size)

	def metal_attn_out_contiguous_backward(self, inputs, outputs, scratch=None):
		x, out = inputs[0], outputs[0]
		if out.grad is None or not x.requires_grad: return
		B, H, T, D = x.shape
		buffer_ids = [out.grad.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id]
		offsets = [out.grad.storage.ptr.offset, x.grad.storage.ptr.offset]
		grid_size = (B * H * T * D) // 4
		self.ctx.dispatch_with_args("attn_out_contiguous_backward_kernel", buffer_ids, offsets, [B, H, T, D], grid_size)

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
	    self.ctx.dispatch_with_args_3d(
	        "swiglu_forward_kernel",
	        [
	            gu.ptr.buffer_id,
	            hidden.ptr.buffer_id,
	        ],
	        [
	            gu.ptr.offset,
	            hidden.ptr.offset,
	        ],
	        [N, H],
	        N*(H+3)//4,
	        1,
	        1,
	        min(N*(H+3)//4 , 256),
	    )

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

	def metal_flash_attention_forward(self, inputs, outputs, scratch=None):
		x, wq, bq, wk, bk, wv, bv = inputs
		out = outputs[0]
		B, T, C = x.shape
		# H, D = out.shape[1], out.shape[3]
		H = out.op.op.n_head
		D = C // H
		is_causal = out.op.op.is_causal

		k = scratch['k']
		v = scratch['v']
		self.ctx.dispatch_sdpa('fa_min_project_kv_d64_kernel',
			[x.storage.ptr.buffer_id, wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id,
			wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[x.storage.ptr.offset, wk.storage.ptr.offset, bk.storage.ptr.offset, 
			wv.storage.ptr.offset, bv.storage.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, T, C, H], B*H*T)
		blocks = ((T + 32 - 1) // 32) * B * H
		self.ctx.dispatch_flash_attn_bwd('fa_pure_fwd_d64_kernel',
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
			k.ptr.buffer_id, v.ptr.buffer_id, out.storage.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
			k.ptr.offset, v.ptr.offset, out.storage.ptr.offset],
			[B, H, T, C, int(is_causal)], blocks)

	def metal_flash_attention_backward(self, inputs, outputs, scratch=None):
		x, wq, bq, wk, bk, wv, bv = inputs
		out = outputs[0]
		if out.grad is None: return

		B, T, C = x.shape
		# H, D = out.shape[1], out.shape[3]
		H = out.op.op.n_head
		D = C // H
		is_causal = out.op.op.is_causal

		k = scratch['k']
		v = scratch['v']
		lse = scratch['lse']
		d_val = scratch['d_val'] 
		q = scratch['q']
		dq = scratch['dq']
		dk = scratch['dk']
		dv = scratch['dv']

		# Step 1: project K/V/Q into scratch
		self.ctx.dispatch_sdpa('fa_min_project_kv_d64_kernel',
			[x.storage.ptr.buffer_id, wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id,
			wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
			[x.storage.ptr.offset, wk.storage.ptr.offset, bk.storage.ptr.offset,
			wv.storage.ptr.offset, bv.storage.ptr.offset, k.ptr.offset, v.ptr.offset],
			[B, T, C, H], B*H*T)
		self.ctx.dispatch_sdpa('fa_min_recompute_q_d64_kernel',
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id, q.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset, q.ptr.offset],
			[B, T, C, H], B * H * T)

		# Step 2: dQ
		q_blocks = ((T + 32 - 1) // 32) * B * H
		self.ctx.dispatch_with_args("fa_bwd_calc_d_kernel",
			[out.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, d_val.ptr.buffer_id],
			[out.storage.ptr.offset, out.grad.storage.ptr.offset, d_val.ptr.offset],
			[B, H, T, C], B*H*T)
		self.ctx.dispatch_flash_attn('fa_min_recompute_lse_d64_kernel',
			[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id, k.ptr.buffer_id, lse.ptr.buffer_id],
			[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset, k.ptr.offset, lse.ptr.offset],
			[B, H, T, C, int(is_causal)], q_blocks)

		self.ctx.dispatch_flash_attn_bwd('fa_pure_bwd_dq_d64_kernel',
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, 
				out.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id,
				dq.ptr.buffer_id, lse.ptr.buffer_id, d_val.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, 
				out.storage.ptr.offset, out.grad.storage.ptr.offset,
				dq.ptr.offset, lse.ptr.offset, d_val.ptr.offset],
			[B, H, T, C, int(is_causal)], q_blocks)

		# Step 3: dK/dV
		k_blocks = ((T + 32 - 1) // 32) * B * H
		self.ctx.dispatch_flash_attn_bwd('fa_pure_bwd_dkv_d64_kernel',
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, 
			 out.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id,
             dk.ptr.buffer_id, dv.ptr.buffer_id, lse.ptr.buffer_id, d_val.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, 
			 out.storage.ptr.offset, out.grad.storage.ptr.offset,
             dk.ptr.offset, dv.ptr.offset, lse.ptr.offset, d_val.ptr.offset],
			[B, H, T, C, int(is_causal)], k_blocks)

		if x.requires_grad:
			M_2d = B * T 
			K_2d = H * D
			N_2d = C
			self.fill_storage(q, 0)
			self.fill_storage(k, 0)
			self.fill_storage(v, 0)
			# dx_q = dQ * Wq, reuse scratch q
			self.ctx.gemm_backward_dx_batched_mps(dq.ptr.buffer_id, dq.ptr.offset,
				wq.storage.ptr.buffer_id, wq.storage.ptr.offset,
				q.ptr.buffer_id, q.ptr.offset, 1, M_2d, K_2d, N_2d, 0)
			# dx_k = dK * Wk, reuse scratch k
			self.ctx.gemm_backward_dx_batched_mps(dk.ptr.buffer_id, dk.ptr.offset,
				wk.storage.ptr.buffer_id, wk.storage.ptr.offset,
				k.ptr.buffer_id, k.ptr.offset, 1, M_2d, K_2d, N_2d, 0)
			# dx_v = dV * Wv, reuse scratch v
			self.ctx.gemm_backward_dx_batched_mps(dv.ptr.buffer_id, dv.ptr.offset,
				wv.storage.ptr.buffer_id, wv.storage.ptr.offset,
				v.ptr.buffer_id, v.ptr.offset, 1, M_2d, K_2d, N_2d, 0)

			self.ctx.dispatch_with_args('fa_min_accumulate_dx_kernel',
				[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, x.grad.storage.ptr.buffer_id if x.requires_grad else 0],
				[q.ptr.offset, k.ptr.offset, v.ptr.offset, x.grad.storage.ptr.offset if x.requires_grad else 0],
				[B, H, T, C], B * T * C)

		if wq.requires_grad: # dWq = X*dQ
			self.ctx.gemm_backward_dw_batched_mps(x.storage.ptr.buffer_id, x.storage.ptr.offset,
				dq.ptr.buffer_id, dq.ptr.offset,
				wq.grad.storage.ptr.buffer_id if wq.requires_grad else 0, 
				wq.grad.storage.ptr.offset if wq.requires_grad else 0,
				1, B*T, H*D, C, 0)

		if wk.requires_grad: # dWk = X*dK
			self.ctx.gemm_backward_dw_batched_mps(x.storage.ptr.buffer_id, x.storage.ptr.offset,
				dk.ptr.buffer_id, dk.ptr.offset,
				wk.grad.storage.ptr.buffer_id if wk.requires_grad else 0, 
				wk.grad.storage.ptr.offset if wk.requires_grad else 0,
				1, B*T, H*D, C, 0)

		if wv.requires_grad: # dWv = X*dV
			self.ctx.gemm_backward_dw_batched_mps(x.storage.ptr.buffer_id, x.storage.ptr.offset,
				dv.ptr.buffer_id, dv.ptr.offset,
				wv.grad.storage.ptr.buffer_id if wv.requires_grad else 0, 
				wv.grad.storage.ptr.offset if wv.requires_grad else 0,
				1, B*T, H*D, C, 0)

		if bq.requires_grad:
			self.ctx.dispatch_with_args('bias_grad_reduce', [dq.ptr.buffer_id, bq.grad.storage.ptr.buffer_id],
				[dq.ptr.offset, bq.grad.storage.ptr.offset], [B*T, H*D, int(bool(getattr(self, "accumulate_parameter_grads", False)))], H*D)
		if bk.requires_grad:
			self.ctx.dispatch_with_args('bias_grad_reduce', [dk.ptr.buffer_id, bk.grad.storage.ptr.buffer_id],
				[dk.ptr.offset, bk.grad.storage.ptr.offset], [B*T, H*D, int(bool(getattr(self, "accumulate_parameter_grads", False)))], H*D)
		if bv.requires_grad:
			self.ctx.dispatch_with_args('bias_grad_reduce', [dv.ptr.buffer_id, bv.grad.storage.ptr.buffer_id],
				[dv.ptr.offset, bv.grad.storage.ptr.offset], [B*T, H*D, int(bool(getattr(self, "accumulate_parameter_grads", False)))], H*D)

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

	def metal_recompute_fused_rms_fa(self, act, scratch, out_buf):
		x, wq, bq, wk, bk, wv, bv = act.op.inputs
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
	        	# [prod(x.shape[:-1]), N_dim, accumulate], N_dim, 1, 1, min(N_dim, 256))
	        	[prod(x.shape[:-1]), N_dim, accumulate], 256, (N_dim+15)//16, 1, 256)

	def get_forward_kernel(self, op):
		op_type = type(op)
		if op_type not in self.forward_kernels:
			return None
			# raise RuntimeError(f"forward kernel not found: {op_type}")
		return self.forward_kernels[op_type]

	def get_backward_kernel(self, op):
		op_type = type(op)
		if op_type not in self.backward_kernels:
			# raise RuntimeError(f"backward kernel not found: {op_type}")
			return None
		return self.backward_kernels[op_type]

	def bind_matmul_forward(self, node):
		return self.metal_matmul_forward

	def bind_matmul_backward(self, node, arena):
		return self.metal_matmul_backward

	def bind_add_forward(self, node):
		x, y = node.inputs[0], node.inputs[1]
		out = node.outputs[0]
		grid_size = prod(out.shape)
		if x.shape == y.shape:
			def _fwd(inputs, outputs, scratch=None):
				if grid_size % 4 == 0 and \
				   inputs[0].storage.ptr.offset % 16 == 0 and \
				   inputs[1].storage.ptr.offset % 16 == 0 and \
				   outputs[0].storage.ptr.offset % 16 == 0:
					self.ctx.dispatch_with_args_3d('add_residual_forward_vec4',
						[x.storage.ptr.buffer_id, y.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
						[x.storage.ptr.offset, y.storage.ptr.offset, out.storage.ptr.offset],
						[], grid_size//4, 1, 1, min(grid_size//4, 256))
				else:
					self.ctx.dispatch_with_args_3d('add_residual_forward',
						[x.storage.ptr.buffer_id, y.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
						[x.storage.ptr.offset, y.storage.ptr.offset, out.storage.ptr.offset],
						[], grid_size, 1, 1, min(grid_size, 256))
			return _fwd
		# [B, T, C] + [T, C] or [B, n_head, T, T] + [1, 1, T, T]
		elif len(y.shape) == 2 or (len(y.shape) == 3 and y.shape[0] == 1) or (len(y.shape) == 4 and y.shape[0] == 1 and y.shape[1] == 1):
			T_val, C_val = out.shape[-2], out.shape[-1]
			def _fwd(inputs, outputs, scratch=None):
				self.ctx.dispatch_with_args_3d('add_pos_forward',
					[x.storage.ptr.buffer_id, y.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
					[x.storage.ptr.offset, y.storage.ptr.offset, out.storage.ptr.offset],
					[T_val, C_val], grid_size, 1, 1, min(grid_size, 256))
			return _fwd
		# [B, T, C] + [C]
		elif len(y.shape) == 1 or (len(y.shape) == 3 and y.shape[0] == 1 and y.shape[1] == 1) or (len(y.shape) == 4 and y.shape[0] == 1 and y.shape[1] == 1 and y.shape[2] == 1):
			C_val = out.shape[-1]
			def _fwd(inputs, outputs, scratch=None):
				self.ctx.dispatch_with_args_3d('add_bias_forward',
					[x.storage.ptr.buffer_id, y.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
					[x.storage.ptr.offset, y.storage.ptr.offset, out.storage.ptr.offset],
					[C_val], grid_size, 1, 1, min(grid_size, 256))
			return _fwd
		else:
			raise RuntimeError(f"Unsupported Add shape: x.shape={x.shape}, y.shape={y.shape}")

	def bind_add_backward(self, node):
		x, y = node.inputs[0], node.inputs[1]
		out = node.outputs[0]
		grid_size = prod(out.shape)
		if x.shape == y.shape:
			def _bwd(inputs, outputs, scratch=None):
				if outputs[0].grad is None: return

				do_x = inputs[0].requires_grad
				do_y = inputs[1].requires_grad
				if do_x and inputs[0].grad.storage.ptr.buffer_id == outputs[0].grad.storage.ptr.buffer_id \
					and inputs[0].grad.storage.ptr.offset == outputs[0].grad.storage.ptr.offset:
					do_x = False 
				if do_y and inputs[1].grad.storage.ptr.buffer_id == outputs[0].grad.storage.ptr.buffer_id \
					and inputs[1].grad.storage.ptr.offset == outputs[0].grad.storage.ptr.offset:
					do_y = False
				if not do_x and not do_y: return 

				buf_dx = inputs[0].grad.storage.ptr.buffer_id if do_x else 0
				off_dx = inputs[0].grad.storage.ptr.offset if do_x else 0
				buf_dy = inputs[1].grad.storage.ptr.buffer_id if do_y else 0
				off_dy = inputs[1].grad.storage.ptr.offset if do_y else 0

				if grid_size % 4 == 0 and \
				   outputs[0].grad.storage.ptr.offset % 16 == 0 and \
				   (off_dx % 16 == 0 if inputs[0].requires_grad else True) and \
				   (off_dy % 16 == 0 if inputs[1].requires_grad else True):
					self.ctx.dispatch_with_args_3d('add_residual_backward_vec4',
						[out.grad.storage.ptr.buffer_id, buf_dx, buf_dy],
						[out.grad.storage.ptr.offset, off_dx, off_dy],
						[], grid_size//4, 1, 1, min(grid_size//4, 256))
				else:
					self.ctx.dispatch_with_args_3d('add_residual_backward',
						[out.grad.storage.ptr.buffer_id, buf_dx, buf_dy],
						[out.grad.storage.ptr.offset, off_dx, off_dy],
						[], grid_size, 1, 1, min(grid_size, 256))
			return _bwd
		elif len(y.shape) == 2 or (len(y.shape) == 3 and y.shape[0] == 1) or (len(y.shape) == 4 and y.shape[0] == 1 and y.shape[1] == 1):
			B_val, T_val, C_val = prod(out.shape[:-2]), out.shape[-2], out.shape[-1]
			def _bwd(inputs, outputs, scratch=None):
				if outputs[0].grad is None: return
				if inputs[0].requires_grad:
					self.ctx.dispatch_with_args_3d('contiguous_backward_kernel',
						[out.grad.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
						[out.grad.storage.ptr.offset, x.grad.storage.ptr.offset],
						[], grid_size, 1, 1, min(grid_size, 256))
				if inputs[1].requires_grad:
					self.ctx.dispatch_with_args_3d('pos_emb_grad',
						[out.grad.storage.ptr.buffer_id, y.grad.storage.ptr.buffer_id],
						[out.grad.storage.ptr.offset, y.grad.storage.ptr.offset],
						[B_val, T_val, C_val], T_val*C_val, 1, 1, min(T_val*C_val, 256))
			return _bwd
		elif len(y.shape) == 1 or (len(y.shape) == 3 and y.shape[0] == 1 and y.shape[1] == 1) or (len(y.shape) == 4 and y.shape[0] == 1 and y.shape[1] == 1 and y.shape[2] == 1):
			rows_val, C_val = prod(out.shape[:-1]), out.shape[-1]
			def _bwd(inputs, outputs, scratch=None):
				if outputs[0].grad is None: return
				if inputs[0].requires_grad:
					self.ctx.dispatch_with_args_3d('contiguous_backward_kernel',
						[out.grad.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
						[out.grad.storage.ptr.offset, x.grad.storage.ptr.offset],
						[], grid_size, 1, 1, min(grid_size, 256))
				if inputs[1].requires_grad:
					self.ctx.dispatch_with_args_3d('bias_grad_reduce',
						[out.grad.storage.ptr.buffer_id, y.grad.storage.ptr.buffer_id],
						[out.grad.storage.ptr.offset, y.grad.storage.ptr.offset],
						# [rows_val, C_val, int(bool(getattr(self, "accumulate_parameter_grads", False)))], C_val, 1, 1, min(C_val, 256))
						[rows_val, C_val, int(bool(getattr(self, "accumulate_parameter_grads", False)))], 256, (C_val+15)//16, 1, 256)
			return _bwd
		else:
			raise RuntimeError(f"Unsupported Add shape: x.shape={x.shape}, y.shape={y.shape}")

	def fill_storage(self, storage, value):
		buffer_id = storage.ptr.buffer_id
		offset = storage.ptr.offset 
		grid_size = prod(storage.shape)
		self.ctx.fill_scalar("fill_scalar_kernel", [buffer_id], [offset], float(value), grid_size)
		# self.ctx.dispatch_with_args_3d('fill_scalar_kernel', [buffer_id], [offset], [f32_bits(value)], grid_size, 1, 1, min(grid_size, 256))

	def numpy_view(self, tensor):
		buffer_id = tensor.storage.ptr.buffer_id
		offset = tensor.storage.ptr.offset
		view_func = self.numpy_view_funcs.get(np.dtype(tensor.dtype))
		if view_func is None:
			raise RuntimeError(f"Unsupported dtype from numpy_view: {tensor.dtype}")
		return view_func(buffer_id, offset, tuple(tensor.shape))

	def copy_from_host(self, tensor, host_np):
		view = self.numpy_view(tensor)
		view[:] = host_np

	def copy_to_host(self, tensor):
		buffer_id = tensor.storage.ptr.buffer_id
		offset = tensor.storage.ptr.offset
		shape = tensor.shape 
		return self.ctx.copy_to_host(buffer_id, offset, shape)

	def adamw_step(self, p_storage, g_storage, m_storage, v_storage, tmp_storage,
		lr, lr_t, b1, b2, eps, weight_decay, numel=None, decay_numel=None):
		storages = [p_storage, g_storage, m_storage, v_storage]
		buffer_ids = [s.ptr.buffer_id for s in storages]
		offsets = [s.ptr.offset for s in storages]
		decay_factor = 1.0 - float(lr) * float(weight_decay)
		if numel is None:
			numel = prod(p_storage.shape)
		if decay_numel is None:
			decay_numel = numel if weight_decay != 0.0 else 0
		numel = int(numel)
		decay_numel = int(decay_numel)
		self.ctx.dispatch_with_args_3d('adamw_step',
			[p_storage.ptr.buffer_id, g_storage.ptr.buffer_id, m_storage.ptr.buffer_id, v_storage.ptr.buffer_id],
			[p_storage.ptr.offset, g_storage.ptr.offset, m_storage.ptr.offset, v_storage.ptr.offset],
			[f32_bits(decay_factor), f32_bits(lr_t), f32_bits(b1), f32_bits(b2), f32_bits(eps), numel, decay_numel],
			(numel + 3)// 4, 1, 1, min((numel+3)//4, 256))

	def adamw8bit_step(self, p, g, qm, qv, absmax_m, absmax_v, qmap_m, qmap_v,
			lr, lr_t, b1, b2, eps, weight_decay, numel, decay_numel, first_step):
		self.ctx.dispatch_with_args_3d("adamw8bit_step",
			[p.ptr.buffer_id, g.ptr.buffer_id, qm.ptr.buffer_id, qv.ptr.buffer_id, absmax_m.ptr.buffer_id, absmax_v.ptr.buffer_id,
				qmap_m.ptr.buffer_id, qmap_v.ptr.buffer_id],
			[p.ptr.offset, g.ptr.offset, qm.ptr.offset, qv.ptr.offset, absmax_m.ptr.offset, absmax_v.ptr.offset,
				qmap_m.ptr.offset, qmap_v.ptr.offset],
			[f32_bits(1.0 - float(lr)*float(weight_decay)), f32_bits(lr_t), f32_bits(b1), f32_bits(b2), f32_bits(eps), 
				int(numel), int(decay_numel), int(first_step)],
				((numel + 255)//256)*256, 1, 1, 256)

	def synchronize(self):
		self.ctx.synchronize()

	def begin_pass(self):
		self.ctx.begin_pass()

	def end_pass(self):
		self.ctx.end_pass()