from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from tensor import Tensor
from utils import broadcast_shape, resolve_reshape_shape, infer_reshape_view
from math import prod
import numpy as np

class Op(ABC):
	op_name = "Op"
	recompute_output = False

	def __call__(self, *inputs):
		shape = self.infer_shape(*inputs)
		dtype = self.infer_dtype(*inputs)
		requires_grad = any(x.requires_grad for x in inputs)
		out = Tensor(shape=shape, dtype=dtype, requires_grad=requires_grad, device=inputs[0].device)
		node = OpNode(op=self, inputs=tuple(inputs), outputs=(out,))
		out.op = node 
		return out

	@abstractmethod
	def infer_shape(self, *inputs):
		pass

	def infer_dtype(self, *inputs):
		return inputs[0].dtype

	def __repr__(self):
		return self.op_name

	def save_for_backward(self, inputs, outputs):
		return list(inputs) + list(outputs)

	def is_recompute_output(self):
		return self.recompute_output

class OpNode:
	__slots__ = ("op", "inputs", "outputs", "forward", "backward", 
		"scratch_fwd_layout", "scratch_bwd_layout",
		"name")
	def __init__(self, op, inputs, outputs, name=""):
		self.op = op
		self.inputs = inputs
		self.outputs = outputs
		self.forward = None
		self.backward = None
		self.scratch_fwd_layout = None
		self.scratch_bwd_layout = None
		self.name = name

class AddOp(Op):
	op_name = "Add"

	def infer_shape(self, x, y):
		return broadcast_shape(x.shape, y.shape)

	def save_for_backward(self, inputs, outputs):
		return []

class MulOp(Op):
	op_name = "Mul"

	def infer_shape(self, x, y):
		return broadcast_shape(x.shape, y.shape)

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class MatMulOp(Op):
	op_name = "MatMul"

	def infer_shape(self, x, y):
		if x.ndim < 2 or y.ndim < 2:
			raise ValueError(f"{self.op_name} requires ndims >=2")

		if x.shape[-1] != y.shape[-2]:
			raise ValueError(f"{self.op_name} dimension dismatch")

		batch_x = x.shape[:-2]
		batch_y = y.shape[:-2]
		batch = broadcast_shape(batch_x, batch_y)

		return (batch + (x.shape[-2], y.shape[-1]))

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class ViewOp(Op):
	op_name = "View"

	def __init__(self, shape):
		self.shape = tuple(shape)

	def infer_shape(self, x):
		return resolve_reshape_shape(x.numel, self.shape)

	def __call__(self, x):
		if not x.is_contiguous:
			raise RuntimeError("view requires contiguous")

		shape, strides, offset, is_contiguous = infer_reshape_view(x, self.shape)
		out = Tensor(shape=shape, dtype=x.dtype, strides=strides, offset=offset, is_contiguous=is_contiguous,
			requires_grad=x.requires_grad, device=x.device)
		out.op = OpNode(op=self, inputs=(x,), outputs=(out,),)
		return out

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class ReshapeOp(Op):
	op_name = "Reshape"

	def __init__(self, shape):
		self.shape = tuple(shape)

	def infer_shape(self, x):
		return resolve_reshape_shape(x.numel, self.shape)

	def __call__(self, x):
		shape, strides, offset, is_contiguous = infer_reshape_view(x, self.shape)
		out = Tensor(shape=shape, dtype=x.dtype, strides=strides, offset=offset, is_contiguous=is_contiguous, 
			requires_grad=x.requires_grad, device=x.device)
		out.op = OpNode(op=self, inputs=(x,), outputs=(out,),)
		return out

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class PermuteOp(Op):
	op_name = "Permute"

	def __init__(self, dims):
		self.dims = tuple(dims)

	def infer_shape(self, x):
		return tuple(x.shape[d] for d in self.dims)

	def __call__(self, x):
		shape = tuple(x.shape[d] for d in self.dims)
		strides = tuple(x.strides[d] for d in self.dims)
		out = Tensor(shape=shape, dtype=x.dtype, strides=strides, offset=x.offset, is_contiguous=False, requires_grad=x.requires_grad, device=x.device)
		out.op = OpNode(op=self, inputs=(x,), outputs=(out,),)
		return out

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class TransposeOp(Op):
	op_name = "Transpose"

	def __init__(self, dim0, dim1):
		self.dim0 = dim0
		self.dim1 = dim1 

	def infer_shape(self, x):
		shape = list(x.shape)
		shape[self.dim0], shape[self.dim1] = shape[self.dim1], shape[self.dim0]
		return tuple(shape)

	def __call__(self, x):
		shape = list(x.shape)
		strides = list(x.strides)
		shape[self.dim0], shape[self.dim1] = shape[self.dim1], shape[self.dim0]
		strides[self.dim0], strides[self.dim1] = strides[self.dim1], strides[self.dim0]
		out = Tensor(shape=tuple(shape), dtype=x.dtype, strides=tuple(strides), offset=x.offset, is_contiguous=False, requires_grad=x.requires_grad, device=x.device,)
		out.op = OpNode(op=self, inputs=(x,), outputs=(out,),)
		return out

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class ContiguousOp(Op):
	op_name = "Contiguous"

	def infer_shape(self, x):
		return x.shape

	def __call__(self, x):
		out = Tensor(shape=x.shape, dtype=x.dtype, requires_grad=x.requires_grad, device=x.device)
		out.op = OpNode(op=self, inputs=(x,), outputs=(out,),)
		return out

	def save_for_backward(self, inputs, outputs):
		return []

class ExpOp(Op):
	op_name = "Exp"

	def infer_shape(self, x):
		return x.shape 

class LogOp(Op):
	op_name = "Log"

	def infer_shape(self, x):
		return x.shape 

class RsqrtOp(Op):
	op_name = "Rsqrt"

	def infer_shape(self, x):
		return x.shape 

class SoftmaxOp(Op):
	op_name = "Softmax"

	def __init__(self, dim):
		self.dim = dim 

	def infer_shape(self, x):
		return x.shape 

	def __call__(self, x):
		out = Tensor(shape=x.shape, dtype=x.dtype, requires_grad=x.requires_grad, device=x.device,)
		out.op = OpNode(op=self, inputs=(x,), outputs=(out,),)
		return out 

class GatherOp(Op):
	op_name = "Gather"

	def infer_shape(self, indices, weight):
		if indices.ndim != 2:
			raise ValueError("Gather indices must be 2D (B, T)")
		if weight.ndim != 2:
			raise ValueError("Gather weight must be 2D (V, C)")
		B, T = indices.shape 
		C = weight.shape[1]
		return (B, T, C)

	def __call__(self, indices, weight):
		shape = self.infer_shape(indices, weight)
		out = Tensor(shape=shape, dtype=weight.dtype, requires_grad=weight.requires_grad, device=weight.device)
		out.op = OpNode(op=self, inputs=(indices, weight), outputs=(out,))
		return out 

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class GELUOp(Op):
	op_name = "GELU"

	def infer_shape(self, x):
		return x.shape 

class LayerNormOp(Op):
	op_name = "LayerNorm"

	def __init__(self, normalized_dim=-1, eps=1e-6):
		self.normalized_dim = normalized_dim
		self.eps = eps

	def infer_shape(self, x, gamma, beta):
		if gamma.shape != beta.shape:
			raise ValueError("gamma/beta shape mismatch")
		if x.shape[self.normalized_dim] != gamma.shape[0]:
			raise ValueError("LayerNorm C mismatch")
		return x.shape 

	def __call__(self, x, gamma, beta):
		shape = self.infer_shape(x, gamma, beta)
		out = Tensor(shape=shape, dtype=x.dtype, requires_grad=any(t.requires_grad for t in (x, gamma, beta)), device=x.device)
		out.op = OpNode(op=self, inputs=(x, gamma, beta), outputs=(out,))
		return out

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class CrossEntropyLossOp(Op):
	op_name = "CrossEntropyLoss"

	def __init__(self, dim=-1):
		self.dim = dim 

	def infer_shape(self, logits, labels):
		if logits.shape[:-1] != labels.shape:
			raise ValueError("CE shape mismatch")
		return labels.shape

	def __call__(self, logits, labels):
		shape = self.infer_shape(logits, labels)
		out = Tensor(shape=shape, dtype=logits.dtype, requires_grad=logits.requires_grad, device=logits.device,)
		out.op = OpNode(op=self, inputs=(logits, labels), outputs=(out,))
		return out

class FusedLinearCrossEntropyOp(Op):
	op_name = "FusedLinearCrossEntropy"

	def __init__(self, ignore_index=-1):
		self.ignore_index = ignore_index

	def infer_shape(self, x, w, targets, row_scale=None):
		return (1,)

	def __call__(self, x, w, targets, row_scale=None):
		if row_scale is None:
			inputs = (x, w, targets)
		else:
			inputs = (x, w, targets, row_scale)
		shape = self.infer_shape(x, w, targets, row_scale)
		req_grad = x.requires_grad or w.requires_grad
		out = Tensor(shape=shape, dtype=x.dtype, requires_grad=req_grad, device=x.device)
		out.op = OpNode(op=self, inputs=inputs, outputs=(out,))
		return out

class FusedRMSNormLinearCrossEntropyOp(Op):
	op_name = "FusedRMSNormLinearCrossEntropy"

	def __init__(self, eps=1e-6):
		self.eps = eps

	def infer_shape(self, x, w, norm_weight, targets, row_scale=None):
		return (1,)

	def __call__(self, x, w, norm_weight, targets, row_scale=None):
		shape = self.infer_shape(x, w, norm_weight, targets, row_scale)
		inputs = (x, w, norm_weight, targets)
		if row_scale is not None:
			inputs = inputs + (row_scale,)
		out = Tensor(shape=shape, dtype=x.dtype, 
			         requires_grad=(x.requires_grad or w.requires_grad or norm_weight.requires_grad),
			         device=x.device)
		out.op = OpNode(op=self, inputs=inputs, outputs=(out,))
		return out

	def save_for_backward(self, inputs, outputs):
		return []

class ScaledDotProductAttenOp(Op):
	op_name = "ScaledDotProductAttention"

	def __init__(self, is_causal=True):
		self.is_causal = is_causal

	def infer_shape(self, q, k, v):
		return q.shape

	def __call__(self, q, k, v):
		shape = self.infer_shape(q, k, v)
		out = Tensor(shape=shape, dtype=q.dtype, requires_grad=any(t.requires_grad for t in (q,k,v)), device=q.device)
		out.op = OpNode(op=self, inputs=(q,k,v), outputs=(out,))
		return out

class FusedMLPOp(Op):
	op_name = "FusedMLP"

	def infer_shape(self, x, w1, b1, w2, b2):
		return x.shape

	def save_for_backward(self, inputs, outputs):
		return [inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]]

class FusedQKVProjOp(Op):
	op_name = "FusedQKVProj"

	def __init__(self, n_head):
		self.n_head = n_head

	def infer_shape(self, x, wq, bq, wk, bk, wv, bv):
		B, T, C = x.shape 
		D = C // self.n_head
		return [(B, self.n_head, T, D), (B, self.n_head, T, D), (B, self.n_head, T, D)]

	def __call__(self, x, wq, bq, wk, bk, wv, bv):
		shapes = self.infer_shape(x, wq, bq, wk, bk, wv, bv)
		req_grad = any(t.requires_grad for t in (x, wq, bq, wk, bk, wv, bv))

		q = Tensor(shape=shapes[0], dtype=x.dtype, requires_grad=req_grad, device=x.device)
		k = Tensor(shape=shapes[1], dtype=x.dtype, requires_grad=req_grad, device=x.device)
		v = Tensor(shape=shapes[2], dtype=x.dtype, requires_grad=req_grad, device=x.device)

		node = OpNode(op=self, inputs=(x, wq, bq, wk, bk, wv, bv), outputs=(q, k, v))
		q.op = k.op = v.op = node
		return q, k, v

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class AttnOutContiguousOp(Op):
	op_name = "AttnOutContiguous"

	def infer_shape(self, x):
		B, H, T, D = x.shape 
		return (B, T, H * D)

	def __call__(self, x):
		shape = self.infer_shape(x)
		out = Tensor(shape=shape, dtype=x.dtype, requires_grad=x.requires_grad, device=x.device)
		out.op = OpNode(op=self, inputs=(x,), outputs=(out,))
		return out

	def save_for_backward(self, inputs, outputs):
		return []

class FusedRMSNormMLPOp(Op):
	op_name = "FusedRMSNormMLP"
	recompute_output = True

	def __init__(self, eps=1e-6):
		self.eps = eps 

	def infer_shape(self, x, *args):
		if len(args) == 2:
			norm_weight = None
			wgu, wdown = args
		elif len(args) == 3:
			norm_weight, wgu, wdown = args

		C = x.shape[-1]
		if wgu.ndim != 2 or wgu.shape[0] != C:
			raise ValueError(f"wgu share must be ({C}, 2H), got {wgu.shape}")
		if wgu.shape[1] % 2 != 0:
			raise ValueError("wgu second dimension must be even")
		H = wgu.shape[1] // 2
		if wdown.shape != (H, C):
			raise ValueError(f"wdown shape must be ({H}, {C}), got {wdown.shape}")

		if norm_weight is not None and norm_weight.shape != (C,):
			raise ValueError(f"norm_weight shape must be ({C},), got {norm_weight.shape}")
			
		return x.shape

	def __call__(self, x, *args):
		shape = self.infer_shape(x, *args)
		if len(args) == 2:
			inputs = (x, args[0], args[1])
		else:
			inputs = (x, args[0], args[1], args[2])
		requires_grad = any(t.requires_grad for t in inputs)
		out = Tensor(shape=shape, dtype=x.dtype, requires_grad=requires_grad, device=x.device)
		out.op = OpNode(op=self, inputs=inputs, outputs=(out,))
		return out

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class FlashAttentionOp(Op):
	op_name = "FlashAttention"

	def __init__(self, n_head, is_causal=True):
		self.n_head = n_head
		self.is_causal = is_causal

	def infer_shape(self, x, wq, bq, wk, bk, wv, bv):
		B, T, C = x.shape 
		return (B, T, C)

	def __call__(self, x, wq, bq, wk, bk, wv, bv):
		shape_out = self.infer_shape(x, wq, bq, wk, bk, wv, bv)
		req_grad = any(t.requires_grad for t in (x, wq, bq, wk, bk, wv, bv))
		out = Tensor(shape=shape_out, dtype=x.dtype, requires_grad=req_grad, device=x.device)
		node = OpNode(op=self, inputs=(x, wq, bq, wk, bk, wv, bv), outputs=(out,))
		out.op = node
		return out

	def save_for_backward(self, inputs, outputs):
		return list(inputs) + list(outputs)

class FusedRMSNormFlashAttentionOp(Op):
	op_name = "FusedRMSNormFlashAttention"
	recompute_output = True

	def __init__(self, n_head, eps=1e-6, is_causal=True):
		self.n_head = n_head
		self.eps = eps
		self.is_causal = is_causal

	def infer_shape(self, x, wq, bq, wk, bk, wv, bv):
		B, T, C = x.shape
		return (B, T, C)

	def __call__(self, x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin):
		shape_out = self.infer_shape(x, wq, bq, wk, bk, wv, bv)
		req_grad = any(t.requires_grad for t in (x, wq, bq, wk, bk, wv, bv))
		out = Tensor(shape=shape_out, dtype=x.dtype, requires_grad=req_grad, device=x.device)
		out.op = OpNode(op=self, inputs=(x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin), outputs=(out,),)
		return out

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class FusedRMSNormGQAOp(Op):
	op_name = "FusedRMSNormGQA"
	recompute_output = True

	def __init__(self, n_head, n_kv_head, eps=1e-6, is_causal=True, rope_sign=1):
		self.n_head = n_head
		self.n_kv_head = n_kv_head
		self.eps = eps
		self.is_causal = is_causal
		self.rope_sign = rope_sign

	def infer_shape(self, x, norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin):
		B, T, C = x.shape
		if C % self.n_head or C % 4:
			raise ValueError("require C % n_head == 0 and C % 4 == 0")
		return x.shape

	def __call__(self, x, norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin):
		inputs = (x, norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin)
		shape = self.infer_shape(*inputs)
		out = Tensor(shape=shape, dtype=x.dtype, device=x.device, requires_grad=any(t.requires_grad for t in inputs[:8]))
		out.op = OpNode(op=self, inputs=inputs, outputs=(out,))
		return out

	def save_for_backward(self, inputs, outputs):
		return list(inputs)

class KVCacheAttentionOp(Op):
	op_name = "KVCacheAttention"

	def __init__(self, n_head, layer_idx, eps=1e-6, mode="decode"):
		self.n_head = n_head
		self.layer_idx = layer_idx
		self.eps = eps
		self.mode = mode

	def infer_shape(self, x):
		B, T, C = x.shape
		if self.mode == "decode" and T != 1:
			raise ValueError("KV cache decode requires T=1")
		return (B, T, C)

	def __call__(self, x, wq, bq, wk, bk, wv, bv, k_cache, v_cache, position, rope_cos, rope_sin):
		out = Tensor(shape=self.infer_shape(x), dtype=x.dtype, requires_grad=False, device=x.device)
		out.op = OpNode(op=self, inputs=(x, wq, bq, wk, bk, wv, bv, k_cache, v_cache, position, rope_cos, rope_sin), outputs=(out,))
		return out

class FusedLinearAddOp(Op):
	op_name = "FusedLinearAdd"

	def infer_shape(self, x, act, w, b):
		return x.shape

	def __call__(self, x, act, w, b):
		shape = self.infer_shape(x, act, w, b)
		req_grad = any(t.requires_grad for t in (x, act, w, b))
		out = Tensor(shape=shape, dtype=x.dtype, requires_grad=req_grad, device=x.device)
		out.op = OpNode(op=self, inputs=(x, act, w, b), outputs=(out,))
		return out 

	def save_for_backward(self, inputs, outputs):
		return []  

class LMHeadOp(Op):
	op_name = "LMHead"

	def infer_shape(self, x, weight):
		if x.shape[-1] != weight.shape[-1]:
			raise ValueError(f"LMHead dimension mismatch: x={x.shape}, weight={weight.shape}")
		return x.shape[:-1] + (weight.shape[0],)

	def __call__(self, x, weight):
		out = Tensor(shape=self.infer_shape(x, weight), dtype=x.dtype, requires_grad=x.requires_grad or weight.requires_grad, device=x.device,)
		out.op = OpNode(op=self, inputs=(x, weight), outputs=(out,),)
		return out

	def save_for_backward(self, inputs, outputs):
		return []

class LastPositionLMHeadOp(Op):
	op_name = "LastPositionLMHead"

	def infer_shape(self, x, weight, position):
		if x.ndim != 3 or x.shape[0] != 1:
			raise ValueError("LastPositionLMHead requires x with shape [1, T, C]")
		if weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
			raise ValueError(f"LastPositionLMHead dimension mismatch: x={x.shape}, weight={weight.shape}")
		if tuple(position.shape) != (1,) or np.dtype(position.dtype) != np.dtype(np.int32):
			raise ValueError("position must have shape (1,) or dtype int32")
		if not x.is_contiguous or not weight.is_contiguous:
			raise ValueError("LastPositionLMHead requires contiguous inputs")

		return (1, 1, weight.shape[0])

	def __call__(self, x, weight, position):
		out = Tensor(shape=self.infer_shape(x, weight, position), dtype=x.dtype, device=x.device, requires_grad=False, name="last_position_lm_head")
		out.op = OpNode(op=self, inputs=(x, weight, position), outputs=(out,))
		return out

	def save_for_backward(self, inputs, outputs):
		return []

class RolloutTopkOp(Op):
	op_name = "RolloutTopk"

	def __init__(self, max_top_k=64, mode="decode"):
		self.max_top_k = max_top_k
		self.mode_id = 0 if mode == "prefill" else 1

	def infer_shape(self, logits, top_k=None, logits_position=None):
		B = logits.shape[0]
		V = logits.shape[-1]
		K = min(64, V)

		return (B, K), (B, K)

	def __call__(self, logits, top_k, logits_position):
		values_shape, indices_shape = self.infer_shape(logits, top_k, logits_position)

		values = Tensor(shape=values_shape, dtype=logits.dtype, requires_grad=False, device=logits.device,
						name="rollout_topk_values")
		indices = Tensor(shape=indices_shape, dtype=np.int32, requires_grad=False, device=logits.device,
						name="rollout_topk_indices")
		values.op = OpNode(op=self, inputs=(logits, top_k, logits_position), outputs=(values, indices))
		indices.op = values.op 
		return values, indices

	def save_for_backward(self, inputs, outputs):
		return []


class RolloutSampleOp(Op):
	op_name = "RolloutSample"

	def __init__(self, eos_token, bos_token, max_tokens, mode, step_offset=0):
		self.eos_token = eos_token
		self.bos_token = bos_token
		self.max_tokens = max_tokens
		self.mode_id = 0 if mode == "prefill" else 1
		self.step_offset = step_offset

	def infer_shape(self, topk_values, topk_indices):
		if topk_indices.ndim != 2:
			raise ValueError(f"RolloutSample expects top-k indices [B, K], got {topk_indices.shape}")
		if topk_values.shape != topk_indices.shape:
			raise ValueError(f"top-k shape mismatch values={topk_values.shape}, indices={topk_indices.shape}")
		return (1,)

	def __call__(self, topk_values, topk_indices, rng_state, rng_raw, finished, temperature, top_k,
				 next_tokens, generated_ids, lengths, position, step, logits):
		out = Tensor(shape=self.infer_shape(topk_values, topk_indices),
					 dtype=np.int32,
					 requires_grad=False,
					 device=topk_indices.device,
					 name="rollout_finished_all")
		out.op = OpNode(op=self, 
			            inputs=(topk_values, topk_indices, rng_state, rng_raw, finished, temperature, top_k, next_tokens, generated_ids, lengths, position, step, logits), 
			            outputs=(out,),)
		return out

	def save_for_backward(self, inputs, outputs):
		return []

class DecodeStateOp(Op):
	op_name = "DecodeState"

	def __init__(self, phase, offset):
		self.phase = phase
		self.offset = offset

	def infer_shape(self, dependency, tokens, masks, metadata, finished, config, positions, steps, candidates, next_tokens, global_step):
	    return next_tokens.shape

	def __call__(self, dependency, tokens, masks, metadata, finished, config, positions, steps, candidates, next_tokens, global_step):
	    inputs = (dependency, tokens, masks, metadata, finished, config, positions, steps, candidates, next_tokens, global_step)
	    out = Tensor(shape=self.infer_shape(*inputs), dtype=np.int32, device=tokens.device, 
	    	         requires_grad=False, name="decode_work_ids")
	    out.op = OpNode(op=self, inputs=inputs, outputs=(out,),)
	    return out

	def save_for_backward(self, inputs, outputs):
		return []

class DecodeFinishedOp(Op):
	op_name = "DecodeFinish"

	def infer_shape(self, next_ids, finished, step, config):
		return (1,)

	def __call__(self, next_ids, finished, step, config):
		inputs = (next_ids, finished, step, config)

		out = Tensor(shape=self.infer_shape(*inputs), dtype=np.int32, device=next_ids.device, requires_grad=False, name="decode_finished_all")
		out.op = OpNode(op=self, inputs=inputs, outputs=(out,),)
		return out

	def save_for_backward(self, inputs, outputs):
		return []













