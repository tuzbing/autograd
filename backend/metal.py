
from math import prod
import struct
import numpy as np
from .base import Backend
from op import (
	MulOp, ViewOp, ReshapeOp, PermuteOp, TransposeOp, ContiguousOp,
	ExpOp, LogOp, RsqrtOp, SoftmaxOp, GELUOp, LayerNormOp, GatherOp,
	CrossEntropyLossOp, ScaledDotProductAttenOp, FusedLinearCrossEntropyOp,
	FusedRMSNormLinearCrossEntropyOp, FusedMLPOp, FusedQKVProjOp,
	AttnOutContiguousOp, FusedRMSNormMLPOp, FlashAttentionOp,
	FusedRMSNormFlashAttentionOp, FusedRMSNormGQAOp, FusedLinearAddOp,
	KVCacheAttentionOp, LMHeadOp, LastPositionLMHeadOp,
	RolloutTopkOp, RolloutSampleOp, DecodeStateOp, DecodeFinishedOp,
)
from utils import f32_bits, bf16_bits
from gemm import fp16_gemm_kernel_name, get_fp16_gemm_config
from scratch import align_up

def noop_forward(inputs, outputs, scratch=None):
	pass

class MetalBackend(Backend):
	def __init__(self):
		import metal_ext
		self.ctx = metal_ext.MetalContext()
		self._rollout_cache = None
		self._rollout_cache_dirty = True
		self.numpy_view_funcs = {
			np.dtype(np.float32): self.ctx.numpy_view_float32,
			np.dtype(np.float16): self.ctx.numpy_view_float16,
			np.dtype(np.int32): self.ctx.numpy_view_int32,
			np.dtype(np.uint8): self.ctx.numpy_view_uint8,
			np.dtype(np.uint32): self.ctx.numpy_view_uint32,
		}
		self.forward_kernels = {
			# AddOp: self.metal_add_forward,
			MulOp: self.metal_mul_forward,
			ViewOp: noop_forward,
			ReshapeOp: noop_forward,
			PermuteOp: noop_forward,
			TransposeOp: noop_forward,
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
			FusedRMSNormLinearCrossEntropyOp: self.metal_fused_rmsnorm_linear_ce_forward,
			FusedMLPOp: self.metal_fused_mlp_forward,
			FusedQKVProjOp: self.metal_fused_qkv_proj_forward,
			AttnOutContiguousOp: self.metal_attn_out_contiguous_forward,
			FusedRMSNormMLPOp: self.metal_fused_rmsnorm_mlp_forward,
			FlashAttentionOp: self.metal_flash_attention_forward,
			FusedRMSNormFlashAttentionOp: self.metal_fused_rms_fa_forward,
			FusedRMSNormGQAOp: self.metal_fused_rms_gqa_forward,
			FusedLinearAddOp: self.metal_fused_linear_add_forward,
			KVCacheAttentionOp: self.metal_kv_cache_attention_forward,
			LMHeadOp: self.metal_lm_head_forward,
			LastPositionLMHeadOp: self.metal_last_position_lm_head_forward,
			RolloutTopkOp: self.metal_rollout_topk_forward,
			RolloutSampleOp: self.metal_rollout_sample_forward,
			DecodeStateOp: self.metal_decode_state_forward,
			DecodeFinishedOp: self.metal_decode_finished_forward,
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
			FusedRMSNormLinearCrossEntropyOp: self.metal_fused_rmsnorm_linear_ce_backward,
			FusedMLPOp: self.metal_fused_mlp_backward,
			FusedQKVProjOp: self.metal_fused_qkv_proj_backward,
			AttnOutContiguousOp: self.metal_attn_out_contiguous_backward,
			FusedRMSNormMLPOp: self.metal_fused_rmsnorm_mlp_backward,
			FlashAttentionOp: self.metal_flash_attention_backward,
			FusedRMSNormFlashAttentionOp: self.metal_fused_rms_fa_backward,
			FusedRMSNormGQAOp: self.metal_fused_rms_gqa_backward,
			FusedLinearAddOp: self.metal_fused_linear_add_backward,
			KVCacheAttentionOp: self.metal_kv_cache_attention_backward,
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

		if out.grad is not None and weight.requires_grad:
			if not getattr(self, "accumulate_parameter_grads", False):
				self.ctx.dispatch_with_args_3d("fill_scalar_kernel2",
					[weight.grad.storage.ptr.buffer_id], [weight.grad.storage.ptr.offset],
					[0, np.dtype(weight.grad.storage.dtype).itemsize], prod(weight.grad.shape), 1, 1, min(prod(weight.grad.shape), 256))
				# self.ctx.dispatch_with_args_3d("fill_scalar_kernel",
				# 	[weight.grad.storage.ptr.buffer_id], [weight.grad.storage.ptr.offset],
				# 	[0], prod(weight.grad.shape), 1, 1, min(prod(weight.grad.shape), 256))

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
		# if not getattr(self, "accumulate_parameter_grads", False):
		# 	self.ctx.dispatch_with_args_3d('fill_scalar_kernel',
		# 		[weight.grad.storage.ptr.buffer_id], [weight.grad.storage.ptr.offset], 
		# 		[0], prod(weight.grad.shape), 1, 1, min(prod(weight.grad.shape), 256))
		buffer_ids = [indices.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, weight.grad.storage.ptr.buffer_id]
		offsets = [indices.storage.ptr.offset, out.grad.storage.ptr.offset, weight.grad.storage.ptr.offset]
		C = weight.shape[-1]
		grad_bf16 = np.dtype(weight.grad.storage.dtype) == np.dtype(np.uint16)
		if grad_bf16:
			rows = prod(indices.shape)
			self.ctx.dispatch_with_args_3d("gather_backward_bf16_kernel",
				buffer_ids, offsets, [rows, C], 32, (C+127)//128, rows, 32)
		else:
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
		self.ctx.dispatch_with_args_3d("fp16_quant_colwise_fused",
			[src_id, q_id, q_id],
			[src_off, q_off, s_off],
			[rows, cols, src_ld], ((cols + 63) // 64) * 256, 1, 1, 256)

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

	def _dispatch_fp16_gemm_raw(self, cfg, with_accum, buf_ids, offsets, m, n, k, out_bf16=False):
		kernel = fp16_gemm_kernel_name(cfg, with_accum, prefix="fp16")
		# print(f"Need kernel {kernel}")
		if cfg["is_splitk"] and not with_accum:
			self.ctx.fill_scalar("fill_scalar_kernel", [buf_ids[4]], [offsets[4]], 0.0, m * n)
		swizzle = cfg["swizzle_log"] | (0x80000000 if out_bf16 else 0) if not cfg["is_splitk"] else 0
		args = (
			[m, n, k, cfg["lda"], cfg["ldb"], cfg["ldd"], cfg["tm"], cfg["tn"], cfg["split_k_partitions"], cfg["split_k_partition_stride"], cfg["split_k_partition_size"], cfg["k_iter_aligned"]]
			if cfg["is_splitk"] else
			[m, n, k, cfg["lda"], cfg["ldb"], cfg["ldd"], cfg["tm"], cfg["tn"], swizzle, cfg["k_iter_aligned"], cfg["m_aligned"], cfg["n_aligned"], cfg["k_aligned"]]
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

	def metal_fused_rmsnorm_linear_ce_forward(self, inputs, outputs, scratch=None):
		x, w, norm_weight, targets = inputs[:4]
		row_scale = inputs[4] if len(inputs) == 5 else None
		out_loss = outputs[0]
		op = out_loss.op.op

		N = prod(x.shape[:-1])
		C = x.shape[-1]
		V = w.shape[0]

		row_loss = scratch["row_loss"]
		lse = scratch["lse"]
		logits = scratch["logits"]
		x_norm = scratch["x_norm"]
		dx_norm = scratch["dx_norm"]
		inv_rms = scratch["inv_rms"]

		raw_rows = min(logits.shape[0], N)
		chunk_rows = (raw_rows // 16) * 16 if raw_rows >= 16 else raw_rows
		itemsize = np.dtype(x.dtype).itemsize
		eps_bits = f32_bits(op.eps)

		accumulate = bool(getattr(self, "accumulate_parameter_grads", False))
		need_grad = out_loss.grad is not None and (x.requires_grad or w.requires_grad or norm_weight.requires_grad)
		need_dz = need_grad and (x.requires_grad or norm_weight.requires_grad)
		dw_bf16 = need_grad and w.requires_grad and np.dtype(w.grad.storage.dtype) == np.dtype(np.uint16)
		dgamma_bf16 = need_grad and norm_weight.requires_grad and np.dtype(norm_weight.grad.storage.dtype) == np.dtype(np.uint16)
		if dgamma_bf16:
			dgamma_acc_offset = x.grad.storage.ptr.offset + (N-1)*C*4

		if row_scale is None:
			kernel_name = "fused_linear_ce_logits_to_dlogits"
			scale = float(getattr(self, "grad_scale", 1.0)) / float(N)
			scale_bits = f32_bits(scale)
			scale_storage = row_loss
		else:
			kernel_name = "fused_linear_ce_logits_to_dlogits_row_scaled"
			scale_bits = 0
			scale_storage = row_scale.storage

		for row_base in range(0, N, chunk_rows):
			rows = min(chunk_rows, N - row_base)
			x_offset = x.storage.ptr.offset + row_base*C*itemsize

			# 1) Z = RMSNorm(X_chunk, gamma), 同时保存每行inv_rms
			self.ctx.dispatch_with_args_3d("gqa_rms_recompute",
				[x.storage.ptr.buffer_id, norm_weight.storage.ptr.buffer_id, x_norm.ptr.buffer_id, inv_rms.ptr.buffer_id],
				[x_offset, norm_weight.storage.ptr.offset, x_norm.ptr.offset, inv_rms.ptr.offset],
				[rows, C, eps_bits], 128, (rows+3)//4, 1, 128)

			# 2) logits = Z @ W.T
			logits_cfg = get_fp16_gemm_config(M=rows, N=V, K=C, with_accum=False, b_transpose=True, dtype="fp32")
			self._dispatch_fp32_gemm(logits_cfg, False,
				[x_norm.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
				[x_norm.ptr.offset, w.storage.ptr.offset, logits.ptr.offset],
				rows, V, C)

			# 3) row_loss/lse以及原地logits->dlogits
			if row_scale is None:
				self.ctx.dispatch_with_args_3d(kernel_name,
					[logits.ptr.buffer_id, targets.storage.ptr.buffer_id, row_loss.ptr.buffer_id, lse.ptr.buffer_id],
					[logits.ptr.offset, targets.storage.ptr.offset, row_loss.ptr.offset, lse.ptr.offset],
					[rows, V, row_base, scale_bits, int(need_grad)], 256, rows, 1, 256)
			else:
				inp_vars = [logits, targets.storage, row_loss, lse, row_scale.storage]
				self.ctx.dispatch_with_args_3d(kernel_name,
					[e.ptr.buffer_id for e in inp_vars], [e.ptr.offset for e in inp_vars],
					[rows, V, row_base, N, int(need_grad)], 256, rows, 1, 256)

			if not need_grad:
				continue

			# 4) 先完成dW, 后覆盖Z
			if w.requires_grad:
				add_dw = bool(accumulate or row_base != 0)
				dw_cfg = get_fp16_gemm_config(M=V, N=C, K=rows, with_accum=add_dw, a_transpose=True, b_transpose=False, dtype="fp32")
				if dw_bf16:
					if not dw_cfg["is_splitk"]:
						tg = dw_cfg["wm"] * dw_cfg["wn"] * 32
						dw_flags = int(add_dw) | (dw_cfg["swizzle_log"] << 2) | (dw_cfg["bm"] << 8)
						self.ctx.dispatch_with_args_3d("fused_rmsnorm_linear_ce_dw_bf16",
							[logits.ptr.buffer_id, x_norm.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
							[logits.ptr.offset, x_norm.ptr.offset, w.grad.storage.ptr.offset],
							[rows, C, V, dw_flags], dw_cfg["grid_x"] * tg, dw_cfg["grid_y"], dw_cfg["grid_z"], tg)
					else:
						self.ctx.dispatch_with_args_3d("fused_rmsnorm_linear_ce_dw_bf16",
							[logits.ptr.buffer_id, x_norm.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
							[logits.ptr.offset, x_norm.ptr.offset, w.grad.storage.ptr.offset],
							[rows, C, V, int(add_dw)], C, V, 1, min(C, 256))
				else:
					self._dispatch_fp32_gemm(dw_cfg, add_dw,
						[logits.ptr.buffer_id, x_norm.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
						[logits.ptr.offset, x_norm.ptr.offset, w.grad.storage.ptr.offset],
						V, C, rows)

			if not need_dz:
				continue

			# 恢复target概率: dlogits[target] = scale * P[target]
			self.ctx.dispatch_with_args_3d("fused_rmsnorm_linear_ce_restore_target",
				[x_norm.ptr.buffer_id, w.storage.ptr.buffer_id, targets.storage.ptr.buffer_id, lse.ptr.buffer_id, scale_storage.ptr.buffer_id, logits.ptr.buffer_id],
				[x_norm.ptr.offset, w.storage.ptr.offset, targets.storage.ptr.offset, lse.ptr.offset, scale_storage.ptr.offset, logits.ptr.offset],
				[rows, C, V, row_base, scale_bits, int(row_scale is not None)], 32, rows, 1, 32)

			# 5) dZ = dlogits @ W
			dz_cfg = get_fp16_gemm_config(M=rows, N=C, K=V, with_accum=False, b_transpose=False, dtype="fp32")
			self._dispatch_fp32_gemm(dz_cfg, False,
				[logits.ptr.buffer_id, w.storage.ptr.buffer_id, dx_norm.ptr.buffer_id],
				[logits.ptr.offset, w.storage.ptr.offset, dx_norm.ptr.offset],
				rows, C, V)

			# 7) dZ = scale * P @ W - scale * W[target]
			self.ctx.dispatch_with_args_3d("fused_rmsnorm_linear_ce_subtract_target",
				[w.storage.ptr.buffer_id, targets.storage.ptr.buffer_id, scale_storage.ptr.buffer_id, dx_norm.ptr.buffer_id],
				[w.storage.ptr.offset, targets.storage.ptr.offset, scale_storage.ptr.offset, dx_norm.ptr.offset],
				[rows, C, V, row_base, scale_bits, int(row_scale is not None)], C, rows, 1, min(C, 256))

			# 8) dgamma = sum(dZ * X * inv_rms)
			if norm_weight.requires_grad:
				if dgamma_bf16:
					self.ctx.dispatch_with_args_3d("gqa_rms_dgamma_bf16",
						[x.storage.ptr.buffer_id, dx_norm.ptr.buffer_id, inv_rms.ptr.buffer_id, norm_weight.grad.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
						[x_offset, dx_norm.ptr.offset, inv_rms.ptr.offset, norm_weight.grad.storage.ptr.offset, dgamma_acc_offset],
						[rows, C, int(accumulate), int(row_base == 0), int(row_base+rows == N)], C, 1, 1, min(C, 256))
				else:
					inp_vars = [x.storage, dx_norm, inv_rms, norm_weight.grad.storage]
					self.ctx.dispatch_with_args_3d("gqa_rms_dgamma",
						[x.storage.ptr.buffer_id, dx_norm.ptr.buffer_id, inv_rms.ptr.buffer_id, norm_weight.grad.storage.ptr.buffer_id,],
						[x_offset, dx_norm.ptr.offset, inv_rms.ptr.offset, norm_weight.grad.storage.ptr.offset],
						[rows, C, int(accumulate or row_base != 0)], C, 1, 1, min(C, 256))

			# 7) 直接写入
			if x.requires_grad:
				self.ctx.dispatch_with_args_3d("fused_rmsnorm_linear_ce_dx",
					[x.storage.ptr.buffer_id, dx_norm.ptr.buffer_id, norm_weight.storage.ptr.buffer_id, inv_rms.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
					[x_offset, dx_norm.ptr.offset, norm_weight.storage.ptr.offset, inv_rms.ptr.offset, x.grad.storage.ptr.offset + row_base*C*itemsize],
					[rows, C], 32, rows, 1, 32)

		self.ctx.dispatch_with_args_3d("fused_linear_ce_forward_reduce",
			[row_loss.ptr.buffer_id, out_loss.storage.ptr.buffer_id],
			[row_loss.ptr.offset, out_loss.storage.ptr.offset],
			[N], 256, 1, 1, 256)

	def metal_fused_rmsnorm_linear_ce_backward(self, inputs, outputs, scratch=None):
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

	def _dispatch_decode_linear(self, x_id, x_off, q_id, wq_off, ws_off, out0_id, out0_off, M, N, K,
		                        flags=0, eps_bits=0, gamma_id=0, gamma_off=0, bias_id=0, bias_off=0,
		                        res_id=0, res_off=0, out1_id=0, out1_off=0, out2_id=0, out2_off=0, c_out=0):
		self.ctx.dispatch_with_args_3d("fp16_decode_linear",
			[x_id, q_id, q_id, gamma_id or x_id, bias_id or x_id, res_id or x_id,
			 out0_id, out1_id or out0_id, out2_id or out0_id],
			[x_off, wq_off, ws_off, gamma_off, bias_off, res_off,
			 out0_off, out1_off, out2_off],
			[M, N, K, flags, eps_bits, c_out],
			((N + 15) // 16) * 128, (M + 7) // 8, 1, 128)

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

		if not self._rollout_cache_dirty and out.grad is None and N <= 16 and self._rollout_cache is not None:
			c_gu = self._rollout_cache.get(id(wgu))
			c_dn = self._rollout_cache.get(id(wdown))
			if c_gu is not None and c_dn is not None:
				flags = 1 | (2 if norm_weight is not None else 0)
				g_id = norm_weight.storage.ptr.buffer_id if norm_weight is not None else 0
				g_off = norm_weight.storage.ptr.offset if norm_weight is not None else 0
				self._dispatch_decode_linear(
					x.storage.ptr.buffer_id, x.storage.ptr.offset,
					c_gu[0], c_gu[1], c_gu[2], gu.ptr.buffer_id, gu.ptr.offset,
					N, H2, C, flags=flags, eps_bits=f32_bits(out.op.op.eps), gamma_id=g_id, gamma_off=g_off)
				self.ctx.dispatch_with_args_3d("swiglu_forward_kernel",
					[gu.ptr.buffer_id, hidden.ptr.buffer_id],
					[gu.ptr.offset, hidden.ptr.offset],
					[N, H], N * ((H + 3) // 4), 1, 1, min(N * ((H + 3) // 4), 256))
				self._dispatch_decode_linear(
					hidden.ptr.buffer_id, hidden.ptr.offset,
					c_dn[0], c_dn[1], c_dn[2], out.storage.ptr.buffer_id, out.storage.ptr.offset,
					N, C, H)
				return

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
			self.ctx.dispatch_with_args_3d("rmsnorm_forward",
				[fa_x.storage.ptr.buffer_id, tmp_dx.ptr.buffer_id],
				[fa_x.storage.ptr.offset, tmp_dx.ptr.offset],
				[C, fa_eps_bits], (B * T) * 256, 1, 1, 256)
			tm, tn = (B * T + 63) // 64, (C + 63) // 64
			sw_log = 3 if (tm >= 8 and tn >= 8) else (2 if (tm >= 4 and tn >= 4) else (1 if (tm >= 2 and tn >= 4) else 0))
			sw = 1 << sw_log
			self.ctx.dispatch_with_args_3d('fused_rms_fa_project_kv2',
				[tmp_dx.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
					wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
					gu.ptr.buffer_id, gu.ptr.buffer_id, gu.ptr.buffer_id],
				[tmp_dx.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
					wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
					fa_out_offset, fa_k_offset, fa_v_offset],
				[B, T, C, n_head | 0x80000000, fa_eps_bits], 128 * tn * sw, (tm + sw - 1) // sw, 3, 128)

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

			tm, tn = (B * T + 63) // 64, (C + 63) // 64 + 2 * ((CKV + 63) // 64)
			sw_log = 3 if (tm >= 8 and tn >= 8) else (2 if (tm >= 4 and tn >= 4) else (1 if (tm >= 2 and tn >= 4) else 0))
			sw = 1 << sw_log
			self.ctx.dispatch_with_args_3d("gqa_rms_qkv",
				[fa_x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id, wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id,
				 wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id, gu.ptr.buffer_id, gu.ptr.buffer_id, gu.ptr.buffer_id, fa_norm_weight.storage.ptr.buffer_id],
				[fa_x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset, wk.storage.ptr.offset, bk.storage.ptr.offset,
				 wv.storage.ptr.offset, bv.storage.ptr.offset, fa_q_offset, fa_k_offset, fa_v_offset, fa_norm_weight.storage.ptr.offset],
				[B, T, C, Hq, fa_eps_bits, Hkv], 128 * tn * sw, (tm + sw - 1) // sw, 1, 128)

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

		if use_fp16_gemm:
			h2_tile, chunk_rows, h_tile, c_tile, gu_tile_r, head_bytes, xnorm_bytes = scratch.gemm_configs["mlp_shape"]
			scale_len = max(chunk_rows, C)
			sdgu_bytes = align_up(N * 4, 16)
			if "gemm_quant" in scratch._views:
				q_buf = scratch["gemm_quant"]
				q_id = q_buf.ptr.buffer_id
				sdgu_off = q_buf.ptr.offset
				xq_off = dout_q_off = p3_off = sdgu_off + sdgu_bytes
			else:
				q_id = hidden.ptr.buffer_id
				sdgu_off = hidden.ptr.offset + ((N * H * 4) & ~15) - sdgu_bytes
				xq_off = hidden.ptr.offset
				dout_q_off = hidden.ptr.offset + head_bytes
				p3_off = hidden.ptr.offset + xnorm_bytes
			s_dout_off = dout_q_off + align_up(N * C * 2, 16)
			wq_off = s_dout_off + align_up(scale_len * 4, 16)
			sw_off = wq_off + align_up(h_tile * C * 2, 16)

			# 5 & 6. X_norm(FP16) & GU = X_norm @ Wgu
			self.ctx.fill_scalar("fill_scalar_kernel", [q_id], [sdgu_off], 1.0, N)
			if norm_weight is None:
				self.ctx.dispatch_with_args_3d("rmsnorm_forward",
					[x_buffer_id, q_id],
					[x_offset, xq_off],
					[C | 0x80000000, eps_bits], N * 256, 1, 1, 256)
			else:
				self.ctx.dispatch_with_args_3d("rmsnorm_weighted_forward",
					[x_buffer_id, norm_weight.storage.ptr.buffer_id, q_id],
					[x_offset, norm_weight.storage.ptr.offset, xq_off],
					[C | 0x80000000, eps_bits], N * 32, 1, 1, 32)
			wgu_q_off = xq_off + align_up(N * C * 2, 16)
			swgu_off = wgu_q_off + align_up(C * h2_tile * 2, 16)
			gu_cfgs = scratch.gemm_configs["gu"]
			for j0 in range(0, H2, h2_tile):
				h2_sub = min(h2_tile, H2 - j0)
				self._quantize_fp16_colwise(wgu.storage.ptr.buffer_id, wgu.storage.ptr.offset + j0 * 4,
					q_id, wgu_q_off, swgu_off, C, h2_sub, H2)
				self._dispatch_fp16_gemm_raw(gu_cfgs[h2_sub], False,
					[q_id, q_id, q_id, q_id, gu.ptr.buffer_id],
					[xq_off, wgu_q_off, sdgu_off, swgu_off, gu.ptr.offset + j0 * 4],
					N, h2_sub, C)
		else:
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
					[C, eps_bits], N * 32, 1, 1, 32)

			# 6. GU = X_norm @ Wgu
			self.ctx.dispatch_gemm_mps([x_norm_storage.ptr.buffer_id, wgu.storage.ptr.buffer_id, gu.ptr.buffer_id],
				[x_norm_storage.ptr.offset, wgu.storage.ptr.offset, gu.ptr.offset],
				[1, N, H2, C, 0, 0, 0, 0], 1.0, 0.0)

		if use_fp16_gemm:
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
					H, C, N, out_bf16=down_bf16)
		elif down_bf16:
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

		if wgu.requires_grad and (recompute_x or use_fp16_gemm):
			if norm_weight is None:
				self.ctx.dispatch_with_args_3d("rmsnorm_forward",
					[x_buffer_id, x_norm_storage.ptr.buffer_id],
					[x_offset, x_norm_storage.ptr.offset],
					[C, eps_bits], N * 256, 1, 1, 256)
			else:
				self.ctx.dispatch_with_args_3d("rmsnorm_weighted_forward",
					[x_buffer_id, norm_weight.storage.ptr.buffer_id, x_norm_storage.ptr.buffer_id],
					[x_offset, norm_weight.storage.ptr.offset, x_norm_storage.ptr.offset],
					[C, eps_bits], N * 32, 1, 1, 32)

		# 11. dWgu = X_norm.T @ dGU
		if wgu.requires_grad:
			if use_fp16_gemm:
				zq_off = p3_off
				sz_off = zq_off + align_up(N * C * 2, 16)
				s1_off = sz_off + align_up(C * 4, 16)
				self._quantize_fp16_tensor(x_norm_storage.ptr.buffer_id, x_norm_storage.ptr.offset,
					q_id, zq_off, sz_off, s1_off, N * C, C, q_id, sdgu_off, C)
				self.ctx.fill_scalar("fill_scalar_kernel", [q_id], [s1_off], 1.0, H2)
				self._dispatch_fp16_gemm_raw(scratch.gemm_configs["dwgu"], bool(accumulate),
					[q_id, gu.ptr.buffer_id, q_id, q_id, wgu.grad.storage.ptr.buffer_id],
					[zq_off, gu.ptr.offset, sz_off, s1_off, wgu.grad.storage.ptr.offset],
					C, H2, N, out_bf16=gu_bf16)
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
			split_n = min(32, (N + 31) // 32) if (not norm_grad_bf16 or gu.nbytes >= C * 4) else 1
			if split_n > 1 and norm_grad_bf16:
				self.ctx.fill_scalar("fill_scalar_kernel", [gu.ptr.buffer_id], [gu.ptr.offset], 0.0, C)
				self.ctx.dispatch_with_args_3d("rmsnorm_weighted_dgamma",
					[x_buffer_id, dx_norm_storage.ptr.buffer_id, gu.ptr.buffer_id],
					[x_offset, dx_norm_storage.ptr.offset, gu.ptr.offset],
					[N, C, eps_bits, 0, split_n << 1], 256, 1, split_n, 256)
				self.ctx.dispatch_with_args_3d("fp32_to_bf16",
					[gu.ptr.buffer_id, norm_weight.grad.storage.ptr.buffer_id],
					[gu.ptr.offset, norm_weight.grad.storage.ptr.offset],
					[int(accumulate)], C, 1, 1, min(C, 256))
			else:
				if split_n > 1 and not accumulate:
					self.ctx.fill_scalar("fill_scalar_kernel",
						[norm_weight.grad.storage.ptr.buffer_id], [norm_weight.grad.storage.ptr.offset], 0.0, C)
				self.ctx.dispatch_with_args_3d("rmsnorm_weighted_dgamma",
					[x_buffer_id, dx_norm_storage.ptr.buffer_id, norm_weight.grad.storage.ptr.buffer_id],
					[x_offset, dx_norm_storage.ptr.offset, norm_weight.grad.storage.ptr.offset],
					[N, C, eps_bits, int(accumulate), int(norm_grad_bf16) | (split_n << 1)], 256, 1, split_n, 256)

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

	def metal_fused_rms_gqa_forward(self, inputs, outputs, scratch=None):
		x, norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
		out = outputs[0]
		op = out.op.op 

		B, T, C = x.shape
		H, HKV = op.n_head, op.n_kv_head
		D = C // H
		eps_bits = struct.unpack("<i", struct.pack("<f", op.eps))[0]
		q, k, v = scratch["q"], scratch["k"], scratch["v"]
		inp_vars = [x.storage, wq.storage, bq.storage, wk.storage, bk.storage, wv.storage, bv.storage, q, k, v, norm_weight.storage]

		tm, tn = (B * T + 63) // 64, (C + 63) // 64 + 2 * ((HKV * D + 63) // 64)
		sw_log = 3 if (tm >= 8 and tn >= 8) else (2 if (tm >= 4 and tn >= 4) else (1 if (tm >= 2 and tn >= 4) else 0))
		sw = 1 << sw_log
		self.ctx.dispatch_with_args_3d("gqa_rms_qkv",
			[e.ptr.buffer_id for e in inp_vars], [e.ptr.offset for e in inp_vars],
			[B, T, C, H, eps_bits, HKV], 128 * tn * sw, (tm + sw - 1) // sw, 1, 128)

		negative_sin = int(op.rope_sign < 0)
		inp_vars2 = [q, rope_cos.storage, rope_sin.storage]
		self.ctx.dispatch_with_args_3d("gqa_rope_inplace",
			[e.ptr.buffer_id for e in inp_vars2], [e.ptr.offset for e in inp_vars2],
			[B, H, T, D, 0, negative_sin], D//2, T, B*H, D//2)
		inp_vars3 = [k, rope_cos.storage, rope_sin.storage]
		self.ctx.dispatch_with_args_3d("gqa_rope_inplace",
			[e.ptr.buffer_id for e in inp_vars3], [e.ptr.offset for e in inp_vars3],
			[B, HKV, T, D, 0, negative_sin], D//2, T, B*HKV, D//2)
		fa_kernel = "gqa_from_qkv_d64" if D <= 64 else "gqa_from_qkv_d128"
		self.ctx.dispatch_with_args_3d(fa_kernel,
			[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out.storage.ptr.buffer_id, q.ptr.buffer_id],
			[q.ptr.offset, k.ptr.offset, v.ptr.offset, out.storage.ptr.offset, q.ptr.offset],
			[B, H, T, C, int(op.is_causal), 0, HKV], 128, (T+7)//8, B*H, 128)

	def metal_fused_rms_gqa_backward(self, inputs, outputs, scratch=None):
		x, norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
		out = outputs[0]
		if out.grad is None:
			return
		op = out.op.op
		B, T, C = x.shape
		H, HKV = op.n_head, op.n_kv_head
		D = C // H
		eps_bits = struct.unpack("<i", struct.pack("<f", op.eps))[0]
		N, CKV = B*T, HKV*D

		accumulate = int(bool(getattr(self, 'accumulate_parameter_grads', False)))
		q, k, v = scratch['q'], scratch['k'], scratch['v']
		dq, dk, dv = scratch['dq'], scratch['dk'], scratch['dv']
		lse, d_val = scratch['lse'], scratch['d_val']

		# 1. 重算 QKV 以及 RoPE
		tm, tn = (N + 63) // 64, (C + 63) // 64 + 2 * ((CKV + 63) // 64)
		sw_log = 3 if (tm >= 8 and tn >= 8) else (2 if (tm >= 4 and tn >= 4) else (1 if (tm >= 2 and tn >= 4) else 0))
		sw = 1 << sw_log
		self.ctx.dispatch_with_args_3d('gqa_rms_qkv',
		    [x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id, wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id, q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, norm_weight.storage.ptr.buffer_id],
		    [x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset, wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset, q.ptr.offset, k.ptr.offset, v.ptr.offset, norm_weight.storage.ptr.offset],
		    [B, T, C, H, eps_bits, HKV], 128 * tn * sw, (tm + sw - 1) // sw, 1, 128)
		negative_sin = int(op.rope_sign < 0)
		self.ctx.dispatch_with_args_3d('gqa_rope_inplace',
		    [q.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
		    [q.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
		    [B, H, T, D, 0, negative_sin], D//2, T, B*H, D//2)
		self.ctx.dispatch_with_args_3d('gqa_rope_inplace',
		    [k.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
		    [k.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
		    [B, HKV, T, D, 0, negative_sin], D//2, T, B*HKV, D//2)

		# 2. 重算 O/LSE：O 暂存到完整大小的 dq，不能复用较小的 dk。
		fa_kernel = "gqa_from_qkv_d64" if D <= 64 else "gqa_from_qkv_d128"
		self.ctx.dispatch_with_args_3d(fa_kernel,
		    [q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, dq.ptr.buffer_id, lse.ptr.buffer_id],
		    [q.ptr.offset, k.ptr.offset, v.ptr.offset, dq.ptr.offset, lse.ptr.offset],
		    [B, H, T, C, int(op.is_causal), 1, HKV], 128, (T+7)//8, B*H, 128)
		self.ctx.dispatch_with_args_3d('fused_fa_bwd_prepare',
		    [dq.ptr.buffer_id, out.grad.storage.ptr.buffer_id, d_val.ptr.buffer_id],
		    [dq.ptr.offset, out.grad.storage.ptr.offset, d_val.ptr.offset],
		    [B, H, T, C], 128, (T+3)//4, B*H, 128)

		# 3. O 已被消费，dq 可以写入 dQ。
		self.ctx.dispatch_with_args_3d('gqa_bwd_dq',
		    [q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out.grad.storage.ptr.buffer_id, dq.ptr.buffer_id, lse.ptr.buffer_id, d_val.ptr.buffer_id],
		    [q.ptr.offset, k.ptr.offset, v.ptr.offset, out.grad.storage.ptr.offset, dq.ptr.offset, lse.ptr.offset, d_val.ptr.offset],
		    [B, H, T, C, int(op.is_causal), HKV], 128*((D+63)//64), (T+31)//32, B*H, 128)

		# 4. 每个 KV 头/key tile 的 threadgroup 内累加所属 Q 头，直接写紧凑 dK/dV。
		self.ctx.dispatch_with_args_3d('gqa_bwd_dkv',
		    [q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out.grad.storage.ptr.buffer_id, dk.ptr.buffer_id, dv.ptr.buffer_id, lse.ptr.buffer_id, d_val.ptr.buffer_id],
		    [q.ptr.offset, k.ptr.offset, v.ptr.offset, out.grad.storage.ptr.offset, dk.ptr.offset, dv.ptr.offset, lse.ptr.offset, d_val.ptr.offset],
		    # [B, H, T, C, int(op.is_causal), HKV], 128, (T+31)//32, B*HKV, 128)
		    [B, H, T, C, int(op.is_causal), HKV], 128*((D+63)//64), (T+31)//32, B*HKV, 128)
		
		negative_sin = int(op.rope_sign > 0)
		self.ctx.dispatch_with_args_3d('gqa_rope_inplace',
		    [dq.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
		    [dq.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
		    [B, H, T, D, 1, negative_sin], D//2, T, B*H, D//2)
		self.ctx.dispatch_with_args_3d('gqa_rope_inplace',
		    [dk.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
		    [dk.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
		    [B, HKV, T, D, 1, negative_sin], D//2, T, B*HKV, D//2)

		# 5. attention 反向结束，q 复用为带权 Z，lse 复用为 inv_rms。
		self.ctx.dispatch_with_args_3d('gqa_rms_recompute',
		    [x.storage.ptr.buffer_id, norm_weight.storage.ptr.buffer_id, q.ptr.buffer_id, lse.ptr.buffer_id],
		    [x.storage.ptr.offset, norm_weight.storage.ptr.offset, q.ptr.offset, lse.ptr.offset],
		    [N, C, eps_bits], 128, (N+3)//4, 1, 128)

		for weight, dproj, cols in ((wq, dq, C), (wk, dk, CKV), (wv, dv, CKV)):
			if not weight.requires_grad:
				continue
			dw_bf16 = np.dtype(weight.grad.storage.dtype) == np.dtype(np.uint16)
			if dw_bf16:
				dw_capacity = (v.ptr.offset + v.nbytes - k.ptr.offset) // 4
				dw_tile_rows = min(C, dw_capacity // cols)

				if N <= 32 or dw_tile_rows == 0:
					# [N, V] = Z, [N, C_kernel] = dproj
					self.ctx.dispatch_with_args_3d("fused_rmsnorm_linear_ce_dw_bf16",
						[q.ptr.buffer_id, dproj.ptr.buffer_id, weight.grad.storage.ptr.buffer_id],
						[q.ptr.offset, dproj.ptr.offset, weight.grad.storage.ptr.offset],
						[N, cols, C, accumulate], cols, C, 1, min(cols, 256))
					continue

				for row0 in range(0, C, dw_tile_rows):
					tile_rows = min(dw_tile_rows, C - row0)
					self.ctx.dispatch_gemm_mps([q.ptr.buffer_id, dproj.ptr.buffer_id, k.ptr.buffer_id],
						[q.ptr.offset + row0*4, dproj.ptr.offset, k.ptr.offset],
						[1, tile_rows, cols, N, 1, 0, 0, 0, 0, C*4], 1.0, 0.0)
					self.ctx.dispatch_with_args_3d("fp32_to_bf16",
						[k.ptr.buffer_id, weight.grad.storage.ptr.buffer_id],
						[k.ptr.offset, weight.grad.storage.ptr.offset + row0*cols*2],
						[accumulate], tile_rows*cols, 1, 1, min(tile_rows*cols, 256))
			else:
				self.ctx.dispatch_gemm_mps([q.ptr.buffer_id, dproj.ptr.buffer_id, weight.grad.storage.ptr.buffer_id],
					[q.ptr.offset, dproj.ptr.offset, weight.grad.storage.ptr.offset],
					[1, C, cols, N, 1, 0, 0, 0], 1.0, float(accumulate))

		for bias, dproj, cols in ((bq, dq, C), (bk, dk, CKV), (bv, dv, CKV)):
			if not bias.requires_grad:
				continue
			db_bf16 = np.dtype(bias.grad.storage.dtype) == np.dtype(np.uint16)
			bias_kernel = "bias_grad_reduce_bf16" if db_bf16 else "bias_grad_reduce"
			self.ctx.dispatch_with_args_3d(bias_kernel,
				[dproj.ptr.buffer_id, bias.grad.storage.ptr.buffer_id],
				[dproj.ptr.offset, bias.grad.storage.ptr.offset],
				[N, cols, accumulate], 256, (cols+15)//16, 1, 256)

		# 6. 所有 dW 完成后，q 的同一空间继续复用为 dZ。
		# 即使 x 冻结，也要为 norm_weight 计算 dZ。
		if x.requires_grad or norm_weight.requires_grad:
		    self.ctx.dispatch_gemm_mps(
		        [dq.ptr.buffer_id, wq.storage.ptr.buffer_id, q.ptr.buffer_id],
		        [dq.ptr.offset, wq.storage.ptr.offset, q.ptr.offset],
		        [1, N, C, C, 0, 1, 0, 0], 1.0, 0.0)
		    self.ctx.dispatch_gemm_mps(
		        [dk.ptr.buffer_id, wk.storage.ptr.buffer_id, q.ptr.buffer_id],
		        [dk.ptr.offset, wk.storage.ptr.offset, q.ptr.offset],
		        [1, N, C, CKV, 0, 1, 0, 0], 1.0, 1.0)
		    self.ctx.dispatch_gemm_mps(
		        [dv.ptr.buffer_id, wv.storage.ptr.buffer_id, q.ptr.buffer_id],
		        [dv.ptr.offset, wv.storage.ptr.offset, q.ptr.offset],
		        [1, N, C, CKV, 0, 1, 0, 0], 1.0, 1.0)
		    if norm_weight.requires_grad:
		    	dnorm_weight_bf16 = np.dtype(norm_weight.grad.storage.dtype) == np.dtype(np.uint16)
		    	if dnorm_weight_bf16:
		    		self.ctx.dispatch_with_args_3d("gqa_rms_dgamma_bf16",
		    			[x.storage.ptr.buffer_id, q.ptr.buffer_id, lse.ptr.buffer_id, norm_weight.grad.storage.ptr.buffer_id, k.ptr.buffer_id],
		    			[x.storage.ptr.offset, q.ptr.offset, lse.ptr.offset, norm_weight.grad.storage.ptr.offset, k.ptr.offset],
		    			[N, C, accumulate, 1, 1], C, 1, 1, 256)
		    	else:
			        self.ctx.dispatch_with_args_3d('gqa_rms_dgamma',
			            [x.storage.ptr.buffer_id, q.ptr.buffer_id, lse.ptr.buffer_id, norm_weight.grad.storage.ptr.buffer_id],
			            [x.storage.ptr.offset, q.ptr.offset, lse.ptr.offset, norm_weight.grad.storage.ptr.offset],
			            [N, C, accumulate], C, 1, 1, 256)
		    if x.requires_grad:
		        self.ctx.dispatch_with_args_3d('gqa_rms_dx',
		            [x.storage.ptr.buffer_id, q.ptr.buffer_id, norm_weight.storage.ptr.buffer_id, lse.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
		            [x.storage.ptr.offset, q.ptr.offset, norm_weight.storage.ptr.offset, lse.ptr.offset, x.grad.storage.ptr.offset],
		            [N, C], 32, N, 1, 32)

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
			[B, T, C, H, eps_bits], 128, (B*T + 31) // 32, 1, 128)

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

	def metal_recompute_fused_rms_gqa(self, act, scratch, out_buf):
		inputs = act.op.inputs
		op = act.op.op
		x, norm_weight, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = inputs
		B, T, C = x.shape
		H, HKV = op.n_head, op.n_kv_head
		D = C // H
		eps_bits = struct.unpack('<i', struct.pack('<f', op.eps))[0]
		q, k, v = scratch['fa_q'], scratch['fa_k'], scratch['fa_v']
		tm, tn = (B * T + 63) // 64, (C + 63) // 64 + 2 * ((HKV * D + 63) // 64)
		sw_log = 3 if (tm >= 8 and tn >= 8) else (2 if (tm >= 4 and tn >= 4) else (1 if (tm >= 2 and tn >= 4) else 0))
		sw = 1 << sw_log
		self.ctx.dispatch_with_args_3d('gqa_rms_qkv',
		    [x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id, wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id, q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, norm_weight.storage.ptr.buffer_id],
		    [x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset, wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset, q.ptr.offset, k.ptr.offset, v.ptr.offset, norm_weight.storage.ptr.offset],
		    [B, T, C, H, eps_bits, HKV], 128 * tn * sw, (tm + sw - 1) // sw, 1, 128)
		negative_sin = int(op.rope_sign < 0)
		self.ctx.dispatch_with_args_3d('gqa_rope_inplace',
		    [q.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
		    [q.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
		    [B, H, T, D, 0, negative_sin], D//2, T, B*H, D//2)
		self.ctx.dispatch_with_args_3d('gqa_rope_inplace',
		    [k.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
		    [k.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
		    [B, HKV, T, D, 0, negative_sin], D//2, T, B*HKV, D//2)
		fa_kernel = "gqa_from_qkv_d64" if D <= 64 else "gqa_from_qkv_d128"
		self.ctx.dispatch_with_args_3d(fa_kernel,
		    [q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out_buf.ptr.buffer_id, q.ptr.buffer_id],
		    [q.ptr.offset, k.ptr.offset, v.ptr.offset, out_buf.ptr.offset, q.ptr.offset],
		    [B, H, T, C, int(op.is_causal), 0, HKV], 128, (T+7)//8, B*H, 128)

	def metal_fused_linear_add_forward(self, inputs, outputs, scratch=None):
	    x, act, w, b = inputs
	    out = outputs[0]
	    M = prod(x.shape[:-1])
	    K = act.shape[-1]
	    N = w.shape[-1]

	    if not self._rollout_cache_dirty and out.grad is None and M <= 16 and self._rollout_cache is not None:
	        c_w = self._rollout_cache.get(id(w))
	        if c_w is not None:
	            b_id = b.storage.ptr.buffer_id if b is not None else 0
	            b_off = b.storage.ptr.offset if b is not None else 0
	            self._dispatch_decode_linear(
	                act.storage.ptr.buffer_id, act.storage.ptr.offset,
	                c_w[0], c_w[1], c_w[2], out.storage.ptr.buffer_id, out.storage.ptr.offset,
	                M, N, K, flags=8 | (4 if b is not None else 0),
	                bias_id=b_id, bias_off=b_off,
	                res_id=x.storage.ptr.buffer_id, res_off=x.storage.ptr.offset)
	            return

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
	    M_total = prod(x.shape[:-1])
	    K_dim = act.shape[-1]
	    N_dim = w.shape[-1]
	    accumulate = int(bool(getattr(self, "accumulate_parameter_grads", False)))

	    dw_bf16 = w.requires_grad and np.dtype(w.grad.storage.dtype) == np.dtype(np.uint16)
	    db_bf16 = b.requires_grad and np.dtype(b.grad.storage.dtype) == np.dtype(np.uint16)

	    if act.op is not None and act.op.op.op_name == "FusedRMSNormFlashAttention":
	    	cached = getattr(self, "_recomputed_attention_cache", None)
	    	if cached is not None and id(act) in cached:
	    		cached.remove(id(act))
	    		act_buf = act.grad.storage
	    	else:
	    		act_buf = act.grad.storage
	    		self.metal_recompute_fused_rms_fa(act, scratch, act_buf)
	    elif act.op is not None and act.op.op.op_name == "FusedRMSNormGQA":
	    	cached = getattr(self, "_recomputed_attention_cache", None)
	    	if cached is not None and id(act) in cached:
	    		cached.remove(id(act))
	    		act_buf = act.grad.storage
	    	else:
	    		act_buf = scratch["fa_out"]
	    		self.metal_recompute_fused_rms_gqa(act,scratch,act_buf)
	    else:
	        act_buf = act.storage
	    if w.requires_grad:
	        dw_cfg = get_fp16_gemm_config(M=K_dim, N=N_dim, K=M_total, with_accum=bool(accumulate), a_transpose=True, b_transpose=False, dtype="fp32")
	        if dw_bf16:
	            if not dw_cfg["is_splitk"]:
	                tg = dw_cfg["wm"] * dw_cfg["wn"] * 32
	                dw_flags = accumulate | (dw_cfg["swizzle_log"] << 2) | (dw_cfg["bm"] << 8)
	                self.ctx.dispatch_with_args_3d("fused_rmsnorm_linear_ce_dw_bf16",
	                    [act_buf.ptr.buffer_id, out.grad.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
	                    [act_buf.ptr.offset, out.grad.storage.ptr.offset, w.grad.storage.ptr.offset],
	                    [M_total, N_dim, K_dim, dw_flags], dw_cfg["grid_x"] * tg, dw_cfg["grid_y"], dw_cfg["grid_z"], tg)
	            else:
	                self.ctx.dispatch_with_args_3d("fused_rmsnorm_linear_ce_dw_bf16",
	                    [act_buf.ptr.buffer_id, out.grad.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
	                    [act_buf.ptr.offset, out.grad.storage.ptr.offset, w.grad.storage.ptr.offset],
	                    [M_total, N_dim, K_dim, accumulate], N_dim, K_dim, 1, min(N_dim, 256))
	        else:
	            self._dispatch_fp32_gemm(dw_cfg, bool(accumulate),
	                [act_buf.ptr.buffer_id, out.grad.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
	                [act_buf.ptr.offset, out.grad.storage.ptr.offset, w.grad.storage.ptr.offset],
	                K_dim, N_dim, M_total)
	    if x.requires_grad:
	        numel = prod(x.shape)
	        og, xg = out.grad.storage, x.grad.storage
	        if og.ptr.buffer_id != xg.ptr.buffer_id or og.ptr.offset != xg.ptr.offset:
	            if x.is_contiguous:
	                if numel % 4 == 0 and og.ptr.offset % 16 == 0 and xg.ptr.offset % 16 == 0:
	                    self.ctx.dispatch_with_args_3d("add_residual_backward_vec4",
	                        [og.ptr.buffer_id, xg.ptr.buffer_id, 0],
	                        [og.ptr.offset, xg.ptr.offset, 0],
	                        [], numel // 4, 1, 1, min(numel // 4, 256))
	                else:
	                    self.ctx.dispatch_with_args_3d("contiguous_backward_kernel",
	                        [og.ptr.buffer_id, xg.ptr.buffer_id],
	                        [og.ptr.offset, xg.ptr.offset],
	                        [], numel, 1, 1, min(numel, 256))
	            else:
	                self.ctx.dispatch_strided("strided_add_backward_kernel",
	                    og.ptr.buffer_id, og.ptr.offset,
	                    xg.ptr.buffer_id, xg.ptr.offset,
	                    list(x.shape), x.strides, numel)
	    if act.requires_grad:
	        dact_cfg = get_fp16_gemm_config(M=M_total, N=K_dim, K=N_dim, with_accum=False, a_transpose=False, b_transpose=True, dtype="fp32")
	        self._dispatch_fp32_gemm(dact_cfg, False,
	            [out.grad.storage.ptr.buffer_id, w.storage.ptr.buffer_id, act.grad.storage.ptr.buffer_id],
	            [out.grad.storage.ptr.offset, w.storage.ptr.offset, act.grad.storage.ptr.offset],
	            M_total, K_dim, N_dim)

	    if b.requires_grad:
	        bias_kernel = "bias_grad_reduce_bf16" if db_bf16 else "bias_grad_reduce"
	        self.ctx.dispatch_with_args_3d(bias_kernel,
	        	[out.grad.storage.ptr.buffer_id, b.grad.storage.ptr.buffer_id],
	        	[out.grad.storage.ptr.offset, b.grad.storage.ptr.offset], 
	        	[prod(x.shape[:-1]), N_dim, accumulate], 256, (N_dim+15)//16, 1, 256)

	def metal_kv_cache_attention_forward(self, inputs, outputs, scratch=None):
		x, wq, bq, wk, bk, wv, bv, k_cache, v_cache, position, rope_cos, rope_sin = inputs

		out = outputs[0]
		B, T, C = x.shape
		H = out.op.op.n_head
		D = C // H 
		S = k_cache.shape[1]
		cache_B = k_cache.shape[0]

		q = scratch["q"]
		k = scratch["k"]
		v = scratch["v"]
		eps_bits = struct.unpack("<i", struct.pack("<f", out.op.op.eps))[0]

		# RMSNorm + QKV projection
		c_qkv = (self._rollout_cache.get(id(wq)) if (not self._rollout_cache_dirty and T == 1 and B <= 16 and self._rollout_cache is not None)
		         else None)
		if c_qkv is not None:
			self._dispatch_decode_linear(
				x.storage.ptr.buffer_id, x.storage.ptr.offset,
				c_qkv[0], c_qkv[1], c_qkv[2], q.ptr.buffer_id, q.ptr.offset,
				B, 3 * C, C, flags=1 | (4 if c_qkv[3] >= 0 else 0) | 16,
				eps_bits=f32_bits(out.op.op.eps),
				bias_id=c_qkv[0] if c_qkv[3] >= 0 else 0, bias_off=max(0, c_qkv[3]),
				out1_id=k.ptr.buffer_id, out1_off=k.ptr.offset,
				out2_id=v.ptr.buffer_id, out2_off=v.ptr.offset, c_out=C)
		else:
			self.ctx.dispatch_with_args_3d("fused_rms_fa_qkv",
				[x.storage.ptr.buffer_id, wq.storage.ptr.buffer_id, bq.storage.ptr.buffer_id,
				wk.storage.ptr.buffer_id, bk.storage.ptr.buffer_id, wv.storage.ptr.buffer_id, bv.storage.ptr.buffer_id,
				q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id],
				[x.storage.ptr.offset, wq.storage.ptr.offset, bq.storage.ptr.offset,
				wk.storage.ptr.offset, bk.storage.ptr.offset, wv.storage.ptr.offset, bv.storage.ptr.offset,
				q.ptr.offset, k.ptr.offset, v.ptr.offset],
				[B, T, C, H, eps_bits], 256, (B*T+15)//16, 1, 256)

		# Q/K RoPE
		row_positions = out.op.op.mode == "decode" and position.shape != (1,)
		if out.op.op.mode == "prefill" and cache_B > B:
			rope_name = "kv_cache_rope_store_broadcast"
			half_dim = D // 2
			source_work_items = H * T * half_dim
			inp_vars = [q, k, v, k_cache.storage, v_cache.storage, position.storage, rope_cos.storage, rope_sin.storage]
			self.ctx.dispatch_with_args_3d("kv_cache_rope_store_broadcast",
				[e.ptr.buffer_id for e in inp_vars],
				[e.ptr.offset for e in inp_vars],
				[B, H, T, D, S, cache_B], 128, (source_work_items + 127) // 128, 1, 128)
		else:
			rope_name = "kv_cache_rope_store_rows" if row_positions else "kv_cache_rope_store"
			self.ctx.dispatch_with_args_3d(rope_name,
				[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, k_cache.storage.ptr.buffer_id, v_cache.storage.ptr.buffer_id,
					position.storage.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
				[q.ptr.offset, k.ptr.offset, v.ptr.offset, k_cache.storage.ptr.offset, v_cache.storage.ptr.offset,
					position.storage.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
				[B, H, T, D, S], 128, (B*H*T*(D//2)+127)//128, 1, 128)

		if out.op.op.mode == "prefill":
			name, gx, gy, threads = self.fa_from_qkv_dispatch(D, T)
			self.ctx.dispatch_with_args_3d(name,
				[q.ptr.buffer_id, k.ptr.buffer_id, v.ptr.buffer_id, out.storage.ptr.buffer_id, q.ptr.buffer_id],
				[q.ptr.offset, k.ptr.offset, v.ptr.offset, out.storage.ptr.offset, q.ptr.offset],
				[B, H, T, C, 1, 0], gx, gy, B*H, threads)
		else:
			if row_positions:
				name = "kv_cache_fa_forward_rows_d64" if D <= 64 else "kv_cache_fa_forward_rows_d128"
			else:
				name = "kv_cache_fa_forward_d64" if D <= 64 else "kv_cache_fa_forward_d128"
			self.ctx.dispatch_with_args_3d(name,
				[q.ptr.buffer_id, k_cache.storage.ptr.buffer_id, v_cache.storage.ptr.buffer_id, position.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
				[q.ptr.offset, k_cache.storage.ptr.offset, v_cache.storage.ptr.offset, position.storage.ptr.offset, out.storage.ptr.offset],
				[B, H, T, C, S], 128, T, B*H, 128)

	def metal_kv_cache_attention_backward(self, inputs, outputs, scratch=None):
		return

	def metal_lm_head_forward(self, inputs, outputs, scratch=None):
		x, weight = inputs 
		out = outputs[0]

		M = prod(x.shape[:-1])
		K = x.shape[-1]
		N = weight.shape[0]

		if not self._rollout_cache_dirty and out.grad is None and M <= 16 and self._rollout_cache is not None:
			c_lm = self._rollout_cache.get(id(weight))
			if c_lm is not None:
				self._dispatch_decode_linear(
					x.storage.ptr.buffer_id, x.storage.ptr.offset,
					c_lm[0], c_lm[1], c_lm[2], out.storage.ptr.buffer_id, out.storage.ptr.offset,
					M, N, K)
				return

		self.ctx.dispatch_gemm_mps([x.storage.ptr.buffer_id, weight.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[x.storage.ptr.offset, weight.storage.ptr.offset, out.storage.ptr.offset],
			[1, M, N, K, 0, 1, 0, 0], 1.0, 0.0)

	def metal_last_position_lm_head_forward(self, inputs, outputs, scratch=None):
		x, weight, position = inputs
		out = outputs[0]

		row = int(self.numpy_view(position)[0])
		T = x.shape[1]
		C = x.shape[-1]
		V = weight.shape[0]

		x_offset = x.storage.ptr.offset + row * C * 4
		out_offset = out.storage.ptr.offset

		if not self._rollout_cache_dirty and out.grad is None and self._rollout_cache is not None:
			c_lm = self._rollout_cache.get(id(weight))
			if c_lm is not None:
				self._dispatch_decode_linear(
					x.storage.ptr.buffer_id, x_offset,
					c_lm[0], c_lm[1], c_lm[2], out.storage.ptr.buffer_id, out_offset,
					1, V, C)
				return

		self.ctx.dispatch_gemm_mps([x.storage.ptr.buffer_id, weight.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[x_offset, weight.storage.ptr.offset, out_offset],
			[1, 1, V, C, 0, 1, 0, 0], 1.0, 0.0)

	def metal_rollout_topk_forward(self, inputs, outputs, scratch=None):
		logits, top_k, logits_position = inputs
		topk_values, topk_indices = outputs

		op = topk_values.op.op

		batch = logits.shape[0]
		vocab = logits.shape[1]

		if logits.ndim == 2:
			batch, vocab = logits.shape
			time = 1
		elif logits.ndim == 3:
			batch, time, vocab = logits.shape

		k_max = topk_values.shape[-1]

		self.ctx.dispatch_with_args_3d("rollout_topk_select",
			[logits.storage.ptr.buffer_id, topk_values.storage.ptr.buffer_id, topk_indices.storage.ptr.buffer_id, 
				top_k.storage.ptr.buffer_id, logits_position.storage.ptr.buffer_id],
			[logits.storage.ptr.offset, topk_values.storage.ptr.offset, topk_indices.storage.ptr.offset, 
				top_k.storage.ptr.offset, logits_position.storage.ptr.offset],
			[batch, vocab, k_max, op.mode_id, time], batch*128, 1, 1, 128)

	def metal_rollout_sample_forward(self, inputs, outputs, scratch=None):
		topk_values, topk_indices, rng_state, rng_raw, finished, temperature, top_k, next_tokens, generated_ids, lengths, position, step, logits = inputs
		finished_all = outputs[0]
		op = finished_all.op.op 

		B = next_tokens.shape[0]
		K = topk_values.shape[1]
		max_tokens = generated_ids.shape[1]
		V = logits.shape[-1]
		T = logits.shape[1] if logits.ndim == 3 else 1
		values, indices = scratch["sort_values"], scratch["sort_indices"]
		P = values.shape[1]

		self.ctx.dispatch_with_args_3d("rollout_sample_from_topk",
			[topk_values.storage.ptr.buffer_id, topk_indices.storage.ptr.buffer_id, rng_state.storage.ptr.buffer_id, rng_raw.storage.ptr.buffer_id,
				finished.storage.ptr.buffer_id, next_tokens.storage.ptr.buffer_id, generated_ids.storage.ptr.buffer_id,
				lengths.storage.ptr.buffer_id, position.storage.ptr.buffer_id, step.storage.ptr.buffer_id,
				finished_all.storage.ptr.buffer_id, temperature.storage.ptr.buffer_id, top_k.storage.ptr.buffer_id,
				logits.storage.ptr.buffer_id, position.storage.ptr.buffer_id, values.ptr.buffer_id, indices.ptr.buffer_id],
			[topk_values.storage.ptr.offset, topk_indices.storage.ptr.offset, rng_state.storage.ptr.offset, rng_raw.storage.ptr.offset,
				finished.storage.ptr.offset, next_tokens.storage.ptr.offset, generated_ids.storage.ptr.offset,
				lengths.storage.ptr.offset, position.storage.ptr.offset, step.storage.ptr.offset + op.step_offset,
				finished_all.storage.ptr.offset, temperature.storage.ptr.offset, top_k.storage.ptr.offset,
				logits.storage.ptr.offset, position.storage.ptr.offset, values.ptr.offset, indices.ptr.offset],
			[B, K, op.mode_id, max_tokens, op.eos_token, op.bos_token, V, T, P], B*256, 1, 1, 256)

	def metal_decode_state_forward(self, inputs, outputs, scratch=None):
		_, tokens, masks, metadata, finished, config, positions, steps, candidates, next_tokens, global_step = inputs
		out = outputs[0]

		op = out.op.op
		B, S = tokens.shape
		inp_vars = [tokens.storage, masks.storage, metadata.storage, finished.storage, config.storage, positions.storage,
		            steps.storage, candidates.storage, next_tokens.storage, global_step.storage, out.storage]
		self.ctx.dispatch_with_args_3d("rollout_decode_state",
			[e.ptr.buffer_id for e in inp_vars], [e.ptr.offset for e in inp_vars],
			[B, S, op.phase, op.offset], B, 1, 1, min(B, 256))

	def metal_decode_finished_forward(self, inputs, outputs, scratch=None):
		next_ids, finished, step, config = inputs
		out = outputs[0]

		B = finished.shape[0]

		self.ctx.dispatch_with_args_3d("rollout_decode_finished",
			[finished.storage.ptr.buffer_id, step.storage.ptr.buffer_id, config.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
			[finished.storage.ptr.offset, step.storage.ptr.offset, config.storage.ptr.offset, out.storage.ptr.offset],
			[B], 1, 1, 1, 1)

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

	def fill_storage2(self, storage, value):
		dtype = np.dtype(storage.dtype)
		value_bits = bf16_bits(value) if dtype == np.dtype(np.uint16) else f32_bits(value)
		self.ctx.fill_scalar2("fill_scalar_kernel2",
			[storage.ptr.buffer_id], [storage.ptr.offset], value_bits, dtype.itemsize, prod(storage.shape))

	def numpy_view(self, tensor):
		buffer_id = tensor.storage.ptr.buffer_id
		offset = tensor.storage.ptr.offset
		view_func = self.numpy_view_funcs.get(np.dtype(tensor.dtype))
		if view_func is None:
			raise RuntimeError(f"Unsupported dtype from numpy_view: {tensor.dtype}")
		return view_func(buffer_id, offset, tuple(tensor.shape))

	def copy_from_host(self, tensor, host_np):
		if tensor.requires_grad or (tensor.ndim >= 2 and np.dtype(tensor.dtype) == np.dtype(np.float32)):
			self._rollout_cache_dirty = True

		if np.dtype(tensor.storage.dtype) == np.dtype(np.uint16):
			dst = self.ctx.numpy_view_uint8(tensor.storage.ptr.buffer_id, tensor.storage.ptr.offset, (prod(tensor.shape)*2,)).view(np.uint16).reshape(tuple(tensor.shape))
			values = np.asarray(host_np, dtype=np.float32)
			bits = values.view(np.uint32)
			# FP32 -> BF16
			rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
			special = ((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000))
			nan = special & ((bits & np.uint32(0x007FFFFF)) != np.uint32(0))
			special_bits = ((bits >> np.uint32(16)) | np.where(nan, np.uint32(0x0040), np.uint32(0)))
			dst[...] = np.where(special, special_bits, rounded).astype(np.uint16)
			return

		view = self.numpy_view(tensor)
		view[:] = host_np

	def copy_to_host(self, tensor):
		if np.dtype(tensor.storage.dtype) == np.dtype(np.uint16):
			src = self.ctx.numpy_view_uint8(tensor.storage.ptr.buffer_id, tensor.storage.ptr.offset, (prod(tensor.shape)*2,)).view(np.uint16).reshape(tuple(tensor.shape))
			bits = src.astype(np.uint32)
			bits <<= np.uint32(16)
			return bits.view(np.float32)

		return np.array(self.numpy_view(tensor), copy=True)

	def grad_norm_sq(self, p_storage, partial, numel):
		grad_dtype = np.dtype(p_storage.dtype)
		n_groups = (numel + 255) // 256
		self.ctx.dispatch_with_args_3d("grad_norm_sq",
			[p_storage.ptr.buffer_id, partial.ptr.buffer_id], [p_storage.ptr.offset, partial.ptr.offset],
			[numel, int(grad_dtype == np.dtype(np.uint16))], n_groups*256, 1, 1, 256)

	def adamw_step(self, p_storage, g_storage, m_storage, v_storage, tmp_storage,
				   lr, lr_t, b1, b2, eps, weight_decay, numel=None, decay_numel=None, 
				   grad_scale=1.0, lr_bc1=None, inv_sqrt_bc2=1.0):
		self._rollout_cache_dirty = True
		storages = [p_storage, g_storage, m_storage, v_storage]
		buffer_ids = [s.ptr.buffer_id for s in storages]
		offsets = [s.ptr.offset for s in storages]
		decay_factor = 1.0 - float(lr) * float(weight_decay)
		grad_dtype = np.dtype(g_storage.dtype)
		if numel is None:
			numel = prod(p_storage.shape)
		if decay_numel is None:
			decay_numel = numel if weight_decay != 0.0 else 0
		numel = int(numel)
		decay_numel = int(decay_numel)
		update_lr = lr_t if lr_bc1 is None else lr_bc1
		self.ctx.dispatch_with_args_3d('adamw_step',
			[p_storage.ptr.buffer_id, g_storage.ptr.buffer_id, m_storage.ptr.buffer_id, v_storage.ptr.buffer_id],
			[p_storage.ptr.offset, g_storage.ptr.offset, m_storage.ptr.offset, v_storage.ptr.offset],
			[f32_bits(decay_factor), f32_bits(update_lr), f32_bits(b1), f32_bits(b2), f32_bits(eps), numel, decay_numel, f32_bits(grad_scale), f32_bits(inv_sqrt_bc2), int(grad_dtype == np.dtype(np.uint16))],
			(numel + 3)// 4, 1, 1, min((numel+3)//4, 256))

	def adamw8bit_step(self, p, g, qm, qv, absmax_m, absmax_v, qmap_m, qmap_v,
					   lr, lr_bc1, inv_sqrt_bc2, b1, b2, eps, weight_decay, numel, decay_numel, first_step, grad_scale=1.0):
		self._rollout_cache_dirty = True
		self.ctx.dispatch_with_args_3d("adamw8bit_step",
			[p.ptr.buffer_id, g.ptr.buffer_id, qm.ptr.buffer_id, qv.ptr.buffer_id, absmax_m.ptr.buffer_id, absmax_v.ptr.buffer_id,
				qmap_m.ptr.buffer_id, qmap_v.ptr.buffer_id],
			[p.ptr.offset, g.ptr.offset, qm.ptr.offset, qv.ptr.offset, absmax_m.ptr.offset, absmax_v.ptr.offset,
				qmap_m.ptr.offset, qmap_v.ptr.offset],
			[f32_bits(1.0 - float(lr)*float(weight_decay)), f32_bits(lr_bc1), f32_bits(b1), f32_bits(b2), f32_bits(eps), 
				int(numel), int(decay_numel), int(first_step), f32_bits(inv_sqrt_bc2), int(np.dtype(g.dtype) == np.dtype(np.uint16)), f32_bits(grad_scale)],
				((numel + 255)//256)*256, 1, 1, 256)

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
			if numel % cols:
				raise ValueError("Q/K numel must be divisible by cols")
			if cols % head_cols:
				raise ValueError("cols must be divisible by head_cols")

			rows = numel // cols 
			if block_elems != rows * head_cols:
				raise ValueError("Q/K block_elems must equal rows*head_cols")

			return cols // head_cols
		return (numel + block_elems - 1) // block_elems

	def mini_update_v(self, g, v, b2, numel, block_elems, *, cols, head_cols, first_step, grad_scale=1.0):
		n_v_blocks = self._mini_v_group_count(numel, block_elems, cols, head_cols)
		grad_dtype = np.dtype(g.dtype)
		grad_bf16 = grad_dtype == np.dtype(np.uint16)
		if g.nbytes < numel * grad_dtype.itemsize or v.nbytes < n_v_blocks * 4:
			raise ValueError("AdamW-mini gradient/v storage is too small")
		self.ctx.dispatch_with_args_3d("mini_update_v",
			[g.ptr.buffer_id, v.ptr.buffer_id],
			[g.ptr.offset, v.ptr.offset],
			[f32_bits(b2), f32_bits(grad_scale), numel, block_elems, cols, head_cols, int(first_step), int(grad_bf16)],
			n_v_blocks*256, 1, 1, 256)

	def adamw_mini8bit_step(self, p, g, m, meta, v, lr, lr_bc1, inv_sqrt_bc2, b1, eps, weight_decay, 
			numel, decay_numel, block_elems, *, cols, head_cols, first_step, seed, stochastic_round, 
			transpose_m, quantized, grad_scale=1.0):
		self._rollout_cache_dirty = True
		n_v = self._mini_v_group_count(numel, block_elems, cols, head_cols)
		n_blocks = (numel + 255) // 256

		if not 0 <= decay_numel <= numel:
			raise ValueError("invalid decay_numel")
		if transpose_m and numel % cols:
			raise ValueError("Invalid transposed matrix shape")
		if meta is None:
			raise ValueError("mini optimizer requires metadata storage in the current Metal kernel interface")

		grad_bf16 = np.dtype(g.dtype) == np.dtype(np.uint16)

		storages = [p, g, m, meta, v]
		self.ctx.dispatch_with_args_3d("adamw_mini8bit_step",
			[s.ptr.buffer_id for s in storages],
			[s.ptr.offset for s in storages],
			[f32_bits(1.0 - float(lr) * float(weight_decay)), f32_bits(lr_bc1), f32_bits(inv_sqrt_bc2), f32_bits(b1), f32_bits(eps), f32_bits(grad_scale), 
				numel, decay_numel, block_elems, cols, head_cols, int(first_step), int(seed)&0xFFFFFFFF, int(stochastic_round), 
				int(transpose_m), int(quantized), int(grad_bf16)],
			n_blocks*256, 1, 1, 256)

	def synchronize(self):
		self.ctx.synchronize()

	def begin_pass(self):
		self.ctx.begin_pass()

	def end_pass(self):
		self.ctx.end_pass()