
from tensor import Tensor
from op import LayerNormOp, ScaledDotProductAttenOp, FusedMLPOp, FusedQKVProjOp, FusedRMSNormMLPOp, FlashAttentionOp
from op import FusedRMSNormFlashAttentionOp, FusedLinearAddOp, KVCacheAttentionOp, RolloutTopkOp, RolloutSampleOp
from op import LastPositionLMHeadOp, DecodeStateOp, DecodeFinishedOp
from op import FusedRMSNormGQAOp
import numpy as np

class Module:
	def parameters(self):
		params = []
		visited = set()

		def collect(v):
			if isinstance(v, Tensor) and v.requires_grad:
				if id(v) not in visited:
					visited.add(id(v))
					params.append(v)
			elif isinstance(v, Module):
				for vv in v.__dict__.values():
					collect(vv)
			elif isinstance(v, list):
				for item in v:
					collect(item)

		for v in self.__dict__.values():
			collect(v)

		return params

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

class Linear(Module):
	def __init__(self, in_f, out_f, dtype, device, bias=True):
		self.weight = Tensor(shape=(in_f, out_f), init_val="xavier", dtype=dtype, requires_grad=True, device=device, name="weight")
		self.has_bias = bias
		if bias:
			self.bias = Tensor(shape=(out_f,), init_val="zeros", dtype=dtype, requires_grad=True, device=device, name="bias")

	def forward(self, x):
		out = x @ self.weight
		if self.has_bias:
			out = out + self.bias
		return out

class LayerNorm(Module):
	def __init__(self, C, dtype, device, eps=1e-5):
		self.gamma = Tensor(shape=(C,), init_val=1, dtype=dtype, device=device, requires_grad=True, name="gamma")
		self.beta = Tensor((C,), init_val=0, dtype=dtype, device=device, requires_grad=True, name="beta")
		self.eps = eps  

	def forward(self, x):
		return LayerNormOp(eps=self.eps)(x, self.gamma, self.beta)

class RMSNorm(Module):
	def __init__(self, eps=1e-5):
		self.eps = eps 

	def forward(self, x):
		return RMSNormOp(eps=self.eps)

class Embedding(Module):
	def __init__(self, num_embeds, embed_dim, dtype, device):
		self.weight = Tensor(shape=(num_embeds, embed_dim), init_val="xavier", dtype=dtype, device=device, requires_grad=True)

	def forward(self, indices):
		return indices.gather(self.weight)

class CausalSelfAttention(Module):
	def __init__(self, C, n_head, max_T, batch_size, dtype, device):
		self.n_head = n_head
		self.head_dim = C // n_head 
		self.max_T = max_T
		self.batch_size = batch_size

		self.q_proj = Linear(C, C, dtype, device)
		self.k_proj = Linear(C, C, dtype, device)
		self.v_proj = Linear(C, C, dtype, device)
		self.proj = Linear(C, C, dtype, device)

	def forward(self, x):
		B, T, C = x.shape 
		ctx = FlashAttentionOp(n_head=self.n_head, is_causal=True)(
			x, self.q_proj.weight, self.q_proj.bias,
			self.k_proj.weight, self.k_proj.bias,
			self.v_proj.weight, self.v_proj.bias)
		return self.proj(ctx)

class MLP(Module):
	def __init__(self, C, dtype, device, H=None):
		if H is None:
			H = (8*C)//3
		self.fc1 = Linear(C, 2*H, dtype, device, bias=False)
		self.fc2 = Linear(H, C, dtype, device, bias=False)

	def forward(self, x):
		return FusedMLPOp()(x, self.fc1.weight, self.fc2.weight)

class GPT2Block(Module):
	def __init__(self, C, n_head, max_T, batch_size, dtype, device, eps=1e-5, layer_idx=-1):
		self.norm_eps = eps
		self.layer_idx = layer_idx
		self.attn = CausalSelfAttention(C, n_head, max_T, batch_size, dtype, device)
		self.mlp = MLP(C, dtype, device)

	def forward(self, x, rope_cos, rope_sin, mode="train", kv_cache=None, position=None):
		if mode == "train":
			attn_out = FusedRMSNormFlashAttentionOp(n_head=self.attn.n_head, eps=self.norm_eps, is_causal=True)(
				x, self.attn.q_proj.weight, self.attn.q_proj.bias,
				self.attn.k_proj.weight, self.attn.k_proj.bias,
				self.attn.v_proj.weight, self.attn.v_proj.bias,
				rope_cos, rope_sin)
		elif mode in ("prefill", "decode"):
			k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
			attn_out = KVCacheAttentionOp(n_head=self.attn.n_head, layer_idx=self.layer_idx, eps=self.norm_eps, mode=mode)(
				x, self.attn.q_proj.weight, self.attn.q_proj.bias,
				self.attn.k_proj.weight, self.attn.k_proj.bias,
				self.attn.v_proj.weight, self.attn.v_proj.bias,
				k_cache, v_cache, position, rope_cos, rope_sin)
		else:
			raise ValueError(f"unsupported block mode: {mode}")
		x = FusedLinearAddOp()(x, attn_out, self.attn.proj.weight, self.attn.proj.bias)

		mlp_out = FusedRMSNormMLPOp(eps=self.norm_eps)(x, self.mlp.fc1.weight, self.mlp.fc2.weight)
		x = x + mlp_out
		return x 

class LMHead(Module):
	def __init__(self, vocab_size, embed_dim, dtype, device, weight=None):
		if weight is None:
			self.weight = Tensor(shape=(vocab_size, embed_dim), init_val='xavier', 
				dtype=dtype, requires_grad=True, device=device, name='weight')
		else:
			self.weight = weight # for weight tying

	def forward(self, x):
		# return x @ self.weight.transpose(0, 1)
		from op import LMHeadOp
		return LMHeadOp()(x, self.weight)

class GPT2(Module):
	def __init__(self, V, C, max_T, n_head, batch_size, n_layer, dtype, device, rollout_max_topk=64):
		self.tok_emb = Embedding(V, C, dtype, device)
		self.rollout_max_topk = rollout_max_topk
		# RoPE
		self.rope_theta = 100000.0
		positions = np.arange(max_T, dtype=np.float32)[:, None]
		rotary_dim = C // n_head
		channels = np.arange(0, rotary_dim, 2, dtype=np.float32)[None,:]
		inv_freq = self.rope_theta ** (-channels / rotary_dim)
		freqs = positions * inv_freq
		self.rope_cos = Tensor(shape=freqs.shape, dtype=dtype, device=device, requires_grad=False,
			init_val=np.cos(freqs).astype(dtype), name="rope_cos")
		self.rope_sin = Tensor(shape=freqs.shape, dtype=dtype, device=device, requires_grad=False,
			init_val=np.sin(freqs).astype(dtype), name="rope_sin")

		self.blocks = [GPT2Block(C, n_head, max_T, batch_size, dtype, device, 1e-6, layer_idx=i)
			for i in range(n_layer)]
		self.lm_head = LMHead(V, C, dtype, device)

	def forward(self, input_ids, targets=None, mode="train", 
				kv_cache=None, position=None, top_k=None,
				temperature=None, eos_token=None, bos_token=None,
				rng_state=None, rng_raw=None, finished=None, next_tokens=None, generated_ids=None,
				lengths=None, decode_position=None, decode_step=None, row_scale=None):
		x = self.tok_emb(input_ids)
		for blk in self.blocks:
			x = blk(x, self.rope_cos, self.rope_sin, kv_cache=kv_cache, position=position, mode=mode)
		if mode == "train":
			if targets is not None:
				return x.fused_linear_cross_entropy(self.lm_head.weight, targets, row_scale=row_scale)
		if mode == "prefill":
			logits = LastPositionLMHeadOp()(x, self.lm_head.weight, decode_position)
		else:
			logits = self.lm_head(x)
		topk_values, topk_indices = RolloutTopkOp(max_top_k=self.rollout_max_topk, mode=mode)(logits, top_k, decode_position)
		return RolloutSampleOp(eos_token=eos_token, bos_token=bos_token, max_tokens=generated_ids.shape[1], mode=mode)(
			topk_values, topk_indices, rng_state, rng_raw, finished, temperature, top_k, 
			next_tokens, generated_ids, lengths, decode_position, decode_step, logits)

	def decode(self, input_ids, kv_cache, top_k, temperature, eos_token, bos_token, rng_state, rng_raw, 
		       finished, generated_ids, lengths, state, steps):
		dependency = input_ids
		for j in range(steps):
			args = (state.tokens, state.mask, state.metadata, state.finished, state.decode_config, 
				    state.decode_position, state.decode_step, state.decode_candidates, state.next_tokens, state.step)
			ids = DecodeStateOp(0, j)(dependency, *args)
			sampled = self.forward(ids, mode="decode", kv_cache=kv_cache, position=state.decode_position,
				                   top_k=top_k, temperature=temperature, eos_token=eos_token, bos_token=bos_token,
				                   rng_state=rng_state, rng_raw=rng_raw, finished=finished, next_tokens=state.decode_candidates,
				                   generated_ids=generated_ids, lengths=lengths, decode_position=state.position,
				                   decode_step=state.decode_step)
			dependency = DecodeStateOp(1, j)(sampled, *args)

		return DecodeFinishedOp()(dependency, finished, state.decode_step, state.decode_config)

class QwenBlock(Module):
	def __init__(self, C, H, n_head, n_kv_head, dtype, device, eps=1e-6, layer_idx=-1):
		self.n_head = n_head
		self.n_kv_head = n_kv_head
		self.norm_eps = eps
		self.layer_idx = layer_idx

		D = C // n_head
		kv_dim = n_kv_head * D

		self.q_proj = Linear(C, C, dtype, device)
		self.k_proj = Linear(C, kv_dim, dtype, device)
		self.v_proj = Linear(C, kv_dim, dtype, device)
		self.proj = Linear(C, C, dtype, device, bias=False)

		self.input_norm_weight = Tensor(shape=(C,), dtype=dtype, device=device, requires_grad=True, 
				init_val=1.0, name="input_layernorm.weight")
		self.post_norm_weight = Tensor(shape=(C,), dtype=dtype, device=device, requires_grad=True, 
				init_val=1.0, name="post_attention_layernorm.weight")
		self.mlp = MLP(C, dtype, device, H=H)
		self.proj_zero_bias = Tensor(shape=(C,), dtype=dtype, device=device, requires_grad=False,
				init_val=0.0, name="proj_zero_bias")

	def forward(self, x, rope_cos, rope_sin):
		attn_out = FusedRMSNormGQAOp(n_head=self.n_head, n_kv_head=self.n_kv_head, eps=self.norm_eps, is_causal=True, rope_sign=1)(
				x, self.input_norm_weight, self.q_proj.weight, self.q_proj.bias, self.k_proj.weight, self.k_proj.bias,
				self.v_proj.weight, self.v_proj.bias, rope_cos, rope_sin)
		x = FusedLinearAddOp()(x, attn_out, self.proj.weight, self.proj_zero_bias)
		mlp_out = FusedRMSNormMLPOp(eps=self.norm_eps)(x, self.post_norm_weight, self.mlp.fc1.weight, self.mlp.fc2.weight)
		return x + mlp_out

class QwenCoder(Module):
	def __init__(self, V=151936, C=1536, max_T=32768, n_head=12, n_layer=28, dtype=np.float32, 
				 device="mps", H=8690, n_kv_head=2, eps=1e-6, rope_theta=1000000.0):
		if C % n_head != 0 or C % 4 != 0:
			raise ValueError("require C % n_head == 0 and C % 4 == 0")
		if n_head % n_kv_head != 0:
			raise ValueError("n_head must be divisible by n_kv_head")

		D = C // n_head
		if D % 2 != 0 or D > 128:
			raise ValueError("current GQA Metal kernels require even head_dim <= 128")

		self.max_T = max_T
		self.norm_eps = eps
		self.rope_theta = rope_theta

		self.tok_emb = Embedding(V, C, dtype, device)

		positions = np.arange(max_T, dtype=np.float32)[:, None]
		channels = np.arange(0, D, 2, dtype=np.float32)[None, :]
		inv_freq = np.float32(rope_theta) ** (-channels/np.float32(D))
		freqs = positions * inv_freq

		self.rope_cos = Tensor(shape=freqs.shape, dtype=dtype, device=device, requires_grad=False, 
			                   init_val=np.cos(freqs).astype(dtype), name="rope_cos")
		self.rope_sin = Tensor(shape=freqs.shape, dtype=dtype, device=device, requires_grad=False,
			                   init_val=np.sin(freqs).astype(dtype), name="rope_sin")
		self.blocks = [QwenBlock(C, H, n_head, n_kv_head, dtype, device, eps=eps, layer_idx=i) for i in range(n_layer)]
		self.final_norm_weight = Tensor(shape=(C,), dtype=dtype, device=device, requires_grad=True,
			                            init_val=1.0, name="norm.weight")
		self.lm_head = LMHead(V, C, dtype, device, weight=self.tok_emb.weight)

	def forward(self, input_ids, targets=None, mode="train", row_scale=None):
		x = self.tok_emb(input_ids)
		for block in self.blocks:
			x = block(x, self.rope_cos, self.rope_sin)

		if targets != None:
			return x.fused_rmsnorm_linear_cross_entropy(self.lm_head.weight, self.final_norm_weight, targets, 
				                                        row_scale=row_scale, eps=self.norm_eps)

		return self.lm_head(x)









