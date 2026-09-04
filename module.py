
from tensor import Tensor
from op import LayerNormOp, ScaledDotProductAttenOp, FusedMLPOp, FusedQKVProjOp, FusedRMSNormMLPOp, FlashAttentionOp
from op import FusedRMSNormFlashAttentionOp, FusedLinearAddOp
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
	def __init__(self, C, dtype, device):
		H = (8*C)//3
		self.fc1 = Linear(C, 2*H, dtype, device, bias=False)
		self.fc2 = Linear(H, C, dtype, device, bias=False)

	def forward(self, x):
		return FusedMLPOp()(x, self.fc1.weight, self.fc2.weight)

class GPT2Block(Module):
	def __init__(self, C, n_head, max_T, batch_size, dtype, device, eps=1e-5):
		self.norm_eps = eps
		self.attn = CausalSelfAttention(C, n_head, max_T, batch_size, dtype, device)
		self.mlp = MLP(C, dtype, device)

	def forward(self, x, rope_cos, rope_sin):
		attn_out = FusedRMSNormFlashAttentionOp(n_head=self.attn.n_head, eps=self.norm_eps, is_causal=True)(
			x, self.attn.q_proj.weight, self.attn.q_proj.bias,
			self.attn.k_proj.weight, self.attn.k_proj.bias,
			self.attn.v_proj.weight, self.attn.v_proj.bias,
			rope_cos, rope_sin)
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
		return x @ self.weight.transpose(0, 1)

class GPT2(Module):
	def __init__(self, V, C, max_T, n_head, batch_size, n_layer, dtype, device):
		self.tok_emb = Embedding(V, C, dtype, device)
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

		self.blocks = [GPT2Block(C, n_head, max_T, batch_size, dtype, device, 1e-6)
			for _ in range(n_layer)
		]
		self.lm_head = LMHead(V, C, dtype, device)

	def forward(self, input_ids, targets=None):
		x = self.tok_emb(input_ids)
		for blk in self.blocks:
			x = blk(x, self.rope_cos, self.rope_sin)
		if targets is not None:
			return x.fused_linear_cross_entropy(self.lm_head.weight, targets)

		return self.lm_head(x)






