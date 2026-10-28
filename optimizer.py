
import math
import numpy as np
from math import prod
from tensor import Tensor
import random

np.random.seed(42)
random.seed(42)

def get_lr_cosine_schedule(step, max_steps, warmup_steps, max_lr, min_lr):
	if step < warmup_steps:
		return max_lr * (step+1) / warmup_steps
	if step > max_steps:
		return min_lr
	decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return min_lr + coeff * (max_lr - min_lr)

def clip_grad_norm(params, max_norm):
	total_norm_sq = 0.0
	grads = [p.grad for p in params if p.requires_grad and p.grad is not None]

	from runtime import Runtime
	backend = Runtime.get_default().backend

	for g in grads:
		if hasattr(g.storage.data, 'ravel'):
			data = g.storage.data  
		else:
			shape = tuple(g.shape) if not isinstance(g.shape, tuple) else g.shape
			data = backend.ctx.copy_to_host(g.storage.ptr.buffer_id, g.storage.ptr.offset, shape)
		total_norm_sq += float(np.vdot(data.ravel(), data.ravel()))

	total_norm = math.sqrt(total_norm_sq)
	clip_coef = max_norm / (total_norm + 1e-6)
	if clip_coef < 1.0:
		for g in grads:
			if hasattr(g.storage.data, 'ravel'):
				np.multiply(g.storage.data, clip_coef, out=g.storage.data)
			else:
				shape = tuple(g.shape) if not isinstance(g.shape, tuple) else g.shape
				data = backend.ctx.copy_to_host(g.storage.ptr.buffer_id, g.storage.ptr.offset, shape)
				np.multiply(data, clip_coef, out=data)
				backend.ctx.copy_from_host(g.storage.ptr.buffer_id, g.storage.ptr.offset, data)
	return total_norm

VALID_MODE = {"fp32", "8bit", "mini"}

class AdamW:
	BLOCK_SIZE = 256
	__slots__ = ('params', 'lr', 'betas', 'eps', 'weight_decay', 't', 'm', 'v', 
		'dispatches', 'backend', "qmap_m", "qmap_v", "state_mode")
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, state_mode="fp32"):
		self.params = [p for p in params if p.requires_grad]
		self.lr = lr 
		self.betas = betas
		self.eps = eps
		self.weight_decay = weight_decay
		self.t = 0
		self.m = []
		self.v = []
		self.dispatches = None
		self.backend = None
		self.qmap_m = None
		self.qmap_v = None
		if state_mode not in VALID_MODE:
			raise ValueError(f"invalid state_mode={state_mode!r}, expected one of {VALID_MODE}")
		self.state_mode = state_mode

		if not self.params:
			return 

		from runtime import Runtime
		rt = Runtime.get_default()
		self.backend = rt.backend
		# max_size = max(p.shape[0] if p.ndim == 1 else prod(p.shape) for p in self.params)
		metal_ok = (hasattr(self.backend, "ctx") and all(p.grad is not None and p.grad.storage is not None
				and np.dtype(p.dtype) == np.dtype(np.float32) 
				and np.dtype(p.grad.dtype) == np.dtype(np.float32)
				for p in self.params))
		if metal_ok:
			if self.state_mode == "mini":
				self._prepare_mini8bit_meta(rt)
			elif self.state_mode == "8bit":
				self._prepare_8bit_meta(rt)
			else:
				self._prepare_meta(rt)
		else:
			max_size = max(p.shape[0] if p.ndim == 1 else prod(p.shape) for p in self.params)
			for p in self.params:
				self.m.append(rt.allocator.zeros(p.shape, p.dtype))
				self.v.append(rt.allocator.zeros(p.shape, p.dtype))
			self.dispatches = rt.allocator.zeros((max_size,), self.params[0].dtype)

	def _prepare_meta(self, rt):
		groups = {}
		for i, p in enumerate(self.params):
			groups.setdefault(p.storage.ptr.buffer_id, []).append(i)

		self.m = [None] * len(self.params)
		self.v = [None] * len(self.params)
		dispatches = []
		for indices in groups.values():
			indices.sort(key=lambda i: self.params[i].storage.ptr.offset)
			first = self.params[indices[0]]
			start = first.storage.ptr.offset
			end = max(self.params[i].storage.ptr.offset + self.params[i].storage.nbytes for i in indices)
			nbytes = end - start
			numel = nbytes // 4
			m_storage = rt.allocator.zeros((numel,), np.float32)
			v_storage = rt.allocator.zeros((numel,), np.float32)

			merge = True
			decay_numel = numel
			saw_vector = False
			for i in indices:
				p = self.params[i]
				offset = p.storage.ptr.offset - start
				self.m[i] = rt.allocator.view(m_storage, offset, p.shape, np.float32)
				self.v[i] = rt.allocator.view(v_storage, offset, p.shape, np.float32)
				if np.dtype(p.grad.storage.dtype) != np.dtype(np.float32) or p.grad.storage.ptr.buffer_id != first.grad.storage.ptr.buffer_id or (p.grad.storage.ptr.offset - first.grad.storage.ptr.offset) != offset:
					merge = False
				if p.ndim <= 1:
					if not saw_vector:
						decay_numel = offset // 4
						saw_vector = True
				elif saw_vector:
					merge = False

			if merge:
				p_storage = rt.allocator.view(first.storage, 0, (numel,), np.float32)
				g_storage = rt.allocator.view(first.grad.storage, 0, (numel,), np.float32)
				dispatches.append((p_storage, g_storage, m_storage, v_storage, numel, decay_numel))
			else:
				for i in indices:
					p = self.params[i]
					n = int(p.numel)
					dispatches.append((p.storage, p.grad.storage, self.m[i], self.v[i], n, n if p.ndim > 1 else 0))

		self.dispatches = tuple(dispatches)

	def _prepare_8bit_meta(self, rt, min_8bit_size=4096):
	    def build_dynamic_map(signed, max_exponent_bits=7, total_bits=8):
	        non_sign_bits = total_bits - 1 if signed else total_bits
	        additional_items = (2 ** (non_sign_bits - max_exponent_bits) - 1)
	        values = []

	        for exponent in range(max_exponent_bits):
	            fraction_items = (2 ** (exponent + non_sign_bits - max_exponent_bits) + 1)
	            boundaries = np.linspace(0.1, 1.0, fraction_items, dtype=np.float32,)
	            means = (boundaries[:-1] + boundaries[1:]) * np.float32(0.5)
	            scale = np.float32(10.0 ** (exponent - (max_exponent_bits-1)))
	            scaled = means * scale
	            values.extend(scaled.tolist())
	            if signed:
	                values.extend((-scaled).tolist())

	        if additional_items > 0:
	            boundaries = np.linspace(0.0, 0.1, additional_items + 1, dtype=np.float32,)
	            means = (boundaries[:-1] + boundaries[1:]) * np.float32(0.5)
	            scale = np.float32(10.0 ** (-(max_exponent_bits-1)))
	            scaled = means * scale
	            values.extend(scaled.tolist())
	            if signed:
	                values.extend((-scaled).tolist())

	        # signed map 使用两个零 code，正好填满 256 项。
	        values.append(0.0)
	        while len(values) < (1 << total_bits):
	            values.append(0.0)

	        if len(values) != (1 << total_bits):
	        	raise RuntimeError(f"invalid dynamic-map size: {len(values)}")

	        qmap = np.asarray(values, dtype=np.float32)
	        qmap.sort()
	        qmap /= np.max(np.abs(qmap))
	        return np.ascontiguousarray(qmap, dtype=np.float32)

	    qmap_m_host = build_dynamic_map(signed=True)
	    qmap_v_host = build_dynamic_map(signed=False)

	    if qmap_m_host.shape != (256,) or qmap_v_host.shape != (256,):
	        raise RuntimeError("dynamic maps must contain 256 values")

	    self.qmap_m = rt.allocator.empty((256,), np.float32)
	    self.qmap_v = rt.allocator.empty((256,), np.float32)

	    rt.backend.ctx.numpy_view_float32(
	        self.qmap_m.ptr.buffer_id,
	        self.qmap_m.ptr.offset,
	        (256,),
	    )[:] = qmap_m_host

	    rt.backend.ctx.numpy_view_float32(
	        self.qmap_v.ptr.buffer_id,
	        self.qmap_v.ptr.offset,
	        (256,),
	    )[:] = qmap_v_host

	    self.m = [None] * len(self.params)
	    self.v = [None] * len(self.params)
	    dispatches = []

	    for index, p in enumerate(self.params):
	        numel = p.numel
	        apply_wd = p.ndim > 1

	        # 主流实现：小 state 不量化。
	        if numel < min_8bit_size:
	            m = rt.allocator.zeros(p.shape, np.float32)
	            v = rt.allocator.zeros(p.shape, np.float32)

	            self.m[index] = m
	            self.v[index] = v
	            dispatches.append(
	                ("fp32", p.storage, p.grad.storage, m, v, numel, apply_wd)
	            )
	            continue

	        nblocks = (numel + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
	        qm = rt.allocator.empty((numel,), np.uint8)
	        qv = rt.allocator.empty((numel,), np.uint8)
	        absmax_m = rt.allocator.empty((nblocks,), np.float32)
	        absmax_v = rt.allocator.empty((nblocks,), np.float32)

	        self.m[index] = qm
	        self.v[index] = qv
	        dispatches.append(("q8", p.storage, p.grad.storage, qm, qv, absmax_m, absmax_v, numel, apply_wd))

	    self.dispatches = tuple(dispatches)

	def _build_qmap_m(self, rt):
	    def build_dynamic_map(signed, max_exponent_bits=7, total_bits=8):
	        non_sign_bits = total_bits - 1 if signed else total_bits
	        additional_items = (2 ** (non_sign_bits - max_exponent_bits) - 1)
	        values = []

	        for exponent in range(max_exponent_bits):
	            fraction_items = (2 ** (exponent + non_sign_bits - max_exponent_bits) + 1)
	            boundaries = np.linspace(0.1, 1.0, fraction_items, dtype=np.float32,)
	            means = (boundaries[:-1] + boundaries[1:]) * np.float32(0.5)
	            scale = np.float32(10.0 ** (exponent - (max_exponent_bits-1)))
	            scaled = means * scale
	            values.extend(scaled.tolist())
	            if signed:
	                values.extend((-scaled).tolist())

	        if additional_items > 0:
	            boundaries = np.linspace(0.0, 0.1, additional_items + 1, dtype=np.float32,)
	            means = (boundaries[:-1] + boundaries[1:]) * np.float32(0.5)
	            scale = np.float32(10.0 ** (-(max_exponent_bits-1)))
	            scaled = means * scale
	            values.extend(scaled.tolist())
	            if signed:
	                values.extend((-scaled).tolist())

	        # signed map 使用两个零 code，正好填满 256 项。
	        values.append(0.0)
	        while len(values) < (1 << total_bits):
	            values.append(0.0)

	        if len(values) != (1 << total_bits):
	        	raise RuntimeError(f"invalid dynamic-map size: {len(values)}")

	        qmap = np.asarray(values, dtype=np.float32)
	        qmap.sort()
	        qmap /= np.max(np.abs(qmap))
	        return np.ascontiguousarray(qmap, dtype=np.float32)

	    qmap_m_host = build_dynamic_map(signed=True)
	    self.qmap_m = rt.allocator.empty((256,), np.float32)
	    rt.backend.ctx.numpy_view_float32(self.qmap_m.ptr.buffer_id, self.qmap_m.ptr.offset, (256,))[:] = qmap_m_host

	def _prepare_mini8bit_meta(self, rt, min_8bit_size=4096):
		stochastic_ids = set()
		column_ids = set()
		linear_ids = set()
		head_counts = {}

		total_state_bytes = 0
		total_m_bytes = 0
		total_v_bytes = 0
		total_meta_bytes = 0

		for node in rt.topo:
			op_name = node.op.op_name
			if op_name == "Gather":
				# inputs = (indices, embedding_weight)
				stochastic_ids.add(id(node.inputs[1]))
			elif op_name == "FusedLinearCrossEntropy":
				# inputs = (x, weight, targets, ...)
				stochastic_ids.add(id(node.inputs[1]))
			elif op_name == "FusedRMSNormLinearCrossEntropy":
				# inputs = (x, weight, norm_weight, targets, ...)
				stochastic_ids.add(id(node.inputs[1]))
			elif op_name == "FusedRMSNormMLP":
				if len(node.inputs) == 3:
					# inputs = (x, gate_up_weight, down_weight)
					gate_up_weight = node.inputs[1]
					down_weight = node.inputs[2]
				else:
					# inputs = (x, norm_weight, gate_up_weight, down_weight)
					gate_up_weight = node.inputs[2]
					down_weight = node.inputs[3]
				column_ids.add(id(gate_up_weight))
				column_ids.add(id(down_weight))
				linear_ids.add(id(gate_up_weight))
				linear_ids.add(id(down_weight))
			elif op_name == "FusedLinearAdd":
				# inputs = (residual, activation, projection_weight, bias)
				column_ids.add(id(node.inputs[2]))
				linear_ids.add(id(node.inputs[2]))
			elif op_name == "FusedRMSNormFlashAttention":
				n_head = int(node.op.n_head)
				q_weight = node.inputs[1]
				k_weight = node.inputs[3]
				v_weight = node.inputs[5]
				for p in (q_weight, k_weight, v_weight):
					linear_ids.add(id(p))
				for p in (q_weight, k_weight, v_weight):
					if p.ndim != 2:
						raise ValueError(f"Q/K/V weight must be 2D, got {p.shape}")
					rows, cols = map(int, p.shape)
					if cols % n_head:
						raise ValueError(f"Q/K/V cols={cols} is not divisible by n_head={n_head}")
					previous = head_counts.get(id(p))
					if previous is not None and previous != n_head:
						raise ValueError("shape Q/K/V parameter has inconsistent head counts")
					head_counts[id(p)] = n_head
			elif op_name == "FusedRMSNormGQA":
				q_weight = node.inputs[2]
				k_weight = node.inputs[4]
				v_weight = node.inputs[6]
				n_head = node.op.n_head
				n_kv_head = node.op.n_kv_head
				if n_head % n_kv_head:
					raise ValueError(f"n_head={n_head} must be divisible by n_kv_head={n_kv_head}")
				linear_ids.add(id(q_weight))
				linear_ids.add(id(k_weight))
				linear_ids.add(id(v_weight))

				for p, n_groups in ((q_weight, n_head), (k_weight, n_kv_head), (v_weight, n_kv_head)):
					if p.ndim != 2:
						raise ValueError(f"GQA weight must be 2D, got {p.shape}")
					rows, cols = map(int, p.shape)
					if cols % n_groups:
						raise ValueError(f"GQA weight shape={p.shape} is not divisible by group count={n_groups}")
					previous = head_counts.get(id(p))
					if previous is not None and previous != n_groups:
						raise ValueError(f"shared GQA parameter has inconsistent head counts")
					head_counts[id(p)] = n_groups

		self.qmap_m = None
		self.qmap_v = None
		self.m = [None] * len(self.params)
		self.v = [None] * len(self.params)
		dispatches = []

		for index, p in enumerate(self.params):
			if p.storage is None or p.grad is None or p.grad.storage is None:
				raise RuntimeError("Build and bind the training graph before constructing AdamW-mini")
			if np.dtype(p.dtype) != np.dtype(np.float32) or np.dtype(p.grad.dtype) != np.dtype(np.float32):
				raise ValueError("AdamW-mini currently requires FP32 parameters and gradients")

			n = int(p.numel)
			if n <= 0:
				raise ValueError("Empty parameter")

			cols = int(p.shape[-1]) if p.ndim else 1
			transpose_m = id(p) in linear_ids
			stochastic = id(p) in stochastic_ids

			if p.ndim > 1 and not(transpose_m or stochastic or id(p) in column_ids or id(p) in head_counts):
				raise ValueError(f"Unrecoginized matrix parameter: {p.shape}")

			if id(p) in head_counts:
				n_groups = int(head_counts[id(p)])
				if cols % n_groups:
					raise ValueError(f"parameter shape={p.shape} has cols={cols}, which is not divisible by groups={n_groups}")

				head_cols = cols // n_groups
				rows = n // cols
				block_elems = rows * head_cols
				n_v = n_groups
			elif id(p) in column_ids:
				# MLP/projection权重
				head_cols = 1
				rows = n // cols
				block_elems = rows
				n_v = cols 
			elif stochastic:
				# embedding/lm_head
				head_cols = 0
				block_elems = cols
				n_v = n // cols 
			else:
				# 普通向量,如bias,RMSNorm weight
				head_cols = 0
				block_elems = 1
				n_v = n

			quantized = n >= min_8bit_size
			n_blocks = (n + 255) // 256
			m = rt.allocator.zeros((n,), np.uint8 if quantized else np.float32)
			# 每块两个FP32数, scale, zero point
			meta = rt.allocator.zeros((n_blocks, 2), np.float32 if quantized else None)
			v = rt.allocator.zeros((n_v,), np.float32)

			total_m_bytes += m.nbytes
			total_v_bytes += v.nbytes
			total_meta_bytes += meta.nbytes if meta is not None else 0
			total_state_bytes += (m.nbytes + v.nbytes + meta.nbytes if meta is not None else 0)

			self.m[index] = m
			self.v[index] = v

			apply_decay = p.ndim > 1
			dispatches.append(("mini8bit", p.storage, p.grad.storage, m, meta, v, n, apply_decay, block_elems, 
				cols, head_cols, stochastic, transpose_m, quantized))

		self.dispatches = tuple(dispatches)

		param_bytes = sum(
		    int(p.storage.nbytes)
		    for p in self.params
		    if p.storage is not None
		)

		print(
		    "[OPTIMIZER-STATE-TOTAL] "
		    f"state_mode=mini "
		    f"total_bytes={total_state_bytes} "
		    f"m_bytes={total_m_bytes} "
		    f"v_bytes={total_v_bytes} "
		    f"meta_bytes={total_meta_bytes} "
		    f"param_bytes={param_bytes} "
		    f"bytes_per_param="
		    f"{total_state_bytes / param_bytes if param_bytes else 0.0:.6f}", flush=True
		)

	def step(self, grad_scale=1.0):
		self.t += 1
		b1, b2 = self.betas
		lr_t = self.lr * math.sqrt(1.0 - b2**self.t) / (1.0 - b1**self.t)

		self.backend.begin_pass()

		if isinstance(self.dispatches, tuple):
			if self.state_mode == "mini":
				first_step = (self.t == 1)
				lr_bc1 = self.lr / (1.0 - b1 ** self.t)
				inv_sqrt_bc2 = 1.0 / math.sqrt(1.0 - b2 ** self.t)
				for item in self.dispatches:
					_, p, g, m, meta, v, n, apply_decay, block_elems, cols, head_cols, stochastic, transpose_m, quantized = item
					wd = self.weight_decay if apply_decay else 0.0
					self.backend.mini_update_v(g, v, b2, n, block_elems, cols=cols, head_cols=head_cols, first_step=first_step, grad_scale=grad_scale)
					seed = random.getrandbits(32) if stochastic and quantized else 0
					self.backend.adamw_mini8bit_step(p, g, m, meta, v, self.lr, lr_bc1, inv_sqrt_bc2, b1, self.eps, wd, n,
							n if apply_decay else 0, block_elems, cols=cols, head_cols=head_cols, 
							first_step=first_step, seed=seed, stochastic_round=stochastic,
							transpose_m=transpose_m, quantized=quantized, grad_scale=grad_scale)
			elif self.state_mode == "8bit":
				lr_bc1 = self.lr / (1.0 - b1**self.t)
				inv_sqrt_bc2 = 1.0 / math.sqrt(1.0 - b2**self.t)
				for item in self.dispatches:
					if item[0] == "fp32":
						_, p, g, m, v, numel, apply_decay = item
						wd = self.weight_decay if apply_decay else 0.0
						self.backend.adamw_step(p, g, m, v, None, self.lr, lr_t, b1, b2, self.eps, wd, 
							numel=numel, decay_numel=numel if apply_decay else 0, grad_scale=grad_scale,
							lr_bc1=lr_bc1, inv_sqrt_bc2=inv_sqrt_bc2)
					else:
						_, p, g, qm, qv, absmax_m, absmax_v, numel, apply_decay = item
						wd = self.weight_decay if apply_decay else 0.0
						self.backend.adamw8bit_step(p, g, qm, qv, absmax_m, absmax_v, self.qmap_m, self.qmap_v,
							self.lr, lr_bc1, inv_sqrt_bc2, b1, b2, self.eps, wd,
							numel, numel if apply_decay else 0, first_step=(self.t == 1), grad_scale=grad_scale)
			else:
				lr_bc1 = self.lr / (1.0 - b1**self.t)
				inv_sqrt_bc2 = 1.0 / math.sqrt(1.0 - b2**self.t)
				for p_storage, g_storage, m_storage, v_storage, numel, decay_numel in self.dispatches:
					self.backend.adamw_step(p_storage, g_storage, m_storage, v_storage, None,
						self.lr, lr_t, b1, b2, self.eps, self.weight_decay, 
						numel=numel, decay_numel=decay_numel, grad_scale=grad_scale, lr_bc1=lr_bc1, inv_sqrt_bc2=inv_sqrt_bc2)
		else:
			for i,p in enumerate(self.params):
				if p.grad is None:
					continue
				apply_wd = self.weight_decay if p.ndim > 1 else 0.0
				if hasattr(self.dispatches.data, 'size'):
					tmp_view = self.dispatches.data[:p.storage.data.nbytes // p.storage.data.dtype.itemsize].reshape(p.shape)
				else:
					tmp_view = None
				self.backend.adamw_step(p.storage, p.grad.storage,
					self.m[i], self.v[i], tmp_view,
					self.lr, lr_t, b1, b2, self.eps, apply_wd, grad_scale=grad_scale)

		self.backend.end_pass()

	def zero_grad(self):
		for p in self.params:
			if p.grad is not None:
				if p.grad.storage.device == "mps":
					self.backend.fill_storage2(p.grad.storage, 0.0)
				else:
					self.backend.fill(p.grad, 0.0)
				# self.backend.fill(p.grad, 0.0)

	def build_grad_clip(self):
		grad_views = []
		if not isinstance(self.dispatches, tuple):
			grad_views = [p.grad for p in self.params if p.grad is not None]
			def clip_grad_norm(max_norm, eps=1e-6):
				norm_sq = 0.0
				for grad in grad_views:
					data = grad.storage.data
					flat = np.asarray(data).reshape(-1)
					norm_sq += float(np.dot(flat, flat))
				grad_norm = math.sqrt(norm_sq)
				if not math.isfinite(grad_norm):
					return grad_norm, 0.0
				clip_coef = min(1.0, float(max_norm) / (grad_norm + eps))
				return grad_norm, clip_coef
			return clip_grad_norm

		if self.state_mode == "fp32":
			# (param_storage, grad_storage, m, v, numel, decay_numel)
			items = [(item[1], item[4]) for item in self.dispatches]
		elif self.state_mode == "8bit":
			# ("fp32"/"q8", param_storage, grad_storage, ...)
			# items = ((item[2], item[7]) for item in self.dispatches)
			items = []
			for item in self.dispatches:
				kind = item[0]
				if kind == "fp32":
					if len(item) != 7:
						raise RuntimeError(f"Unexpected fp32 dispatch layout: {len(item)} fields")
					items.append((item[2], int(item[5])))
				elif kind == "q8":
					if len(item) != 9:
						raise RuntimeError(f"Unexpected q8 dispatch layout: {len(item)} fields")
					items.append((item[2], int(item[7])))
				else:
					raise RuntimeError(f"Unexpected 8bit dispatch kind: {kind!r}")
		else:
			# ("mini8bit", param_storage, grad_storage, ..., numel, ...)
			items = [(item[2], item[6]) for item in self.dispatches]

		partial_bufs = []
		from runtime import Runtime
		rt = Runtime.get_default()
		for _, numel in items:
			n_groups = (int(numel) + 255) // 256
			partial_bufs.append(rt.allocator.empty((n_groups,), np.float32))

		def clip_grad_norm(max_norm, eps=1e-6):
			norm_sq = 0.0

			self.backend.begin_pass()
			for (grad_storage, numel), partial in zip(items, partial_bufs):
				self.backend.grad_norm_sq(grad_storage, partial, int(numel))
			self.backend.end_pass()
			for partial in partial_bufs:
				view = self.backend.ctx.numpy_view_float32(partial.ptr.buffer_id, partial.ptr.offset, partial.shape)
				norm_sq += float(np.sum(view, dtype=np.float64))
			grad_norm = math.sqrt(norm_sq)
			if not math.isfinite(grad_norm):
				return grad_norm, 0.0
			clip_coef = min(1.0, float(max_norm) / (grad_norm + eps))

			# if clip_coef < 1.0:
			# 	for grad in grad_views:
			# 		np.multiply(grad, np.float32(clip_coef), out=grad)

			return grad_norm, clip_coef

		return clip_grad_norm

































