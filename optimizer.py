
import math
import numpy as np
from math import prod
from tensor import Tensor

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

class AdamW:
	BLOCK_SIZE = 256
	__slots__ = ('params', 'lr', 'betas', 'eps', 'weight_decay', 't', 'm', 'v', 'dispatches', 'backend', 'quant8bit', "qmap_m", "qmap_v")
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, quant_8bit=False):
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
		self.quant8bit = quant_8bit
		self.qmap_m = None
		self.qmap_v = None

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
			if self.quant8bit:
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


			decay_numel = numel
			saw_vector = False
			for i in indices:
				p = self.params[i]
				offset = p.storage.ptr.offset - start
				if p.grad.storage.ptr.buffer_id != first.grad.storage.ptr.buffer_id:
					raise RuntimeError("parameter group uses multiple gradient buffers")
				if p.grad.storage.ptr.offset != p.storage.ptr.offset:
					raise RuntimeError("parameter and gradient arena offsets differ")
				if p.ndim <= 1:
					if not saw_vector:
						decay_numel = offset // 4
						saw_vector = True 
				elif saw_vector:
					raise RuntimeError("matrix parameter follows vector in Metal arena")
				self.m[i] = rt.allocator.view(m_storage, offset, p.shape, p.dtype)
				self.v[i] = rt.allocator.view(v_storage, offset, p.shape, p.dtype)

			dispatches.append((first.storage, first.grad.storage, m_storage, v_storage, numel, decay_numel))
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
	        apply_wd = self.weight_decay if p.ndim > 1 else 0.0

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

	def step(self):
		self.t += 1
		b1, b2 = self.betas
		lr_t = self.lr * math.sqrt(1.0 - b2**self.t) / (1.0 - b1**self.t)

		self.backend.begin_pass()

		if isinstance(self.dispatches, tuple):
			if self.quant8bit:
				for item in self.dispatches:
					if item[0] == "fp32":
						_, p, g, m, v, numel, wd = item
						self.backend.adamw_step(p, g, m, v, None, self.lr, lr_t, b1, b2, self.eps, wd, numel=numel, decay_numel=numel if wd else 0)
					else:
						_, p, g, qm, qv, absmax_m, absmax_v, numel, wd = item
						self.backend.adamw8bit_step(p, g, qm, qv, absmax_m, absmax_v, self.qmap_m, self.qmap_v,
							self.lr, lr_t, b1, b2, self.eps, wd,
							numel, numel if wd else 0, first_step=(self.t == 1))
			else:
				for p_storage, g_storage, m_storage, v_storage, numel, decay_numel in self.dispatches:
					self.backend.adamw_step(p_storage, g_storage, m_storage, v_storage, None,
						self.lr, lr_t, b1, b2, self.eps, self.weight_decay, numel=numel, decay_numel=decay_numel)
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
					self.lr, lr_t, b1, b2, self.eps, apply_wd)

		self.backend.end_pass()

	def zero_grad(self):
		for p in self.params:
			if p.grad is not None:
				self.backend.fill(p.grad, 0.0)
































