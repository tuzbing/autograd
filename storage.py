import numpy as np
from math import prod

import metal_ext

class Storage:
	__slots__ = ("ptr", "shape", "dtype", "device", "nbytes")
	def __init__(self, ptr, shape, dtype, device, nbytes):
		self.ptr = ptr
		self.shape = tuple(shape)
		self.dtype = dtype
		self.device = device
		self.nbytes = nbytes

	@property
	def data(self):
		return self.ptr

class Allocator:
	def __init__(self, device):
		self.device = device

	def alloc(self, shape, dtype):
		raise NotImplementedError()

	def free(self):
		raise NotImplementedError()

	def view(self, storage, offset, shape, dtype):
		raise NotImplementedError()

	def empty(self, shape, dtype=np.float32):
		return self.alloc(shape=shape, dtype=dtype)

	def zeros(self, shape, dtype=np.float32):
		storage = self.alloc(shape=shape, dtype=dtype)
		from device import Device
		backend = Device.get_backend(self.device)
		backend.fill_storage(storage, 0)
		return storage

	def ones(self, shape, dtype=np.float32):
		storage = self.alloc(shape=shape, dtype=dtype)
		from device import Device
		backend = Device.get_backend(self.device)
		backend.fill_storage(storage, 1)
		return storage

class CPUAllocator(Allocator):
	def __init__(self):
		super().__init__(device="cpu")

	def alloc(self, shape, dtype=np.float32):
		ptr = np.empty(shape, dtype=dtype)
		storage = Storage(ptr=ptr, shape=shape, dtype=dtype, device=self.device, nbytes=ptr.nbytes)
		return storage

	def view(self, storage, offset, shape, dtype):
		dt = np.dtype(dtype)
		nbytes = prod(shape) * dt.itemsize
		if offset + nbytes > storage.nbytes:
			raise RuntimeError(f"View overflow: {offset}+{nbytes}>{storage.nbytes}")
		view_data = storage.data[offset:offset+nbytes].view(dt).reshape(shape)
		return Storage(ptr=view_data, shape=shape, dtype=dtype, device=self.device, nbytes=nbytes)

class MetalMemPointer:
	__slots__ = ("buffer_id", "offset")
	def __init__(self, buffer_id, offset):
		self.buffer_id = buffer_id
		self.offset = offset

class MetalAllocator(Allocator):
	def __init__(self):
		super().__init__("mps")
		self.ctx = metal_ext.MetalContext()

	def alloc(self, shape, dtype):
		nbytes = prod(shape) * np.dtype(dtype).itemsize
		buffer_id = self.ctx.alloc(nbytes)
		ptr = MetalMemPointer(buffer_id, 0)
		return Storage(ptr, shape, dtype, self.device, nbytes)

	def empty(self, shape, dtype):
		return self.alloc(shape, dtype)

	def view(self, storage, offset_bytes, shape, dtype):
		dt = np.dtype(dtype)
		nbytes = prod(shape) * dt.itemsize
		new_ptr = MetalMemPointer(storage.ptr.buffer_id, storage.ptr.offset+offset_bytes)
		return Storage(new_ptr, shape, dtype, self.device, nbytes)