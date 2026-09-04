import numpy as np
from math import prod

class MemoryPlanner:
	def __init__(self):
		# used memory block [offset, size, tensor_id]
		self.active_blocks = []
		# tensor offsets
		self.offsets = {}
		self.peak = 0

	def alloc(self, tensor, align=256):
		t_id = id(tensor)
		if t_id in self.offsets:
			return

		size = prod(tensor.shape) * np.dtype(tensor.dtype).itemsize
		size = (size + align - 1) // align * align

		# sort by offset
		self.active_blocks.sort(key=lambda x: x[0])
		curr_offset = 0
		alloc_offset = -1
		for off, sz, _ in self.active_blocks:
			if curr_offset + size <= off:
				alloc_offset = curr_offset
				break
			curr_offset = max(curr_offset, off+sz)

		if alloc_offset == -1:
			alloc_offset = curr_offset

		self.active_blocks.append((alloc_offset, size, t_id))
		self.offsets[t_id] = alloc_offset
		self.peak = max(self.peak, alloc_offset+size)

	def free(self, tensor):
		t_id = id(tensor)
		if t_id in self.offsets:
			self.active_blocks = [b for b in self.active_blocks if b[2] != t_id]

	def get_offset(self, tensor):
		return self.offsets.get(id(tensor), None)














