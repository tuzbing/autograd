
import numpy as np
from utils import xavier_uniform

class Backend:
	mixed_precision = False

	def get_forward_kernel(self, op):
		raise NotImplementedError()

	def get_backward_kernel(self, op):
		raise NotImplementedError()

	def fill_storage(self, storage, value):
		raise NotImplementedError()

	def fill_storage2(self, storage, value):
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