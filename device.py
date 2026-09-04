
from storage import CPUAllocator, MetalAllocator
from backend import CPUBackend, MetalBackend

class Device:
	_allocators = {}
	_backends = {}

	@classmethod
	def get_allocator(cls, device_name):
		if device_name not in cls._allocators:
			if device_name == "cpu":
				cls._allocators[device_name] = CPUAllocator()
			elif device_name == "mps":
				cls._allocators[device_name] = MetalAllocator()
			else:
				raise ValueError(f"Unknown device: {device_name}")
		return cls._allocators[device_name]

	@classmethod
	def get_backend(cls, device_name):
		if device_name not in cls._backends:
			if device_name == "cpu":
				cls._backends[device_name] = CPUBackend()
			elif device_name == "mps":
				cls._backends[device_name] = MetalBackend()
			else:
				raise ValueError(f"Unknown device: {device_name}")
		return cls._backends[device_name]
