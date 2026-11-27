
class Device:
	_allocators = {}
	_backends = {}

	@classmethod
	def get_allocator(cls, device_name):
		if device_name not in cls._allocators:
			if device_name == "cpu":
				from storage import CPUAllocator
				cls._allocators[device_name] = CPUAllocator()
			elif device_name == "mps":
				from storage import MetalAllocator
				cls._allocators[device_name] = MetalAllocator()
			elif device_name == "cuda":
				from storage import CudaAllocator
				cls._allocators[device_name] = CudaAllocator()
			else:
				raise ValueError(f"Unknown device: {device_name}")
		return cls._allocators[device_name]

	@classmethod
	def get_backend(cls, device_name):
		if device_name not in cls._backends:
			if device_name == "cpu":
				from backend import CPUBackend
				cls._backends[device_name] = CPUBackend()
			elif device_name == "mps":
				from backend import MetalBackend
				cls._backends[device_name] = MetalBackend()
			elif device_name == "cuda":
				from backend import CudaBackend
				cls._backends[device_name] = CudaBackend()
			else:
				raise ValueError(f"Unknown device: {device_name}")
		return cls._backends[device_name]
