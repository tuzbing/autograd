
__all__ = ["Backend", "CPUBackend", "MetalBackend", "CudaBackend"]

def __getattr__(name):
	if name == "Backend":
		from .base import Backend
		return Backend
	if name == "CPUBackend":
		from .cpu import CPUBackend
		return CPUBackend
	if name == "MetalBackend":
		from .metal import MetalBackend
		return MetalBackend
	if name == "CudaBackend":
		from .cuda import CudaBackend
		return CudaBackend
	raise AttributeError(f"Module 'backend' has no attribute {name!r}") 