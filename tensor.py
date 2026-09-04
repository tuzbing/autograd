from math import prod
from utils import contiguous_strides

class Tensor():
	__slots__ = ("shape", "dtype", "device", 
				"strides", "offset", "is_contiguous",
				"requires_grad", "grad", "op", "init_val",
				"storage", "name")

	def __init__(self, shape, dtype, strides = None, 
				offset=0, is_contiguous=True,
				requires_grad=True, init_val=None, device=None, name=''):
		self.shape = shape
		self.dtype = dtype
		self.device = device
		self.strides = (strides if strides is not None else contiguous_strides(shape)) 
		self.offset = offset
		self.is_contiguous = is_contiguous
		self.requires_grad = requires_grad
		self.grad = None
		self.op = None
		self.init_val = init_val
		self.storage = None
		self.name = name

	def backward(self, grad_scale=1.0, accumulate_parameter_grads=False):
		from runtime import Runtime
		rt = Runtime.get_default()
		rt.backward(self, grad_scale=grad_scale, accumulate_parameter_grads=accumulate_parameter_grads)

	@property
	def ndim(self):
		return len(self.shape)

	@property
	def numel(self):
		return prod(self.shape)

	def __repr__(self):
		grad_flag = ", requires_grad=" if self.requires_grad else ""
		name = f"{self.name}: " if self.name else ""
		return (f"Tensor({name} shape={self.shape} dtype={self.dtype} device={self.device} {grad_flag})")

	def __add__(self, other):
		from op import AddOp
		return AddOp()(self, other)

	def __mul__(self, other):
		from op import MulOp
		return MulOp()(self, other)

	def __matmul__(self, other):
		from op import MatMulOp
		return MatMulOp()(self, other)

	def view(self, *shape):
		from op import ViewOp
		return ViewOp(shape)(self)

	def reshape(self, *shape):
		from op import ReshapeOp
		return ReshapeOp(shape)(self)

	def transpose(self, dim0, dim1):
		from op import TransposeOp
		return TransposeOp(dim0, dim1)(self)

	def permute(self, *dims):
		from op import PermuteOp
		return PermuteOp(dims)(self)

	def contiguous(self):
		from op import ContiguousOp
		return ContiguousOp()(self)

	def exp(self):
		from op import ExpOp
		return ExpOp()(self)

	def log(self):
		from op import LogOp
		return LogOp()(self)

	def rsqrt(self):
		from op import RsqrtOp
		return RsqrtOp()(self)

	def softmax(self, dim):
		from op import SoftmaxOp
		return SoftmaxOp(dim)(self)

	def gather(self, weight):
		from op import GatherOp 
		return GatherOp()(self, weight)

	def layer_norm(self, gamma, beta, eps=1e-5):
		from op import LayerNormOp
		return LayerNormOp(eps=eps)(self, gamma, beta)

	def gelu(self):
		from op import GELUOp
		return GELUOp()(self)

	def cross_entropy(self, labels, dim=-1):
		from op import CrossEntropyLossOp
		return CrossEntropyLossOp(dim)(self, labels)

	def fused_linear_cross_entropy(self, weight, targets):
		from op import FusedLinearCrossEntropyOp
		return FusedLinearCrossEntropyOp()(self, weight, targets)

	def attn_out_contiguous(self):
		from op import AttnOutContiguousOp
		return AttnOutContiguousOp()(self)