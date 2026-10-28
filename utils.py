
import numpy as np
from math import prod
import struct

def f32_bits(x):
	return struct.unpack('<I', struct.pack('<f', float(x)))[0] 

def f16_bits(x):
    return struct.unpack("<H", struct.pack("<e", float(x)))[0]

def bf16_bits(x):
    bits = f32_bits(x)
    if (bits & 0x7f800000) == 0x7f800000:
        if bits & 0x007fffff:
            return ((bits >> 16) | 0x0040) & 0xffff
        return bits >> 16
    bias = 0x7fff + ((bits >> 16) & 1)
    return ((bits + bias) >> 16) & 0xffff

def contiguous_strides(shape):
	strides = []
	acc = 1
	for s in reversed(shape):
		strides.append(acc)
		acc *= s 
	return tuple(reversed(strides))

def is_contiguous_strides(shape, strides):
	return tuple(strides) == contiguous_strides(shape)

def is_materialized(t):
	if t.storage is None:
		return False
	if t.op is None:
		return t.storage.data.shape == t.shape
	if t.op.op.op_name in ('View', 'Reshape', 'Permute', 'Transpose'):
		return False
	return t.storage.data.shape == t.shape

def broadcast_shape(a, b):
	out = []
	ra = list(reversed(a))
	rb = list(reversed(b))
	n = max(len(ra), len(rb))
	for i in range(n):
		da = ra[i] if i < len(ra) else 1
		db = rb[i] if i < len(rb) else 1
		if da == 1:
			out.append(db)
		elif db == 1:
			out.append(da)
		elif da == db:
			out.append(da)
		else:
			raise RuntimeError(f"broadcast mismatch: {a} {b}")

	return tuple(reversed(out))

def xavier_uniform(shape, dtype):
	fan_in, fan_out = shape[-2], shape[-1]
	std = np.sqrt(2.0 / (fan_in + fan_out))
	return np.random.normal(0, std, shape).astype(dtype)

def reduce_broadcast_grad(grad, target_shape):
	if grad.shape == target_shape:
		return grad
	pad = grad.ndim - len(target_shape)
	if pad < 0:
		raise ValueError(f"grad ndim {grad.ndim} < target ndim {len(target_shape)}")
	target_ext = (1,) * pad + target_shape
	axes = tuple(i for i, (g,t) in enumerate(zip(grad.shape, target_ext)) if t == 1 and g != 1)
	if axes:
		grad = grad.sum(axis=axes, keepdims=True)
	return grad.reshape(target_shape)

def resolve_reshape_shape(input_numel, target_shape):
	shape = list(target_shape)
	minus_one = [i for i, d in enumerate(shape) if d == -1]
	if len(minus_one) > 1:
		raise ValueError(f"reshape: at most one -1, got {target_shape}")
	known = prod(d for d in shape if d != -1)
	if known == 0:
		raise ValueError(f"reshape: invalid target_shape {target_shape}")

	if minus_one:
		if input_numel % known != 0:
			raise ValueError(f"reshape: numel {input_numel} not divisible by {known},target={target_shape}")
		shape[minus_one[0]] = input_numel // known
	elif prod(shape) != input_numel:
		raise ValueError(f"reshape: numel mismatch: {input_numel} -> {tuple(shape)}")
	return tuple(shape)

def infer_reshape_view(x, new_shape):
	new_shape = resolve_reshape_shape(x.numel, new_shape)
	if x.numel != prod(new_shape):
		raise ValueError(f"reshape view numel mismatch: {x.shape}->{new_shape}")
	if not x.is_contiguous:
		raise RuntimeError("reshape view requires contiguous input")

	return new_shape, contiguous_strides(new_shape), x.offset, True

def tensor_to_numpy(t):
	if t.storage is None:
		raise RuntimeError("tensor_to_numpy: storage is None")
	buf = t.storage.data
	numel = t.numel 
	if (t.offset == 0 and buf.shape == t.shape and is_contiguous_strides(t.shape, t.strides)):
		return buf
	flat = buf.reshape(-1)
	if t.offset:
		flat = flat[t.offset:]
	if t.is_contiguous:
		return flat[:numel].reshape(t.shape)
	itemsize = np.dtype(t.dtype).itemsize
	byte_strides = tuple(s*itemsize for s in t.strides)
	return np.lib.stride_tricks.as_strided(flat, shape=t.shape, strides=byte_strides, writeable=True,)

def grad_shape_matches(grad_shape, target_shape):
	if grad_shape == target_shape:
		return True
	pad = len(grad_shape) - len(target_shape)
	target_ext = (1,) * pad + target_shape
	return all(g == t or t == 1 for g,t in zip(grad_shape, target_ext))

def mul_bcast_kind(x_shape, y_shape):
	if len(y_shape) == 0 or prod(y_shape) == 1:
		return 'scalar'
	if len(y_shape) == 1 and x_shape[-1] == y_shape[0]:
		return 'row'
	if len(y_shape) == len(x_shape) and y_shape[:-1] == x_shape[:-1] and y_shape[-1] == 1:
		return 'col'
	if y_shape == x_shape:
		return 'same'
	raise RuntimeError(f"Mul backward: unsupported broadcast x={x_shape} y={y_shape}")
