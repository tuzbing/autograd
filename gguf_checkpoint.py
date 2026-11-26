"""Minimal F32 GGUF checkpoints for this repository's GPT2 model."""

import json
import os
import struct
import tempfile

import numpy as np

from tensor import Tensor


MAGIC = b"GGUF"
VERSION = 3
ALIGNMENT = 32
GGML_TYPE_F32 = 0

UINT32 = 4
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9
UINT64 = 10

_SCALAR_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
    6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _write_string(file, value):
    data = value.encode("utf-8")
    file.write(struct.pack("<Q", len(data)))
    file.write(data)


def _read_exact(file, size):
    data = file.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF file")
    return data


def _read_string(file):
    size = struct.unpack("<Q", _read_exact(file, 8))[0]
    return _read_exact(file, size).decode("utf-8")


def _write_value(file, value_type, value):
    if value_type == STRING:
        _write_string(file, value)
    elif value_type in _SCALAR_FORMATS:
        file.write(struct.pack("<" + _SCALAR_FORMATS[value_type], value))
    else:
        raise TypeError(f"unsupported GGUF metadata type: {value_type}")


def _read_value(file, value_type):
    if value_type == STRING:
        return _read_string(file)
    if value_type == ARRAY:
        item_type = struct.unpack("<I", _read_exact(file, 4))[0]
        count = struct.unpack("<Q", _read_exact(file, 8))[0]
        return [_read_value(file, item_type) for _ in range(count)]
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"unsupported GGUF metadata type: {value_type}")
    return struct.unpack("<" + fmt, _read_exact(file, struct.calcsize(fmt)))[0]


def named_parameters(model):
    """Yield stable dotted names without requiring changes to Module."""
    seen = set()

    def visit(value, prefix):
        if isinstance(value, Tensor):
            if value.requires_grad and id(value) not in seen:
                seen.add(id(value))
                yield prefix, value
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from visit(item, f"{prefix}.{index}")
        elif isinstance(value, dict):
            for key in sorted(value):
                yield from visit(value[key], f"{prefix}.{key}")
        elif hasattr(value, "__dict__"):
            for key, item in value.__dict__.items():
                name = f"{prefix}.{key}" if prefix else key
                yield from visit(item, name)

    yield from visit(model, "")


def save_gguf(path, model, backend, model_config, step):
    """Atomically save all trainable model tensors as unquantized F32 GGUF."""
    parameters = list(named_parameters(model))
    if not parameters:
        raise ValueError("model has no trainable parameters")

    metadata = [
        ("general.architecture", STRING, "autograd-gpt2"),
        ("general.name", STRING, os.path.basename(path)),
        ("general.alignment", UINT32, ALIGNMENT),
        ("general.file_type", UINT32, 0),
        ("autograd.checkpoint_version", UINT32, 1),
        ("autograd.training_step", UINT64, int(step)),
        ("autograd.model_config", STRING, json.dumps(model_config, sort_keys=True)),
    ]

    tensor_infos = []
    data_offset = 0
    for name, tensor in parameters:
        if np.dtype(tensor.dtype) != np.dtype(np.float32):
            raise TypeError(f"{name}: only float32 parameters are supported")
        data_offset = _align(data_offset, ALIGNMENT)
        tensor_infos.append((name, tensor, data_offset))
        data_offset += tensor.numel * np.dtype(np.float32).itemsize

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".gguf-", dir=directory)

    try:
        backend.synchronize()
        with os.fdopen(fd, "wb") as file:
            file.write(MAGIC)
            file.write(struct.pack("<IQQ", VERSION, len(tensor_infos), len(metadata)))

            for key, value_type, value in metadata:
                _write_string(file, key)
                file.write(struct.pack("<I", value_type))
                _write_value(file, value_type, value)

            for name, tensor, offset in tensor_infos:
                shape = tuple(int(x) for x in tensor.shape)
                _write_string(file, name)
                file.write(struct.pack("<I", len(shape)))
                file.write(struct.pack("<" + "Q" * len(shape), *reversed(shape)))
                file.write(struct.pack("<IQ", GGML_TYPE_F32, offset))

            data_start = _align(file.tell(), ALIGNMENT)
            file.write(b"\0" * (data_start - file.tell()))

            for _, tensor, offset in tensor_infos:
                target = data_start + offset
                file.write(b"\0" * (target - file.tell()))
                array = np.asarray(backend.copy_to_host(tensor), dtype="<f4")
                np.ascontiguousarray(array).tofile(file)

            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def load_gguf(path, model, backend, model_config=None, strict=True):
    """Load an F32 checkpoint produced by save_gguf into allocated tensors."""
    with open(path, "rb") as file:
        if _read_exact(file, 4) != MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        version, tensor_count, metadata_count = struct.unpack(
            "<IQQ", _read_exact(file, 20)
        )
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {version}")

        metadata = {}
        for _ in range(metadata_count):
            key = _read_string(file)
            value_type = struct.unpack("<I", _read_exact(file, 4))[0]
            metadata[key] = _read_value(file, value_type)

        tensor_infos = {}
        for _ in range(tensor_count):
            name = _read_string(file)
            n_dims = struct.unpack("<I", _read_exact(file, 4))[0]
            dimensions = struct.unpack(
                "<" + "Q" * n_dims, _read_exact(file, 8 * n_dims)
            )
            tensor_type, offset = struct.unpack("<IQ", _read_exact(file, 12))
            tensor_infos[name] = (tuple(reversed(dimensions)), tensor_type, offset)

        alignment = int(metadata.get("general.alignment", ALIGNMENT))
        data_start = _align(file.tell(), alignment)

    architecture = metadata.get("general.architecture")
    if architecture != "autograd-gpt2":
        raise ValueError(f"unsupported GGUF architecture: {architecture!r}")
    if model_config is not None:
        saved_config = json.loads(metadata.get("autograd.model_config", "{}"))
        if saved_config != model_config:
            raise ValueError(
                f"GGUF model config does not match: saved={saved_config}, "
                f"current={model_config}"
            )

    parameters = dict(named_parameters(model))
    missing = sorted(set(parameters) - set(tensor_infos))
    unexpected = sorted(set(tensor_infos) - set(parameters))
    if strict and (missing or unexpected):
        raise ValueError(
            f"GGUF tensor names do not match model; missing={missing}, "
            f"unexpected={unexpected}"
        )

    file_size = os.path.getsize(path)
    loaded = 0
    for name, tensor in parameters.items():
        if name not in tensor_infos:
            continue
        shape, tensor_type, offset = tensor_infos[name]
        if tensor_type != GGML_TYPE_F32:
            raise TypeError(f"{name}: GGUF tensor type {tensor_type} is not F32")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: checkpoint shape {shape}, model shape {tensor.shape}")
        byte_offset = data_start + offset
        byte_count = tensor.numel * 4
        if byte_offset + byte_count > file_size:
            raise ValueError(f"{name}: tensor data extends past end of GGUF file")
        source = np.memmap(
            path, mode="r", dtype="<f4", offset=byte_offset, shape=shape
        )
        backend.copy_from_host(tensor, np.asarray(source, dtype=np.float32))
        loaded += 1

    backend.synchronize()
    metadata["autograd.loaded_tensor_count"] = loaded
    return metadata

def read_gguf_metadata(path):
    """Read metadata header only without allocating model tensors."""
    with open(path, "rb") as file:
        if _read_exact(file, 4) != MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        version, _, metadata_count = struct.unpack("<IQQ", _read_exact(file, 20))
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {version}")

        metadata = {}
        for _ in range(metadata_count):
            key = _read_string(file)
            value_type = struct.unpack("<I", _read_exact(file, 4))[0]
            metadata[key] = _read_value(file, value_type)

    if metadata.get("general.architecture") != "autograd-gpt2":
        raise ValueError("expected autograd-gpt2 GGUF checkpoint")
    return metadata


def storage_view(backend, storage):
    if hasattr(storage.ptr, "buffer_id"):
        tensor = Tensor(
            storage.shape, storage.dtype, device="mps", requires_grad=False
        )
        tensor.storage = storage
        return backend.numpy_view(tensor)
    return storage.data.reshape(storage.shape)


def optimizer_storages(optimizer):
    if not isinstance(optimizer.dispatches, tuple):
        raise TypeError("Optimizer checkpoint requires Metal tuple dispatches")

    result = {}
    for i, storage in enumerate(optimizer.m):
        result[f"m_{i}"] = storage
    for i, storage in enumerate(optimizer.v):
        result[f"v_{i}"] = storage

    if optimizer.state_mode != "fp32":
        for i, item in enumerate(optimizer.dispatches):
            kind = item[0]
            if kind == "mini8bit":
                # item[13] 为 quantized；n 小于 4096 时 meta 的 dtype 为 None，无需保存
                if bool(item[13]):
                    result[f"absmax_m_{i}"] = item[4]
            elif kind == "q8":
                result[f"absmax_m_{i}"] = item[5]
                result[f"absmax_v_{i}"] = item[6]
            elif kind != "fp32":
                raise ValueError(f"Unsupported optimizer dispatch: {kind!r}")

    if optimizer.qmap_m is not None:
        result["qmap_m"] = optimizer.qmap_m
    if optimizer.qmap_v is not None:
        result["qmap_v"] = optimizer.qmap_v

    return result


def optimizer_signature(optimizer, model):
    if not isinstance(optimizer.dispatches, tuple):
        raise TypeError("Optimizer checkpoint requires Metal tuple dispatches")

    dispatches = []
    for item in optimizer.dispatches:
        if optimizer.state_mode == "fp32":
            if len(item) != 6:
                raise ValueError("Unexpected fp32 dispatch layout")
            dispatches.append(["fp32_group", int(item[4]), int(item[5])])
            continue

        kind = item[0]
        if kind == "mini8bit":
            if len(item) != 14:
                raise ValueError("Unexpected mini8bit dispatch layout")
            dispatches.append([kind, *map(int, item[6:])])
        elif kind == "q8":
            if len(item) != 9:
                raise ValueError("Unexpected q8 dispatch layout")
            dispatches.append([kind, int(item[7]), int(item[8])])
        elif kind == "fp32":
            if len(item) != 7:
                raise ValueError("Unexpected fp32 dispatch layout")
            dispatches.append([kind, int(item[5]), int(item[6])])
        else:
            raise ValueError(f"Unsupported optimizer dispatch: {kind!r}")

    return {
        "state_mode": optimizer.state_mode,
        "parameters": [
            [name, list(tensor.shape)] for name, tensor in named_parameters(model)
        ],
        "dispatches": dispatches,
    }


def save_optimizer(path, optimizer, model, backend, extra_meta=None):
    """Atomically save AdamW state (fp32 / 8bit / mini) to .optim.npz."""
    backend.synchronize()
    metadata = optimizer_signature(optimizer, model)
    metadata.update({
        "t": int(optimizer.t),
        "betas": list(optimizer.betas),
        "eps": float(optimizer.eps),
    })
    if extra_meta:
        metadata.update(extra_meta)

    arrays = {
        name: storage_view(backend, storage).copy()
        for name, storage in optimizer_storages(optimizer).items()
    }

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".optim-", suffix=".npz", dir=directory)
    try:
        with os.fdopen(fd, "wb") as file:
            np.savez(file, metadata=np.array(json.dumps(metadata)), **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def load_optimizer(path, optimizer, model, backend):
    """Load AdamW state from .optim.npz if present; return saved metadata dict."""
    if not os.path.isfile(path):
        print(f"Optimizer checkpoint not found; starting fresh AdamW ({optimizer.state_mode})")
        return {}

    backend.synchronize()
    storages = optimizer_storages(optimizer)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        expected = optimizer_signature(optimizer, model)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Incompatible optimizer state: {key}")
        if set(data.files) != set(storages) | {"metadata"}:
            raise ValueError("Optimizer buffer names do not match")

        for name, storage in storages.items():
            array = data[name]
            if array.shape != storage.shape or array.dtype != np.dtype(storage.dtype):
                raise ValueError(f"Optimizer buffer mismatch: {name}")

        for name, storage in storages.items():
            storage_view(backend, storage)[:] = data[name]

    optimizer.t = int(metadata["t"])
    optimizer.betas = tuple(metadata["betas"])
    optimizer.eps = float(metadata["eps"])
    print(f"Loaded AdamW ({optimizer.state_mode}) state at optimizer step {optimizer.t}")
    return metadata
