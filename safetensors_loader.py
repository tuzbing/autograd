"""Load Qwen2.5-Coder safetensors weights into module.QwenCoder.

This loader intentionally has no dependency on torch or safetensors. It reads
 the safetensors header directly, decodes BF16/F16/F32 data to float32, converts
 HuggingFace Qwen parameter names/layouts to this repository's Tensor layout,
and copies each parameter into the supplied backend.
"""

import json
import mmap
import os
import struct
from pathlib import Path

import numpy as np

from gguf_checkpoint import named_parameters


_HEADER_SIZE = 8
_SUPPORTED_DTYPES = {
    "BOOL": (np.dtype("u1"), 1),
    "U8": (np.dtype("u1"), 1),
    "I8": (np.dtype("i1"), 1),
    "U16": (np.dtype("<u2"), 2),
    "I16": (np.dtype("<i2"), 2),
    "U32": (np.dtype("<u4"), 4),
    "I32": (np.dtype("<i4"), 4),
    "U64": (np.dtype("<u8"), 8),
    "I64": (np.dtype("<i8"), 8),
    "F16": (np.dtype("<f2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "F64": (np.dtype("<f8"), 8),
    # BF16 is represented as uint16 on disk; conversion is handled below.
    "BF16": (np.dtype("<u2"), 2),
}


def _read_header(path):
    path = os.fspath(path)
    with open(path, "rb") as file:
        raw = file.read(_HEADER_SIZE)
        if len(raw) != _HEADER_SIZE:
            raise ValueError("truncated safetensors header length")
        header_size = struct.unpack("<Q", raw)[0]
        file_size = os.fstat(file.fileno()).st_size
        if header_size > file_size - _HEADER_SIZE:
            raise ValueError("safetensors header extends past end of file")
        header_bytes = file.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("truncated safetensors header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header_size, header, file_size


def _numel(shape):
    result = 1
    for value in shape:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid tensor shape: {shape!r}")
        result *= value
    return result


def _tensor_from_file(mm, data_start, entry, file_size, path):
    if not isinstance(entry, dict):
        raise ValueError("invalid safetensors tensor entry")
    dtype_name = entry.get("dtype")
    shape = tuple(entry.get("shape", ()))
    offsets = entry.get("data_offsets")
    if dtype_name not in _SUPPORTED_DTYPES:
        raise TypeError(f"{path}: unsupported safetensors dtype {dtype_name!r}")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"{path}: invalid data_offsets")
    start, end = offsets
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError(f"{path}: invalid data_offsets values")

    dtype, itemsize = _SUPPORTED_DTYPES[dtype_name]
    expected_bytes = _numel(shape) * itemsize
    if end - start != expected_bytes:
        raise ValueError(
            f"{path}: byte range {end - start} does not match "
            f"shape {shape} and dtype {dtype_name}"
        )
    absolute_start = data_start + start
    absolute_end = data_start + end
    if absolute_end > file_size:
        raise ValueError(f"{path}: tensor data extends past end of file")

    if expected_bytes == 0:
        raw = np.empty(shape, dtype=dtype)
    else:
        raw = np.frombuffer(mm, dtype=dtype, count=_numel(shape), offset=absolute_start)
        raw = raw.reshape(shape)

    if dtype_name == "BF16":
        # BF16 bits occupy the high 16 bits of an IEEE754 float32.
        words = np.asarray(raw, dtype=np.uint16)
        result = (words.astype(np.uint32) << 16).view(np.float32)
        return np.ascontiguousarray(result)
    if dtype_name == "BOOL":
        return np.ascontiguousarray(raw.astype(np.float32))
    return np.ascontiguousarray(raw.astype(np.float32))


def _check_config(config, header):
    if config is None:
        return
    checks = {
        "vocab_size": ("vocab_size", 151936),
        "hidden_size": ("n_embd", 1536),
        "num_hidden_layers": ("n_layer", 28),
        "num_attention_heads": ("n_head", 12),
        "num_key_value_heads": ("n_kv_head", 2),
        "intermediate_size": ("intermediate_size", 8960),
    }
    for hf_key, (local_key, default) in checks.items():
        if hf_key in config and local_key in config and int(config[hf_key]) != int(config[local_key]):
            raise ValueError(
                f"configuration mismatch for {hf_key}: "
                f"source={config[hf_key]}, model={config[local_key]}"
            )


def _target_arrays(source, model):
    def get(name):
        if name not in source:
            raise KeyError(f"missing source tensor: {name}")
        return source[name]

    result = {}
    result["tok_emb.weight"] = get("model.embed_tokens.weight")
    result["final_norm_weight"] = get("model.norm.weight")

    layers = sorted({
        int(name.split(".")[2])
        for name in source
        if name.startswith("model.layers.")
    })

    if not layers or layers != list(range(max(layers) + 1)):
        raise ValueError("safetensors layers are missing or not contiguous")

    for i in layers:
        prefix = f"model.layers.{i}"
        target = f"blocks.{i}"

        result[f"{target}.input_norm_weight"] = get(
            f"{prefix}.input_layernorm.weight"
        )
        result[f"{target}.post_norm_weight"] = get(
            f"{prefix}.post_attention_layernorm.weight"
        )

        for proj in ("q_proj", "k_proj", "v_proj"):
            result[f"{target}.{proj}.weight"] = get(
                f"{prefix}.self_attn.{proj}.weight"
            ).T
            result[f"{target}.{proj}.bias"] = get(
                f"{prefix}.self_attn.{proj}.bias"
            )

        result[f"{target}.proj.weight"] = get(
            f"{prefix}.self_attn.o_proj.weight"
        ).T

        gate = get(f"{prefix}.mlp.gate_proj.weight").T
        up = get(f"{prefix}.mlp.up_proj.weight").T
        result[f"{target}.mlp.fc1.weight"] = np.ascontiguousarray(
            np.concatenate((gate, up), axis=1)
        )
        result[f"{target}.mlp.fc2.weight"] = get(
            f"{prefix}.mlp.down_proj.weight"
        ).T

    return result


def load_safetensors(path, model, backend, config=None, strict=True):
    """Load Qwen2.5-Coder weights into an already-built QwenCoder model.

    Parameters
    ----------
    path : str or pathlib.Path
        HuggingFace `model.safetensors` file.
    model : module.QwenCoder
        Allocated model whose Tensor shapes must match the checkpoint.
    backend : Runtime backend
        Backend providing `copy_from_host(tensor, numpy_array)` and
        `synchronize()`.
    config : dict, optional
        HuggingFace config.json contents, optionally augmented with local keys.
    strict : bool
        Require every source tensor and every trainable model tensor to match.

    Returns
    -------
    dict
        Loader metadata and the number of copied tensors.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    header_size, header, file_size = _read_header(path)
    data_start = _HEADER_SIZE + header_size
    tensor_entries = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensor_entries:
        raise ValueError("safetensors file contains no tensors")

    # Validate all ranges before allocating or copying model data.
    ranges = []
    for name, entry in tensor_entries.items():
        if not isinstance(name, str):
            raise ValueError("safetensors tensor name must be a string")
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{name}: invalid data_offsets")
        ranges.append((int(offsets[0]), int(offsets[1]), name))
    ranges.sort()
    previous_end = 0
    for start, end, name in ranges:
        if start < previous_end:
            raise ValueError(f"overlapping safetensors ranges near {name}")
        previous_end = end
        if data_start + end > file_size:
            raise ValueError(f"{name}: data extends past end of file")

    _check_config(config, config or {})

    with open(path, "rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        source = {
            name: _tensor_from_file(mm, data_start, entry, file_size, name)
            for name, entry in tensor_entries.items()
        }

    targets = _target_arrays(source, model)
    parameters = dict(named_parameters(model))
    target_names = set(parameters)
    source_target_names = set(targets)
    missing = sorted(target_names - source_target_names)
    unexpected = sorted(source_target_names - target_names)
    if strict and (missing or unexpected):
        raise ValueError(
            "converted tensor names do not match model; "
            f"missing={missing}, unexpected={unexpected}"
        )

    copied = 0
    for name, tensor in parameters.items():
        if name not in targets:
            continue
        values = np.asarray(targets[name], dtype=np.float32)
        if tuple(values.shape) != tuple(tensor.shape):
            raise ValueError(
                f"{name}: checkpoint shape {values.shape}, "
                f"model shape {tensor.shape}"
            )
        backend.copy_from_host(tensor, values)
        copied += 1

    if strict:
        layer_ids = sorted({
            int(name.split(".")[2])
            for name in source
            if name.startswith("model.layers.")
        })
        expected_source = {"model.embed_tokens.weight", "model.norm.weight"}
        for i in layer_ids:
            prefix = f"model.layers.{i}"
            expected_source.update({
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            })
        for i in layer_ids:
            prefix = f"model.layers.{i}"
            expected_source.update({
                f"{prefix}.self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj")
            })
            expected_source.update({
                f"{prefix}.self_attn.{p}.bias" for p in ("q_proj", "k_proj", "v_proj")
            })
            expected_source.update({
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            })
        extra_source = sorted(set(source) - expected_source)
        missing_source = sorted(expected_source - set(source))
        if extra_source or missing_source:
            raise ValueError(
                "safetensors source names mismatch; "
                f"missing={missing_source}, unexpected={extra_source}"
            )

    backend.synchronize()
    return {
        "path": str(path),
        "tensor_count": len(tensor_entries),
        "loaded_tensor_count": copied,
        "metadata": header.get("__metadata__", {}),
    }


if __name__ == "__main__":
    raise SystemExit(
        "Import load_safetensors() from this module; model/backend are required."
    )
