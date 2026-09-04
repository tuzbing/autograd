#!/usr/bin/env python3
"""Correctness checker for the exact GPT training graph used by train_autograd.py.

CPU is a correctness oracle with independent activation/gradient storage.
MPS keeps Runtime.bind_memory() unchanged, including arena reuse, aliases and
recomputation.  Only operators present in module.GPT2.forward() are checked.
"""

import argparse
import functools
import gc
import math
import os
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np


ATOL = 1e-6
RTOL = 1e-6
SEED = 1333
FAILURES = []
PRINT_PASSES = True


def _locate_project_root():
    candidates = []
    env_root = os.environ.get("AUTOGRAD_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve().parent
    candidates.extend((Path.cwd(), Path.cwd().parent, here, here.parent))
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "debug" / "runtime.py").is_file() and (candidate / "train_autograd.py").is_file():
            return candidate
        if candidate.name == "debug" and (candidate / "runtime.py").is_file():
            return candidate.parent
    raise RuntimeError(
        "Cannot locate autograd project. Run from the repository/debug directory "
        "or set AUTOGRAD_ROOT=/absolute/path/to/autograd."
    )


PROJECT_ROOT = _locate_project_root()
DEBUG_ROOT = PROJECT_ROOT / "debug"
os.chdir(DEBUG_ROOT)
sys.path.insert(0, str(DEBUG_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

import runtime  # noqa: E402
from module import GPT2  # noqa: E402
from op import (  # noqa: E402
    FusedRMSNormFlashAttentionOp,
    FusedRMSNormMLPOp,
    FusedLinearAddOp,
    FusedLinearCrossEntropyOp,
)
from optimizer import AdamW # noqa: E402
from tensor import Tensor  # noqa: E402

Runtime = runtime.Runtime


def make_tensor(shape, dtype, device, requires_grad, init_val):
    return Tensor(
        shape=shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        init_val=np.asarray(init_val, dtype=dtype).copy(),
    )


def iter_gpt2_named_tensors(model):
    yield "tok_emb.weight", model.tok_emb.weight

    for index, block in enumerate(model.blocks):
        p = f"blocks[{index}]"

        yield f"{p}.attn.q_proj.weight", block.attn.q_proj.weight
        yield f"{p}.attn.q_proj.bias", block.attn.q_proj.bias
        yield f"{p}.attn.k_proj.weight", block.attn.k_proj.weight
        yield f"{p}.attn.k_proj.bias", block.attn.k_proj.bias
        yield f"{p}.attn.v_proj.weight", block.attn.v_proj.weight
        yield f"{p}.attn.v_proj.bias", block.attn.v_proj.bias
        yield f"{p}.attn.proj.weight", block.attn.proj.weight
        yield f"{p}.attn.proj.bias", block.attn.proj.bias

        yield f"{p}.mlp.fc1.weight", block.mlp.fc1.weight
        yield f"{p}.mlp.fc2.weight", block.mlp.fc2.weight

    yield "lm_head.weight", model.lm_head.weight


@dataclass(frozen=True)
class Config:
    B: int
    T: int
    V: int
    C: int
    H: int
    D: int
    L: int
    dtype: object = np.float32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the exact GPT2 training operator path on CPU and MPS"
    )
    # Small defaults keep the CPU oracle practical.  The dimension formula and
    # option names match train_autograd.py, so production shapes can be supplied
    # directly on the command line.
    parser.add_argument("--depth", "--L", dest="depth", type=int, default=2)
    parser.add_argument("--aspect-ratio", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-seq-len", "--T", dest="max_seq_len", type=int, default=32)
    parser.add_argument("--device-batch-size", "--B", dest="device_batch_size", type=int, default=2)
    parser.add_argument("--vocab-size", "--V", dest="vocab_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--kernel-block",
        type=int,
        default=0,
        help="block used for internal fused-kernel diagnostics (default: 0)",
    )
    parser.add_argument(
        "--skip-internal-kernels",
        action="store_true",
        help="run exact graph and per-operator checks only",
    )
    parser.add_argument(
        "--skip-optimizer",
        action="store_true",
        help="skip the AdamW training-step check",
    )
    parser.add_argument(
        "--skip-gradient-accumulation",
        action="store_true",
        help="skip the no-extra-buffer gradient accumulation checks",
    )
    parser.add_argument(
        "--gradient-accumulation-only",
        action="store_true",
        help="run only the no-extra-buffer gradient accumulation checks",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="print failures and summary events, but omit individual passing comparisons",
    )
    args = parser.parse_args()

    if min(
        args.depth,
        args.aspect_ratio,
        args.head_dim,
        args.max_seq_len,
        args.device_batch_size,
        args.vocab_size,
    ) <= 0:
        parser.error("all model dimensions must be positive")

    if args.head_dim > 512:
        parser.error(
            "--head-dim must be <= 512: the current FA dispatch "
            "only registers 64/128/256/512 capacity kernels"
        )

    base_dim = args.depth * args.aspect_ratio
    model_dim = (
        (base_dim + args.head_dim - 1)
        // args.head_dim
        * args.head_dim
    )

    if model_dim % 4 != 0:
        parser.error(
            f"model dimension C={model_dim} is not divisible by 4; "
            "fused_ln_fa_qkv and fused_ln_fa_project_kv currently "
            "require C % 4 == 0"
        )

    heads = model_dim // args.head_dim
    cfg = Config(
        B=args.device_batch_size,
        T=args.max_seq_len,
        V=args.vocab_size,
        C=model_dim,
        H=heads,
        D=args.head_dim,
        L=args.depth,
    )

    if not 0 <= args.kernel_block < cfg.L:
        parser.error(f"--kernel-block must be in [0, {cfg.L})")

    return args, cfg


def reset_runtime(device):
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device)
    Runtime._default_runtime = rt
    return rt


def copy_to_host(rt, tensor):
    return np.asarray(rt.backend.copy_to_host(tensor)).copy()


def read_storage_f32(rt, storage, shape=None):
    shape = tuple(shape if shape is not None else storage.shape)
    if rt.allocator.device == "cpu":
        return np.asarray(storage.data).reshape(shape).copy()
    return np.asarray(
        rt.backend.ctx.numpy_view_float32(
            storage.ptr.buffer_id, storage.ptr.offset, shape
        )
    ).copy()


def check(name, got, ref, *, fail_fast=False):
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        raise AssertionError(f"{name}: shape {got.shape} != {ref.shape}")
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-30)
    max_abs = float(diff.max(initial=0.0))
    max_rel = float((diff / denom).max(initial=0.0))
    ok = bool(np.allclose(got, ref, atol=ATOL, rtol=RTOL, equal_nan=False))
    if PRINT_PASSES or not ok:
        print(
            f"  {'PASS' if ok else 'FAIL'} {name}: max_abs={max_abs:.8e}, "
            f"max_rel={max_rel:.8e}, atol={ATOL:g}, rtol={RTOL:g}"
        )
    if fail_fast and not ok:
        bad = np.argwhere(~np.isclose(got, ref, atol=ATOL, rtol=RTOL, equal_nan=False))
        first = tuple(int(v) for v in bad[0]) if bad.size else ()
        raise AssertionError(
            f"{name} mismatch at {first}: got={got[first]!r}, ref={ref[first]!r}, "
            f"max_abs={max_abs:.8e}, max_rel={max_rel:.8e}"
        )
    if not ok:
        FAILURES.append((name, max_abs, max_rel))
    return ok


def op_name(node):
    return node.op.op_name


def topology_labels(topo):
    counts = defaultdict(int)
    labels = []
    for node in topo:
        name = op_name(node)
        labels.append(f"{name}[{counts[name]}]")
        counts[name] += 1
    return labels


def verify_training_topology(topo, cfg):
    expected = Counter({
        "Gather": 1,
        "Add": cfg.L,
        "FusedRMSNormFlashAttention": cfg.L,
        "FusedLinearAdd": cfg.L,
        "FusedRMSNormMLP": cfg.L,
        "FusedLinearCrossEntropy": 1,
    })

    actual = Counter(op_name(node) for node in topo)

    if actual != expected:
        raise AssertionError(
            "GPT2.forward topology differs from expected training graph:\n"
            f"  expected={dict(expected)}\n"
            f"  actual={dict(actual)}"
        )

    allowed = set(expected)
    unexpected = [
        op_name(node)
        for node in topo
        if op_name(node) not in allowed
    ]

    if unexpected:
        raise AssertionError(
            f"unexpected training operators: {unexpected}"
        )

    print(
        "Training topology:",
        " -> ".join(topology_labels(topo)),
    )


def build_parameter_arrays(cfg, seed):
    rng = np.random.default_rng(seed)
    model = GPT2(cfg.V, cfg.C, cfg.T, cfg.H, cfg.B, cfg.L, cfg.dtype, "cpu")
    arrays = {}
    for name, tensor in iter_gpt2_named_tensors(model):
        arrays[name] = (rng.standard_normal(tensor.shape).astype(np.float32) * 0.02).copy()
    return arrays


def initialize_model(model, arrays):
    names = []
    for name, tensor in iter_gpt2_named_tensors(model):
        if name not in arrays:
            raise KeyError(f"missing parameter {name}")
        tensor.init_val = arrays[name].copy()
        names.append(name)
    if set(names) != set(arrays):
        raise AssertionError("parameter array set does not match GPT2")


def build_inputs(cfg, seed):
    rng = np.random.default_rng(seed + 1)
    input_ids = rng.integers(0, cfg.V, (cfg.B, cfg.T), dtype=np.int32)
    # pos_ids = np.broadcast_to(np.arange(cfg.T, dtype=np.int32), (cfg.B, cfg.T)).copy()
    labels = rng.integers(0, cfg.V, (cfg.B, cfg.T), dtype=np.int32)
    return input_ids, labels


def all_graph_tensors(rt):
    seen = set()
    tensors = []
    for node in rt.topo:
        for tensor in node.inputs + node.outputs:
            if id(tensor) not in seen:
                seen.add(id(tensor))
                tensors.append(tensor)
    return tensors


def isolate_cpu_reference_memory(rt):
    """Replace every planner-owned CPU value and scratch region with private storage."""
    if rt.allocator.device != "cpu":
        raise ValueError("CPU isolation called for non-CPU runtime")
    view_names = {"View", "Reshape", "Permute", "Transpose"}
    tensors = all_graph_tensors(rt)
    for tensor in tensors:
        if tensor.op is None or op_name(tensor.op) in view_names:
            continue
        tensor.storage = rt.allocator.alloc(tensor.shape, tensor.dtype)
        if tensor.grad is not None:
            tensor.grad.storage = rt.allocator.alloc(tensor.grad.shape, tensor.grad.dtype)
            rt.backend.fill(tensor.grad, 0.0)
    for tensor in tensors:
        if tensor.op is None or op_name(tensor.op) not in view_names:
            continue
        base = tensor.op.inputs[0]
        tensor.storage = rt.allocator.view(base.storage, 0, tensor.shape, tensor.dtype)
        if tensor.grad is not None and base.grad is not None:
            tensor.grad.storage = rt.allocator.view(base.grad.storage, 0, tensor.grad.shape, tensor.grad.dtype)
    for node in rt.topo:
        for phase in ("forward", "backward"):
            layout = getattr(node, f"scratch_{'fwd' if phase == 'forward' else 'bwd'}_layout", None)
            if layout is None:
                continue
            private_storage = rt.allocator.alloc((layout.nbytes,), np.uint8)
            private_scratch = rt.scratch_bind(private_storage, layout)
            bound = getattr(node, phase)
            if isinstance(bound, functools.partial):
                keywords = dict(bound.keywords or {})
                keywords["scratch"] = private_scratch
                bound = functools.partial(bound.func, *bound.args, **keywords)
            else:
                bound = functools.partial(bound, scratch=private_scratch)
            setattr(node, phase, bound)
    rt.unified_arena = None


@dataclass
class GraphState:
    rt: object
    model: object
    loss: object
    input_ids: object
    labels: object
    param_by_name: dict
    param_name_by_id: dict
    labels_by_node: dict


def build_graph(device, cfg, arrays, input_ids, labels):
    rt = reset_runtime(device)
    model = GPT2(cfg.V, cfg.C, cfg.T, cfg.H, cfg.B, cfg.L, cfg.dtype, device)
    initialize_model(model, arrays)
    input_t = make_tensor(input_ids.shape, np.int32, device, False, input_ids)
    labels_t = make_tensor(labels.shape, np.int32, device, False, labels)
    # This is deliberately the model call used by train_autograd.py.  Do not
    # manually reconstruct the graph here.
    loss = model(input_t, targets=labels_t)
    rt.loss = loss
    rt.build_topo(loss)
    verify_training_topology(rt.topo, cfg)
    rt.bind_memory()
    if device == "cpu":
        isolate_cpu_reference_memory(rt)
    rt.is_compiled = True
    param_by_name = dict(iter_gpt2_named_tensors(model))
    return GraphState(
        rt=rt,
        model=model,
        loss=loss,
        input_ids=input_t,
        labels=labels_t,
        param_by_name=param_by_name,
        param_name_by_id={id(t): n for n, t in param_by_name.items()},
        labels_by_node={id(node): label for node, label in zip(rt.topo, topology_labels(rt.topo))},
    )


def zero_every_grad(state):
    seen = set()
    for tensor in all_graph_tensors(state.rt):
        if tensor.grad is not None and id(tensor.grad) not in seen:
            seen.add(id(tensor.grad))
            state.rt.backend.fill(tensor.grad, 0.0)


def run_exact_training_graph(state, repeats=2):
    """Call Runtime.execute() exactly as the compiled training loop does."""
    results = []
    for _ in range(repeats):
        state.rt.execute()
        state.rt.backend.synchronize()
        results.append({
            "loss": copy_to_host(state.rt, state.loss),
            "parameter_grads": {
                name: copy_to_host(state.rt, tensor.grad)
                for name, tensor in state.param_by_name.items()
            },
        })
    return results


def compare_exact_graph(cpu, mps):
    print("\n=== Exact Runtime.execute() training path ===")
    if len(cpu) != len(mps):
        raise AssertionError("CPU/MPS repeat count differs")
    for repeat, (cpu_result, mps_result) in enumerate(zip(cpu, mps), 1):
        print(f"  [training execution {repeat}]")
        check(f"repeat[{repeat}]/loss", mps_result["loss"], cpu_result["loss"])
        for name in sorted(cpu_result["parameter_grads"]):
            check(
                f"repeat[{repeat}]/parameter.grad/{name}",
                mps_result["parameter_grads"][name],
                cpu_result["parameter_grads"][name],
            )


def run_staged_graph(state, reference_inputs=None):
    rt = state.rt
    forward = {}
    forward_inputs = {}
    backward = {}

    rt.backend.begin_pass()

    for node in rt.topo:
        label = state.labels_by_node[id(node)]

        if reference_inputs is not None:
            expected_inputs = reference_inputs[label]

            if len(expected_inputs) != len(node.inputs):
                raise AssertionError(
                    f"{label}: input count differs"
                )

            for tensor, value in zip(
                node.inputs,
                expected_inputs,
            ):
                rt.backend.copy_from_host(
                    tensor,
                    np.asarray(
                        value,
                        dtype=tensor.dtype,
                    ),
                )

        forward_inputs[label] = [
            copy_to_host(rt, tensor)
            for tensor in node.inputs
        ]

        node.forward(node.inputs, node.outputs)
        rt.backend.end_pass()
        rt.backend.synchronize()

        forward[label] = [
            copy_to_host(rt, out)
            for out in node.outputs
        ]

        rt.backend.begin_pass()

    rt.backend.end_pass()

    if rt.allocator.device == "cpu":
        zero_every_grad(state)

    rt.backend.fill(state.loss.grad, 1.0)
    rt.backend.begin_pass()

    for node in reversed(rt.topo):
        label = state.labels_by_node[id(node)]

        output_grads = [
            None if out.grad is None
            else copy_to_host(rt, out.grad)
            for out in node.outputs
        ]

        input_grads_before = [
            None if tensor.grad is None
            else copy_to_host(rt, tensor.grad)
            for tensor in node.inputs
        ]

        node.backward(node.inputs, node.outputs)
        rt.backend.end_pass()
        rt.backend.synchronize()

        inputs = []

        for index, tensor in enumerate(node.inputs):
            if tensor.grad is None:
                continue

            inputs.append({
                "index": index,
                "parameter": state.param_name_by_id.get(
                    id(tensor)
                ),
                "grad": copy_to_host(rt, tensor.grad),
            })

        backward[label] = {
            "output_grads": output_grads,
            "input_grads_before": input_grads_before,
            "inputs": inputs,
        }

        rt.backend.begin_pass()

    rt.backend.end_pass()

    return {
        "forward": forward,
        "forward_inputs": forward_inputs,
        "backward": backward,
    }


def compare_staged_graph(cpu, mps):
    print("\n=== Per-operator exact training path ===")
    if list(cpu["forward"]) != list(mps["forward"]):
        raise AssertionError("CPU/MPS forward topology labels differ")
    for label in cpu["forward"]:
        for i, (mps_value, cpu_value) in enumerate(zip(mps["forward"][label], cpu["forward"][label])):
            check(f"{label}/forward/output[{i}]", mps_value, cpu_value)
    for label in reversed(list(cpu["forward"])):
        cpu_bwd = cpu["backward"][label]
        mps_bwd = mps["backward"][label]
        for i, (mps_value, cpu_value) in enumerate(zip(mps_bwd["output_grads"], cpu_bwd["output_grads"])):
            if cpu_value is not None:
                check(f"{label}/backward/output[{i}].grad", mps_value, cpu_value)
        if len(cpu_bwd["inputs"]) != len(mps_bwd["inputs"]):
            raise AssertionError(f"{label}: different differentiable input count")
        for cpu_item, mps_item in zip(cpu_bwd["inputs"], mps_bwd["inputs"]):
            if (cpu_item["index"], cpu_item["parameter"]) != (mps_item["index"], mps_item["parameter"]):
                raise AssertionError(f"{label}: input mapping differs")
            suffix = cpu_item["parameter"] or f"input[{cpu_item['index']}]"
            check(f"{label}/backward/{suffix}.grad", mps_item["grad"], cpu_item["grad"])


def layernorm_values(x, gamma, beta, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    inv_std = np.float32(1.0) / np.sqrt(np.var(x, axis=-1, keepdims=True) + np.float32(eps))
    x_hat = (x - mean) * inv_std
    x_norm = x_hat * gamma + beta
    return mean.astype(np.float32), inv_std.astype(np.float32), x_hat.astype(np.float32), x_norm.astype(np.float32)

def rmsnorm_values(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)

    mean_square = np.mean(
        x * x,
        axis=-1,
        keepdims=True,
    ).astype(np.float32)

    inv_rms = (
        np.float32(1.0)
        / np.sqrt(mean_square + np.float32(eps))
    ).astype(np.float32)

    x_norm = (x * inv_rms).astype(np.float32)

    return mean_square, inv_rms, x_norm


def gelu(x):
    x = np.asarray(x, dtype=np.float32)
    inner = np.sqrt(np.float32(2.0 / np.pi)) * (x + np.float32(0.044715) * x * x * x)
    inner = np.clip(inner, -20.0, 20.0)
    return (np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)


def gelu_grad(x):
    x = np.asarray(x, dtype=np.float32)
    c = np.sqrt(np.float32(2.0 / np.pi))
    inner = np.clip(c * (x + np.float32(0.044715) * x * x * x), -20.0, 20.0)
    t = np.tanh(inner)
    return (
        np.float32(0.5) * (np.float32(1.0) + t)
        + np.float32(0.5) * x * (np.float32(1.0) - t * t)
        * c * (np.float32(1.0) + np.float32(3.0 * 0.044715) * x * x)
    ).astype(np.float32)

def apply_rope_reference(q, k, cos, sin):
    # q/k: (B, H, T, D)
    d2 = q.shape[-1] // 2

    q1 = q[..., :d2].copy()
    q2 = q[..., d2:].copy()
    k1 = k[..., :d2].copy()
    k2 = k[..., d2:].copy()

    c = cos[None, None, :, :]
    s = sin[None, None, :, :]

    q[..., :d2] = q1 * c + q2 * s
    q[..., d2:] = -q1 * s + q2 * c

    k[..., :d2] = k1 * c + k2 * s
    k[..., d2:] = -k1 * s + k2 * c

def apply_rope_backward_reference(dq, dk, cos, sin):
    d2 = dq.shape[-1] // 2

    dq1 = dq[..., :d2].copy()
    dq2 = dq[..., d2:].copy()
    dk1 = dk[..., :d2].copy()
    dk2 = dk[..., d2:].copy()

    c = cos[None, None, :, :]
    s = sin[None, None, :, :]

    dq[..., :d2] = dq1 * c - dq2 * s
    dq[..., d2:] = dq1 * s + dq2 * c

    dk[..., :d2] = dk1 * c - dk2 * s
    dk[..., d2:] = dk1 * s + dk2 * c

def attention_oracle(
    x,
    params,
    upstream,
    heads,
    eps=1e-6,
    causal=True,
    residual_grad=None,
    rope_theta=100000.0,
):
    x = np.asarray(x, dtype=np.float32)
    upstream = np.asarray(upstream, dtype=np.float32)

    B, T, C = x.shape
    H = heads
    D = C // H

    if D % 2 != 0:
        raise ValueError(
            f"RoPE requires an even head dimension, got D={D}"
        )

    positions = np.arange(
        T,
        dtype=np.float32,
    )[:, None]

    channels = np.arange(
        0,
        D,
        2,
        dtype=np.float32,
    )[None, :]

    inv_freq = np.asarray(
        rope_theta ** (-channels / D),
        dtype=np.float32,
    )

    freqs = positions * inv_freq

    rope_cos = np.cos(freqs).astype(np.float32)
    rope_sin = np.sin(freqs).astype(np.float32)

    mean_square, inv_rms, x_norm = rmsnorm_values(
        x,
        eps=eps,
    )

    x_flat = x_norm.reshape(B * T, C)

    q_flat = (
        x_flat @ params["wq"] + params["bq"]
    ).astype(np.float32)

    k_flat = (
        x_flat @ params["wk"] + params["bk"]
    ).astype(np.float32)

    v_flat = (
        x_flat @ params["wv"] + params["bv"]
    ).astype(np.float32)

    q = (
        q_flat
        .reshape(B, T, H, D)
        .transpose(0, 2, 1, 3)
        .copy()
    )
    k = (
        k_flat
        .reshape(B, T, H, D)
        .transpose(0, 2, 1, 3)
        .copy()
    )
    v = (
        v_flat
        .reshape(B, T, H, D)
        .transpose(0, 2, 1, 3)
        .copy()
    )

    q_raw = q.copy()
    k_raw = k.copy()
    apply_rope_reference(q, k, rope_cos, rope_sin)

    scale = np.float32(1.0 / math.sqrt(D))

    scores = np.matmul(
        q,
        k.transpose(0, 1, 3, 2),
    )
    scores *= scale

    if causal:
        mask = np.triu(
            np.ones((T, T), dtype=bool),
            1,
        )
        scores[..., mask] = -np.inf

    row_max = np.max(
        scores,
        axis=-1,
        keepdims=True,
    )

    exp_scores = np.exp(scores - row_max)
    row_sum = np.sum(
        exp_scores,
        axis=-1,
        keepdims=True,
    )
    probs = exp_scores / row_sum

    out_h = np.matmul(probs, v)
    out = (
        out_h
        .transpose(0, 2, 1, 3)
        .reshape(B, T, C)
        .astype(np.float32)
    )

    dO = (
        upstream
        .reshape(B, T, H, D)
        .transpose(0, 2, 1, 3)
    )

    work = np.matmul(
        dO,
        v.transpose(0, 1, 3, 2),
    )

    d_scores = (
        work
        - np.sum(
            probs * work,
            axis=-1,
            keepdims=True,
        )
    )
    d_scores *= probs
    d_scores *= scale

    dq_rot = np.matmul(d_scores, k).astype(np.float32)
    dk_rot = np.matmul(
        d_scores.transpose(0, 1, 3, 2),
        q,
    ).astype(np.float32)
    dv = np.matmul(
        probs.transpose(0, 1, 3, 2),
        dO,
    )

    dq = dq_rot.copy()
    dk = dk_rot.copy()
    apply_rope_backward_reference(dq, dk, rope_cos, rope_sin)

    dq2 = (
        dq
        .transpose(0, 2, 1, 3)
        .reshape(B * T, C)
    )
    dk2 = (
        dk
        .transpose(0, 2, 1, 3)
        .reshape(B * T, C)
    )
    dv2 = (
        dv
        .transpose(0, 2, 1, 3)
        .reshape(B * T, C)
    )

    dx_norm = (
        dq2 @ params["wq"].T
        + dk2 @ params["wk"].T
        + dv2 @ params["wv"].T
    ).reshape(B, T, C).astype(np.float32)

    mean_dy_xnorm = np.mean(
        dx_norm * x_norm,
        axis=-1,
        keepdims=True,
    )

    dx_local = (
        inv_rms
        * (
            dx_norm
            - x_norm * mean_dy_xnorm
        )
    ).astype(np.float32)

    if residual_grad is None:
        dx = dx_local
    else:
        dx = (
            dx_local
            + np.asarray(residual_grad, dtype=np.float32)
        ).astype(np.float32)

    return {
        "mean_square": mean_square,
        "inv_rms": inv_rms,
        "x_norm": x_norm,
        "q": q.astype(np.float32),
        "k": k.astype(np.float32),
        "v": v.astype(np.float32),
        "q_raw": q_raw.astype(np.float32),
        "k_raw": k_raw.astype(np.float32),
        "probs": probs.astype(np.float32),
        "out": out,
        "lse": (
            row_max[..., 0]
            + np.log(row_sum[..., 0])
        ).astype(np.float32),
        "d_val": np.sum(
            out_h * dO,
            axis=-1,
        ).astype(np.float32),
        "dq": dq.astype(np.float32),
        "dk": dk.astype(np.float32),
        "dv": dv.astype(np.float32),
        "dq_rot": dq_rot,
        "dk_rot": dk_rot,
        "dx_norm": dx_norm,
        "dx_local": dx_local,
        "dx": dx,
        "dwq": (
            x_flat.T @ dq2
        ).astype(np.float32),
        "dwk": (
            x_flat.T @ dk2
        ).astype(np.float32),
        "dwv": (
            x_flat.T @ dv2
        ).astype(np.float32),
        "dbq": np.sum(
            dq2,
            axis=0,
        ).astype(np.float32),
        "dbk": np.sum(
            dk2,
            axis=0,
        ).astype(np.float32),
        "dbv": np.sum(
            dv2,
            axis=0,
        ).astype(np.float32),
    }


def heads_to_btc(value):
    value = np.asarray(value)
    B, H, T, D = value.shape
    return value.transpose(0, 2, 1, 3).reshape(B, T, H*D).copy()

def mlp_oracle(
    x,
    params,
    upstream,
    residual_grad,
    eps=1e-6,
):
    x = np.asarray(x, dtype=np.float32)
    upstream = np.asarray(upstream, dtype=np.float32)
    residual_grad = np.asarray(residual_grad, dtype=np.float32)

    B, T, C = x.shape
    N = B * T

    wgu = np.asarray(params["w1"], dtype=np.float32)
    wdown = np.asarray(params["w2"], dtype=np.float32)

    if wgu.shape[0] != C:
        raise AssertionError(
            f"wgu shape mismatch: expected first dimension {C}, got {wgu.shape}"
        )

    if wgu.shape[1] % 2 != 0:
        raise AssertionError(
            f"wgu second dimension must be even, got {wgu.shape}"
        )

    H = wgu.shape[1] // 2

    if wdown.shape != (H, C):
        raise AssertionError(
            f"wdown shape mismatch: expected {(H, C)}, got {wdown.shape}"
        )

    _, inv_rms, x_norm = rmsnorm_values(x, eps)

    XN = x_norm.reshape(N, C)

    # GU = X_norm @ Wgu
    GU = XN @ wgu

    G = GU[:, :H]
    U = GU[:, H:]

    sigmoid_g = 1.0 / (1.0 + np.exp(-G))
    silu_g = G * sigmoid_g

    # HIDDEN = SiLU(G) * U
    HIDDEN = silu_g * U

    # OUT = HIDDEN @ Wdown
    out = (
        HIDDEN @ wdown
    ).reshape(B, T, C).astype(np.float32)

    DOUT = upstream.reshape(N, C)

    # dWdown
    dwdown = (
        HIDDEN.T @ DOUT
    ).astype(np.float32)

    # dHIDDEN
    d_hidden = DOUT @ wdown.T

    # dG
    dsilu_dg = sigmoid_g * (
        1.0 + G * (1.0 - sigmoid_g)
    )

    d_gate = (
        d_hidden * U * dsilu_dg
    )

    # dU
    d_up = (
        d_hidden * silu_g
    )

    # dGU = [dG, dU]
    dGU = np.concatenate(
        (d_gate, d_up),
        axis=-1,
    ).astype(np.float32)

    # dWgu
    dwgu = (
        XN.T @ dGU
    ).astype(np.float32)

    # dX_norm
    dx_norm = (
        dGU @ wgu.T
    ).reshape(B, T, C).astype(np.float32)

    mean_dy_xnorm = np.mean(
        dx_norm * x_norm,
        axis=-1,
        keepdims=True,
    )

    dx_local = (
        inv_rms
        * (
            dx_norm
            - x_norm * mean_dy_xnorm
        )
    ).astype(np.float32)

    # 这里保留 residual_grad，用于验证 MLP 后的 Add 梯度传播
    dx = (
        dx_local + residual_grad
    ).astype(np.float32)

    return {
        "out": out,
        "dx": dx,
        "dx_local": dx_local,
        "dw1": dwgu,
        "dw2": dwdown,
        "gu": GU.reshape(B, T, 2 * H).astype(np.float32),
        "hidden": HIDDEN.reshape(B, T, H).astype(np.float32),
        "dgu": dGU.reshape(B, T, 2 * H).astype(np.float32),
        "dhidden": d_hidden.reshape(B, T, H).astype(np.float32),
    }

def block_parameter_dict(arrays, block, which):
    p = f"blocks[{block}]"

    if which == "fa":
        return {
            "wq": arrays[f"{p}.attn.q_proj.weight"],
            "bq": arrays[f"{p}.attn.q_proj.bias"],
            "wk": arrays[f"{p}.attn.k_proj.weight"],
            "bk": arrays[f"{p}.attn.k_proj.bias"],
            "wv": arrays[f"{p}.attn.v_proj.weight"],
            "bv": arrays[f"{p}.attn.v_proj.bias"],
        }

    if which == "mlp":
        return {
            "w1": arrays[f"{p}.mlp.fc1.weight"],
            "w2": arrays[f"{p}.mlp.fc2.weight"],
        }

    raise ValueError(which)


def _eps_bits(eps=1e-6):
    return struct.unpack("<i", struct.pack("<f", np.float32(eps)))[0]


def _dispatch(rt, callback):
    rt.backend.begin_pass()
    callback()
    rt.backend.end_pass()
    rt.backend.synchronize()


def _zero(rt, tensor):
    rt.backend.fill(tensor, 0.0)


def _read_at(rt, storage, byte_offset, shape):
    return np.asarray(
        rt.backend.ctx.numpy_view_float32(
            storage.ptr.buffer_id, int(byte_offset), tuple(shape)
        )
    ).copy()


def _scratch_from_partial(bound):
    if not isinstance(bound, functools.partial) or "scratch" not in (bound.keywords or {}):
        raise AssertionError("expected Runtime.bind_memory() to bind scratch via functools.partial")
    return bound.keywords["scratch"]

def build_rope_cache(cfg, theta=100000.0):
    positions = np.arange(cfg.T, dtype=np.float32)[:, None]
    channels = np.arange(
        0,
        cfg.D,
        2,
        dtype=np.float32,
    )[None, :]

    inv_freq = theta ** (-channels / cfg.D)
    freqs = positions * inv_freq

    return (
        np.cos(freqs).astype(np.float32),
        np.sin(freqs).astype(np.float32),
    )

def _make_fa_inputs(device, cfg, x_np, params):
    x = make_tensor(
        x_np.shape,
        np.float32,
        device,
        True,
        x_np,
    )

    names = (
        "wq",
        "bq",
        "wk",
        "bk",
        "wv",
        "bv",
    )

    tensors = [
        make_tensor(
            params[name].shape,
            np.float32,
            device,
            True,
            params[name],
        )
        for name in names
    ]

    rope_cos_np, rope_sin_np = build_rope_cache(cfg)

    rope_cos = make_tensor(
        rope_cos_np.shape,
        np.float32,
        device,
        False,
        rope_cos_np,
    )

    rope_sin = make_tensor(
        rope_sin_np.shape,
        np.float32,
        device,
        False,
        rope_sin_np,
    )

    out = FusedRMSNormFlashAttentionOp(
        n_head=cfg.H,
        eps=1e-6,
        is_causal=True,
    )(x, *tensors, rope_cos, rope_sin)

    tracked = {
        "x": x,
        **dict(zip(names, tensors)),
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "out": out,
    }

    return out, tracked


def probe_fa_internal(cfg, x_np, params, upstream):
    print(
        "\n=== Internal kernels: "
        "FusedRMSNormFlashAttention ==="
    )

    oracle = attention_oracle(
        x_np,
        params,
        upstream,
        cfg.H,
    )

    rt = reset_runtime("mps")
    out, t = _make_fa_inputs(
        "mps",
        cfg,
        x_np,
        params,
    )

    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()

    node = next(
        n
        for n in rt.topo
        if op_name(n) == "FusedRMSNormFlashAttention"
    )

    sf = _scratch_from_partial(node.forward)
    sb = _scratch_from_partial(node.backward)

    B, T, C = cfg.B, cfg.T, cfg.C
    H, D = cfg.H, cfg.D
    eps_bits = _eps_bits()

    def dispatch_qkv(scratch):
        input_names = (
            "x",
            "wq",
            "bq",
            "wk",
            "bk",
            "wv",
            "bv",
        )

        rt.backend.ctx.dispatch_with_args_3d(
            "fused_rms_fa_qkv",
            [
                t[name].storage.ptr.buffer_id
                for name in input_names
            ] + [
                scratch[name].ptr.buffer_id
                for name in ("q", "k", "v")
            ],
            [
                t[name].storage.ptr.offset
                for name in input_names
            ] + [
                scratch[name].ptr.offset
                for name in ("q", "k", "v")
            ],
            [B, T, C, H, eps_bits],
            256,
            (B * T + 15) // 16,
            1,
            256,
        )

    def dispatch_rope(scratch, mode):
        rt.backend.ctx.dispatch_with_args_3d(
            "rope_qk_inplace",
            [
                scratch["q"].ptr.buffer_id,
                scratch["k"].ptr.buffer_id,
                t["rope_cos"].storage.ptr.buffer_id,
                t["rope_sin"].storage.ptr.buffer_id,
            ],
            [
                scratch["q"].ptr.offset,
                scratch["k"].ptr.offset,
                t["rope_cos"].storage.ptr.offset,
                t["rope_sin"].storage.ptr.offset,
            ],
            [B, H, T, D, mode],
            D // 2,
            T,
            B * H,
            D // 2,
        )

    # Forward QKV
    _dispatch(rt, lambda: dispatch_qkv(sf))

    check(
        "fused_rms_fa_qkv/Q",
        read_storage_f32(rt, sf["q"]),
        oracle["q_raw"],
    )
    check(
        "fused_rms_fa_qkv/K",
        read_storage_f32(rt, sf["k"]),
        oracle["k_raw"],
    )
    check(
        "fused_rms_fa_qkv/V",
        read_storage_f32(rt, sf["v"]),
        oracle["v"],
    )

    _dispatch(rt, lambda: dispatch_rope(sf, 0))

    fa_kernel, fa_grid_x, fa_grid_y, fa_threads = (
        rt.backend.fa_from_qkv_dispatch(D, T)
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            fa_kernel,
            [
                sf["q"].ptr.buffer_id,
                sf["k"].ptr.buffer_id,
                sf["v"].ptr.buffer_id,
                out.storage.ptr.buffer_id,
                sf["q"].ptr.buffer_id,
            ],
            [
                sf["q"].ptr.offset,
                sf["k"].ptr.offset,
                sf["v"].ptr.offset,
                out.storage.ptr.offset,
                sf["q"].ptr.offset,
            ],
            [B, H, T, C, 1, 0],
            fa_grid_x,
            fa_grid_y,
            B * H,
            fa_threads,
        ),
    )

    check(
        f"{fa_kernel}/output",
        copy_to_host(rt, out),
        oracle["out"],
    )

    zero_every = [
        t["x"],
        t["wq"],
        t["bq"],
        t["wk"],
        t["bk"],
        t["wv"],
        t["bv"],
        out,
    ]

    for tensor in zero_every:
        if tensor.grad is not None:
            _zero(rt, tensor.grad)

    rt.backend.copy_from_host(
        out.grad,
        np.asarray(upstream, dtype=np.float32),
    )

    # Backward QKV recomputation
    _dispatch(rt, lambda: dispatch_qkv(sb))

    check(
        "backward/recompute_qkv/Q",
        read_storage_f32(rt, sb["q"]),
        oracle["q_raw"],
    )
    check(
        "backward/recompute_qkv/K",
        read_storage_f32(rt, sb["k"]),
        oracle["k_raw"],
    )
    check(
        "backward/recompute_qkv/V",
        read_storage_f32(rt, sb["v"]),
        oracle["v"],
    )

    _dispatch(rt, lambda: dispatch_rope(sb, 0))

    recompute_kernel, gx, gy, threads = (
        rt.backend.fa_from_qkv_dispatch(D, T)
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            recompute_kernel,
            [
                sb["q"].ptr.buffer_id,
                sb["k"].ptr.buffer_id,
                sb["v"].ptr.buffer_id,
                sb["dk"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
            ],
            [
                sb["q"].ptr.offset,
                sb["k"].ptr.offset,
                sb["v"].ptr.offset,
                sb["dk"].ptr.offset,
                sb["lse"].ptr.offset,
            ],
            [B, H, T, C, 1, 1],
            gx,
            gy,
            B * H,
            threads,
        ),
    )

    check(
        f"backward/{recompute_kernel}/output",
        read_storage_f32(
            rt,
            sb["dk"],
            (B, T, C),
        ),
        oracle["out"],
    )
    check(
        f"backward/{recompute_kernel}/lse",
        read_storage_f32(rt, sb["lse"]),
        oracle["lse"],
    )

    # FA backward preparation
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "fused_fa_bwd_prepare",
            [
                sb["dk"].ptr.buffer_id,
                out.grad.storage.ptr.buffer_id,
                sb["d_val"].ptr.buffer_id,
            ],
            [
                sb["dk"].ptr.offset,
                out.grad.storage.ptr.offset,
                sb["d_val"].ptr.offset,
            ],
            [B, H, T, C],
            128,
            (T + 3) // 4,
            B * H,
            128,
        ),
    )

    check(
        "fused_fa_bwd_prepare/d_val",
        read_storage_f32(rt, sb["d_val"]),
        oracle["d_val"],
    )

    dq_kernel, dq_gx, dq_gy, dq_threads = (
        rt.backend.fa_bwd_dq_dispatch(D, T)
    )

    dkv_kernel, dkv_gx, dkv_gy, dkv_threads = (
        rt.backend.fa_bwd_dkv_dispatch(D, T)
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            dq_kernel,
            [
                sb["q"].ptr.buffer_id,
                sb["k"].ptr.buffer_id,
                sb["v"].ptr.buffer_id,
                out.grad.storage.ptr.buffer_id,
                sb["dq"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
                sb["d_val"].ptr.buffer_id,
            ],
            [
                sb["q"].ptr.offset,
                sb["k"].ptr.offset,
                sb["v"].ptr.offset,
                out.grad.storage.ptr.offset,
                sb["dq"].ptr.offset,
                sb["lse"].ptr.offset,
                sb["d_val"].ptr.offset,
            ],
            [B, H, T, C, 1],
            dq_gx,
            dq_gy,
            B * H,
            dq_threads,
        ),
    )

    check(
        f"{dq_kernel}/dQ",
        read_storage_f32(
            rt,
            sb["dq"],
            (B, T, C),
        ),
        heads_to_btc(oracle["dq_rot"]),
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            dkv_kernel,
            [
                sb["q"].ptr.buffer_id,
                sb["k"].ptr.buffer_id,
                sb["v"].ptr.buffer_id,
                out.grad.storage.ptr.buffer_id,
                sb["dk"].ptr.buffer_id,
                sb["dv"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
                sb["d_val"].ptr.buffer_id,
            ],
            [
                sb["q"].ptr.offset,
                sb["k"].ptr.offset,
                sb["v"].ptr.offset,
                out.grad.storage.ptr.offset,
                sb["dk"].ptr.offset,
                sb["dv"].ptr.offset,
                sb["lse"].ptr.offset,
                sb["d_val"].ptr.offset,
            ],
            [B, H, T, C, 1],
            dkv_gx,
            dkv_gy,
            B * H,
            dkv_threads,
        ),
    )

    check(
        f"{dkv_kernel}/dK",
        read_storage_f32(
            rt,
            sb["dk"],
            (B, T, C),
        ),
        heads_to_btc(oracle["dk_rot"]),
    )
    check(
        f"{dkv_kernel}/dV",
        read_storage_f32(
            rt,
            sb["dv"],
            (B, T, C),
        ),
        heads_to_btc(oracle["dv"]),
    )

    # Recompute RMSNorm(X).
    # q is reused for x_norm; the first B*T elements of lse store inv_rms.
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "fused_rms_recompute",
            [
                t["x"].storage.ptr.buffer_id,
                sb["q"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
            ],
            [
                t["x"].storage.ptr.offset,
                sb["q"].ptr.offset,
                sb["lse"].ptr.offset,
            ],
            [B, T, C, eps_bits],
            128,
            (B * T + 3) // 4,
            1,
            128,
        ),
    )

    check(
        "fused_rms_recompute/x_norm",
        read_storage_f32(
            rt,
            sb["q"],
            (B, T, C),
        ),
        oracle["x_norm"],
    )
    check(
        "fused_rms_recompute/inv_rms",
        _read_at(
            rt,
            sb["lse"],
            sb["lse"].ptr.offset,
            (B, T, 1),
        ),
        oracle["inv_rms"],
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "rope_qk_inplace",
            [
                sb["dq"].ptr.buffer_id,
                sb["dk"].ptr.buffer_id,
                t["rope_cos"].storage.ptr.buffer_id,
                t["rope_sin"].storage.ptr.buffer_id,
            ],
            [
                sb["dq"].ptr.offset,
                sb["dk"].ptr.offset,
                t["rope_cos"].storage.ptr.offset,
                t["rope_sin"].storage.ptr.offset,
            ],
            [B, H, T, D, 1],
            D // 2,
            T,
            B * H,
            D // 2,
        ),
    )

    check(
        "rope_qk_inplace_backward/dQ",
        read_storage_f32(rt, sb["dq"], (B, T, C)),
        heads_to_btc(oracle["dq"]),
    )

    check(
        "rope_qk_inplace_backward/dK",
        read_storage_f32(rt, sb["dk"], (B, T, C)),
        heads_to_btc(oracle["dk"]),
    )

    # Projection weight gradients
    for weight, dproj, expected in (
        ("wq", "dq", "dwq"),
        ("wk", "dk", "dwk"),
        ("wv", "dv", "dwv"),
    ):
        _dispatch(
            rt,
            lambda weight=weight, dproj=dproj:
            rt.backend.ctx.dispatch_gemm_mps(
                [
                    sb["q"].ptr.buffer_id,
                    sb[dproj].ptr.buffer_id,
                    t[weight].grad.storage.ptr.buffer_id,
                ],
                [
                    sb["q"].ptr.offset,
                    sb[dproj].ptr.offset,
                    t[weight].grad.storage.ptr.offset,
                ],
                [1, C, C, B * T, 1, 0, 0, 0],
                1.0,
                0.0,
            ),
        )

        check(
            f"gemm/{weight}.grad",
            copy_to_host(rt, t[weight].grad),
            oracle[expected],
        )

    # Projection bias gradients
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "fused_qkv_bias_grad",
            [
                sb["dq"].ptr.buffer_id,
                sb["dk"].ptr.buffer_id,
                sb["dv"].ptr.buffer_id,
                t["bq"].grad.storage.ptr.buffer_id,
                t["bk"].grad.storage.ptr.buffer_id,
                t["bv"].grad.storage.ptr.buffer_id,
            ],
            [
                sb["dq"].ptr.offset,
                sb["dk"].ptr.offset,
                sb["dv"].ptr.offset,
                t["bq"].grad.storage.ptr.offset,
                t["bk"].grad.storage.ptr.offset,
                t["bv"].grad.storage.ptr.offset,
            ],
            [B * T, C, 0],
            128,
            (C + 3) // 4,
            1,
            128,
        ),
    )

    for bias in ("bq", "bk", "bv"):
        check(
            f"fused_qkv_bias_grad/{bias}.grad",
            copy_to_host(rt, t[bias].grad),
            oracle[f"d{bias}"],
        )

    # dRMSNormOutput
    for index, (dproj, weight, beta) in enumerate((
        ("dq", "wq", 0.0),
        ("dk", "wk", 1.0),
        ("dv", "wv", 1.0),
    )):
        _dispatch(
            rt,
            lambda dproj=dproj, weight=weight, beta=beta:
            rt.backend.ctx.dispatch_gemm_mps(
                [
                    sb[dproj].ptr.buffer_id,
                    t[weight].storage.ptr.buffer_id,
                    sb["k"].ptr.buffer_id,
                ],
                [
                    sb[dproj].ptr.offset,
                    t[weight].storage.ptr.offset,
                    sb["k"].ptr.offset,
                ],
                [1, B * T, C, C, 0, 1, 0, 0],
                1.0,
                beta,
            ),
        )

        if index == 2:
            check(
                "three projection GEMMs/dx_norm",
                read_storage_f32(
                    rt,
                    sb["k"],
                    (B, T, C),
                ),
                oracle["dx_norm"],
            )

    # RMSNorm backward
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "fused_rmsnorm_bwd_dx",
            [
                t["x"].storage.ptr.buffer_id,
                sb["k"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
                t["x"].grad.storage.ptr.buffer_id,
            ],
            [
                t["x"].storage.ptr.offset,
                sb["k"].ptr.offset,
                sb["lse"].ptr.offset,
                t["x"].grad.storage.ptr.offset,
            ],
            [B * T, C],
            32,
            B * T,
            1,
            32,
        ),
    )

    check(
        "fused_rmsnorm_bwd_dx/x.grad",
        copy_to_host(rt, t["x"].grad),
        oracle["dx_local"],
    )

def ce_oracle(x, w, labels):
    B, T, C = x.shape
    N, V = B * T, w.shape[0]

    X = x.reshape(N, C)
    y = labels.reshape(N)

    logits = X @ w.T

    row_max = np.max(
        logits,
        axis=1,
        keepdims=True,
    )
    exp_logits = np.exp(logits - row_max)
    sums = np.sum(
        exp_logits,
        axis=1,
        keepdims=True,
    )

    lse = row_max + np.log(sums)

    row_loss = (
        lse[:, 0]
        - logits[np.arange(N), y]
    ).astype(np.float32)

    probs = exp_logits / sums
    probs[np.arange(N), y] -= 1.0

    dlogits = probs / np.float32(N)

    return {
        "logits": logits.astype(np.float32),
        "target_logits": (
            logits[np.arange(N), y]
            .astype(np.float32)
            .reshape(N, 1)
        ),
        "row_loss": row_loss.reshape(N, 1),
        "loss": np.asarray(
            [np.mean(row_loss)],
            dtype=np.float32,
        ),
        "lse": lse.astype(np.float32),
        "dlogits": dlogits.astype(np.float32),
        "dx": (
            dlogits @ w
        ).reshape(B, T, C).astype(np.float32),
        "dw": (
            dlogits.T @ X
        ).astype(np.float32),
    }

def probe_ce_internal(cfg, x_np, w_np, labels_np):
    print(
        "\n=== Internal kernels: "
        "Optimized FusedLinearCrossEntropy ==="
    )

    oracle = ce_oracle(
        x_np,
        w_np,
        labels_np,
    )

    rt = reset_runtime("mps")

    x = make_tensor(
        x_np.shape,
        np.float32,
        "mps",
        True,
        x_np,
    )
    w = make_tensor(
        w_np.shape,
        np.float32,
        "mps",
        True,
        w_np,
    )
    labels = make_tensor(
        labels_np.shape,
        np.int32,
        "mps",
        False,
        labels_np,
    )

    loss = FusedLinearCrossEntropyOp()(
        x,
        w,
        labels,
    )

    rt.loss = loss
    rt.build_topo(loss)
    rt.bind_memory()

    node = rt.topo[-1]
    sf = _scratch_from_partial(node.forward)
    # 融合后 fwd/bwd 共用同一份 scratch，backward 不再绑 scratch
    assert not isinstance(node.backward, functools.partial), \
        "fused CE backward should not carry scratch anymore"
    N, C, V = cfg.B * cfg.T, cfg.C, cfg.V
    itemsize = np.dtype(x.dtype).itemsize
    logits, row_loss, lse = sf["logits"], sf["row_loss"], sf["lse"]
    chunk_rows = min(logits.shape[0], N)
    # 与算子内一致：grad_scale/N 作为 host 常量传入，不再读 grad_out[0]
    scale_bits = struct.unpack('<i', struct.pack('<f', 1.0 / N))[0]
    accumulate = bool(getattr(rt.backend, "accumulate_parameter_grads", False))
    _zero(rt, x.grad)
    _zero(rt, w.grad)
    for row_base in range(0, N, chunk_rows):
        rows = min(chunk_rows, N - row_base)
        x_offset = x.storage.ptr.offset + row_base * C * itemsize
        grad_x_offset = x.grad.storage.ptr.offset + row_base * C * itemsize
        # --- 阶段 1：只提交 logits，便于在被覆盖前校验 ---
        def submit_logits(rows=rows, x_offset=x_offset):
            rt.backend.ctx.dispatch_with_args_3d(
                "fused_linear_ce_chunk_logits",
                [x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
                [x_offset, w.storage.ptr.offset, logits.ptr.offset],
                [rows, C, V], 256, (rows + 63) // 64, (V + 95) // 96, 256)
        _dispatch(rt, submit_logits)
        check(f"fused_ce/logits[{row_base}:{row_base + rows}]",
              read_storage_f32(rt, logits, (rows, V)),
              oracle["logits"][row_base:row_base + rows])
        # --- 阶段 2：row_loss + lse + 就地 dlogits + dX + dW ---
        def submit_grads(rows=rows, row_base=row_base,
                         x_offset=x_offset, grad_x_offset=grad_x_offset):
            rt.backend.ctx.dispatch_with_args_3d(
                "fused_linear_ce_logits_to_dlogits",
                [logits.ptr.buffer_id, labels.storage.ptr.buffer_id,
                 row_loss.ptr.buffer_id, lse.ptr.buffer_id],
                [logits.ptr.offset, labels.storage.ptr.offset,
                 row_loss.ptr.offset, lse.ptr.offset],
                [rows, V, row_base, scale_bits, 1], 256, rows, 1, 256)

            rt.backend.ctx.dispatch_gemm_mps(
                [logits.ptr.buffer_id, w.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
                [logits.ptr.offset, w.storage.ptr.offset, grad_x_offset],
                [1, rows, C, V, 0, 0, 0, 0], 1.0, 0.0)
            rt.backend.ctx.dispatch_gemm_mps(
                [logits.ptr.buffer_id, x.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
                [logits.ptr.offset, x_offset, w.grad.storage.ptr.offset],
                [1, V, C, rows, 1, 0, 0, 0], 1.0,
                1.0 if (accumulate or row_base != 0) else 0.0)
        _dispatch(rt, submit_grads)
        check(f"fused_ce/row_loss[{row_base}:{row_base + rows}]",
              _read_at(rt, row_loss, row_loss.ptr.offset + row_base * itemsize, (rows, 1)),
              oracle["row_loss"][row_base:row_base + rows])
        check(f"fused_ce/dlogits[{row_base}:{row_base + rows}]",
              read_storage_f32(rt, logits, (rows, V)),
              oracle["dlogits"][row_base:row_base + rows])
        check(f"fused_ce/x.grad[{row_base}:{row_base + rows}]",
              _read_at(rt, x.grad.storage, grad_x_offset, (rows, C)),
              oracle["dx"].reshape(N, C)[row_base:row_base + rows])
    # 全部 row_loss 就绪后再 reduce（保持原样）
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_with_args_3d(
        "fused_linear_ce_forward_reduce",
        [row_loss.ptr.buffer_id, loss.storage.ptr.buffer_id],
        [row_loss.ptr.offset, loss.storage.ptr.offset],
        [N], 256, 1, 1, 256))
    check("fused_ce/loss",   copy_to_host(rt, loss),   oracle["loss"])
    check("fused_ce/lse",    read_storage_f32(rt, lse, (N, 1)), oracle["lse"])
    check("fused_ce/x.grad", copy_to_host(rt, x.grad), oracle["dx"])
    check("fused_ce/w.grad", copy_to_host(rt, w.grad), oracle["dw"])

def _make_block_prefix(
    device,
    cfg,
    x_np,
    arrays,
    block,
    include_mlp=False,
):
    p = f"blocks[{block}]"

    fa_params = block_parameter_dict(
        arrays,
        block,
        "fa",
    )

    attn, t = _make_fa_inputs(
        device,
        cfg,
        x_np,
        fa_params,
    )

    proj_w = make_tensor(
        arrays[f"{p}.attn.proj.weight"].shape,
        np.float32,
        device,
        True,
        arrays[f"{p}.attn.proj.weight"],
    )

    proj_b = make_tensor(
        arrays[f"{p}.attn.proj.bias"].shape,
        np.float32,
        device,
        True,
        arrays[f"{p}.attn.proj.bias"],
    )

    x_after = FusedLinearAddOp()(
        t["x"],
        attn,
        proj_w,
        proj_b,
    )

    t.update({
        "attn": attn,
        "proj_w": proj_w,
        "proj_b": proj_b,
        "x_after": x_after,
    })

    if not include_mlp:
        return x_after, t

    mp = block_parameter_dict(
        arrays,
        block,
        "mlp",
    )

    wgu = make_tensor(
        mp["w1"].shape,
        np.float32,
        device,
        True,
        mp["w1"],
    )

    wdown = make_tensor(
        mp["w2"].shape,
        np.float32,
        device,
        True,
        mp["w2"],
    )

    mlp = FusedRMSNormMLPOp(
        eps=1e-6,
    )(
        x_after,
        wgu,
        wdown,
    )

    t.update({
        "w1": wgu,
        "w2": wdown,
        "mlp": mlp,
    })

    return mlp, t


def _dispatch_fa_recompute_project(
    rt,
    cfg,
    act,
    scratch,
    out_storage,
):
    x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = act.op.inputs

    rt.backend.ctx.dispatch_with_args_3d(
        "fused_rms_fa_project_kv",
        [
            x.storage.ptr.buffer_id,
            wq.storage.ptr.buffer_id,
            bq.storage.ptr.buffer_id,
            wk.storage.ptr.buffer_id,
            bk.storage.ptr.buffer_id,
            wv.storage.ptr.buffer_id,
            bv.storage.ptr.buffer_id,
            out_storage.ptr.buffer_id,
            scratch["fa_k"].ptr.buffer_id,
            scratch["fa_v"].ptr.buffer_id,
        ],
        [
            x.storage.ptr.offset,
            wq.storage.ptr.offset,
            bq.storage.ptr.offset,
            wk.storage.ptr.offset,
            bk.storage.ptr.offset,
            wv.storage.ptr.offset,
            bv.storage.ptr.offset,
            out_storage.ptr.offset,
            scratch["fa_k"].ptr.offset,
            scratch["fa_v"].ptr.offset,
        ],
        [
            cfg.B,
            cfg.T,
            cfg.C,
            cfg.H,
            _eps_bits(),
        ],
        512,
        (cfg.B * cfg.T + 31) // 32,
        1,
        512,
    )

def probe_linear_add_internal(cfg, x_np, arrays, block, upstream):
    print("\n=== Internal kernels: FusedLinearAdd (real FA producer path) ===")
    p = f"blocks[{block}]"
    fa_params = block_parameter_dict(arrays, block, "fa")
    fa_ref = attention_oracle(x_np, fa_params, np.zeros_like(x_np), cfg.H)
    proj_w_np = arrays[f"{p}.attn.proj.weight"]
    proj_b_np = arrays[f"{p}.attn.proj.bias"]
    expected_out = (
        x_np.reshape(-1, cfg.C)
        + fa_ref["out"].reshape(-1, cfg.C) @ proj_w_np
        + proj_b_np
    ).reshape(x_np.shape).astype(np.float32)
    dO = np.asarray(upstream, dtype=np.float32)
    expected_dw = (fa_ref["out"].reshape(-1, cfg.C).T @ dO.reshape(-1, cfg.C)).astype(np.float32)
    expected_dact = (dO.reshape(-1, cfg.C) @ proj_w_np.T).reshape(x_np.shape).astype(np.float32)
    expected_db = np.sum(dO.reshape(-1, cfg.C), axis=0).astype(np.float32)

    rt = reset_runtime("mps")
    out, t = _make_block_prefix("mps", cfg, x_np, arrays, block, include_mlp=False)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()
    fla = next(n for n in rt.topo if op_name(n) == "FusedLinearAdd")
    scratch = _scratch_from_partial(fla.backward)
    rt.forward(); rt.backend.synchronize()
    check("fused_linear_add_forward_kernel/output", copy_to_host(rt, out), expected_out)
    for tensor in t.values():
        if isinstance(tensor, Tensor) and tensor.grad is not None:
            _zero(rt, tensor.grad)
    rt.backend.copy_from_host(out.grad, dO)

    _dispatch(rt, lambda: _dispatch_fa_recompute_project(rt, cfg, t["attn"], scratch, t["attn"].grad.storage))
    # 这里只检查 fused_rms_fa_project_kv 的原始投影结果。
    check(
        "fused_rms_fa_project_kv/Q",
        read_storage_f32(
            rt,
            t["attn"].grad.storage,
            (cfg.B, cfg.T, cfg.C),
        ),
        heads_to_btc(fa_ref["q_raw"]),
    )

    check(
        "fused_rms_fa_project_kv/K",
        read_storage_f32(
            rt,
            scratch["fa_k"],
        ),
        fa_ref["k_raw"],
    )

    check(
        "fused_rms_fa_project_kv/V",
        read_storage_f32(
            rt,
            scratch["fa_v"],
        ),
        fa_ref["v"],
    )

    rope_cos = t["attn"].op.inputs[-2]
    rope_sin = t["attn"].op.inputs[-1]

    # project_kv 的 Q 是 BTHD，K 是 BHTD，因此使用 mode 2。
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "rope_qk_inplace",
            [
                t["attn"].grad.storage.ptr.buffer_id,
                scratch["fa_k"].ptr.buffer_id,
                rope_cos.storage.ptr.buffer_id,
                rope_sin.storage.ptr.buffer_id,
            ],
            [
                t["attn"].grad.storage.ptr.offset,
                scratch["fa_k"].ptr.offset,
                rope_cos.storage.ptr.offset,
                rope_sin.storage.ptr.offset,
            ],
            [
                cfg.B,
                cfg.H,
                cfg.T,
                cfg.D,
                2,
            ],
            cfg.D // 2,
            cfg.T,
            cfg.B * cfg.H,
            cfg.D // 2,
        ),
    )

    kernel, gx, gy, threads = (
        rt.backend.fa_recompute_dispatch(
            cfg.D,
            cfg.T,
        )
    )

    kernel, gx, gy, threads = rt.backend.fa_recompute_dispatch(cfg.D, cfg.T)
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_with_args_3d(
        kernel,
        [t["attn"].grad.storage.ptr.buffer_id, scratch["fa_k"].ptr.buffer_id, scratch["fa_v"].ptr.buffer_id],
        [t["attn"].grad.storage.ptr.offset, scratch["fa_k"].ptr.offset, scratch["fa_v"].ptr.offset],
        [cfg.B, cfg.H, cfg.T, cfg.C, 1], gx, gy, cfg.B*cfg.H, threads,
    ))
    check(f"{kernel}/attention_output", copy_to_host(rt, t["attn"].grad), fa_ref["out"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch_gemm_mps(
        [t["attn"].grad.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, t["proj_w"].grad.storage.ptr.buffer_id],
        [t["attn"].grad.storage.ptr.offset, out.grad.storage.ptr.offset, t["proj_w"].grad.storage.ptr.offset],
        [1, cfg.C, cfg.C, cfg.B*cfg.T, 1, 0, 0, 0], 1.0, 0.0,
    ))
    check("FusedLinearAdd GEMM/proj_w.grad", copy_to_host(rt, t["proj_w"].grad), expected_dw)
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_strided(
        "strided_add_backward_kernel",
        out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset,
        t["x"].grad.storage.ptr.buffer_id, t["x"].grad.storage.ptr.offset,
        list(t["x"].shape), t["x"].strides, prod(t["x"].shape),
    ))
    check("strided_add_backward_kernel/residual.grad", copy_to_host(rt, t["x"].grad), dO)
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_gemm_mps(
        [out.grad.storage.ptr.buffer_id, t["proj_w"].storage.ptr.buffer_id, t["attn"].grad.storage.ptr.buffer_id],
        [out.grad.storage.ptr.offset, t["proj_w"].storage.ptr.offset, t["attn"].grad.storage.ptr.offset],
        [1, cfg.B*cfg.T, cfg.C, cfg.C, 0, 1, 0, 0], 1.0, 0.0,
    ))
    check("FusedLinearAdd GEMM/attention.grad", copy_to_host(rt, t["attn"].grad), expected_dact)
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_with_args_3d(
        "bias_grad_reduce",
        [out.grad.storage.ptr.buffer_id, t["proj_b"].grad.storage.ptr.buffer_id],
        [out.grad.storage.ptr.offset, t["proj_b"].grad.storage.ptr.offset],
        # [cfg.B*cfg.T, cfg.C], cfg.C, 1, 1, min(cfg.C, 256),
        [cfg.B*cfg.T, cfg.C, 0], 256, (cfg.C+15)//16, 1, 256,
    ))
    check("bias_grad_reduce/proj_b.grad", copy_to_host(rt, t["proj_b"].grad), expected_db)


def _dispatch_context_recompute(rt, cfg, t, h, dh):
    del dh

    B, T, C, H = cfg.B, cfg.T, cfg.C, cfg.H
    N = B * T
    x_nbytes = N * C * 4

    q_offset = h.ptr.offset
    k_offset = q_offset + x_nbytes
    v_offset = q_offset + 2 * x_nbytes

    x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = (
        t["attn"].op.inputs
    )

    rt.backend.ctx.dispatch_with_args_3d(
        "fused_rms_fa_project_kv",
        [
            x.storage.ptr.buffer_id,
            wq.storage.ptr.buffer_id,
            bq.storage.ptr.buffer_id,
            wk.storage.ptr.buffer_id,
            bk.storage.ptr.buffer_id,
            wv.storage.ptr.buffer_id,
            bv.storage.ptr.buffer_id,
            h.ptr.buffer_id,
            h.ptr.buffer_id,
            h.ptr.buffer_id,
        ],
        [
            x.storage.ptr.offset,
            wq.storage.ptr.offset,
            bq.storage.ptr.offset,
            wk.storage.ptr.offset,
            bk.storage.ptr.offset,
            wv.storage.ptr.offset,
            bv.storage.ptr.offset,
            q_offset,
            k_offset,
            v_offset,
        ],
        [
            B,
            T,
            C,
            H,
            _eps_bits(),
        ],
        512,
        (B * T + 31) // 32,
        1,
        512,
    )

    return q_offset, k_offset, v_offset


def _dispatch_context_attention(rt, cfg, h, q_off, k_off, v_off):
    kernel, gx, gy, threads = rt.backend.fa_recompute_dispatch(cfg.D, cfg.T)
    rt.backend.ctx.dispatch_with_args_3d(
        kernel, [h.ptr.buffer_id]*3, [q_off, k_off, v_off],
        [cfg.B, cfg.H, cfg.T, cfg.C, 1], gx, gy, cfg.B*cfg.H, threads,
    )
    return kernel


def _dispatch_context_linear_add(rt, cfg, t, h, dh, q_off):
    rt.backend.ctx.dispatch_with_args_3d(
        "fused_linear_add_forward_kernel",
        [t["x"].storage.ptr.buffer_id, h.ptr.buffer_id, t["proj_w"].storage.ptr.buffer_id,
         t["proj_b"].storage.ptr.buffer_id, dh.ptr.buffer_id],
        [t["x"].storage.ptr.offset, q_off, t["proj_w"].storage.ptr.offset,
         t["proj_b"].storage.ptr.offset, dh.ptr.offset],
        [cfg.B*cfg.T, cfg.C, cfg.C, 0], 256, (cfg.C+63)//64, (cfg.B*cfg.T+63)//64, 256,
    )


def probe_mlp_internal(
    cfg,
    x_np,
    arrays,
    block,
    upstream,
    residual_grad,
):
    print(
        "\n=== FusedRMSNormMLP SwiGLU validation ==="
    )

    p = f"blocks[{block}]"

    upstream = np.asarray(
        upstream,
        dtype=np.float32,
    ).copy()

    residual_grad = np.asarray(
        residual_grad,
        dtype=np.float32,
    ).copy()

    # 独立计算 FusedRMSNormFlashAttention 的参考输出。
    fa_params = block_parameter_dict(
        arrays,
        block,
        "fa",
    )

    fa_ref = attention_oracle(
        x_np,
        fa_params,
        np.zeros_like(x_np, dtype=np.float32),
        cfg.H,
    )

    # 独立计算 FusedLinearAdd 输出。
    x_after_ref = (
        np.asarray(x_np, dtype=np.float32).reshape(-1, cfg.C)
        + fa_ref["out"].reshape(-1, cfg.C)
        @ np.asarray(
            arrays[f"{p}.attn.proj.weight"],
            dtype=np.float32,
        )
        + np.asarray(
            arrays[f"{p}.attn.proj.bias"],
            dtype=np.float32,
        )
    ).reshape(
        np.asarray(x_np).shape
    ).astype(np.float32)

    mp = block_parameter_dict(
        arrays,
        block,
        "mlp",
    )

    # MLP oracle 的输入必须是 x_after_ref，而不是 block 输入 x_np。
    oracle = mlp_oracle(
        x_after_ref,
        mp,
        upstream,
        residual_grad,
        eps=1e-6,
    )

    results = {}

    for device in ("cpu", "mps"):
        print(
            f"\n=== FusedRMSNormMLP SwiGLU "
            f"{device.upper()} validation ==="
        )

        rt = reset_runtime(device)

        mlp, t = _make_block_prefix(
            device,
            cfg,
            x_np,
            arrays,
            block,
            include_mlp=True,
        )

        rt.loss = mlp
        rt.build_topo(mlp)
        rt.bind_memory()

        node = next(
            n
            for n in rt.topo
            if op_name(n) == "FusedRMSNormMLP"
        )

        wgu = t["w1"]
        wdown = t["w2"]
        C = cfg.C
        H = wgu.shape[1] // 2

        if wgu.shape != (C, 2 * H):
            raise AssertionError(
                f"{device}: invalid wgu shape {wgu.shape}"
            )

        if wdown.shape != (H, C):
            raise AssertionError(
                f"{device}: invalid wdown shape {wdown.shape}"
            )

        if device == "mps":
            N = cfg.B * cfg.T
            itemsize = np.dtype(np.float32).itemsize
            old_mps_bytes = 8 * N * C * itemsize

            new_fwd_bytes = (
                node.scratch_fwd_layout.nbytes
                if node.scratch_fwd_layout is not None
                else 0
            )
            new_bwd_bytes = (
                node.scratch_bwd_layout.nbytes
                if node.scratch_bwd_layout is not None
                else 0
            )
            new_peak_bytes = max(
                new_fwd_bytes,
                new_bwd_bytes,
            )

            if new_peak_bytes > old_mps_bytes:
                raise AssertionError(
                    "MPS SwiGLU scratch memory increased: "
                    f"new={new_peak_bytes}, old={old_mps_bytes}"
                )

        rt.forward()
        rt.backend.synchronize()

        # 先验证 MLP 收到的输入正确。
        check(
            f"{device}/FusedLinearAdd/x_after_vs_oracle",
            copy_to_host(rt, t["x_after"]),
            x_after_ref,
            fail_fast=True,
        )

        got_out = copy_to_host(rt, mlp)

        check(
            f"{device}/FusedRMSNormMLP/forward_vs_oracle",
            got_out,
            oracle["out"],
            fail_fast=True,
        )

        for tensor in t.values():
            if (
                isinstance(tensor, Tensor)
                and tensor.grad is not None
            ):
                _zero(rt, tensor.grad)

        rt.backend.copy_from_host(
            mlp.grad,
            upstream,
        )

        rt.backend.copy_from_host(
            t["x_after"].grad,
            residual_grad,
        )

        _dispatch(
            rt,
            lambda: node.backward(
                node.inputs,
                node.outputs,
            ),
        )

        got_dx = copy_to_host(
            rt,
            t["x_after"].grad,
        )
        got_dwgu = copy_to_host(
            rt,
            t["w1"].grad,
        )
        got_dwdown = copy_to_host(
            rt,
            t["w2"].grad,
        )

        check(
            f"{device}/FusedRMSNormMLP/backward/x.grad_vs_oracle",
            got_dx,
            oracle["dx"],
            fail_fast=True,
        )
        check(
            f"{device}/FusedRMSNormMLP/backward/wgu.grad_vs_oracle",
            got_dwgu,
            oracle["dw1"],
            fail_fast=True,
        )
        check(
            f"{device}/FusedRMSNormMLP/backward/wdown.grad_vs_oracle",
            got_dwdown,
            oracle["dw2"],
            fail_fast=True,
        )

        results[device] = {
            "out": got_out,
            "dx": got_dx,
            "dwgu": got_dwgu,
            "dwdown": got_dwdown,
        }

    check(
        "FusedRMSNormMLP/CPU_vs_Metal/forward",
        results["mps"]["out"],
        results["cpu"]["out"],
        fail_fast=True,
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/x.grad",
        results["mps"]["dx"],
        results["cpu"]["dx"],
        fail_fast=True,
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/wgu.grad",
        results["mps"]["dwgu"],
        results["cpu"]["dwgu"],
        fail_fast=True,
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/wdown.grad",
        results["mps"]["dwdown"],
        results["cpu"]["dwdown"],
        fail_fast=True,
    )

def _make_optimizer_leaf(rt, array, grad):
    tensor = Tensor(array.shape, np.float32, device=rt.allocator.device,
                    requires_grad=True, init_val=np.asarray(array, dtype=np.float32).copy())
    tensor.storage = rt.allocator.alloc(tensor.shape, tensor.dtype)
    rt.backend.init(tensor)
    tensor.grad = Tensor(tensor.shape, tensor.dtype, device=rt.allocator.device,
                         requires_grad=False, init_val=np.asarray(grad, dtype=np.float32).copy())
    tensor.grad.storage = rt.allocator.alloc(tensor.shape, tensor.dtype)
    rt.backend.init(tensor.grad)
    return tensor


def _adamw_numpy_step(p, g, m, v, step, lr, betas, eps, wd):
    b1, b2 = betas
    lr_t = np.float32(lr * math.sqrt(1.0 - b2**step) / (1.0 - b1**step))
    p = p.copy(); m = m.copy(); v = v.copy()
    if wd:
        p -= np.float32(lr * wd) * p
    m = np.float32(b1) * m + np.float32(1.0-b1) * g
    v = np.float32(b2) * v + np.float32(1.0-b2) * g * g
    p -= lr_t * m / (np.sqrt(v) + np.float32(eps))
    return p.astype(np.float32), m.astype(np.float32), v.astype(np.float32)

def read_adamw8bit_state(rt, optimizer, index):
    p = optimizer.params[index]

    for item in optimizer.dispatches:
        if item[0] != "q8":
            continue
        _, p0, _, qm, qv, absmax_m, absmax_v, _, _ = item
        if (p0.ptr.buffer_id == p.storage.ptr.buffer_id and p0.ptr.offset == p.storage.ptr.offset):
            break
    else:
        return (
            read_storage_f32(rt, optimizer.m[index], p.shape),
            read_storage_f32(rt, optimizer.v[index], p.shape),
        )

    q_m = np.asarray(
        rt.backend.ctx.numpy_view_uint8(
            qm.ptr.buffer_id,
            qm.ptr.offset,
            tuple(p.shape),
        )
    ).copy()

    q_v = np.asarray(
        rt.backend.ctx.numpy_view_uint8(
            qv.ptr.buffer_id,
            qv.ptr.offset,
            tuple(p.shape),
        )
    ).copy()

    m_absmax = np.asarray(
        rt.backend.ctx.numpy_view_float32(
            absmax_m.ptr.buffer_id,
            absmax_m.ptr.offset,
            tuple(absmax_m.shape),
        )
    ).copy()

    v_absmax = np.asarray(
        rt.backend.ctx.numpy_view_float32(
            absmax_v.ptr.buffer_id,
            absmax_v.ptr.offset,
            tuple(absmax_v.shape),
        )
    ).copy()

    qmap_m = np.asarray(
        rt.backend.ctx.numpy_view_float32(
            optimizer.qmap_m.ptr.buffer_id,
            optimizer.qmap_m.ptr.offset,
            (256,),
        )
    ).copy()

    qmap_v = np.asarray(
        rt.backend.ctx.numpy_view_float32(
            optimizer.qmap_v.ptr.buffer_id,
            optimizer.qmap_v.ptr.offset,
            (256,),
        )
    ).copy()

    blocks = np.arange(
        p.numel,
        dtype=np.int64,
    ) // optimizer.BLOCK_SIZE

    m = (
        m_absmax[blocks] * qmap_m[q_m.reshape(-1)]
    ).reshape(p.shape)

    v = (
        v_absmax[blocks] * qmap_v[q_v.reshape(-1)]
    ).reshape(p.shape)

    return m, v

def probe_adamw():
    print("\n=== Training step: AdamW (matrix decay, vector no decay) ===")
    rng = np.random.default_rng(SEED + 77)
    p0 = [
        (rng.standard_normal((17, 13))*0.02).astype(np.float32),
        (rng.standard_normal((13,))*0.02).astype(np.float32),
    ]
    grads = [
        (rng.standard_normal(x.shape)*0.01).astype(np.float32) for x in p0
    ]
    lr, wd, betas, eps = 0.002, 0.234903, (0.9, 0.999), 1e-8
    refs = []
    for index, (p, g) in enumerate(zip(p0, grads)):
        m = np.zeros_like(p); v = np.zeros_like(p)
        per_param_wd = wd if p.ndim > 1 else 0.0
        history = []
        for step in (1, 2):
            p, m, v = _adamw_numpy_step(p, g, m, v, step, lr, betas, eps, per_param_wd)
            history.append((p.copy(), m.copy(), v.copy()))
        refs.append(history)
    for device in ("cpu", "mps"):
        print(f"  [{device}]")
        rt = reset_runtime(device)
        params = [_make_optimizer_leaf(rt, p, g) for p, g in zip(p0, grads)]
        use_8bit = (device == "mps")
        optimizer = AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps, quant_8bit=use_8bit)
        for step in (1, 2):
            optimizer.step(); rt.backend.synchronize()
            for index, tensor in enumerate(params):
                expected_p, expected_m, expected_v = refs[index][step-1]
                kind = "matrix" if tensor.ndim > 1 else "vector"
                if use_8bit:
                    got_m, got_v = read_adamw8bit_state(rt, optimizer, index)
                    check(f"AdamW8bit/{device}/step[{step}]/{kind}/parameter", copy_to_host(rt, tensor), expected_p)
                    check(f"AdamW8bit/{device}/step[{step}]/{kind}/m", got_m, expected_m)
                    check(f"AdamW8bit/{device}/step[{step}]/{kind}/v", got_v, expected_v)
                else:
                    check(f"AdamW/{device}/step[{step}]/{kind}/parameter", copy_to_host(rt, tensor), expected_p)
                    check(f"AdamW/{device}/step[{step}]/{kind}/m", read_storage_f32(rt, optimizer.m[index]), expected_m)
                    check(f"AdamW/{device}/step[{step}]/{kind}/v", read_storage_f32(rt, optimizer.v[index]), expected_v)


def _copy_training_batch(state, input_ids, labels):
    state.rt.backend.copy_from_host(state.input_ids, input_ids)
    state.rt.backend.copy_from_host(state.labels, labels)


def _parameter_snapshot(state):
    return {
        name: copy_to_host(state.rt, tensor)
        for name, tensor in state.param_by_name.items()
    }


def _gradient_snapshot(state):
    return {
        name: copy_to_host(state.rt, tensor.grad)
        for name, tensor in state.param_by_name.items()
    }


def _optimizer_snapshot(state, optimizer):
    moments = {}
    use_8bit_state = (
        optimizer.quant8bit
        and hasattr(state.rt.backend, "ctx")
    )

    for index, tensor in enumerate(optimizer.params):
        name = state.param_name_by_id[id(tensor)]

        if use_8bit_state:
            m, v = read_adamw8bit_state(
                state.rt,
                optimizer,
                index,
            )
        else:
            m = read_storage_f32(
                state.rt,
                optimizer.m[index],
                tensor.shape,
            )
            v = read_storage_f32(
                state.rt,
                optimizer.v[index],
                tensor.shape,
            )

        moments[name] = {
            "m": m,
            "v": v,
        }

    return moments


def _gradient_storage_layout(state):
    layout = {}
    for name, tensor in state.param_by_name.items():
        storage = tensor.grad.storage
        layout[name] = (
            storage.ptr.buffer_id,
            storage.ptr.offset,
            storage.nbytes,
        )
    return layout


def _assert_step_updated(label, before, after):
    max_change = max(
        float(np.max(np.abs(after[name] - before[name]), initial=0.0))
        for name in before
    )
    ok = max_change > 0.0
    print(f"  {'PASS' if ok else 'FAIL'} {label}: max_parameter_change={max_change:.8e}")
    if not ok:
        FAILURES.append((label, 0.0, 0.0))


def _make_accumulation_batches(cfg, accumulation_steps, training_steps, seed):
    large_cfg = Config(
        B=cfg.B * accumulation_steps,
        T=cfg.T,
        V=cfg.V,
        C=cfg.C,
        H=cfg.H,
        D=cfg.D,
        L=cfg.L,
        dtype=cfg.dtype,
    )
    batches = []
    for step in range(training_steps):
        large_batch = build_inputs(large_cfg, seed + 1000 + step * 17)
        micro_batches = []
        for micro_step in range(accumulation_steps):
            start = micro_step * cfg.B
            end = start + cfg.B
            micro_batches.append(tuple(array[start:end].copy() for array in large_batch))
        batches.append((large_batch, micro_batches))
    return large_cfg, batches


def _run_training_trajectory(cfg, arrays, batches, accumulation_steps, enabled, optimizer_gradients=None):
    initial = batches[0][1][0]
    state = build_graph("mps", cfg, arrays, *initial)
    optimizer = AdamW(
        state.model.parameters(),
        lr=0.002,
        weight_decay=0.234903,
        betas=(0.9, 0.999),
        eps=1e-8,
        quant_8bit=True,
    )
    initial_layout = _gradient_storage_layout(state)
    history = []
    for step, (large_batch, micro_batches) in enumerate(batches, 1):
        before = _parameter_snapshot(state)
        if enabled:
            for accumulation_step, micro_batch in enumerate(micro_batches):
                _copy_training_batch(state, *micro_batch)
                state.loss.backward(
                    grad_scale=1.0 / accumulation_steps,
                    # Step 0 overwrites the existing parameter-gradient
                    # storage. Later micro-steps add into the same storage,
                    # so no zero_grad call or second accumulation buffer is
                    # needed.
                    accumulate_parameter_grads=accumulation_step > 0,
                )
        else:
            micro_gradients = []

            for micro_batch in micro_batches:
                _copy_training_batch(state, *micro_batch)

                # 与真实累加路径一致：先对每个 micro-batch 缩放。
                state.loss.backward(
                    grad_scale=1.0 / accumulation_steps,
                )
                state.rt.backend.synchronize()

                micro_gradients.append(
                    _gradient_snapshot(state)
                )

            # 每个 micro-batch 已经缩放过，因此这里求和而不是求平均。
            gradients = {
                name: np.sum(
                    np.stack(
                        [
                            item[name]
                            for item in micro_gradients
                        ],
                        axis=0,
                    ),
                    axis=0,
                    dtype=np.float32,
                ).astype(np.float32)
                for name in state.param_by_name
            }

            for name, tensor in state.param_by_name.items():
                state.rt.backend.copy_from_host(
                    tensor.grad,
                    gradients[name],
                )
        state.rt.backend.synchronize()
        if enabled:
            gradients = _gradient_snapshot(state)
        if optimizer_gradients is not None:
            reference_gradients = optimizer_gradients[step - 1]

            for name, tensor in state.param_by_name.items():
                state.rt.backend.copy_from_host(
                    tensor.grad,
                    reference_gradients[name],
                )
        optimizer.step()
        state.rt.backend.synchronize()
        parameters = _parameter_snapshot(state)
        _assert_step_updated(
            f"grad_accum[{accumulation_steps}]/{'enabled' if enabled else 'baseline'}/step[{step}]/updated",
            before,
            parameters,
        )
        history.append({
            "gradients": gradients,
            "parameters": parameters,
            "optimizer": _optimizer_snapshot(state, optimizer),
        })
    if _gradient_storage_layout(state) != initial_layout:
        raise AssertionError("gradient accumulation replaced or reallocated parameter gradient storage")
    return history

def _compare_training_trajectories(reference, accumulated, accumulation_steps):
    bitwise_equal = 0
    tensor_count = 0
    max_abs = {"gradient": 0.0, "parameter": 0.0, "optimizer.m": 0.0, "optimizer.v": 0.0}
    for step, (ref_step, acc_step) in enumerate(zip(reference, accumulated), 1):
        for name in sorted(ref_step["gradients"]):
            label = f"grad_accum[{accumulation_steps}]/step[{step}]/gradient/{name}"
            max_abs["gradient"] = max(
                max_abs["gradient"],
                float(np.max(np.abs(
                    acc_step["gradients"][name] - ref_step["gradients"][name]
                ), initial=0.0)),
            )
            tensor_count += 1
            bitwise_equal += int(np.array_equal(
                acc_step["gradients"][name], ref_step["gradients"][name]
            ))
            # Separate MPS graph executions can differ in the last few bits
            # for parallel reductions. K=1 still takes the exact overwrite
            # path (accumulation_step == 0); numerical equality is therefore
            # the meaningful cross-run assertion here.
            check(label, acc_step["gradients"][name], ref_step["gradients"][name])
            check(
                f"grad_accum[{accumulation_steps}]/step[{step}]/parameter/{name}",
                acc_step["parameters"][name],
                ref_step["parameters"][name],
            )
            max_abs["parameter"] = max(
                max_abs["parameter"],
                float(np.max(np.abs(
                    acc_step["parameters"][name] - ref_step["parameters"][name]
                ), initial=0.0)),
            )
            for moment in ("m", "v"):
                key = f"optimizer.{moment}"
                check(
                    f"grad_accum[{accumulation_steps}]/step[{step}]/{key}/{name}",
                    acc_step["optimizer"][name][moment],
                    ref_step["optimizer"][name][moment],
                )
                max_abs[key] = max(
                    max_abs[key],
                    float(np.max(np.abs(
                        acc_step["optimizer"][name][moment]
                        - ref_step["optimizer"][name][moment]
                    ), initial=0.0)),
                )
    print(
        f"  INFO grad_accum[{accumulation_steps}]/max_abs: "
        + ", ".join(f"{key}={value:.8e}" for key, value in max_abs.items())
    )
    if accumulation_steps == 1:
        print(
            "  INFO grad_accum[1]/bitwise-identical-gradient-tensors: "
            f"{bitwise_equal}/{tensor_count}; K=1 equivalence is asserted "
            "with the configured numerical tolerances because independent "
            "Metal reductions are not guaranteed bitwise deterministic"
        )

def probe_gradient_accumulation(cfg, arrays, seed):
    print("\n=== No-extra-buffer parameter gradient accumulation ===")
    training_steps = 2
    for accumulation_steps in (1, 2, 4):
        print(f"  [accumulation_steps={accumulation_steps}]")
        _, batches = _make_accumulation_batches(
            cfg, accumulation_steps, training_steps, seed
        )
        reference = _run_training_trajectory(
            cfg, arrays, batches, accumulation_steps, enabled=False
        )
        accumulated = _run_training_trajectory(
            cfg, arrays, batches, accumulation_steps, enabled=True,
            optimizer_gradients=[item["gradients"] for item in reference],
        )
        _compare_training_trajectories(reference, accumulated, accumulation_steps)

def main():
    global PRINT_PASSES
    args, cfg = parse_args()
    PRINT_PASSES = not args.compact
    print(f"project_root={PROJECT_ROOT}")
    print(f"runtime={runtime.__file__}")
    print(
        "config: "
        f"B={cfg.B}, T={cfg.T}, V={cfg.V}, C={cfg.C}, H={cfg.H}, "
        f"D={cfg.D}, L={cfg.L}, atol={ATOL:g}, rtol={RTOL:g}"
    )
    arrays = build_parameter_arrays(cfg, args.seed)
    input_ids, labels = build_inputs(cfg, args.seed)

    if args.gradient_accumulation_only:
        probe_gradient_accumulation(cfg, arrays, args.seed)
        if FAILURES:
            print(f"\n{len(FAILURES)} gradient accumulation correctness check(s) failed:")
            for name, max_abs, max_rel in FAILURES:
                print(f"  {name}: max_abs={max_abs:.8e}, max_rel={max_rel:.8e}")
            raise AssertionError(
                f"{len(FAILURES)} gradient accumulation correctness check(s) failed"
            )
        print("\nAll gradient accumulation correctness checks passed.")
        return

    # The current extension must initialize its Metal contexts before the first
    # CPUBackend instance is constructed.  Device caches both afterwards.
    mps_exact = run_exact_training_graph(
        build_graph("mps", cfg, arrays, input_ids, labels)
    )
    cpu_exact = run_exact_training_graph(
        build_graph("cpu", cfg, arrays, input_ids, labels)
    )
    compare_exact_graph(cpu_exact, mps_exact)

    cpu_trace = run_staged_graph(
        build_graph("cpu", cfg, arrays, input_ids, labels)
    )
    mps_trace = run_staged_graph(
        build_graph("mps", cfg, arrays, input_ids, labels),
        reference_inputs=cpu_trace["forward_inputs"],
    )
    compare_staged_graph(cpu_trace, mps_trace)

    if not args.skip_internal_kernels:
        block = args.kernel_block
        if block == 0:
            x_in = cpu_trace["forward"][f"Add[{block}]"][0]
        else:
            x_in = cpu_trace["forward"][f"Add[{block-1}]"][0]
        fa_upstream = cpu_trace["backward"][f"FusedRMSNormFlashAttention[{block}]"]["output_grads"][0]
        fla_upstream = cpu_trace["backward"][f"FusedLinearAdd[{block}]"]["output_grads"][0]
        mlp_bwd = cpu_trace["backward"][f"FusedRMSNormMLP[{block}]"]
        mlp_upstream = mlp_bwd["output_grads"][0]
        mlp_residual = mlp_bwd["input_grads_before"][0]
        probe_fa_internal(cfg, x_in, block_parameter_dict(arrays, block, "fa"), fa_upstream)
        probe_linear_add_internal(cfg, x_in, arrays, block, fla_upstream)
        probe_mlp_internal(cfg, x_in, arrays, block, mlp_upstream, mlp_residual)
        final_x = cpu_trace["forward"][f"Add[{cfg.L-1}]"][0]
        probe_ce_internal(cfg, final_x, arrays["lm_head.weight"], labels)

    if not args.skip_gradient_accumulation:
        probe_gradient_accumulation(cfg, arrays, args.seed)

    if not args.skip_optimizer:
        probe_adamw()
    if FAILURES:
        print(f"\n{len(FAILURES)} correctness check(s) failed:")
        for name, max_abs, max_rel in FAILURES:
            print(f"  {name}: max_abs={max_abs:.8e}, max_rel={max_rel:.8e}")
        raise AssertionError("real GPT2 training-path correctness validation failed")
    print("\nAll real GPT2 training-path correctness checks passed.")

if __name__ == "__main__":
    main()
