#!/usr/bin/env python3
"""Profile the exact GPT training callbacks used by train_autograd.py.

/Users/a58/Documents/code/nanochat/.venv/bin/python timing_ops_v2.py --T=256 --iters=5 --operator-threshold-ms 0 \
  --kernel-threshold-ms 0
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import io
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


SEED = 1234


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd()]
    for candidate in candidates:
        if (
            (candidate / "train_autograd.py").is_file()
            and (candidate / "debug" / "backend.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        "cannot locate autograd repository; run from its root or copy this "
        "file to debug/timing_ops_v2.py"
    )

ROOT_DIR = _find_repo_root()
DEBUG_DIR = ROOT_DIR / "debug"
os.chdir(DEBUG_DIR)
sys.path.insert(0, str(DEBUG_DIR))
sys.path.insert(1, str(ROOT_DIR))

# These imports deliberately resolve runtime.py/backend.py from debug first.
from module import GPT2
from optimizer import AdamW
from runtime import Runtime
from tensor import Tensor


def iter_gpt2_named_tensors(model):
    """Yield the parameters created by module.GPT2 in stable model order."""
    yield "tok_emb.weight", model.tok_emb.weight

    for index, block in enumerate(model.blocks):
        prefix = f"blocks[{index}]"

        yield f"{prefix}.attn.q_proj.weight", block.attn.q_proj.weight
        yield f"{prefix}.attn.q_proj.bias", block.attn.q_proj.bias
        yield f"{prefix}.attn.k_proj.weight", block.attn.k_proj.weight
        yield f"{prefix}.attn.k_proj.bias", block.attn.k_proj.bias
        yield f"{prefix}.attn.v_proj.weight", block.attn.v_proj.weight
        yield f"{prefix}.attn.v_proj.bias", block.attn.v_proj.bias
        yield f"{prefix}.attn.proj.weight", block.attn.proj.weight
        yield f"{prefix}.attn.proj.bias", block.attn.proj.bias

        yield f"{prefix}.mlp.fc1.weight", block.mlp.fc1.weight
        yield f"{prefix}.mlp.fc2.weight", block.mlp.fc2.weight

    yield "lm_head.weight", model.lm_head.weight


@dataclass(frozen=True)
class BenchConfig:
    B: int
    T: int
    V: int
    C: int
    H: int
    L: int

    @property
    def D(self) -> int:
        return self.C // self.H

    def validate(self) -> None:
        if min(self.B, self.T, self.V, self.C, self.H, self.L) <= 0:
            raise ValueError(f"all dimensions must be positive: {self}")
        if self.C % self.H:
            raise ValueError(f"C={self.C} must be divisible by H={self.H}")


@dataclass
class Stats:
    median_ms: float
    mean_ms: float
    min_ms: float
    p90_ms: float
    samples_ms: list[float]


@dataclass
class ModelBundle:
    cfg: BenchConfig
    rt: Runtime
    model: GPT2
    loss: Tensor
    optimizer: AdamW


@dataclass(frozen=True)
class NodeInfo:
    index: int
    label: str
    op_name: str
    layer: int | None
    node: Any


@dataclass
class DispatchEvent:
    ordinal: int
    label: str
    gpu_ms: float
    wall_ms: float


class NullWriter(io.TextIOBase):
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


@contextlib.contextmanager
def suppress_output(enabled: bool = True):
    if not enabled:
        yield
        return
    sink = NullWriter()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def stats(values: Iterable[float]) -> Stats:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sample set")
    return Stats(
        median_ms=float(np.median(array)),
        mean_ms=float(np.mean(array)),
        min_ms=float(np.min(array)),
        p90_ms=float(np.percentile(array, 90)),
        samples_ms=[float(x) for x in array],
    )


def reset_runtime(device: str) -> Runtime:
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device)
    Runtime._default_runtime = rt
    return rt


def make_leaf(
    array: np.ndarray,
    device: str,
    *,
    requires_grad: bool = True,
) -> Tensor:
    value = np.asarray(array)
    return Tensor(
        shape=value.shape,
        dtype=value.dtype,
        device=device,
        requires_grad=requires_grad,
        init_val=value.copy(),
    )


def randn(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    scale: float = 0.02,
) -> np.ndarray:
    return (rng.standard_normal(shape).astype(np.float32) * scale).copy()


def prepare_model(device: str, cfg: BenchConfig, use_fp16_gemm: bool = False) -> ModelBundle:
    rt = reset_runtime(device)
    rt.use_fp16_gemm = bool(use_fp16_gemm)
    rt.backend.use_fp16_gemm = bool(use_fp16_gemm)
    model = GPT2(
        cfg.V, cfg.C, cfg.T, cfg.H, cfg.B, cfg.L, np.float32, device
    )
    rng = np.random.default_rng(SEED + 100)
    for name, tensor in iter_gpt2_named_tensors(model):
        if name.endswith(".gamma"):
            tensor.init_val = np.ones(tensor.shape, dtype=np.float32)
        elif name.endswith(".beta") or name.endswith(".bias"):
            tensor.init_val = np.zeros(tensor.shape, dtype=np.float32)
        else:
            tensor.init_val = randn(rng, tensor.shape)
    input_ids = rng.integers(
        0, cfg.V, size=(cfg.B, cfg.T), dtype=np.int32
    )
    labels = rng.integers(
        0, cfg.V, size=(cfg.B, cfg.T), dtype=np.int32
    )
    input_ids_t = make_leaf(input_ids, device, requires_grad=False)
    labels_t = make_leaf(labels, device, requires_grad=False)
    loss = model(input_ids_t, targets=labels_t)
    rt.loss = loss
    rt.build_topo(loss)
    with suppress_output(True):
        rt.bind_memory()
    rt.is_compiled = True
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    return ModelBundle(cfg=cfg, rt=rt, model=model, loss=loss, optimizer=optimizer)

def describe_topology(bundle: ModelBundle) -> list[NodeInfo]:
    expected = Counter(
        {
            "Gather": 1,
            "FusedRMSNormFlashAttention": bundle.cfg.L,
            "FusedLinearAdd": bundle.cfg.L,
            "FusedRMSNormMLP": bundle.cfg.L,
            "Add": bundle.cfg.L,
            "FusedLinearCrossEntropy": 1,
        }
    )

    actual = Counter(
        node.op.op_name
        for node in bundle.rt.topo
    )

    if actual != expected:
        raise AssertionError(
            "GPT2.forward topology changed; "
            "timing coverage must be updated: "
            f"expected={expected}, actual={actual}"
        )

    occurrences: Counter[str] = Counter()
    result = []

    for index, node in enumerate(bundle.rt.topo):
        op_name = node.op.op_name
        occurrence = occurrences[op_name]
        occurrences[op_name] += 1
        layer = None

        if op_name == "Gather":
            label = "embedding.token_gather"

        elif op_name == "FusedRMSNormFlashAttention":
            layer = occurrence
            label = f"layer{layer}.attention"

        elif op_name == "FusedLinearAdd":
            layer = occurrence
            label = f"layer{layer}.linear_add_from_fa"

            producer = node.inputs[1].op
            if (
                producer is None
                or producer.op.op_name
                != "FusedRMSNormFlashAttention"
            ):
                raise AssertionError(
                    f"{label} is not fed by the real FA node"
                )

        elif op_name == "FusedRMSNormMLP":
            layer = occurrence
            label = f"layer{layer}.mlp_from_fa"

            producer = node.inputs[0].op
            if (
                producer is None
                or producer.op.op_name
                != "FusedLinearAdd"
            ):
                raise AssertionError(
                    f"{label} is not fed by FusedLinearAdd"
                )

        elif op_name == "Add":
            layer = occurrence
            label = f"layer{layer}.residual_add_from_mlp"

        elif op_name == "FusedLinearCrossEntropy":
            label = "final.cross_entropy"

        else:
            raise AssertionError(
                f"uncovered training op: {op_name}"
            )

        result.append(
            NodeInfo(
                index,
                label,
                op_name,
                layer,
                node,
            )
        )

    return result


def _gpu_ms(raw: dict[str, Any]) -> float:
    value = float(raw.get("gpu_ms", -1.0))
    return value if value >= 0.0 else 0.0


class PassCaptureContext:
    """Make an existing begin_pass/end_pass callback return GPU timing."""

    def __init__(self, real_ctx):
        self.real = real_ctx
        self.profile = None

    def __getattr__(self, name):
        return getattr(self.real, name)

    def begin_pass(self):
        return self.real.begin_pass_profiled()

    def end_pass(self):
        self.profile = self.real.end_pass_profiled()
        return self.profile


def profile_real_pass_callback(backend, callback: Callable[[], None]):
    real_ctx = backend.ctx
    proxy = PassCaptureContext(real_ctx)
    begin = time.perf_counter_ns()
    backend.ctx = proxy
    try:
        callback()
    finally:
        backend.ctx = real_ctx
    end = time.perf_counter_ns()
    if proxy.profile is None:
        raise RuntimeError("callback did not execute exactly one Metal pass")
    return proxy.profile, (end - begin) / 1e6


def profile_exact_training_passes(
    bundle: ModelBundle,
    *,
    warmup: int,
    iters: int,
):
    """Profile the same passes used by train_autograd.py after compilation."""
    for _ in range(warmup):
        bundle.rt.execute()
        bundle.optimizer.step()

    graph_gpu = []
    graph_wall = []
    optimizer_gpu = []
    optimizer_wall = []
    for _ in range(iters):
        raw, wall = profile_real_pass_callback(bundle.rt.backend, bundle.rt.execute)
        graph_gpu.append(_gpu_ms(raw))
        graph_wall.append(wall)
        raw, wall = profile_real_pass_callback(
            bundle.rt.backend, bundle.optimizer.step
        )
        optimizer_gpu.append(_gpu_ms(raw))
        optimizer_wall.append(wall)
    return {
        "graph_gpu": stats(graph_gpu),
        "graph_wall": stats(graph_wall),
        "optimizer_gpu": stats(optimizer_gpu),
        "optimizer_wall": stats(optimizer_wall),
    }


def _profile_callback_once(backend, callback):
    begin = time.perf_counter_ns()
    backend.ctx.begin_pass_profiled()
    callback()
    raw = backend.ctx.end_pass_profiled()
    end = time.perf_counter_ns()
    return _gpu_ms(raw), (end - begin) / 1e6


def profile_contextual_nodes(
    bundle: ModelBundle,
    *,
    warmup: int,
    iters: int,
):
    """Time every bound node callback in the exact graph order.

    The only structural difference from Runtime.execute is one command buffer
    per callback, which is necessary to obtain a per-node GPU interval.  No
    synthetic operators, gradient clears, or replacement dispatches are used.
    """
    infos = describe_topology(bundle)
    samples = {
        info.index: {"forward_gpu": [], "forward_wall": [],
                     "backward_gpu": [], "backward_wall": []}
        for info in infos
    }

    for iteration in range(warmup + iters):
        record = iteration >= warmup
        for info in infos:
            gpu, wall = _profile_callback_once(
                bundle.rt.backend,
                lambda node=info.node: node.forward(node.inputs, node.outputs),
            )
            if record:
                samples[info.index]["forward_gpu"].append(gpu)
                samples[info.index]["forward_wall"].append(wall)

        # Runtime.execute performs this fill between forward and backward.
        bundle.rt.backend.begin_pass()
        bundle.rt.backend.fill(bundle.loss.grad, 1.0)
        bundle.rt.backend.end_pass()

        for info in reversed(infos):
            gpu, wall = _profile_callback_once(
                bundle.rt.backend,
                lambda node=info.node: node.backward(node.inputs, node.outputs),
            )
            if record:
                samples[info.index]["backward_gpu"].append(gpu)
                samples[info.index]["backward_wall"].append(wall)

    result = []
    for info in infos:
        item = samples[info.index]
        result.append(
            {
                "info": info,
                "forward_gpu": stats(item["forward_gpu"]),
                "forward_wall": stats(item["forward_wall"]),
                "backward_gpu": stats(item["backward_gpu"]),
                "backward_wall": stats(item["backward_wall"]),
            }
        )
    return result


def _gemm_label(args) -> str:
    try:
        dims = list(args[2])
        if len(dims) >= 8:
            batch, m, n, k, ta, tb, _, _ = dims[:8]
            return (
                f"MPS_GEMM[B={batch},M={m},N={n},K={k},"
                f"TA={ta},TB={tb}]"
            )
    except Exception:
        pass
    return "MPS_GEMM"


class DispatchCaptureContext:
    """Profile every dispatch made by one real bound Runtime callback."""

    INTERCEPTED = {
        "dispatch",
        "dispatch_with_args",
        "dispatch_with_args_3d",
        "dispatch_strided",
        "dispatch_contiguous",
        "dispatch_softmax",
        "dispatch_sdpa",
        "dispatch_flash_attn",
        "dispatch_flash_attn_bwd",
        "dispatch_gemm_mps",
        "gemm_backward_dw_batched_mps",
        "gemm_backward_dx_batched_mps",
        "fill_scalar",
    }

    def __init__(self, real_ctx):
        self.real = real_ctx
        self.events: list[DispatchEvent] = []

    def __getattr__(self, name):
        value = getattr(self.real, name)
        if name not in self.INTERCEPTED or not callable(value):
            return value

        def wrapped(*args, **kwargs):
            if name == "dispatch_gemm_mps":
                label = _gemm_label(args)
            elif args and isinstance(args[0], str):
                label = str(args[0])
            else:
                label = name
            begin = time.perf_counter_ns()
            self.real.begin_pass_profiled()
            value_out = value(*args, **kwargs)
            raw = self.real.end_pass_profiled()
            end = time.perf_counter_ns()
            self.events.append(
                DispatchEvent(
                    ordinal=len(self.events),
                    label=label,
                    gpu_ms=_gpu_ms(raw),
                    wall_ms=(end - begin) / 1e6,
                )
            )
            return value_out

        return wrapped


def capture_callback_dispatches(backend, callback) -> list[DispatchEvent]:
    real_ctx = backend.ctx
    proxy = DispatchCaptureContext(real_ctx)
    backend.ctx = proxy
    try:
        callback()
    finally:
        backend.ctx = real_ctx
    return proxy.events


def profile_callback_dispatches(
    backend,
    callback,
    *,
    warmup: int,
    iters: int,
):
    for _ in range(warmup):
        capture_callback_dispatches(backend, callback)

    grouped: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: {"gpu": [], "wall": []}
    )
    expected_sequence = None
    for _ in range(iters):
        events = capture_callback_dispatches(backend, callback)
        sequence = [(event.ordinal, event.label) for event in events]
        if expected_sequence is None:
            expected_sequence = sequence
        elif sequence != expected_sequence:
            raise AssertionError(
                "dispatch sequence changed between samples: "
                f"expected={expected_sequence}, got={sequence}"
            )
        for event in events:
            key = (event.ordinal, event.label)
            grouped[key]["gpu"].append(event.gpu_ms)
            grouped[key]["wall"].append(event.wall_ms)

    result = []
    for (ordinal, label), values in sorted(grouped.items()):
        result.append(
            {
                "ordinal": ordinal,
                "label": label,
                "gpu": stats(values["gpu"]),
                "wall": stats(values["wall"]),
            }
        )
    return result


def select_expensive_nodes(
    node_results,
    *,
    threshold_ms: float,
) -> list[tuple[NodeInfo, str]]:
    """Use last-layer nodes so backward dependency distance is minimal."""
    selected: dict[str, tuple[NodeInfo, float, float]] = {}
    for result in node_results:
        info = result["info"]
        fwd = result["forward_gpu"].median_ms
        bwd = result["backward_gpu"].median_ms
        previous = selected.get(info.op_name)
        if previous is None or (info.layer or -1) >= (previous[0].layer or -1):
            selected[info.op_name] = (info, fwd, bwd)

    phases = []
    for op_name in (
        "FusedRMSNormFlashAttention",
        "FusedLinearAdd",
        "FusedRMSNormMLP",
        "FusedLinearCrossEntropy",
    ):
        if op_name not in selected:
            continue
        info, fwd, bwd = selected[op_name]
        if fwd >= threshold_ms:
            phases.append((info, "forward"))
        if bwd >= threshold_ms:
            phases.append((info, "backward"))
    return phases


def prepare_real_callback_inputs(bundle: ModelBundle) -> None:
    """Populate all activations and gradients once using the exact graph pass."""
    bundle.rt.execute()


def print_node_results(results) -> None:
    print("\nPer-node cost from real bound callbacks")
    print(
        f"{'training node':<43s} {'op':<34s} "
        f"{'forward GPU':>13s} {'backward GPU':>14s}"
    )
    grouped = defaultdict(list)
    for result in results:
        info = result["info"]
        fwd = result["forward_gpu"].median_ms
        bwd = result["backward_gpu"].median_ms
        grouped[info.op_name].append((fwd, bwd))
        print(
            f"{info.label:<43s} {info.op_name:<34s} "
            f"{fwd:11.3f} ms {bwd:12.3f} ms"
        )

    print("\nGrouped contribution across real occurrences")
    print(
        f"{'op':<34s} {'count':>6s} {'forward sum':>14s} "
        f"{'backward sum':>15s} {'total':>12s}"
    )
    for op_name, values in grouped.items():
        fwd = sum(item[0] for item in values)
        bwd = sum(item[1] for item in values)
        print(
            f"{op_name:<34s} {len(values):6d} "
            f"{fwd:12.3f} ms {bwd:13.3f} ms {fwd+bwd:10.3f} ms"
        )


def print_internal_profiles(
    bundle: ModelBundle,
    node_results,
    *,
    warmup: int,
    iters: int,
    operator_threshold_ms: float,
    kernel_threshold_ms: float,
) -> None:
    prepare_real_callback_inputs(bundle)
    phases = select_expensive_nodes(
        node_results, threshold_ms=operator_threshold_ms
    )
    print("\nInternal dispatches actually issued by debug/backend.py")
    if not phases:
        print("  no operator phase exceeded the selected threshold")
        return

    for info, phase in phases:
        callback = (
            functools.partial(info.node.forward, info.node.inputs, info.node.outputs)
            if phase == "forward"
            else functools.partial(info.node.backward, info.node.inputs, info.node.outputs)
        )
        events = profile_callback_dispatches(
            bundle.rt.backend,
            callback,
            warmup=warmup,
            iters=iters,
        )
        print(f"\n  {info.label} / {phase}")
        shown = 0
        total = 0.0
        for event in events:
            gpu = event["gpu"].median_ms
            total += gpu
            if gpu < kernel_threshold_ms:
                continue
            shown += 1
            print(
                f"    {event['ordinal']+1:02d} {event['label']:<66s} "
                f"GPU={gpu:9.3f} ms  wall={event['wall'].median_ms:9.3f} ms"
            )
        if shown == 0:
            print("    all dispatches were below the kernel threshold")
        print(f"    {'dispatch GPU sum':<69s} {total:9.3f} ms")

def profile_kv_cache_attention(
    *,
    batch_size: int,
    max_seq_len: int,
    embed_dim: int,
    num_heads: int,
    warmup: int = 3,
    iters: int = 20,
) -> None:
    from op import KVCacheAttentionOp
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if embed_dim <= 0 or num_heads <= 0:
        raise ValueError("embed_dim and num_heads must be positive")
    if embed_dim % num_heads:
        raise ValueError("embed_dim must be divisible by num_heads")
    if warmup < 0 or iters <= 0:
        raise ValueError("require warmup >= 0 and iters > 0")

    head_dim = embed_dim // num_heads
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")

    previous_runtime = Runtime._default_runtime
    rt = Runtime(device="mps", forward_only=True, owns_params=False)
    Runtime._default_runtime = rt

    try:
        rng = np.random.default_rng(SEED + 200)

        def leaf(array):
            return make_leaf(
                np.asarray(array),
                "mps",
                requires_grad=False,
            )

        x_host = (
            rng.standard_normal((batch_size, 1, embed_dim))
            .astype(np.float32)
        )

        weight_shape = (embed_dim, embed_dim)
        bias_shape = (embed_dim,)

        wq_host = randn(rng, weight_shape)
        bq_host = np.zeros(bias_shape, dtype=np.float32)

        wk_host = randn(rng, weight_shape)
        bk_host = np.zeros(bias_shape, dtype=np.float32)

        wv_host = randn(rng, weight_shape)
        bv_host = np.zeros(bias_shape, dtype=np.float32)

        # KV cache layout must match engine.KVCache:
        # [batch, max_seq_len, num_heads, head_dim]
        k_cache_host = np.zeros(
            (batch_size, max_seq_len, num_heads, head_dim),
            dtype=np.float32,
        )
        v_cache_host = np.zeros_like(k_cache_host)

        position_host = np.asarray([0], dtype=np.int32)

        # Same RoPE layout as module.GPT2:
        # [max_seq_len, head_dim // 2]
        positions = np.arange(
            max_seq_len,
            dtype=np.float32,
        )[:, None]
        channels = np.arange(
            0,
            head_dim,
            2,
            dtype=np.float32,
        )[None, :]
        inv_freq = 100000.0 ** (-channels / head_dim)
        freqs = positions * inv_freq

        rope_cos_host = np.cos(freqs).astype(np.float32)
        rope_sin_host = np.sin(freqs).astype(np.float32)

        x = leaf(x_host)
        wq = leaf(wq_host)
        bq = leaf(bq_host)
        wk = leaf(wk_host)
        bk = leaf(bk_host)
        wv = leaf(wv_host)
        bv = leaf(bv_host)
        k_cache = leaf(k_cache_host)
        v_cache = leaf(v_cache_host)
        position = leaf(position_host)
        rope_cos = leaf(rope_cos_host)
        rope_sin = leaf(rope_sin_host)

        output = KVCacheAttentionOp(
            n_head=num_heads,
            layer_idx=0,
            eps=1e-6,
            mode="decode",
        )(
            x,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            k_cache,
            v_cache,
            position,
            rope_cos,
            rope_sin,
        )

        rt.loss = output
        rt.build_topo(output)

        with suppress_output(True):
            rt.bind_memory()

        rt.is_compiled = True

        callback = functools.partial(
            output.op.forward,
            output.op.inputs,
            output.op.outputs,
        )

        state_tensors = [
            x,
            wq,
            bq,
            wk,
            bk,
            wv,
            bv,
            k_cache,
            v_cache,
            position,
            rope_cos,
            rope_sin,
        ]

        state_hosts = [
            x_host,
            wq_host,
            bq_host,
            wk_host,
            bk_host,
            wv_host,
            bv_host,
            k_cache_host,
            v_cache_host,
            position_host,
            rope_cos_host,
            rope_sin_host,
        ]

        gpu_samples = []
        wall_samples = []

        for iteration in range(warmup + iters):
            # Reset inputs and KV cache outside the measured interval.
            rt.backend.synchronize()

            for tensor, host in zip(state_tensors, state_hosts):
                rt.backend.copy_from_host(tensor, host)

            rt.backend.synchronize()

            gpu_ms, wall_ms = _profile_callback_once(
                rt.backend,
                callback,
            )

            if iteration >= warmup:
                gpu_samples.append(gpu_ms)
                wall_samples.append(wall_ms)

        rt.backend.synchronize()

        gpu = stats(gpu_samples)
        wall = stats(wall_samples)

        print(
            "\nKVCacheAttention operator timing "
            f"(decode, B={batch_size}, S={max_seq_len}, "
            f"C={embed_dim}, H={num_heads}, D={head_dim}, "
            f"warmup={warmup}, iters={iters})"
        )
        print(
            f"{'callback':<34s} "
            f"{'GPU median':>13s} "
            f"{'GPU p90':>13s} "
            f"{'wall median':>14s}"
        )
        print(
            f"{'metal_kv_cache_attention_forward':<34s} "
            f"{gpu.median_ms:10.6f} ms "
            f"{gpu.p90_ms:10.6f} ms "
            f"{wall.median_ms:11.6f} ms"
        )

    finally:
        Runtime._default_runtime = previous_runtime

def profile_rollout_ops(
    *,
    batch_size: int,
    vocab_size: int,
    top_k: int = 50,
    max_top_k: int = 64,
    temperature: float = 1.0,
    warmup: int = 3,
    iters: int = 20,
) -> None:
    from op import RolloutTopkOp, RolloutSampleOp

    if batch_size <= 0 or vocab_size <= 0:
        raise ValueError("require batch_size > 0 and vocab_size > 0")
    # 当前 RolloutTopkOp 的输出容量固定为 min(64, V)。
    if not 1 <= top_k <= min(64, max_top_k, vocab_size):
        raise ValueError(
            "require 1 <= top_k <= min(64, max_top_k, vocab_size)"
        )
    if warmup < 0 or iters <= 0:
        raise ValueError("require warmup >= 0 and iters > 0")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("require finite temperature >= 0")

    previous_runtime = Runtime._default_runtime
    rt = Runtime(device="mps", forward_only=True, owns_params=False)
    Runtime._default_runtime = rt

    try:
        rng = np.random.default_rng(SEED)

        def leaf(array):
            return make_leaf(
                np.asarray(array), "mps", requires_grad=False
            )

        def int_array(shape, value=0):
            return np.full(shape, value, dtype=np.int32)

        # 测量一次 decode；logits 的时间维度为 1，与训练 T 无关。
        max_tokens = 256
        logits_host = rng.standard_normal(
            (batch_size, 1, vocab_size)
        ).astype(np.float32)
        logits = leaf(logits_host)
        k_tensor = leaf(int_array((1,), top_k))
        temperature_tensor = leaf(
            np.array([temperature], dtype=np.float32)
        )

        rng_host = np.zeros((batch_size, 4), dtype=np.uint32)
        rng_host[:, 0] = np.uint32(SEED)

        state_hosts = [
            rng_host,
            np.zeros((max_tokens, batch_size, 2), dtype=np.uint32),
            int_array((batch_size,)),                 # finished
            int_array((batch_size, 1), -1),           # next_tokens
            int_array((batch_size, max_tokens), -1),  # generated_ids
            int_array((batch_size,)),                 # lengths
            int_array((1,)),                         # position
            int_array((1,)),                         # step
        ]
        state_tensors = [leaf(host) for host in state_hosts]
        (
            rng_state,
            rng_raw,
            finished,
            next_tokens,
            generated_ids,
            lengths,
            position,
            step,
        ) = state_tensors

        values, indices = RolloutTopkOp(
            max_top_k=max_top_k,
            mode="decode",
        )(logits, k_tensor, position)

        # 返回的是图输出 finished_all；token 写入 next_tokens。
        finished_all = RolloutSampleOp(
            eos_token=vocab_size - 1,
            bos_token=0,
            max_tokens=max_tokens,
            mode="decode",
        )(
            values,
            indices,
            rng_state,
            rng_raw,
            finished,
            temperature_tensor,
            k_tensor,
            next_tokens,
            generated_ids,
            lengths,
            position,
            step,
            logits,
        )

        rt.loss = finished_all
        rt.build_topo(finished_all)
        with suppress_output(True):
            rt.bind_memory()
        rt.is_compiled = True

        # 使用 Runtime 绑定好的回调，保留 Sample 所需 scratch。
        callbacks = [
            (
                "metal_rollout_topk_forward",
                functools.partial(
                    values.op.forward,
                    values.op.inputs,
                    values.op.outputs,
                ),
            ),
            (
                "metal_rollout_sample_forward",
                functools.partial(
                    finished_all.op.forward,
                    finished_all.op.inputs,
                    finished_all.op.outputs,
                ),
            ),
        ]
        samples = {
            name: {"gpu": [], "wall": []}
            for name, _ in callbacks
        }

        for iteration in range(warmup + iters):
            # 同步及状态重置不计入算子耗时。
            rt.backend.synchronize()
            for tensor, host in zip(state_tensors, state_hosts):
                rt.backend.copy_from_host(tensor, host)
            rt.backend.synchronize()

            # 每轮先生成 top-k，再测量使用该结果的 Sample。
            for name, callback in callbacks:
                gpu_ms, wall_ms = _profile_callback_once(
                    rt.backend, callback
                )
                if iteration >= warmup:
                    samples[name]["gpu"].append(gpu_ms)
                    samples[name]["wall"].append(wall_ms)

        # 所有结果检查均在计时范围之外。
        rt.backend.synchronize()
        actual_values = rt.backend.numpy_view(values).copy()[:, :top_k]
        actual_indices = rt.backend.numpy_view(indices).copy()[:, :top_k]
        tokens = rt.backend.numpy_view(next_tokens).copy().reshape(-1)

        expected_indices = np.argsort(
            -logits_host[:, 0, :], axis=1, kind="stable"
        )[:, :top_k]
        expected_values = np.take_along_axis(
            logits_host[:, 0, :], expected_indices, axis=1
        )
        np.testing.assert_array_equal(actual_indices, expected_indices)
        np.testing.assert_allclose(
            actual_values, expected_values, rtol=0, atol=1e-6
        )
        if not np.all(
            np.any(tokens[:, None] == expected_indices, axis=1)
        ):
            raise RuntimeError(f"sampled tokens outside top-k: {tokens}")
        if temperature == 0:
            np.testing.assert_array_equal(
                tokens, expected_indices[:, 0]
            )

        print(
            "\nRollout operator timing "
            f"(decode, B={batch_size}, V={vocab_size}, "
            f"top_k={top_k}, K_MAX={values.shape[-1]}, "
            f"temperature={temperature}, "
            f"warmup={warmup}, iters={iters})"
        )
        print(
            f"{'callback':<34s} "
            f"{'GPU median':>13s} "
            f"{'GPU p90':>13s} "
            f"{'wall median':>14s}"
        )
        for name, _ in callbacks:
            gpu = stats(samples[name]["gpu"])
            wall = stats(samples[name]["wall"])
            print(
                f"{name:<34s} "
                f"{gpu.median_ms:10.6f} ms "
                f"{gpu.p90_ms:10.6f} ms "
                f"{wall.median_ms:11.6f} ms"
            )
    finally:
        Runtime._default_runtime = previous_runtime

def profile_prepared_callback_dispatches(
    backend,
    callback,
    prepare,
    *,
    warmup: int,
    iters: int,
):
    def capture_once():
        return capture_callback_dispatches(backend, callback)

    for _ in range(warmup):
        prepare()
        capture_once()

    grouped = defaultdict(
        lambda: {"gpu": [], "wall": []}
    )
    expected_sequence = None

    for _ in range(iters):
        prepare()
        events = capture_once()

        sequence = [
            (event.ordinal, event.label)
            for event in events
        ]

        if expected_sequence is None:
            expected_sequence = sequence
        elif sequence != expected_sequence:
            raise AssertionError(
                "dispatch sequence changed between samples: "
                f"expected={expected_sequence}, got={sequence}"
            )

        for event in events:
            key = (event.ordinal, event.label)
            grouped[key]["gpu"].append(event.gpu_ms)
            grouped[key]["wall"].append(event.wall_ms)

    result = []
    for (ordinal, label), values in sorted(grouped.items()):
        result.append(
            {
                "ordinal": ordinal,
                "label": label,
                "gpu": stats(values["gpu"]),
                "wall": stats(values["wall"]),
            }
        )

    return result


def print_qwen_dispatches(phase, events):
    print(f"\n    {phase} internal dispatches")

    total_gpu = 0.0

    for event in events:
        gpu = event["gpu"].median_ms
        wall = event["wall"].median_ms
        total_gpu += gpu

        print(
            f"      {event['ordinal'] + 1:02d} "
            f"{event['label']:<66s} "
            f"GPU={gpu:10.3f} ms "
            f"wall={wall:10.3f} ms"
        )

    print(
        f"      {'dispatch GPU sum':<69s}"
        f"{total_gpu:10.3f} ms"
    )

def profile_qwen_fused_ops(
    *,
    batch_size: int,
    seq_len: int,
    warmup: int = 3,
    iters: int = 20,
) -> None:
    from op import (
        FusedRMSNormGQAOp,
        FusedRMSNormLinearCrossEntropyOp,
        FusedRMSNormMLPOp,
    )

    # Qwen 配置，与 module.QwenCoder / check_qwen_kernel.py 保持一致。
    vocab_size = 151936
    embed_dim = 1536
    mlp_hidden_dim = 8960
    n_head = 12
    n_kv_head = 2
    eps = 1e-6
    rope_theta = 1_000_000.0
    qwen_max_seq_len = 32768

    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    if seq_len > qwen_max_seq_len:
        raise ValueError(
            f"seq_len must be <= Qwen max sequence length "
            f"{qwen_max_seq_len}"
        )
    if warmup < 0 or iters <= 0:
        raise ValueError("require warmup >= 0 and iters > 0")

    previous_runtime = Runtime._default_runtime

    try:
        def run_case(label, build_case):
            rt = Runtime(
                device="mps",
                forward_only=False,
                owns_params=True,
            )
            Runtime._default_runtime = rt

            output = build_case()

            rt.loss = output
            rt.build_topo(output)

            with suppress_output(True):
                rt.bind_memory()

            rt.is_compiled = True

            node = output.op

            forward_callback = functools.partial(
                node.forward,
                node.inputs,
                node.outputs,
            )
            backward_callback = functools.partial(
                node.backward,
                node.inputs,
                node.outputs,
            )

            def prepare_backward():
                # 重新生成 backward 所需的 forward 中间结果。
                rt.backend.begin_pass()
                forward_callback()
                rt.backend.end_pass()

                # 填充输出梯度，保持与 profile_contextual_nodes()
                # 中的 backward 前处理一致。
                rt.backend.begin_pass()
                rt.backend.fill(output.grad, 1.0)
                rt.backend.end_pass()

            # ------------------------------------------------------------
            # 算子整体 forward/backward 时间
            # ------------------------------------------------------------
            forward_gpu = []
            forward_wall = []
            backward_gpu = []
            backward_wall = []

            for iteration in range(warmup + iters):
                rt.backend.synchronize()

                f_gpu, f_wall = _profile_callback_once(
                    rt.backend,
                    forward_callback,
                )

                rt.backend.begin_pass()
                rt.backend.fill(output.grad, 1.0)
                rt.backend.end_pass()

                b_gpu, b_wall = _profile_callback_once(
                    rt.backend,
                    backward_callback,
                )

                if iteration >= warmup:
                    forward_gpu.append(f_gpu)
                    forward_wall.append(f_wall)
                    backward_gpu.append(b_gpu)
                    backward_wall.append(b_wall)

            rt.backend.synchronize()

            forward_stats = stats(forward_gpu)
            forward_wall_stats = stats(forward_wall)
            backward_stats = stats(backward_gpu)
            backward_wall_stats = stats(backward_wall)

            print(
                f"{label:<38s} "
                f"{forward_stats.median_ms:10.6f} ms "
                f"{forward_wall_stats.median_ms:10.6f} ms "
                f"{backward_stats.median_ms:10.6f} ms "
                f"{backward_wall_stats.median_ms:10.6f} ms"
            )

            # ------------------------------------------------------------
            # forward 内部 kernel 计时
            # ------------------------------------------------------------
            forward_dispatches = profile_callback_dispatches(
                rt.backend,
                forward_callback,
                warmup=warmup,
                iters=iters,
            )

            # ------------------------------------------------------------
            # backward 内部 kernel 计时
            # ------------------------------------------------------------
            backward_dispatches = profile_prepared_callback_dispatches(
                rt.backend,
                backward_callback,
                prepare_backward,
                warmup=warmup,
                iters=iters,
            )

            print_qwen_dispatches(
                "forward",
                forward_dispatches,
            )
            print_qwen_dispatches(
                "backward",
                backward_dispatches,
            )

        rng = np.random.default_rng(SEED + 300)

        def leaf(array, requires_grad=True):
            return make_leaf(
                np.asarray(array),
                "mps",
                requires_grad=requires_grad,
            )

        def build_mlp():
            x = leaf(np.zeros(
                (batch_size, seq_len, embed_dim),
                dtype=np.float32,
            ))
            norm_weight = leaf(np.ones(
                (embed_dim,),
                dtype=np.float32,
            ))
            wgu = leaf(np.zeros(
                (embed_dim, 2 * mlp_hidden_dim),
                dtype=np.float32,
            ))
            wdown = leaf(np.zeros(
                (mlp_hidden_dim, embed_dim),
                dtype=np.float32,
            ))

            return FusedRMSNormMLPOp(eps=eps)(
                x,
                norm_weight,
                wgu,
                wdown,
            )

        def build_gqa():
            head_dim = embed_dim // n_head
            kv_dim = n_kv_head * head_dim

            x = leaf(np.zeros(
                (batch_size, seq_len, embed_dim),
                dtype=np.float32,
            ))
            norm_weight = leaf(np.ones(
                (embed_dim,),
                dtype=np.float32,
            ))

            wq = leaf(np.zeros(
                (embed_dim, embed_dim),
                dtype=np.float32,
            ))
            bq = leaf(np.zeros(
                (embed_dim,),
                dtype=np.float32,
            ))
            wk = leaf(np.zeros(
                (embed_dim, kv_dim),
                dtype=np.float32,
            ))
            bk = leaf(np.zeros(
                (kv_dim,),
                dtype=np.float32,
            ))
            wv = leaf(np.zeros(
                (embed_dim, kv_dim),
                dtype=np.float32,
            ))
            bv = leaf(np.zeros(
                (kv_dim,),
                dtype=np.float32,
            ))

            positions = np.arange(
                qwen_max_seq_len,
                dtype=np.float32,
            )[:, None]
            channels = np.arange(
                0,
                head_dim,
                2,
                dtype=np.float32,
            )[None, :]
            inv_freq = (
                rope_theta ** (-channels / np.float32(head_dim))
            )
            freqs = positions * inv_freq

            rope_cos = leaf(
                np.cos(freqs).astype(np.float32),
                requires_grad=False,
            )
            rope_sin = leaf(
                np.sin(freqs).astype(np.float32),
                requires_grad=False,
            )

            return FusedRMSNormGQAOp(
                n_head=n_head,
                n_kv_head=n_kv_head,
                eps=eps,
                is_causal=True,
                rope_sign=1,
            )(
                x,
                norm_weight,
                wq,
                bq,
                wk,
                bk,
                wv,
                bv,
                rope_cos,
                rope_sin,
            )

        def build_linear_cross_entropy():
            x = leaf(np.zeros(
                (batch_size, seq_len, embed_dim),
                dtype=np.float32,
            ))
            weight = leaf(np.zeros(
                (vocab_size, embed_dim),
                dtype=np.float32,
            ))
            norm_weight = leaf(np.ones(
                (embed_dim,),
                dtype=np.float32,
            ))
            targets = leaf(
                np.zeros(
                    (batch_size, seq_len),
                    dtype=np.int32,
                ),
                requires_grad=False,
            )

            return FusedRMSNormLinearCrossEntropyOp(
                eps=eps,
            )(
                x,
                weight,
                norm_weight,
                targets,
            )

        print(
            "\nQwen fused operator timing "
            f"(B={batch_size}, T={seq_len}, V={vocab_size}, "
            f"C={embed_dim}, H={mlp_hidden_dim}, "
            f"n_head={n_head}, n_kv_head={n_kv_head}, "
            f"warmup={warmup}, iters={iters})"
        )
        print(
            f"{'operator':<38s} "
            f"{'forward GPU':>15s} "
            f"{'forward wall':>16s} "
            f"{'backward GPU':>16s} "
            f"{'backward wall':>17s}"
        )

        run_case(
            "FusedRMSNormMLP(norm_weight)",
            build_mlp,
        )
        run_case(
            "FusedRMSNormGQA",
            build_gqa,
        )
        run_case(
            "FusedRMSNormLinearCrossEntropy",
            build_linear_cross_entropy,
        )

    finally:
        Runtime._default_runtime = previous_runtime

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exact train_autograd.py MPS callback timing with automatic "
            "per-dispatch kernel profiling"
        )
    )
    # Defaults and model-dimension derivation match train_autograd.py.
    parser.add_argument("--depth", "--L", dest="depth", type=int, default=6)
    parser.add_argument(
        "--aspect-ratio",
        type=int,
        default=64,
        help="base model dimension is depth * aspect_ratio",
    )
    parser.add_argument(
        "--head-dim", type=int, default=64,
        help="model dimension is rounded up to a multiple of head_dim",
    )
    parser.add_argument(
        "--max-seq-len", "--T", dest="max_seq_len", type=int, default=512,
    )
    parser.add_argument(
        "--device-batch-size", "--B", dest="device_batch_size",
        type=int, default=32,
    )
    parser.add_argument(
        "--vocab-size", "--V", dest="vocab_size", type=int, default=32768,
        help="train_autograd.py obtains this from the tokenizer",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--kernel-warmup", type=int, default=1)
    parser.add_argument("--kernel-iters", type=int, default=3)
    parser.add_argument(
        "--operator-threshold-ms",
        type=float,
        default=1.0,
        help="only break down forward/backward phases at or above this GPU time",
    )
    parser.add_argument(
        "--kernel-threshold-ms",
        type=float,
        default=0.0,
        help="hide individual dispatches below this GPU time",
    )
    parser.add_argument("--use-fp16-gemm", action="store_true", help="enable the fused FP16 GEMM path for GPT timing")
    parser.add_argument("--skip-kernels", action="store_true")
    args = parser.parse_args()

    if min(args.warmup, args.iters, args.kernel_warmup, args.kernel_iters) < 0:
        parser.error("iteration counts must be non-negative")
    if args.iters == 0 or args.kernel_iters == 0:
        parser.error("--iters and --kernel-iters must be positive")

    if min(
        args.depth,
        args.aspect_ratio,
        args.head_dim,
        args.max_seq_len,
        args.device_batch_size,
        args.vocab_size,
    ) <= 0:
        parser.error("all model configuration values must be positive")

    base_dim = args.depth * args.aspect_ratio
    model_dim = (
        (base_dim + args.head_dim - 1)
        // args.head_dim
        * args.head_dim
    )
    num_heads = model_dim // args.head_dim
    cfg = BenchConfig(
        B=args.device_batch_size,
        T=args.max_seq_len,
        V=args.vocab_size,
        C=model_dim,
        H=num_heads,
        L=args.depth,
    )
    cfg.validate()

    print(f"repository: {ROOT_DIR}")
    print("runtime path: debug/runtime.py -> debug/backend.py")
    print(
        f"performance config: B={cfg.B} T={cfg.T} V={cfg.V} "
        f"C={cfg.C} H={cfg.H} D={cfg.D} L={cfg.L}"
    )

    print(f"use_fp16_gemm={args.use_fp16_gemm}")
    bundle = prepare_model("mps", cfg, use_fp16_gemm=args.use_fp16_gemm)
    topology = describe_topology(bundle)
    print("\nTraining topology covered")
    counts = Counter(info.op_name for info in topology)
    for op_name, count in counts.items():
        print(f"  {op_name:<34s} x {count}")

    exact = profile_exact_training_passes(
        bundle, warmup=args.warmup, iters=args.iters
    )
    print("\nExact train_autograd.py Metal passes")
    print(
        f"  loss.backward graph pass  GPU={exact['graph_gpu'].median_ms:9.3f} ms  "
        f"wall={exact['graph_wall'].median_ms:9.3f} ms"
    )
    print(
        f"  optimizer.step pass        GPU={exact['optimizer_gpu'].median_ms:9.3f} ms  "
        f"wall={exact['optimizer_wall'].median_ms:9.3f} ms"
    )

    node_results = profile_contextual_nodes(
        bundle, warmup=args.warmup, iters=args.iters
    )
    print_node_results(node_results)

    if not args.skip_kernels:
        print_internal_profiles(
            bundle,
            node_results,
            warmup=args.kernel_warmup,
            iters=args.kernel_iters,
            operator_threshold_ms=args.operator_threshold_ms,
            kernel_threshold_ms=args.kernel_threshold_ms,
        )

    # profile_kv_cache_attention(
    #     batch_size=cfg.B,
    #     max_seq_len=cfg.T,
    #     embed_dim=cfg.C,
    #     num_heads=cfg.H,
    #     warmup=args.warmup,
    #     iters=args.iters,
    # )

    # profile_rollout_ops(
    #         batch_size=cfg.B,
    #         vocab_size=cfg.V,
    #         top_k=min(50, cfg.V),
    #         max_top_k=64,
    #         temperature=1.0,
    #         warmup=args.warmup,
    #         iters=args.iters,
    #     )

    # profile_qwen_fused_ops(
    #     batch_size=cfg.B,
    #     seq_len=cfg.T,
    #     warmup=args.warmup,
    #     iters=args.iters,
    # )


if __name__ == "__main__":
    main()
