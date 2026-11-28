#!/usr/bin/env python3
"""Profile the exact GPT2 CUDA training callbacks and internal kernels used by train_gpt2.py.

Usage (defaults match train_gpt2.py d12 config: B=8, T=1024, V=32768, L=12, accum=4, mini, bf16):
  python -B debug/timing_cuda_ops.py
  python -B debug/timing_cuda_ops.py --grad-dtype fp32
  python -B debug/timing_cuda_ops.py --operator-threshold-ms 0 --kernel-threshold-ms 0 --verbose-dispatches
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import functools
import gc
import io
import math
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
    env_root = os.environ.get("AUTOGRAD_ROOT")
    candidates = [Path(env_root)] if env_root else []
    here = Path(__file__).resolve()
    candidates.extend((Path.cwd(), Path.cwd().parent, here.parent, here.parent.parent))
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "train_gpt2.py").is_file() and (candidate / "backend" / "cuda.py").is_file():
            return candidate
    raise RuntimeError(
        "cannot locate autograd repository; run from its root or set AUTOGRAD_ROOT=/path/to/autograd"
    )


ROOT_DIR = _find_repo_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from module import GPT2  # noqa: E402
from optimizer import AdamW  # noqa: E402
from runtime import Runtime  # noqa: E402
from tensor import Tensor  # noqa: E402


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
    grad_accum_steps: int
    grad_dtype: str
    optimizer_state: str
    max_norm: float

    @property
    def D(self) -> int:
        return self.C // self.H

    @property
    def total_batch_size(self) -> int:
        return self.B * self.T * self.grad_accum_steps

    def validate(self) -> None:
        if min(self.B, self.T, self.V, self.C, self.H, self.L, self.grad_accum_steps) <= 0:
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
    clip_grad_norm: Callable[[float], tuple[float, float]] | None


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


class CudaEventTimer:
    """Hardware GPU timer using libcudart cudaEvent (automatically falls back to sync + wall clock)."""

    def __init__(self, real_ctx) -> None:
        self.real_ctx = real_ctx
        self._lib = self._load_cudart()
        self._start = ctypes.c_void_p()
        self._end = ctypes.c_void_p()
        if self._lib is not None:
            r0 = self._lib.cudaEventCreate(ctypes.byref(self._start))
            r1 = self._lib.cudaEventCreate(ctypes.byref(self._end))
            if r0 != 0 or r1 != 0:
                self._lib = None

    @property
    def mode(self) -> str:
        return "cudaEvent (hardware GPU clock)" if self._lib is not None else "cudaStreamSynchronize (wall clock)"

    @staticmethod
    def _load_cudart():
        candidates = []
        maps_path = Path("/proc/self/maps")
        if maps_path.is_file():
            for line in maps_path.read_text(errors="ignore").splitlines():
                if "libcudart.so" in line and "/" in line:
                    path = "/" + line.split("/", 1)[1].strip()
                    if path not in candidates:
                        candidates.append(path)
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        candidates.extend(("libcudart.so", "libcudart.so.12", "libcudart.so.11.0", None))

        for candidate in candidates:
            try:
                lib = ctypes.CDLL(candidate)
                lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                lib.cudaEventCreate.restype = ctypes.c_int
                lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
                lib.cudaEventDestroy.restype = ctypes.c_int
                lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                lib.cudaEventRecord.restype = ctypes.c_int
                lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
                lib.cudaEventSynchronize.restype = ctypes.c_int
                lib.cudaEventElapsedTime.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
                lib.cudaEventElapsedTime.restype = ctypes.c_int
                return lib
            except (OSError, AttributeError):
                continue
        return None

    def close(self) -> None:
        if self._lib is not None:
            if self._start.value:
                self._lib.cudaEventDestroy(self._start)
                self._start = ctypes.c_void_p()
            if self._end.value:
                self._lib.cudaEventDestroy(self._end)
                self._end = ctypes.c_void_p()
            self._lib = None

    def begin_interval(self) -> int:
        self.real_ctx.synchronize()
        wall_t0 = time.perf_counter_ns()
        if self._lib is not None:
            if self._lib.cudaEventRecord(self._start, None) != 0:
                raise RuntimeError("cudaEventRecord(start) failed")
        return wall_t0

    def end_interval(self, wall_t0: int) -> tuple[float, float]:
        if self._lib is not None:
            if self._lib.cudaEventRecord(self._end, None) != 0:
                raise RuntimeError("cudaEventRecord(end) failed")
            if self._lib.cudaEventSynchronize(self._end) != 0:
                raise RuntimeError("cudaEventSynchronize(end) failed")
            ms = ctypes.c_float(0.0)
            if self._lib.cudaEventElapsedTime(ctypes.byref(ms), self._start, self._end) != 0:
                raise RuntimeError("cudaEventElapsedTime failed")
            self.real_ctx.synchronize()
            wall_ms = (time.perf_counter_ns() - wall_t0) / 1e6
            return float(ms.value), wall_ms

        self.real_ctx.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_t0) / 1e6
        return wall_ms, wall_ms

    def measure(self, fn: Callable[[], Any]) -> tuple[Any, float, float]:
        wall_t0 = self.begin_interval()
        out = fn()
        gpu_ms, wall_ms = self.end_interval(wall_t0)
        return out, gpu_ms, wall_ms


def reset_runtime(device: str, use_fp16_gemm: bool = False) -> Runtime:
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device, use_fp16_gemm=bool(use_fp16_gemm))
    Runtime._default_runtime = rt
    return rt


def make_leaf(array: np.ndarray, device: str, *, requires_grad: bool = True) -> Tensor:
    value = np.asarray(array)
    return Tensor(
        shape=value.shape,
        dtype=value.dtype,
        device=device,
        requires_grad=requires_grad,
        init_val=value.copy(),
    )


def randn(rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.02) -> np.ndarray:
    return (rng.standard_normal(shape).astype(np.float32) * scale).copy()


def prepare_model(device: str, cfg: BenchConfig) -> ModelBundle:
    rt = reset_runtime(device, use_fp16_gemm=False)
    rt.grad_scale = 1.0 / cfg.grad_accum_steps
    rt.backend.grad_scale = 1.0 / cfg.grad_accum_steps
    rt.accumulate_parameter_grads = False
    rt.backend.accumulate_parameter_grads = False

    model = GPT2(cfg.V, cfg.C, cfg.T, cfg.H, cfg.B, cfg.L, np.float32, device)
    param_grad_dtype = {"fp32": np.float32, "bf16": "bf16"}[cfg.grad_dtype]
    for param in model.parameters():
        param.grad_dtype = param_grad_dtype

    rng = np.random.default_rng(SEED + 100)
    resid_scale = np.float32(1.0 / math.sqrt(2.0 * cfg.L))
    for name, tensor in iter_gpt2_named_tensors(model):
        if name.endswith(".gamma"):
            tensor.init_val = np.ones(tensor.shape, dtype=np.float32)
        elif name.endswith(".beta") or name.endswith(".bias"):
            tensor.init_val = np.zeros(tensor.shape, dtype=np.float32)
        elif name.endswith(".attn.proj.weight") or name.endswith(".mlp.fc2.weight"):
            tensor.init_val = (randn(rng, tensor.shape) * resid_scale).copy()
        else:
            tensor.init_val = randn(rng, tensor.shape)

    input_ids = rng.integers(0, cfg.V, size=(cfg.B, cfg.T), dtype=np.int32)
    labels = rng.integers(0, cfg.V, size=(cfg.B, cfg.T), dtype=np.int32)
    input_ids_t = make_leaf(input_ids, device, requires_grad=False)
    labels_t = make_leaf(labels, device, requires_grad=False)
    loss = model(input_ids_t, targets=labels_t)

    rt.loss = loss
    with suppress_output(True):
        rt.build_topo(loss)
        rt.bind_memory()
    rt.is_compiled = True

    with suppress_output(True):
        optimizer = AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.2,
            state_mode=cfg.optimizer_state,
        )
    clip_fn = optimizer.build_grad_clip() if cfg.max_norm > 0 else None
    return ModelBundle(
        cfg=cfg,
        rt=rt,
        model=model,
        loss=loss,
        optimizer=optimizer,
        clip_grad_norm=clip_fn,
    )


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
    actual = Counter(node.op.op_name for node in bundle.rt.topo)
    if actual != expected:
        raise AssertionError(
            f"GPT2.forward topology changed; expected={expected}, actual={actual}"
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
            if producer is None or producer.op.op_name != "FusedRMSNormFlashAttention":
                raise AssertionError(f"{label} is not fed by the real FA node")
        elif op_name == "FusedRMSNormMLP":
            layer = occurrence
            label = f"layer{layer}.mlp_from_fa"
            producer = node.inputs[0].op
            if producer is None or producer.op.op_name != "FusedLinearAdd":
                raise AssertionError(f"{label} is not fed by FusedLinearAdd")
        elif op_name == "Add":
            layer = occurrence
            label = f"layer{layer}.residual_add_from_mlp"
        elif op_name == "FusedLinearCrossEntropy":
            label = "final.cross_entropy"
        else:
            raise AssertionError(f"uncovered training op: {op_name}")

        result.append(NodeInfo(index, label, op_name, layer, node))
    return result


class PassCaptureContext:
    """Capture exact hardware GPU interval between a begin_pass / end_pass pair."""

    def __init__(self, real_ctx, timer: CudaEventTimer):
        self.real = real_ctx
        self.timer = timer
        self._wall_t0: int | None = None
        self.gpu_ms: float | None = None

    def __getattr__(self, name):
        return getattr(self.real, name)

    def begin_pass(self):
        self._wall_t0 = self.timer.begin_interval()

    def end_pass(self):
        if self._wall_t0 is None:
            raise RuntimeError("end_pass called without matching begin_pass")
        self.gpu_ms, _ = self.timer.end_interval(self._wall_t0)
        self._wall_t0 = None


def profile_real_pass_callback(
    backend,
    timer: CudaEventTimer,
    callback: Callable[[], Any],
) -> tuple[float, float]:
    real_ctx = backend.ctx
    proxy = PassCaptureContext(real_ctx, timer)
    begin = time.perf_counter_ns()
    backend.ctx = proxy
    try:
        callback()
    finally:
        backend.ctx = real_ctx
    end = time.perf_counter_ns()
    if proxy.gpu_ms is None:
        raise RuntimeError("callback did not execute a begin_pass/end_pass pass")
    return proxy.gpu_ms, (end - begin) / 1e6


def profile_exact_training_passes(
    bundle: ModelBundle,
    timer: CudaEventTimer,
    *,
    warmup: int,
    iters: int,
):
    """Profile the exact passes executed by train_gpt2.py."""
    rt = bundle.rt
    accum = bundle.cfg.grad_accum_steps

    def run_micro(accumulate: bool):
        rt.loss.backward(
            grad_scale=1.0 / accum,
            accumulate_parameter_grads=accumulate,
        )

    for _ in range(warmup):
        run_micro(False)
        if accum > 1:
            run_micro(True)
        clip_coef = 1.0
        if bundle.clip_grad_norm is not None:
            _, clip_coef = bundle.clip_grad_norm(bundle.cfg.max_norm)
        bundle.optimizer.step(grad_scale=clip_coef)

    micro0_gpu, micro0_wall = [], []
    micro_acc_gpu, micro_acc_wall = [], []
    clip_gpu, clip_wall = [], []
    opt_gpu, opt_wall = [], []

    for _ in range(iters):
        g0, w0 = profile_real_pass_callback(rt.backend, timer, lambda: run_micro(False))
        micro0_gpu.append(g0)
        micro0_wall.append(w0)

        if accum > 1:
            ga, wa = profile_real_pass_callback(rt.backend, timer, lambda: run_micro(True))
            micro_acc_gpu.append(ga)
            micro_acc_wall.append(wa)

        clip_coef = 1.0
        if bundle.clip_grad_norm is not None:
            def _run_clip():
                nonlocal clip_coef
                _, clip_coef = bundle.clip_grad_norm(bundle.cfg.max_norm)

            gc_ms, wc_ms = profile_real_pass_callback(rt.backend, timer, _run_clip)
            clip_gpu.append(gc_ms)
            clip_wall.append(wc_ms)
        else:
            clip_gpu.append(0.0)
            clip_wall.append(0.0)

        go, wo = profile_real_pass_callback(
            rt.backend, timer, lambda c=clip_coef: bundle.optimizer.step(grad_scale=c)
        )
        opt_gpu.append(go)
        opt_wall.append(wo)

    # Restore steady-state non-accumulating flag for per-node profiling.
    rt.accumulate_parameter_grads = False
    rt.backend.accumulate_parameter_grads = False
    rt.grad_scale = 1.0 / accum
    rt.backend.grad_scale = 1.0 / accum

    return {
        "micro0_gpu": stats(micro0_gpu),
        "micro0_wall": stats(micro0_wall),
        "micro_acc_gpu": stats(micro_acc_gpu) if micro_acc_gpu else None,
        "micro_acc_wall": stats(micro_acc_wall) if micro_acc_wall else None,
        "clip_gpu": stats(clip_gpu),
        "clip_wall": stats(clip_wall),
        "optimizer_gpu": stats(opt_gpu),
        "optimizer_wall": stats(opt_wall),
    }


def _profile_callback_once(timer: CudaEventTimer, callback: Callable[[], None]) -> tuple[float, float]:
    _, gpu_ms, wall_ms = timer.measure(callback)
    return gpu_ms, wall_ms


def profile_contextual_nodes(
    bundle: ModelBundle,
    timer: CudaEventTimer,
    *,
    warmup: int,
    iters: int,
):
    """Time every bound node callback in the exact topological order on unified_arena."""
    infos = describe_topology(bundle)
    samples = {
        info.index: {
            "forward_gpu": [],
            "forward_wall": [],
            "backward_gpu": [],
            "backward_wall": [],
        }
        for info in infos
    }

    scale = 1.0 / bundle.cfg.grad_accum_steps
    bundle.rt.grad_scale = scale
    bundle.rt.backend.grad_scale = scale
    bundle.rt.accumulate_parameter_grads = False
    bundle.rt.backend.accumulate_parameter_grads = False

    for iteration in range(warmup + iters):
        record = iteration >= warmup
        for info in infos:
            gpu, wall = _profile_callback_once(
                timer,
                lambda node=info.node: node.forward(node.inputs, node.outputs),
            )
            if record:
                samples[info.index]["forward_gpu"].append(gpu)
                samples[info.index]["forward_wall"].append(wall)

        bundle.rt.backend.begin_pass()
        bundle.rt.backend.fill(bundle.loss.grad, scale)
        bundle.rt.backend.end_pass()

        for info in reversed(infos):
            gpu, wall = _profile_callback_once(
                timer,
                lambda node=info.node: node.backward(node.inputs, node.outputs),
            )
            if record:
                samples[info.index]["backward_gpu"].append(gpu)
                samples[info.index]["backward_wall"].append(wall)

    return [
        {
            "info": info,
            "forward_gpu": stats(samples[info.index]["forward_gpu"]),
            "forward_wall": stats(samples[info.index]["forward_wall"]),
            "backward_gpu": stats(samples[info.index]["backward_gpu"]),
            "backward_wall": stats(samples[info.index]["backward_wall"]),
        }
        for info in infos
    ]


def _format_cuda_dispatch_label(kernel_name: str, u32_args: Iterable[int]) -> str:
    u32 = list(u32_args)
    if kernel_name in ("cuda_sgemm_nn_kernel", "cuda_sgemm_nt_kernel", "fused_linear_add_forward_kernel") and len(u32) >= 3:
        return f"{kernel_name}[M={u32[0]},N={u32[1]},K={u32[2]}]"
    if kernel_name == "cuda_sgemm_tn_kernel" and len(u32) >= 5:
        return f"{kernel_name}[M={u32[0]},N={u32[1]},K={u32[2]},bf16={u32[4]}]"
    if kernel_name == "fused_wgu_swiglu_forward_kernel" and len(u32) >= 3:
        return f"{kernel_name}[M={u32[0]},H={u32[1]},K={u32[2]}]"
    if kernel_name == "fused_rms_fa_qkv_kernel" and len(u32) >= 4:
        return f"{kernel_name}[B={u32[0]},T={u32[1]},C={u32[2]},H={u32[3]}]"
    if kernel_name == "flash_attn_fwd_kernel" and len(u32) >= 6:
        return f"{kernel_name}[B={u32[0]},H={u32[1]},T={u32[2]},C={u32[3]},lse={u32[5]}]"
    if kernel_name == "flash_attn_bwd_kernel" and len(u32) >= 4:
        return f"{kernel_name}[B={u32[0]},H={u32[1]},T={u32[2]},C={u32[3]}]"
    if kernel_name == "fused_linear_ce_logits_to_dlogits_kernel" and len(u32) >= 2:
        return f"{kernel_name}[rows={u32[0]},V={u32[1]}]"
    if kernel_name == "ce_tile_logits_to_dlogits_kernel" and len(u32) >= 2:
        return f"{kernel_name}[rows={u32[0]},cols={u32[1]}]"
    return kernel_name


class DispatchCaptureContext:
    """Intercept every CUDA kernel dispatch / memset_zero made inside a real callback."""

    def __init__(self, real_ctx, timer: CudaEventTimer):
        self.real = real_ctx
        self.timer = timer
        self.events: list[DispatchEvent] = []

    def __getattr__(self, name):
        return getattr(self.real, name)

    def begin_pass(self):
        return None

    def end_pass(self):
        self.real.synchronize()

    def memset_zero(self, buffer_id, offset, nbytes):
        label = f"memset_zero[bytes={int(nbytes)}]"
        _, gpu_ms, wall_ms = self.timer.measure(
            lambda: self.real.memset_zero(buffer_id, offset, nbytes)
        )
        self.events.append(
            DispatchEvent(ordinal=len(self.events), label=label, gpu_ms=gpu_ms, wall_ms=wall_ms)
        )

    def dispatch(
        self,
        name,
        buf_ids,
        offsets,
        u32_args,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        smem_bytes=0,
    ):
        label = _format_cuda_dispatch_label(str(name), u32_args)
        _, gpu_ms, wall_ms = self.timer.measure(
            lambda: self.real.dispatch(
                name, buf_ids, offsets, u32_args, grid_x, grid_y, grid_z, block_x, smem_bytes
            )
        )
        self.events.append(
            DispatchEvent(ordinal=len(self.events), label=label, gpu_ms=gpu_ms, wall_ms=wall_ms)
        )


def capture_callback_dispatches(
    backend,
    timer: CudaEventTimer,
    callback: Callable[[], Any],
) -> list[DispatchEvent]:
    real_ctx = backend.ctx
    proxy = DispatchCaptureContext(real_ctx, timer)
    backend.ctx = proxy
    try:
        callback()
    finally:
        backend.ctx = real_ctx
    return proxy.events


def profile_callback_dispatches(
    backend,
    timer: CudaEventTimer,
    callback: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
):
    for _ in range(warmup):
        capture_callback_dispatches(backend, timer, callback)

    grouped: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: {"gpu": [], "wall": []}
    )
    expected_sequence = None
    for _ in range(iters):
        events = capture_callback_dispatches(backend, timer, callback)
        sequence = [(event.ordinal, event.label) for event in events]
        if expected_sequence is None:
            expected_sequence = sequence
        elif sequence != expected_sequence:
            raise AssertionError(
                f"dispatch sequence changed between samples: expected={expected_sequence}, got={sequence}"
            )
        for event in events:
            key = (event.ordinal, event.label)
            grouped[key]["gpu"].append(event.gpu_ms)
            grouped[key]["wall"].append(event.wall_ms)

    return [
        {
            "ordinal": ordinal,
            "label": label,
            "gpu": stats(values["gpu"]),
            "wall": stats(values["wall"]),
        }
        for (ordinal, label), values in sorted(grouped.items())
    ]


def select_expensive_nodes(
    node_results,
    *,
    threshold_ms: float,
) -> list[tuple[NodeInfo, str, int]]:
    """Select representative (last-layer) node per operator type and count occurrences."""
    selected: dict[str, tuple[NodeInfo, float, float]] = {}
    counts: Counter[str] = Counter()
    for result in node_results:
        info = result["info"]
        counts[info.op_name] += 1
        fwd = result["forward_gpu"].median_ms
        bwd = result["backward_gpu"].median_ms
        previous = selected.get(info.op_name)
        if previous is None or (info.layer or -1) >= (previous[0].layer or -1):
            selected[info.op_name] = (info, fwd, bwd)

    phases = []
    for op_name in (
        "Gather",
        "FusedRMSNormFlashAttention",
        "FusedLinearAdd",
        "FusedRMSNormMLP",
        "Add",
        "FusedLinearCrossEntropy",
    ):
        if op_name not in selected:
            continue
        info, fwd, bwd = selected[op_name]
        if fwd >= threshold_ms:
            phases.append((info, "forward", counts[op_name]))
        if bwd >= threshold_ms:
            phases.append((info, "backward", counts[op_name]))
    return phases


def print_node_results(results, grad_accum_steps: int) -> None:
    print("\nPer-node cost from real bound callbacks (single micro-step)")
    print(
        f"{'training node':<40s} {'op':<30s} "
        f"{'forward GPU':>13s} {'backward GPU':>14s}"
    )
    grouped = defaultdict(list)
    for result in results:
        info = result["info"]
        fwd = result["forward_gpu"].median_ms
        bwd = result["backward_gpu"].median_ms
        grouped[info.op_name].append((fwd, bwd))
        print(
            f"{info.label:<40s} {info.op_name:<30s} "
            f"{fwd:10.3f} ms {bwd:11.3f} ms"
        )

    micro_total = sum(
        sum(f + b for f, b in vals) for vals in grouped.values()
    )
    print(
        f"\nGrouped contribution across real occurrences "
        f"(1 micro-step vs x{grad_accum_steps} full step)"
    )
    print(
        f"{'op':<30s} {'count':>5s} {'fwd sum':>12s} "
        f"{'bwd sum':>12s} {'1-micro sum':>13s} "
        f"{f'x{grad_accum_steps} step':>13s} {'share':>8s}"
    )
    for op_name, values in grouped.items():
        fwd = sum(item[0] for item in values)
        bwd = sum(item[1] for item in values)
        sub = fwd + bwd
        pct = (100.0 * sub / micro_total) if micro_total > 0 else 0.0
        print(
            f"{op_name:<30s} {len(values):5d} "
            f"{fwd:9.3f} ms {bwd:9.3f} ms {sub:10.3f} ms "
            f"{sub * grad_accum_steps:10.3f} ms {pct:6.2f}%"
        )
    print(
        f"{'TOTAL (graph nodes)':<30s} {len(results):5d} "
        f"{'':>12s} {'':>12s} {micro_total:10.3f} ms "
        f"{micro_total * grad_accum_steps:10.3f} ms 100.00%"
    )


def _print_dispatch_events(
    title: str,
    events: list[dict[str, Any]],
    *,
    multiplier: int,
    kernel_threshold_ms: float,
    verbose_dispatches: bool,
) -> dict[str, float]:
    total = sum(ev["gpu"].median_ms for ev in events)
    print(f"\n  {title} (calls={len(events)}, single-call GPU sum={total:.3f} ms, full-step x{multiplier}={total * multiplier:.3f} ms)")

    if verbose_dispatches or len(events) <= 32:
        shown = 0
        for event in events:
            gpu = event["gpu"].median_ms
            if gpu < kernel_threshold_ms:
                continue
            shown += 1
            print(
                f"    {event['ordinal'] + 1:03d} {event['label']:<64s} "
                f"GPU={gpu:9.3f} ms  wall={event['wall'].median_ms:9.3f} ms"
            )
        if shown == 0:
            print("    all individual dispatches were below --kernel-threshold-ms")
    else:
        print(
            f"    ({len(events)} dispatches in loop; showing grouped kernel breakdown below, "
            "pass --verbose-dispatches to list every ordinal)"
        )

    by_label: dict[str, list[float]] = defaultdict(list)
    for event in events:
        by_label[event["label"]].append(event["gpu"].median_ms)

    if len(events) > 1:
        print("    [Grouped by kernel inside this phase]")
        for label, vals in sorted(by_label.items(), key=lambda kv: sum(kv[1]), reverse=True):
            sub = sum(vals)
            if sub < kernel_threshold_ms:
                continue
            pct = (100.0 * sub / total) if total > 0 else 0.0
            print(
                f"      x{len(vals):<4d} {label:<60s} "
                f"sum={sub:9.3f} ms  step_x{multiplier}={sub * multiplier:10.3f} ms ({pct:5.1f}%)"
            )

    return {label: sum(vals) * multiplier for label, vals in by_label.items()}


def print_internal_profiles(
    bundle: ModelBundle,
    timer: CudaEventTimer,
    node_results,
    *,
    warmup: int,
    iters: int,
    operator_threshold_ms: float,
    kernel_threshold_ms: float,
    verbose_dispatches: bool,
) -> None:
    # Populate real activations and gradients once before isolated kernel replay.
    bundle.rt.loss.backward(
        grad_scale=1.0 / bundle.cfg.grad_accum_steps,
        accumulate_parameter_grads=False,
    )

    phases = select_expensive_nodes(node_results, threshold_ms=operator_threshold_ms)
    print("\nInternal CUDA kernel dispatches issued by backend/cuda.py")
    global_kernel_step_ms: dict[str, float] = defaultdict(float)

    for info, phase, op_count in phases:
        callback = (
            functools.partial(info.node.forward, info.node.inputs, info.node.outputs)
            if phase == "forward"
            else functools.partial(info.node.backward, info.node.inputs, info.node.outputs)
        )
        events = profile_callback_dispatches(
            bundle.rt.backend, timer, callback, warmup=warmup, iters=iters
        )
        step_multiplier = op_count * bundle.cfg.grad_accum_steps
        contribs = _print_dispatch_events(
            f"{info.label} / {phase} (op x{op_count} per micro-step)",
            events,
            multiplier=step_multiplier,
            kernel_threshold_ms=kernel_threshold_ms,
            verbose_dispatches=verbose_dispatches,
        )
        for klabel, ms in contribs.items():
            global_kernel_step_ms[klabel] += ms

    if bundle.clip_grad_norm is not None:
        clip_events = profile_callback_dispatches(
            bundle.rt.backend,
            timer,
            lambda: bundle.clip_grad_norm(bundle.cfg.max_norm),
            warmup=warmup,
            iters=iters,
        )
        contribs = _print_dispatch_events(
            "grad_clip / clip_grad_norm",
            clip_events,
            multiplier=1,
            kernel_threshold_ms=kernel_threshold_ms,
            verbose_dispatches=verbose_dispatches,
        )
        for klabel, ms in contribs.items():
            global_kernel_step_ms[klabel] += ms

    opt_events = profile_callback_dispatches(
        bundle.rt.backend,
        timer,
        lambda: bundle.optimizer.step(grad_scale=1.0),
        warmup=warmup,
        iters=iters,
    )
    contribs = _print_dispatch_events(
        f"optimizer / AdamW.step (state={bundle.cfg.optimizer_state})",
        opt_events,
        multiplier=1,
        kernel_threshold_ms=kernel_threshold_ms,
        verbose_dispatches=verbose_dispatches,
    )
    for klabel, ms in contribs.items():
        global_kernel_step_ms[klabel] += ms

    if global_kernel_step_ms:
        grand_total = sum(global_kernel_step_ms.values())
        print(
            f"\nTop CUDA kernels across full train_gpt2.py step "
            f"(projected total={grand_total:.2f} ms)"
        )
        print(f"{'rank':<6s} {'kernel[shape]':<62s} {'full-step GPU':>15s} {'share':>8s}")
        for rank, (klabel, ms) in enumerate(
            sorted(global_kernel_step_ms.items(), key=lambda kv: kv[1], reverse=True), 1
        ):
            if ms < kernel_threshold_ms:
                continue
            pct = (100.0 * ms / grand_total) if grand_total > 0 else 0.0
            print(f"{rank:<6d} {klabel:<62s} {ms:12.3f} ms {pct:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact train_gpt2.py CUDA operator and per-dispatch kernel timing"
    )
    # Defaults match train_gpt2.py (d12 = 135.3M params, B=8, T=1024, total_batch_size=32768).
    parser.add_argument("--depth", "--L", dest="depth", type=int, default=12)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-seq-len", "--T", dest="max_seq_len", type=int, default=1024)
    parser.add_argument("--device-batch-size", "--B", dest="device_batch_size", type=int, default=8)
    parser.add_argument("--total-batch-size", type=int, default=None)
    parser.add_argument("--vocab-size", "--V", dest="vocab_size", type=int, default=32768)
    parser.add_argument("--optimizer-state", choices=("fp32", "8bit", "mini"), default="mini")
    parser.add_argument("--grad-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--kernel-warmup", type=int, default=1)
    parser.add_argument("--kernel-iters", type=int, default=3)
    parser.add_argument("--operator-threshold-ms", type=float, default=0.5)
    parser.add_argument("--kernel-threshold-ms", type=float, default=0.0)
    parser.add_argument("--verbose-dispatches", action="store_true")
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

    tokens_per_micro = args.device_batch_size * args.max_seq_len
    if args.total_batch_size is None:
        total_batch_size = (
            32768 if (32768 >= tokens_per_micro and 32768 % tokens_per_micro == 0) else tokens_per_micro
        )
    else:
        if args.total_batch_size % tokens_per_micro != 0:
            parser.error(
                f"--total-batch-size={args.total_batch_size} must be divisible by B*T={tokens_per_micro}"
            )
        total_batch_size = args.total_batch_size
    grad_accum_steps = total_batch_size // tokens_per_micro

    base_dim = args.depth * args.aspect_ratio
    model_dim = (base_dim + args.head_dim - 1) // args.head_dim * args.head_dim
    num_heads = model_dim // args.head_dim
    cfg = BenchConfig(
        B=args.device_batch_size,
        T=args.max_seq_len,
        V=args.vocab_size,
        C=model_dim,
        H=num_heads,
        L=args.depth,
        grad_accum_steps=grad_accum_steps,
        grad_dtype=args.grad_dtype,
        optimizer_state=args.optimizer_state,
        max_norm=args.max_norm,
    )
    cfg.validate()

    bundle = prepare_model("cuda", cfg)
    timer = CudaEventTimer(bundle.rt.backend.ctx)
    try:
        print(f"repository: {ROOT_DIR}")
        print(f"timer mode: {timer.mode}")
        print(
            f"config: B={cfg.B} T={cfg.T} V={cfg.V} C={cfg.C} H={cfg.H} D={cfg.D} L={cfg.L} "
            f"total_batch={cfg.total_batch_size} (accum={cfg.grad_accum_steps}) "
            f"optimizer={cfg.optimizer_state} grad_dtype={cfg.grad_dtype}"
        )

        topology = describe_topology(bundle)
        print("\nTraining topology covered")
        for op_name, count in Counter(info.op_name for info in topology).items():
            print(f"  {op_name:<30s} x {count}")

        exact = profile_exact_training_passes(
            bundle, timer, warmup=args.warmup, iters=args.iters
        )
        m0_gpu = exact["micro0_gpu"].median_ms
        m0_wall = exact["micro0_wall"].median_ms
        ma_gpu = exact["micro_acc_gpu"].median_ms if exact["micro_acc_gpu"] else m0_gpu
        ma_wall = exact["micro_acc_wall"].median_ms if exact["micro_acc_wall"] else m0_wall
        clip_gpu = exact["clip_gpu"].median_ms
        clip_wall = exact["clip_wall"].median_ms
        opt_gpu = exact["optimizer_gpu"].median_ms
        opt_wall = exact["optimizer_wall"].median_ms

        full_step_gpu = m0_gpu + max(0, cfg.grad_accum_steps - 1) * ma_gpu + clip_gpu + opt_gpu
        full_step_wall = m0_wall + max(0, cfg.grad_accum_steps - 1) * ma_wall + clip_wall + opt_wall
        est_tok_s = int(cfg.total_batch_size / (full_step_wall / 1000.0)) if full_step_wall > 0 else 0

        print("\nExact train_gpt2.py CUDA passes")
        print(f"  loss.backward (micro_step=0)   GPU={m0_gpu:9.3f} ms  wall={m0_wall:9.3f} ms")
        if cfg.grad_accum_steps > 1:
            print(f"  loss.backward (micro_step>=1)  GPU={ma_gpu:9.3f} ms  wall={ma_wall:9.3f} ms")
        if bundle.clip_grad_norm is not None:
            print(f"  clip_grad_norm                 GPU={clip_gpu:9.3f} ms  wall={clip_wall:9.3f} ms")
        print(f"  optimizer.step ({cfg.optimizer_state:<4s})          GPU={opt_gpu:9.3f} ms  wall={opt_wall:9.3f} ms")
        print(
            f"  => Projected full step (x{cfg.grad_accum_steps})    "
            f"GPU={full_step_gpu:9.2f} ms  wall={full_step_wall:9.2f} ms  (~{est_tok_s:,} tok/sec)"
        )

        node_results = profile_contextual_nodes(
            bundle, timer, warmup=args.warmup, iters=args.iters
        )
        print_node_results(node_results, cfg.grad_accum_steps)

        if not args.skip_kernels:
            print_internal_profiles(
                bundle,
                timer,
                node_results,
                warmup=args.kernel_warmup,
                iters=args.kernel_iters,
                operator_threshold_ms=args.operator_threshold_ms,
                kernel_threshold_ms=args.kernel_threshold_ms,
                verbose_dispatches=args.verbose_dispatches,
            )
    finally:
        timer.close()


if __name__ == "__main__":
    main()