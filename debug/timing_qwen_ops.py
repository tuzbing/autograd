#!/usr/bin/env python3
"""Profile the exact QwenCoder SFT training callbacks used by sft_qwen_coder.py.

/Users/a58/Documents/code/nanochat/.venv/bin/python -B debug/timing_qwen_ops.py \
  --B=4 --T=256 --iters=5 --operator-threshold-ms 0 --kernel-threshold-ms 0
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

sys.dont_write_bytecode = True

import numpy as np


SEED = 1234


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    env_root = os.environ.get("AUTOGRAD_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.extend([here.parent, here.parent.parent, Path.cwd()])
    for candidate in candidates:
        if (
            (candidate / "sft_qwen_coder.py").is_file()
            and (candidate / "debug" / "backend.py").is_file()
        ):
            return candidate
    raise RuntimeError(
        "cannot locate autograd repository; run from its root or copy this "
        "file to debug/timing_qwen_ops.py"
    )


ROOT_DIR = _find_repo_root()
DEBUG_DIR = ROOT_DIR / "debug"
os.chdir(DEBUG_DIR)
sys.path[:] = [
    p for p in sys.path
    if Path(p).resolve() not in (ROOT_DIR.resolve(), DEBUG_DIR.resolve())
]
sys.path.insert(0, str(DEBUG_DIR))
sys.path.insert(1, str(ROOT_DIR))

# sft_qwen_coder.py resolves [invalid URL] from repository root.
from module import QwenCoder
from optimizer import AdamW
from runtime import Runtime
from tensor import Tensor


def iter_qwen_named_tensors(model: QwenCoder):
    """Yield the unique trainable parameters of module.QwenCoder in model order."""
    if model.lm_head.weight is not model.tok_emb.weight:
        raise AssertionError(
            "QwenCoder must share tok_emb.weight and lm_head.weight"
        )

    yield "tok_emb.weight", model.tok_emb.weight

    for index, block in enumerate(model.blocks):
        prefix = f"blocks[{index}]"

        yield f"{prefix}.input_norm_weight", block.input_norm_weight
        yield f"{prefix}.q_proj.weight", block.q_proj.weight
        yield f"{prefix}.q_proj.bias", block.q_proj.bias
        yield f"{prefix}.k_proj.weight", block.k_proj.weight
        yield f"{prefix}.k_proj.bias", block.k_proj.bias
        yield f"{prefix}.v_proj.weight", block.v_proj.weight
        yield f"{prefix}.v_proj.bias", block.v_proj.bias
        yield f"{prefix}.proj.weight", block.proj.weight

        yield f"{prefix}.post_norm_weight", block.post_norm_weight
        yield f"{prefix}.mlp.fc1.weight", block.mlp.fc1.weight
        yield f"{prefix}.mlp.fc2.weight", block.mlp.fc2.weight

    yield "final_norm_weight", getattr(model, "final_norm_weight")


@dataclass(frozen=True)
class BenchConfig:
    B: int
    T: int
    V: int
    C: int
    H: int
    n_head: int
    n_kv_head: int
    L: int
    eps: float = 1e-6
    rope_theta: float = 1000000.0
    grad_dtype: str = "bf16"
    state_mode: str = "8bit"
    max_norm: float = 2.5

    @property
    def D(self) -> int:
        return self.C // self.n_head

    def validate(self) -> None:
        if min(
            self.B,
            self.T,
            self.V,
            self.C,
            self.H,
            self.n_head,
            self.n_kv_head,
            self.L,
        ) <= 0:
            raise ValueError(f"all dimensions must be positive: {self}")
        if self.C % self.n_head or self.C % 4:
            raise ValueError(
                f"C={self.C} must be divisible by n_head={self.n_head} and 4"
            )
        if self.n_head % self.n_kv_head:
            raise ValueError(
                f"n_head={self.n_head} must be divisible by "
                f"n_kv_head={self.n_kv_head}"
            )
        if self.D % 2 or self.D > 128:
            raise ValueError(
                f"GQA Metal kernels require even head_dim <= 128, got D={self.D}"
            )


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
    model: QwenCoder
    loss: Tensor
    optimizer: AdamW
    clip_grad_norm: Callable[[float], tuple[float, float]]


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


def reset_runtime(device: str, *, use_fp16_gemm: bool = True) -> Runtime:
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device)
    rt.use_fp16_gemm = bool(use_fp16_gemm)
    rt.backend.use_fp16_gemm = bool(use_fp16_gemm)
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


def prepare_model(
    device: str,
    cfg: BenchConfig,
    use_fp16_gemm: bool = True,
) -> ModelBundle:
    rt = reset_runtime(device, use_fp16_gemm=use_fp16_gemm)
    model = QwenCoder(
        V=cfg.V,
        C=cfg.C,
        max_T=cfg.T,
        n_head=cfg.n_head,
        n_layer=cfg.L,
        dtype=np.float32,
        device=device,
        H=cfg.H,
        n_kv_head=cfg.n_kv_head,
        eps=cfg.eps,
        rope_theta=cfg.rope_theta,
    )
    parameter_grad_dtype = {"fp32": np.float32, "bf16": "bf16"}[cfg.grad_dtype]
    for parameter in model.parameters():
        parameter.grad_dtype = parameter_grad_dtype

    for name, tensor in iter_qwen_named_tensors(model):
        if name.endswith("norm_weight"):
            tensor.init_val = 1.0
        elif name.endswith(".bias"):
            tensor.init_val = 0.0
        else:
            tensor.init_val = None

    rng = np.random.default_rng(SEED + 100)
    input_ids = rng.integers(0, cfg.V, size=(cfg.B, cfg.T), dtype=np.int32)
    labels = rng.integers(0, cfg.V, size=(cfg.B, cfg.T), dtype=np.int32)
    input_ids_t = make_leaf(input_ids, device, requires_grad=False)
    labels_t = make_leaf(labels, device, requires_grad=False)

    loss = model(input_ids_t, targets=labels_t)
    rt.loss = loss
    rt.build_topo(loss)
    rt.backend.synchronize()

    for name, tensor in iter_qwen_named_tensors(model):
        if not (name.endswith("norm_weight") or name.endswith(".bias")):
            view = rt.backend.numpy_view(tensor)
            rng.standard_normal(dtype=np.float32, out=view)
            view *= np.float32(0.02)

    with suppress_output(True):
        rt.bind_memory()
    rt.is_compiled = True

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        state_mode=cfg.state_mode,
    )
    clip_grad_norm = getattr(optimizer, "build_grad_clip")()
    return ModelBundle(
        cfg=cfg,
        rt=rt,
        model=model,
        loss=loss,
        optimizer=optimizer,
        clip_grad_norm=clip_grad_norm,
    )


def describe_topology(bundle: ModelBundle) -> list[NodeInfo]:
    expected = Counter(
        {
            "Gather": 1,
            "FusedRMSNormGQA": bundle.cfg.L,
            "FusedLinearAdd": bundle.cfg.L,
            "FusedRMSNormMLP": bundle.cfg.L,
            "Add": bundle.cfg.L,
            "FusedRMSNormLinearCrossEntropy": 1,
        }
    )

    actual = Counter(node.op.op_name for node in bundle.rt.topo)

    if actual != expected:
        raise AssertionError(
            "QwenCoder.forward topology changed; "
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

        elif op_name == "FusedRMSNormGQA":
            layer = occurrence
            label = f"layer{layer}.attention"

        elif op_name == "FusedLinearAdd":
            layer = occurrence
            label = f"layer{layer}.linear_add_from_gqa"

            producer = node.inputs[1].op
            if (
                producer is None
                or producer.op.op_name != "FusedRMSNormGQA"
            ):
                raise AssertionError(
                    f"{label} is not fed by the real GQA node"
                )

        elif op_name == "FusedRMSNormMLP":
            layer = occurrence
            label = f"layer{layer}.mlp_from_gqa"

            producer = node.inputs[0].op
            if (
                producer is None
                or producer.op.op_name != "FusedLinearAdd"
            ):
                raise AssertionError(
                    f"{label} is not fed by FusedLinearAdd"
                )

        elif op_name == "Add":
            layer = occurrence
            label = f"layer{layer}.residual_add_from_mlp"

        elif op_name == "FusedRMSNormLinearCrossEntropy":
            label = "final.rmsnorm_cross_entropy"

        else:
            raise AssertionError(f"uncovered training op: {op_name}")

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
    """Profile the exact passes used by sft_qwen_coder.py after compilation."""
    for _ in range(warmup):
        bundle.rt.execute()
        _, clip_coef = bundle.clip_grad_norm(bundle.cfg.max_norm)
        bundle.optimizer.step(grad_scale=clip_coef)

    graph_gpu = []
    graph_wall = []
    clip_gpu = []
    clip_wall = []
    optimizer_gpu = []
    optimizer_wall = []

    for _ in range(iters):
        raw, wall = profile_real_pass_callback(
            bundle.rt.backend, bundle.rt.execute
        )
        graph_gpu.append(_gpu_ms(raw))
        graph_wall.append(wall)

        clip_coef_box = [1.0]

        def _run_clip():
            _, clip_coef_box[0] = bundle.clip_grad_norm(bundle.cfg.max_norm)

        raw, wall = profile_real_pass_callback(bundle.rt.backend, _run_clip)
        clip_gpu.append(_gpu_ms(raw))
        clip_wall.append(wall)

        raw, wall = profile_real_pass_callback(
            bundle.rt.backend,
            lambda coef=clip_coef_box[0]: bundle.optimizer.step(grad_scale=coef),
        )
        optimizer_gpu.append(_gpu_ms(raw))
        optimizer_wall.append(wall)

    return {
        "graph_gpu": stats(graph_gpu),
        "graph_wall": stats(graph_wall),
        "clip_gpu": stats(clip_gpu),
        "clip_wall": stats(clip_wall),
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
        info.index: {
            "forward_gpu": [],
            "forward_wall": [],
            "backward_gpu": [],
            "backward_wall": [],
        }
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
        bundle.rt.backend.fill(bundle.loss.grad, bundle.rt.grad_scale)
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
        "fill_scalar2",
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
    prepare: Callable[[], None] | None = None,
):
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        capture_callback_dispatches(backend, callback)

    grouped: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
        lambda: {"gpu": [], "wall": []}
    )
    expected_sequence = None
    for _ in range(iters):
        if prepare is not None:
            prepare()
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
        prev_layer = (
            previous[0].layer
            if (previous and previous[0].layer is not None)
            else -1
        )
        curr_layer = info.layer if info.layer is not None else -1
        if previous is None or curr_layer >= prev_layer:
            selected[info.op_name] = (info, fwd, bwd)

    phases = []
    for op_name in (
        "Gather",
        "FusedRMSNormGQA",
        "FusedLinearAdd",
        "FusedRMSNormMLP",
        "Add",
        "FusedRMSNormLinearCrossEntropy",
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

    backend = bundle.rt.backend
    for info, phase in phases:
        callback = (
            functools.partial(info.node.forward, info.node.inputs, info.node.outputs)
            if phase == "forward"
            else functools.partial(info.node.backward, info.node.inputs, info.node.outputs)
        )
        prepare = None
        if info.op_name == "FusedLinearAdd" and phase == "backward":
            # In Runtime.execute(), FusedRMSNormMLP.backward precedes
            # FusedLinearAdd.backward and caches the recomputed GQA output in
            # [invalid URL] via backend._recomputed_attention_cache.
            act = info.node.inputs[1]

            def prepare(backend=backend, act_id=id(act)):
                cache = getattr(backend, "_recomputed_attention_cache", None)
                if cache is None:
                    cache = set()
                    backend._recomputed_attention_cache = cache
                cache.add(act_id)

        events = profile_callback_dispatches(
            backend,
            callback,
            warmup=warmup,
            iters=iters,
            prepare=prepare,
        )
        cache = getattr(backend, "_recomputed_attention_cache", None)
        if cache is not None:
            cache.clear()

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exact sft_qwen_coder.py MPS callback timing with automatic "
            "per-dispatch kernel profiling"
        )
    )
    # Defaults match Qwen2.5-Coder-1.5B and sft_qwen_coder.py training settings.
    parser.add_argument("--depth", "--L", dest="depth", type=int, default=28)
    parser.add_argument("--n-embd", "--C", dest="n_embd", type=int, default=1536)
    parser.add_argument(
        "--intermediate-size", "--H", dest="intermediate_size", type=int, default=8960
    )
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument(
        "--max-seq-len", "--T", dest="max_seq_len", type=int, default=512
    )
    parser.add_argument(
        "--device-batch-size", "--B", dest="device_batch_size", type=int, default=4
    )
    parser.add_argument(
        "--vocab-size", "--V", dest="vocab_size", type=int, default=151936
    )
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--rope-theta", type=float, default=1000000.0)
    parser.add_argument(
        "--grad-dtype", choices=["fp32", "bf16"], default="bf16"
    )
    parser.add_argument(
        "--state-mode", choices=["fp32", "8bit", "mini"], default="8bit"
    )
    parser.add_argument("--max-norm", type=float, default=2.5)
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
    parser.add_argument(
        "--use-fp16-gemm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the fused FP16 GEMM path used by root Runtime (default: True)",
    )
    parser.add_argument("--skip-kernels", action="store_true")
    args = parser.parse_args()

    if min(args.warmup, args.iters, args.kernel_warmup, args.kernel_iters) < 0:
        parser.error("iteration counts must be non-negative")
    if args.iters == 0 or args.kernel_iters == 0:
        parser.error("--iters and --kernel-iters must be positive")

    cfg = BenchConfig(
        B=args.device_batch_size,
        T=args.max_seq_len,
        V=args.vocab_size,
        C=args.n_embd,
        H=args.intermediate_size,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        L=args.depth,
        eps=args.rms_norm_eps,
        rope_theta=args.rope_theta,
        grad_dtype=args.grad_dtype,
        state_mode=args.state_mode,
        max_norm=args.max_norm,
    )
    try:
        cfg.validate()
    except ValueError as exc:
        parser.error(str(exc))

    print(f"repository: {ROOT_DIR}")
    print("runtime path: debug/runtime.py -> debug/backend.py")
    print(
        f"performance config: B={cfg.B} T={cfg.T} V={cfg.V} "
        f"C={cfg.C} H={cfg.H} n_head={cfg.n_head} n_kv_head={cfg.n_kv_head} "
        f"D={cfg.D} L={cfg.L} grad_dtype={cfg.grad_dtype} state_mode={cfg.state_mode}"
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
    print("\nExact sft_qwen_coder.py Metal passes")
    print(
        f"  loss.backward graph pass  GPU={exact['graph_gpu'].median_ms:9.3f} ms  "
        f"wall={exact['graph_wall'].median_ms:9.3f} ms"
    )
    print(
        f"  clip_grad_norm pass       GPU={exact['clip_gpu'].median_ms:9.3f} ms  "
        f"wall={exact['clip_wall'].median_ms:9.3f} ms"
    )
    print(
        f"  optimizer.step pass       GPU={exact['optimizer_gpu'].median_ms:9.3f} ms  "
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


if __name__ == "__main__":
    main()