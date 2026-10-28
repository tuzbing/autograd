#!/usr/bin/env python3
"""
Profile kernels used by RL prefill/decode.

Example:

/Users/a58/Documents/code/nanochat/.venv/bin/python -B timing_rl_ops.py \
  --model-in ../checkpoints/base_d6/fp32_model_001000.gguf \
  --data-dir ../sft_data \
  --device-type mps \
  --device-batch-size 8 \
  --max-new-tokens 256 \
  --prompt-index 0 \
  --warmup 1 \
  --iters 10 \
  --kernel-threshold-ms 1
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import gc
import io
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()

    candidates = [
        here.parent,
        here.parent.parent,
        Path.cwd(),
    ]

    for candidate in candidates:
        if (
            (candidate / "train_autograd.py").is_file()
            and (candidate / "debug" / "backend.py").is_file()
            and (candidate / "debug" / "metal_ext.cpython-310-darwin.so").is_file()
        ):
            return candidate

    raise RuntimeError(
        "cannot locate autograd repository; expected train_autograd.py, "
        "debug/backend.py and debug/metal_ext.*"
    )


ROOT_DIR = _find_repo_root()
DEBUG_DIR = ROOT_DIR / "debug"

os.chdir(DEBUG_DIR)

if str(DEBUG_DIR) not in sys.path:
    sys.path.insert(0, str(DEBUG_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(1, str(ROOT_DIR))

from tensor import Tensor
from module import GPT2
from runtime import Runtime, ExecutionPlan
from engine import Engine
from gguf_checkpoint import (
    load_gguf,
    _read_exact,
    _read_string,
    _read_value,
)


NANOCHAT_DIR = os.path.expanduser("~/Documents/code/nanochat")
if NANOCHAT_DIR not in sys.path:
    sys.path.insert(0, NANOCHAT_DIR)

from nanochat.tokenizer import RustBPETokenizer
from tasks.gsm8k import GSM8K


@dataclass
class DispatchEvent:
    ordinal: int
    label: str
    gpu_ms: float
    wall_ms: float


@dataclass
class Stats:
    median_ms: float
    mean_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    samples_ms: list[float]


def stats(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return Stats(
            median_ms=math.nan,
            mean_ms=math.nan,
            p90_ms=math.nan,
            min_ms=math.nan,
            max_ms=math.nan,
            samples_ms=[],
        )

    return Stats(
        median_ms=float(np.median(values)),
        mean_ms=float(np.mean(values)),
        p90_ms=float(np.percentile(values, 90)),
        min_ms=float(np.min(values)),
        max_ms=float(np.max(values)),
        samples_ms=[float(x) for x in values],
    )


def _gpu_ms(raw: dict[str, Any]) -> float:
    value = float(raw.get("gpu_ms", -1.0))
    return value if value >= 0.0 else math.nan


class DispatchCaptureContext:
    """
    Same mechanism as debug/timing_ops_v2.py.

    Every intercepted backend dispatch is executed in one profiled
    Metal command buffer, so its GPU interval can be measured independently.
    """

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
                label = self._gemm_label(args)
            elif args and isinstance(args[0], str):
                label = str(args[0])
            else:
                label = name

            begin = time.perf_counter_ns()

            self.real.begin_pass_profiled()
            try:
                result = value(*args, **kwargs)
            finally:
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
            return result

        return wrapped

    @staticmethod
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


def capture_callback_dispatches(backend, callback):
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
    prepare_iteration,
    warmup: int,
    iters: int,
):
    """
    Profile one real bound callback repeatedly.

    prepare_iteration() resets the real RL state and feeds the real tensors
    before every invocation. It is outside the measured kernel interval.
    """

    for _ in range(warmup):
        prepare_iteration()
        capture_callback_dispatches(backend, callback)

    grouped = collections.defaultdict(
        lambda: {
            "gpu": [],
            "wall": [],
        }
    )

    expected_sequence = None

    for _ in range(iters):
        prepare_iteration()
        events = capture_callback_dispatches(backend, callback)

        sequence = [
            (event.ordinal, event.label)
            for event in events
        ]

        if expected_sequence is None:
            expected_sequence = sequence
        elif sequence != expected_sequence:
            raise RuntimeError(
                "dispatch sequence changed between samples:\n"
                f"expected={expected_sequence}\n"
                f"actual={sequence}"
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


def read_gguf_metadata(path: str):
    with open(path, "rb") as f:
        if _read_exact(f, 4) != b"GGUF":
            raise ValueError(f"not a GGUF checkpoint: {path}")

        version, _, metadata_count = struct.unpack(
            "<IQQ",
            _read_exact(f, 20),
        )

        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {version}")

        metadata = {}

        for _ in range(metadata_count):
            name = _read_string(f)
            kind = struct.unpack(
                "<I",
                _read_exact(f, 4),
            )[0]
            metadata[name] = _read_value(f, kind)

    if metadata.get("general.architecture") != "autograd-gpt2":
        raise ValueError(
            "the checkpoint must be produced by this repository"
        )

    return metadata


def build_train_runtime(
    *,
    model_path: str,
    device: str,
    batch_size: int,
    sequence_len: int,
):
    metadata = read_gguf_metadata(model_path)
    saved_config = json.loads(
        metadata["autograd.model_config"]
    )

    config = dict(saved_config)
    config["sequence_len"] = sequence_len

    runtime = Runtime(
        device=device,
        forward_only=False,
        owns_params=True,
    )
    Runtime._default_runtime = runtime

    model = GPT2(
        config["vocab_size"],
        config["n_embd"],
        config["sequence_len"],
        config["n_head"],
        batch_size,
        config["n_layer"],
        np.float32,
        device,
    )

    input_ids = Tensor(
        shape=(batch_size, sequence_len),
        dtype=np.int32,
        device=device,
        requires_grad=False,
        init_val=None,
        name="input_ids",
    )

    target_ids = Tensor(
        shape=(batch_size, sequence_len),
        dtype=np.int32,
        device=device,
        requires_grad=False,
        init_val=None,
        name="target_ids",
    )

    loss = model(
        input_ids,
        targets=target_ids,
    )

    runtime.loss = loss
    runtime.build_topo(loss)
    runtime.bind_memory()
    runtime.is_compiled = True

    load_gguf(
        model_path,
        model,
        runtime.backend,
        saved_config,
    )

    return runtime, model, config


def make_engine(
    *,
    model_path: str,
    data_dir: str,
    device: str,
    batch_size: int,
    max_new_tokens: int,
):
    rollout_seq_len = max_new_tokens + 300

    runtime, model, config = build_train_runtime(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        sequence_len=rollout_seq_len,
    )

    tokenizer = RustBPETokenizer.from_directory(
        str(Path(data_dir).expanduser() / "tokenizer")
    )

    engine = Engine(
        model,
        tokenizer,
        device=device,
        batch_size=batch_size,
        max_seq_len=rollout_seq_len,
        parameter_runtime=runtime,
        arena=runtime.unified_arena,
        max_top_k=64,
    )

    return engine, tokenizer, config


def node_label(node, index):
    op_name = getattr(
        node.op,
        "op_name",
        type(node.op).__name__,
    )

    return f"node[{index}]::{op_name}"


def make_prefill_callbacks(plan):
    callbacks = []

    for index, node in enumerate(plan.runtime.topo):
        callbacks.append(
            (
                node_label(node, index),
                functools.partial(
                    node.forward,
                    node.inputs,
                    node.outputs,
                ),
            )
        )

    return callbacks


def make_decode_callbacks(plan):
    callbacks = []

    for index, node in enumerate(plan.runtime.topo):
        callbacks.append(
            (
                node_label(node, index),
                functools.partial(
                    node.forward,
                    node.inputs,
                    node.outputs,
                ),
            )
        )

    return callbacks


def print_kernel_results(
    *,
    phase,
    plan,
    callback_name,
    results,
    kernel_threshold_ms,
):
    results = list(results)
    total = sum(item["gpu"].median_ms for item in results)
    if not results:
        return
    if math.isfinite(total) and total < kernel_threshold_ms:
        return
    print()
    print(
        f"[{phase}] "
        f"{callback_name} "
        f"B={plan.input_ids.shape[0]} "
        f"T={plan.seq_len}"
    )

    print(
        f"{'dispatch':>10} "
        f"{'kernel':<60} "
        f"{'GPU median':>14} "
        f"{'GPU p90':>14} "
        f"{'wall median':>14}"
    )

    total = 0.0

    for item in results:
        gpu = item["gpu"]
        wall = item["wall"]

        print(
            f"{item['ordinal']:10d} "
            f"{item['label']:<60.60} "
            f"{gpu.median_ms:11.6f} ms "
            f"{gpu.p90_ms:11.6f} ms "
            f"{wall.median_ms:11.6f} ms"
        )

    print(
        f"{'dispatch GPU median sum':<74} "
        f"{total:11.6f} ms"
    )


def run_prefill_profile(
    *,
    engine,
    prompt_tokens,
    warmup,
    iters,
    kernel_threshold_ms,
):
    prompt_ids = np.asarray(
        [prompt_tokens] * engine.batch_size,
        dtype=np.int32,
    )

    prompt_len = len(prompt_tokens)
    plan = engine._choose(prompt_len)

    padded_prompt = np.full(
        (1, plan.seq_len),
        engine.bos,
        dtype=np.int32,
    )
    padded_prompt[0, :prompt_len] = prompt_ids[0]

    feeds = {
        plan.input_ids: padded_prompt,
        engine.kv_cache.position: np.asarray(
            [0],
            dtype=np.int32,
        ),
        engine.rollout_state.position: np.asarray(
            [prompt_len - 1],
            dtype=np.int32,
        ),
    }

    def prepare_iteration():
        engine._reset_rollout_state(1234)
        engine._start_rollout_policy(
            prompt_tokens,
            engine.batch_size,
        )
        engine._decode_config[:] = (
            prompt_len,
            1,
            engine.decode_steps,
        )

        # engine.prefill_kv_cache.reset()
        engine.kv_cache.reset()

        for tensor, host_value in feeds.items():
            plan.runtime.backend.copy_from_host(
                tensor,
                host_value,
            )

        plan.runtime.backend.synchronize()

    callbacks = make_prefill_callbacks(plan)

    print()
    print(
        f"Prefill plan: bucket_T={plan.seq_len}, "
        f"prompt_T={prompt_len}, "
        f"B={engine.batch_size}, "
        f"nodes={len(callbacks)}"
    )

    for callback_name, callback in callbacks:
        results = profile_callback_dispatches(
            plan.runtime.backend,
            callback,
            prepare_iteration=prepare_iteration,
            warmup=warmup,
            iters=iters,
        )

        print_kernel_results(
            phase="prefill",
            plan=plan,
            callback_name=callback_name,
            results=results,
            kernel_threshold_ms=kernel_threshold_ms,
        )


def run_decode_profile(
    *,
    engine,
    prompt_tokens,
    warmup,
    iters,
    kernel_threshold_ms,
):
    prompt_ids = np.asarray(
        [prompt_tokens] * engine.batch_size,
        dtype=np.int32,
    )

    prompt_len = len(prompt_tokens)
    prefill_plan = engine._choose(prompt_len)
    decode_plan = engine.decode_plan

    padded_prompt = np.full(
        (1, prefill_plan.seq_len),
        engine.bos,
        dtype=np.int32,
    )
    padded_prompt[0, :prompt_len] = prompt_ids[0]

    prefill_feeds = {
        prefill_plan.input_ids: padded_prompt,
        engine.kv_cache.position: np.asarray(
            [0],
            dtype=np.int32,
        ),
        engine.rollout_state.position: np.asarray(
            [prompt_len - 1],
            dtype=np.int32,
        ),
    }

    def prepare_iteration():
        engine._reset_rollout_state(5678)
        engine._start_rollout_policy(
            prompt_tokens,
            engine.batch_size,
        )
        engine._decode_config[:] = (
            prompt_len,
            256,
            engine.decode_steps,
        )

        # engine.prefill_kv_cache.reset()
        engine.kv_cache.reset()

        for tensor, host_value in prefill_feeds.items():
            prefill_plan.runtime.backend.copy_from_host(
                tensor,
                host_value,
            )

        prefill_plan.runtime.backend.synchronize()

        # The prefill is real and uses the same model/KV cache.
        # It is setup for decode and is not included in decode timing.
        prefill_plan.runtime.execute()

        engine._broadcast_prefill_kv(
            prompt_len=prompt_len,
            num_samples=engine.batch_size,
        )

        engine._apply_rollout_policy(prefill_plan)

        decode_plan.runtime.backend.synchronize()

    callbacks = make_decode_callbacks(decode_plan)

    print()
    print(
        f"Decode plan: "
        f"B={decode_plan.input_ids.shape[0]}, "
        f"chunk={engine.decode_steps}, "
        f"nodes={len(callbacks)}"
    )

    for callback_name, callback in callbacks:
        results = profile_callback_dispatches(
            decode_plan.runtime.backend,
            callback,
            prepare_iteration=prepare_iteration,
            warmup=warmup,
            iters=iters,
        )

        print_kernel_results(
            phase="decode",
            plan=decode_plan,
            callback_name=callback_name,
            results=results,
            kernel_threshold_ms=kernel_threshold_ms,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-in",
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
    )
    parser.add_argument(
        "--device-type",
        choices=("mps", "cpu"),
        default="mps",
    )
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--kernel-threshold-ms",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    if args.iters <= 0:
        parser.error("--iters must be positive")

    if args.device_batch_size <= 0:
        parser.error("--device-batch-size must be positive")

    train_task = GSM8K(
        subset="main",
        split="train",
    )

    if not 0 <= args.prompt_index < len(train_task):
        parser.error(
            f"--prompt-index must be in [0, {len(train_task)})"
        )

    engine, tokenizer, config = make_engine(
        model_path=args.model_in,
        data_dir=args.data_dir,
        device=args.device_type,
        batch_size=args.device_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    conversation = train_task[args.prompt_index]
    prompt_tokens = tokenizer.render_for_completion(
        conversation
    )

    print(
        f"checkpoint: {args.model_in}\n"
        f"device: {args.device_type}\n"
        f"prompt_index: {args.prompt_index}\n"
        f"prompt_length: {len(prompt_tokens)}\n"
        f"batch_size: {args.device_batch_size}\n"
        f"warmup: {args.warmup}\n"
        f"iters: {args.iters}"
    )

    run_prefill_profile(
        engine=engine,
        prompt_tokens=prompt_tokens,
        warmup=args.warmup,
        iters=args.iters,
        kernel_threshold_ms=args.kernel_threshold_ms,
    )

    run_decode_profile(
        engine=engine,
        prompt_tokens=prompt_tokens,
        warmup=args.warmup,
        iters=args.iters,
        kernel_threshold_ms=args.kernel_threshold_ms,
    )


if __name__ == "__main__":
    main()