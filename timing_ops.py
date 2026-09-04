#!/usr/bin/env python3
"""
Timing-only benchmark for autograd ops and Tiny GPT2.

This script is intentionally separate from verify_model.py:
- no finite-diff
- no correctness assertions
- large shapes to expose real CPU vs MPS throughput

Usage:
  python debug/bench_timing_compare.py
  python debug/bench_timing_compare.py --warmup 2 --iters 5
  python debug/bench_timing_compare.py --preset heavy
"""

import argparse
import contextlib
import gc
import io
import os
import sys
import time
from math import prod

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor import Tensor
from runtime import Runtime
from op import (
    AddOp,
    MulOp,
    GatherOp,
    FusedLinearAddOp,
    FusedLayerNormMLPOp,
    FusedLayerNormFlashAttentionOp,
    FusedLinearCrossEntropyOp,
)
from module import GPT2
from verify_model import iter_gpt2_named_tensors


SEED = 1234


def reset_runtime(device):
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device)
    Runtime._default_runtime = rt
    return rt


def make_tensor(shape, dtype, device, requires_grad, init_val):
    return Tensor(
        shape=shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        init_val=np.asarray(init_val, dtype=dtype).copy(),
    )


def alloc_tensor(rt, arr, device, requires_grad=True):
    t = make_tensor(arr.shape, np.float32, device, requires_grad, arr)
    t.storage = rt.allocator.alloc(arr.shape, np.float32)
    rt.backend.init(t)
    if requires_grad:
        t.grad = Tensor(arr.shape, np.float32, device=device, requires_grad=False)
        t.grad.storage = rt.allocator.alloc(arr.shape, np.float32)
        rt.backend.fill_scalar(t.grad, 0.0)
    return t


def timed_repeat(rt, loss, warmup=2, iters=5):
    # Warmup to amortize kernel compile / cache setup.
    for _ in range(warmup):
        rt.zero_grad()
        rt.backend.fill(loss.grad, 1.0)
        rt.forward()
        rt.backend.synchronize()
        rt.backward_graph()
        rt.backend.synchronize()

    fwd_ms_list = []
    bwd_ms_list = []
    for _ in range(iters):
        rt.zero_grad()
        rt.backend.fill(loss.grad, 1.0)
        t0 = time.perf_counter()
        rt.forward()
        rt.backend.synchronize()
        t1 = time.perf_counter()
        rt.backward_graph()
        rt.backend.synchronize()
        t2 = time.perf_counter()
        fwd_ms_list.append((t1 - t0) * 1000.0)
        bwd_ms_list.append((t2 - t1) * 1000.0)
    return float(np.mean(fwd_ms_list)), float(np.mean(bwd_ms_list))


def summarize(name, config, cpu, mps, units):
    cpu_total = cpu[0] + cpu[1]
    mps_total = mps[0] + mps[1]
    fwd_speedup = cpu[0] / mps[0] if mps[0] > 0 else float("inf")
    bwd_speedup = cpu[1] / mps[1] if mps[1] > 0 else float("inf")
    total_speedup = cpu_total / mps_total if mps_total > 0 else float("inf")
    total_delta_ms = cpu_total - mps_total
    total_delta_pct = (total_delta_ms / cpu_total * 100.0) if cpu_total > 0 else 0.0
    rate_cpu = units / (cpu_total / 1000.0) if cpu_total > 0 else float("inf")
    rate_mps = units / (mps_total / 1000.0) if mps_total > 0 else float("inf")
    print(f"{name:<28s} {config:<34s} "
          f"cpu={cpu[0]:8.3f}/{cpu[1]:8.3f}ms  "
          f"mps={mps[0]:8.3f}/{mps[1]:8.3f}ms  "
          f"speedup={total_speedup:6.2f}x  "
          f"delta={total_delta_ms:8.3f}ms({total_delta_pct:5.1f}%)  "
          f"rate={rate_cpu:10.2f}->{rate_mps:10.2f} units/s")
    return {
        "cpu_fwd_ms": cpu[0],
        "cpu_bwd_ms": cpu[1],
        "mps_fwd_ms": mps[0],
        "mps_bwd_ms": mps[1],
        "cpu_rate": rate_cpu,
        "mps_rate": rate_mps,
        "speedup": total_speedup,
        "delta_ms": total_delta_ms,
        "delta_pct": total_delta_pct,
        "fwd_speedup": fwd_speedup,
        "bwd_speedup": bwd_speedup,
    }


def run_on_both_devices(build_case, warmup, iters):
    results = {}
    for device in ("cpu", "mps"):
        rt = reset_runtime(device)
        loss, config, units = build_case(rt, device)
        fwd_ms, bwd_ms = timed_repeat(rt, loss, warmup=warmup, iters=iters)
        results[device] = (fwd_ms, bwd_ms)
    return results["cpu"], results["mps"], config, units


def build_add_case(rt, device, warmup, iters):
    rng = np.random.default_rng(SEED)
    B, T, C = 32, 512, 1024
    x = alloc_tensor(rt, (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1), device, True)
    y = alloc_tensor(rt, (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1), device, True)
    out = AddOp()(x, y)
    rt.loss = out
    rt.build_topo(out)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt.bind_memory()
    return out, f"B={B} T={T} C={C}", int(np.prod(out.shape))


def build_mul_case(rt, device, warmup, iters):
    rng = np.random.default_rng(SEED + 1)
    B, T, C = 32, 512, 1024
    x = alloc_tensor(rt, (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1), device, True)
    y = alloc_tensor(rt, (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1), device, True)
    out = MulOp()(x, y)
    rt.loss = out
    rt.build_topo(out)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt.bind_memory()
    return out, f"B={B} T={T} C={C}", int(np.prod(out.shape))


def build_gather_case(rt, device, warmup, iters):
    rng = np.random.default_rng(SEED + 2)
    B, T, V, C = 32, 512, 32768, 256
    idx = make_tensor((B, T), np.int32, device, False, rng.integers(0, V, size=(B, T), dtype=np.int32))
    w = alloc_tensor(rt, (rng.standard_normal((V, C)).astype(np.float32) * 0.1), device, True)
    out = GatherOp()(idx, w)
    rt.loss = out
    rt.build_topo(out)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt.bind_memory()
    return out, f"B={B} T={T} V={V} C={C}", int(np.prod(out.shape))


def build_fused_linear_add_case(rt, device, warmup, iters):
    rng = np.random.default_rng(SEED + 3)
    B, T, C = 8, 1024, 1024
    x = alloc_tensor(rt, (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1), device, True)
    act = alloc_tensor(rt, (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1), device, True)
    w = alloc_tensor(rt, (rng.standard_normal((C, C)).astype(np.float32) * 0.1), device, True)
    b = alloc_tensor(rt, (rng.standard_normal((C,)).astype(np.float32) * 0.1), device, True)
    out = FusedLinearAddOp()(x, act, w, b)
    rt.loss = out
    rt.build_topo(out)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt.bind_memory()
    return out, f"B={B} T={T} C={C}", int(np.prod(out.shape))


def build_fused_ln_mlp_case(rt, device, warmup, iters):
    rng = np.random.default_rng(SEED + 4)
    B, T, C = 8, 512, 1024
    C4 = 4 * C
    x = alloc_tensor(rt, (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1), device, True)
    ln_w = alloc_tensor(rt, (rng.standard_normal((C,)).astype(np.float32) * 0.1), device, True)
    ln_b = alloc_tensor(rt, (rng.standard_normal((C,)).astype(np.float32) * 0.1), device, True)
    w1 = alloc_tensor(rt, (rng.standard_normal((C, C4)).astype(np.float32) * 0.1), device, True)
    b1 = alloc_tensor(rt, (rng.standard_normal((C4,)).astype(np.float32) * 0.1), device, True)
    w2 = alloc_tensor(rt, (rng.standard_normal((C4, C)).astype(np.float32) * 0.1), device, True)
    b2 = alloc_tensor(rt, (rng.standard_normal((C,)).astype(np.float32) * 0.1), device, True)
    out = FusedLayerNormMLPOp(eps=1e-5)(x, ln_w, ln_b, w1, b1, w2, b2)
    rt.loss = out
    rt.build_topo(out)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt.bind_memory()
    return out, f"B={B} T={T} C={C}", int(np.prod(out.shape))


def build_fused_ln_fa_case(rt, device, warmup, iters):
    rng = np.random.default_rng(SEED + 5)
    B, T, C, H = 2, 1024, 512, 8
    x = alloc_tensor(rt, (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1), device, True)
    ln_w = alloc_tensor(rt, (rng.standard_normal((C,)).astype(np.float32) * 0.1), device, True)
    ln_b = alloc_tensor(rt, (rng.standard_normal((C,)).astype(np.float32) * 0.1), device, True)
    wq = alloc_tensor(rt, (rng.standard_normal((C, C)).astype(np.float32) * 0.1), device, True)
    bq = alloc_tensor(rt, (rng.standard_normal((C,)).astype(np.float32) * 0.1), device, True)
    wk = alloc_tensor(rt, (rng.standard_normal((C, C)).astype(np.float32) * 0.1), device, True)
    bk = alloc_tensor(rt, (rng.standard_normal((C,)).astype(np.float32) * 0.1), device, True)
    wv = alloc_tensor(rt, (rng.standard_normal((C, C)).astype(np.float32) * 0.1), device, True)
    bv = alloc_tensor(rt, (rng.standard_normal((C,)).astype(np.float32) * 0.1), device, True)
    out = FusedLayerNormFlashAttentionOp(n_head=H, eps=1e-5, is_causal=True)(
        x, ln_w, ln_b, wq, bq, wk, bk, wv, bv
    )
    rt.loss = out
    rt.build_topo(out)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt.bind_memory()
    return out, f"B={B} T={T} C={C} H={H}", int(np.prod(out.shape))


def build_fused_linear_ce_case(rt, device, warmup, iters):
    rng = np.random.default_rng(SEED + 6)
    B, T, V, C = 8, 1024, 32768, 512
    x = alloc_tensor(rt, (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1), device, True)
    w = alloc_tensor(rt, (rng.standard_normal((V, C)).astype(np.float32) * 0.1), device, True)
    targets = make_tensor((B, T), np.int32, device, False, rng.integers(0, V, size=(B, T), dtype=np.int32))
    out = FusedLinearCrossEntropyOp()(x, w, targets)
    rt.loss = out
    rt.build_topo(out)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt.bind_memory()
    return out, f"B={B} T={T} V={V} C={C}", int(np.prod(out.shape))


def build_tiny_gpt2_case(rt, device, warmup, iters):
    rng = np.random.default_rng(SEED + 99)
    cfg = dict(B=4, T=512, V=50304, C=512, n_head=8, n_layer=6, dtype=np.float32)
    ref_model = GPT2(cfg["V"], cfg["C"], cfg["T"], cfg["n_head"], cfg["B"], cfg["n_layer"], cfg["dtype"], "cpu")
    param_arrays = {}
    for name, tensor in iter_gpt2_named_tensors(ref_model):
        param_arrays[name] = (rng.standard_normal(tensor.shape).astype(np.float32) * 0.02).copy()

    model = GPT2(cfg["V"], cfg["C"], cfg["T"], cfg["n_head"], cfg["B"], cfg["n_layer"], cfg["dtype"], device)
    for name, tensor in iter_gpt2_named_tensors(model):
        tensor.init_val = param_arrays[name].copy()

    input_ids = rng.integers(0, cfg["V"], size=(cfg["B"], cfg["T"]), dtype=np.int32)
    pos_ids = np.tile(np.arange(cfg["T"], dtype=np.int32), (cfg["B"], 1))
    labels = rng.integers(0, cfg["V"], size=(cfg["B"], cfg["T"]), dtype=np.int32)
    input_ids_t = make_tensor((cfg["B"], cfg["T"]), np.int32, device, False, input_ids)
    pos_ids_t = make_tensor((cfg["B"], cfg["T"]), np.int32, device, False, pos_ids)
    labels_t = make_tensor((cfg["B"], cfg["T"]), np.int32, device, False, labels)
    loss = model(input_ids_t, pos_ids_t, targets=labels_t)
    rt.loss = loss
    rt.build_topo(loss)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt.bind_memory()
    return loss, f"B={cfg['B']} T={cfg['T']} V={cfg['V']} C={cfg['C']} n_layer={cfg['n_layer']}", int(np.prod(labels.shape))


def main():
    parser = argparse.ArgumentParser(description="Timing-only CPU vs MPS benchmark suite")
    parser.add_argument("--warmup", type=int, default=2, help="warmup iterations before timing")
    parser.add_argument("--iters", type=int, default=5, help="timed iterations to average")
    args = parser.parse_args()

    benchmarks = [
        ("Add", build_add_case),
        ("Mul", build_mul_case),
        ("Gather", build_gather_case),
        ("FusedLinearAdd", build_fused_linear_add_case),
        ("FusedLayerNormMLP", build_fused_ln_mlp_case),
        ("FusedLayerNormFlashAttention", build_fused_ln_fa_case),
        ("FusedLinearCrossEntropy", build_fused_linear_ce_case),
        ("Tiny GPT2", build_tiny_gpt2_case),
    ]

    print("CPU vs MPS timing benchmark")
    print(f"warmup={args.warmup} iters={args.iters}")
    print("-" * 120)
    print(f"{'Case':<28s} {'Config':<34s} {'CPU':<25s} {'MPS':<25s} {'Speedup':<10s} {'Throughput':<28s}")
    print("-" * 120)

    for name, builder in benchmarks:
        cpu_rt = reset_runtime("cpu")
        cpu_loss, config, units = builder(cpu_rt, "cpu", args.warmup, args.iters)
        cpu = timed_repeat(cpu_rt, cpu_loss, warmup=args.warmup, iters=args.iters)

        mps_rt = reset_runtime("mps")
        mps_loss, config2, units2 = builder(mps_rt, "mps", args.warmup, args.iters)
        mps = timed_repeat(mps_rt, mps_loss, warmup=args.warmup, iters=args.iters)

        assert config == config2
        assert units == units2
        summarize(name, config, cpu, mps, units)

    print("-" * 120)
    print("Done.")


if __name__ == "__main__":
    main()
