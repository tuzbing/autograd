#!/usr/bin/env python3
import gc
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parent))

from op import FusedLinearCrossEntropyOp
from tensor import Tensor


def ensure_core_ext():
    try:
        import metal_ext  # noqa: F401
        return
    except ImportError:
        pass

    subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(ROOT.parent))
    importlib.invalidate_caches()
    import metal_ext  # noqa: F401


ensure_core_ext()

from verify_model import cross_entropy_np
from runtime import Runtime


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


def read_storage(rt, storage, shape, dtype):
    if isinstance(getattr(storage, "data", None), np.ndarray):
        return np.asarray(storage.data).copy().reshape(shape)
    if dtype == np.float32:
        view = rt.backend.ctx.numpy_view_float32(storage.ptr.buffer_id, storage.ptr.offset, shape)
    elif dtype == np.int32:
        view = rt.backend.ctx.numpy_view_int32(storage.ptr.buffer_id, storage.ptr.offset, shape)
    else:
        raise TypeError(f"Unsupported dtype for trace read: {dtype}")
    return np.asarray(view).copy().reshape(shape)

def read_tensor(rt, tensor):
    return read_storage(rt, tensor.storage, tensor.shape, tensor.dtype)


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def print_section(title):
    print(f"\n=== {title} ===")


def print_max_abs(label, got, ref):
    print(f"{label} max_abs={max_abs_diff(got, ref):.6e}")


def print_array_block(label, got, ref):
    got = np.asarray(got)
    ref = np.asarray(ref)
    diff = got - ref
    print(f"{label} got=")
    print(got)
    print(f"{label} cpu=")
    print(ref)
    print(f"{label} diff=")
    print(diff)
    print(f"{label} max_abs={float(np.max(np.abs(diff))):.6e}")


def sync(rt):
    if hasattr(rt.backend, "synchronize"):
        rt.backend.synchronize()


def build_case(rt, device, arrays):
    x = make_tensor(arrays["x"].shape, np.float32, device, True, arrays["x"])
    w = make_tensor(arrays["w"].shape, np.float32, device, True, arrays["w"])
    targets = make_tensor(arrays["targets"].shape, np.int32, device, False, arrays["targets"])
    out = FusedLinearCrossEntropyOp()(x, w, targets)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()
    return out, x, w, targets


def run_runtime(device, arrays):
    rt = reset_runtime(device)
    out, x, w, targets = build_case(rt, device, arrays)

    pre = {
        "x": read_tensor(rt, x),
        "w": read_tensor(rt, w),
        "targets": read_tensor(rt, targets),
    }

    rt.forward()
    sync(rt)
    fwd = {
        "x": read_tensor(rt, x),
        "w": read_tensor(rt, w),
        "targets": read_tensor(rt, targets),
        "loss": read_tensor(rt, out),
    }

    rt.zero_grad()
    if out.grad is not None:
        rt.backend.fill(out.grad, 1.0)
    rt.backward_graph()
    sync(rt)
    bwd = {
        "x_grad": read_tensor(rt, x.grad) if x.grad is not None else None,
        "w_grad": read_tensor(rt, w.grad) if w.grad is not None else None,
    }
    return {"pre": pre, "fwd": fwd, "bwd": bwd}

def build_ce_cpu_refs(x, w, targets, grad_out=1.0):
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int32)
    X = x.reshape(-1, x.shape[-1])
    Y = targets.reshape(-1)
    N, C = X.shape
    V = w.shape[0]
    logits = X @ w.T
    row_max = np.full((N, 1), -np.inf, dtype=np.float32)
    row_sum = np.zeros((N, 1), dtype=np.float32)
    dlogits = np.zeros((N, V), dtype=np.float32)
    scale = np.float32(grad_out / max(N, 1))
    for i in range(N):
        y = int(Y[i])
        if y < 0 or y >= V:
            continue
        lm = np.max(logits[i]).astype(np.float32)
        exp_shift = np.exp(logits[i] - lm).astype(np.float32)
        rs = np.sum(exp_shift).astype(np.float32)
        row_max[i, 0] = lm
        row_sum[i, 0] = rs
        probs = exp_shift / rs
        probs[y] -= 1.0
        dlogits[i] = probs * scale
    dx = (dlogits @ w).reshape(x.shape).astype(np.float32)
    dw = (dlogits.T @ X).astype(np.float32)
    loss = np.asarray([cross_entropy_np(logits, targets)], dtype=np.float32)
    return {
        "loss": loss,
        "row_max": row_max,
        "row_sum": row_sum,
        "dx": dx,
        "dw": dw,
    }

def run_mps_split_trace(arrays):
    rt = reset_runtime("mps")
    out, x, w, targets = build_case(rt, "mps", arrays)
    cpu_refs = build_ce_cpu_refs(arrays["x"], arrays["w"], arrays["targets"], grad_out=1.0)
    backend = rt.backend
    node = next(n for n in rt.topo if type(n.op).__name__ == "FusedLinearCrossEntropyOp")
    scratch = node.backward.keywords["scratch"]
    row_max_st = scratch["row_max"]
    row_sum_st = scratch["row_sum"]
    N = x.shape[0] * x.shape[1] if len(x.shape) == 3 else x.shape[0]
    C = x.shape[-1]
    V = w.shape[0]
    row_blocks = (N + 31) // 32
    vocab_blocks = (V + 63) // 64
    print_section("mps forward")
    rt.forward()
    sync(rt)
    print_max_abs("forward loss", read_tensor(rt, out), cpu_refs["loss"])
    rt.zero_grad()
    backend.fill(out.grad, 1.0)
    if x.grad is not None:
        backend.fill(x.grad, 0.0)
    if w.grad is not None:
        backend.fill(w.grad, 0.0)
    print_section("backward / fused_linear_ce_backward_stats_kernel")
    # print("before: x =", read_tensor(rt, x))
    # print("before: w =", read_tensor(rt, w))
    # print("before: targets =", read_tensor(rt, targets))
    # print("before: x.grad", read_tensor(rt, x.grad))
    # print("before: w.grad", read_tensor(rt, w.grad))
    # print("before: out.grad", read_tensor(rt, out.grad))
    backend.begin_pass()
    backend.ctx.dispatch_with_args_3d(
        "fused_linear_ce_backward_stats_kernel",
        [
            x.storage.ptr.buffer_id,
            w.storage.ptr.buffer_id,
            targets.storage.ptr.buffer_id,
            row_max_st.ptr.buffer_id,
            row_sum_st.ptr.buffer_id,
        ],
        [
            x.storage.ptr.offset,
            w.storage.ptr.offset,
            targets.storage.ptr.offset,
            row_max_st.ptr.offset,
            row_sum_st.ptr.offset,
        ],
        [N, C, V],
        row_blocks * 128,
        1,
        1,
        128,
    )
    backend.end_pass()
    backend.synchronize()
    row_max_got = read_storage(rt, row_max_st, (N, 1), np.float32)
    row_sum_got = read_storage(rt, row_sum_st, (N, 1), np.float32)
    print_max_abs("stats/output row_max", row_max_got, cpu_refs["row_max"])
    print_max_abs("stats/output row_sum", row_sum_got, cpu_refs["row_sum"])
    print_section("backward / fused_linear_ce_backward_dx_kernel")
    backend.fill(x.grad, 0.0)
    backend.begin_pass()
    backend.ctx.dispatch_with_args_3d(
        "fused_linear_ce_backward_dx_kernel",
        [
            x.storage.ptr.buffer_id,
            w.storage.ptr.buffer_id,
            targets.storage.ptr.buffer_id,
            out.grad.storage.ptr.buffer_id,
            row_max_st.ptr.buffer_id,
            row_sum_st.ptr.buffer_id,
            x.grad.storage.ptr.buffer_id,
        ],
        [
            x.storage.ptr.offset,
            w.storage.ptr.offset,
            targets.storage.ptr.offset,
            out.grad.storage.ptr.offset,
            row_max_st.ptr.offset,
            row_sum_st.ptr.offset,
            x.grad.storage.ptr.offset,
        ],
        [N, C, V],
        row_blocks * 128,
        1,
        1,
        128,
    )
    backend.end_pass()
    backend.synchronize()
    dx_got = read_tensor(rt, x.grad)
    print_max_abs("dx/output x.grad", dx_got, cpu_refs["dx"])
    print_section("backward / fused_linear_ce_backward_dw_kernel")
    backend.fill(w.grad, 0.0)
    backend.begin_pass()
    backend.ctx.dispatch_with_args_3d(
        "fused_linear_ce_backward_dw_kernel",
        [
            x.storage.ptr.buffer_id,
            w.storage.ptr.buffer_id,
            targets.storage.ptr.buffer_id,
            out.grad.storage.ptr.buffer_id,
            row_max_st.ptr.buffer_id,
            row_sum_st.ptr.buffer_id,
            w.grad.storage.ptr.buffer_id,
        ],
        [
            x.storage.ptr.offset,
            w.storage.ptr.offset,
            targets.storage.ptr.offset,
            out.grad.storage.ptr.offset,
            row_max_st.ptr.offset,
            row_sum_st.ptr.offset,
            w.grad.storage.ptr.offset,
        ],
        [N, C, V],
        vocab_blocks * 128,
        1,
        1,
        128,
    )
    backend.end_pass()
    backend.synchronize()
    dw_got = read_tensor(rt, w.grad)
    print_max_abs("dw/output w.grad", dw_got, cpu_refs["dw"])
    return {
        "loss": read_tensor(rt, out),
        "row_max": row_max_got,
        "row_sum": row_sum_got,
        "dx": dx_got,
        "dw": dw_got,
        "cpu_refs": cpu_refs,
    }

def dump_ce_backend_inputs(rt, x, w, targets, out, scratch, label):
    print_section(label)
    print("x =", read_tensor(rt, x))
    print("w =", read_tensor(rt, w))
    print("targets =", read_tensor(rt, targets))
    print("out.grad =", read_tensor(rt, out.grad))
    if x.grad is not None:
        print("x.grad before =", read_tensor(rt, x.grad))
    if w.grad is not None:
        print("w.grad before =", read_tensor(rt, w.grad))
    if scratch is not None:
        row_max_st = scratch["row_max"]
        row_sum_st = scratch["row_sum"]
        print("row_max =", read_storage(rt, row_max_st, row_max_st.shape, row_max_st.dtype))
        print("row_sum =", read_storage(rt, row_sum_st, row_sum_st.shape, row_sum_st.dtype))

def run_mps_backend_trace(arrays):
    rt = reset_runtime("mps")
    out, x, w, targets = build_case(rt, "mps", arrays)
    cpu_refs = build_ce_cpu_refs(arrays["x"], arrays["w"], arrays["targets"], grad_out=1.0)
    print_section("mps forward / backend backward")
    rt.forward()
    sync(rt)
    print_max_abs("forward loss", read_tensor(rt, out), cpu_refs["loss"])
    rt.zero_grad()
    rt.backend.fill(out.grad, 1.0)
    if x.grad is not None:
        rt.backend.fill(x.grad, 0.0)
    if w.grad is not None:
        rt.backend.fill(w.grad, 0.0)
    # 这里会走 Runtime 绑定的 node.backward，
    # 对 FusedLinearCrossEntropyOp 来说就是 debug/backend.py 里的 metal_fused_linear_ce_backward
    
    # node = next(n for n in rt.topo if type(n.op).__name__ == "FusedLinearCrossEntropyOp")
    # scratch = node.backward.keywords["scratch"]
    # dump_ce_backend_inputs(rt, x, w, targets, out, scratch, "backend backward / inputs before rt.backward_graph")

    rt.backward_graph()
    sync(rt)
    dx_got = read_tensor(rt, x.grad)
    dw_got = read_tensor(rt, w.grad)
    print_max_abs("backend/output x.grad", dx_got, cpu_refs["dx"])
    print_max_abs("backend/output w.grad", dw_got, cpu_refs["dw"])
    return {
        "loss": read_tensor(rt, out),
        "dx": dx_got,
        "dw": dw_got,
        "cpu_refs": cpu_refs,
    }

def compare_trace(name, got, ref, atol=3e-4, rtol=3e-4, failures=None):
    diff = max_abs_diff(got, ref)
    print(f"{name} max_abs={diff:.6e}")
    ok = np.allclose(np.asarray(got), np.asarray(ref), atol=atol, rtol=rtol, equal_nan=True)
    if not ok:
        print_array_block(name, got, ref)
        if failures is not None:
            failures.append(name)


def main():
    seed = 1234
    # B, T, V, C = 1, 1, 8, 1
    # B, T, V, C = 2, 32, 8, 512
    B, T, V, C = 2, 32, 1024, 512
    rng = np.random.default_rng(seed)
    arrays = {
        "x": (rng.standard_normal((B, T, C)).astype(np.float32) * 0.1).copy(),
        "w": (rng.standard_normal((V, C)).astype(np.float32) * 0.1).copy(),
        "targets": rng.integers(0, V, size=(B, T), dtype=np.int32).copy(),
    }

    print_section("numpy oracle")
    oracle_loss = np.asarray([cross_entropy_np(arrays["x"] @ arrays["w"].T, arrays["targets"])], dtype=np.float32)
    print(f"oracle loss={oracle_loss[0]:.6e}")

    print_section("cpu runtime")
    cpu = run_runtime("cpu", arrays)
    print_max_abs("cpu/oracle loss", cpu["fwd"]["loss"], oracle_loss)

    print_section("mps runtime")
    mps = run_mps_split_trace(arrays)

    print_section("mps runtime / backend backward")
    mps_backend = run_mps_backend_trace(arrays)

    failures = []

    print_section("trace summary")
    compare_trace("loss cpu_vs_oracle", mps["loss"], mps["cpu_refs"]["loss"], failures=failures)
    compare_trace("row_max cpu_vs_ref", mps["row_max"], mps["cpu_refs"]["row_max"], failures=failures)
    compare_trace("row_sum cpu_vs_ref", mps["row_sum"], mps["cpu_refs"]["row_sum"], failures=failures)
    compare_trace("x.grad cpu_vs_ref", mps["dx"], mps["cpu_refs"]["dx"], failures=failures)
    compare_trace("w.grad cpu_vs_ref", mps["dw"], mps["cpu_refs"]["dw"], failures=failures)

    compare_trace("backend x.grad cpu_vs_ref", mps_backend["dx"], mps_backend["cpu_refs"]["dx"], failures=failures)
    compare_trace("split vs backend x.grad", mps["dx"], mps_backend["dx"], failures=failures)
    compare_trace("split vs backend w.grad", mps["dw"], mps_backend["dw"], failures=failures)

    if failures:
        print("\nTrace summary: FAIL")
        print("failed checks:")
        for name in failures:
            print(f"  - {name}")
        raise AssertionError("FusedLinearCrossEntropy trace mismatch")

    print("\nTrace summary: PASS")
    print(f"script_path={Path(__file__).resolve()}")


if __name__ == "__main__":
    main()
