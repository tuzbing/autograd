#!/usr/bin/env python3
"""
/Users/a58/Documents/code/nanochat/.venv/bin/python check_qwen_kernel.py --T=256

"""
import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

import functools
from collections import Counter, defaultdict
from dataclasses import dataclass

sys.dont_write_bytecode = True
ATOL = 1e-6
RTOL = 1e-6
SEED = 1333
FAILURES = []
PRINT_PASSES = True

# The modified operator implementations are in the project root.
# debug/backend.py overrides them with the old three-input implementation.
candidates = [Path(os.environ.get("AUTOGRAD_ROOT", ".")), Path(__file__).resolve().parent.parent]
PROJECT_ROOT = next((p.resolve() for p in candidates if (p / "op.py").is_file() and (p / "backend.py").is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Set AUTOGRAD_ROOT to the autograd project directory")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
from runtime import Runtime
from tensor import Tensor
from op import FusedRMSNormMLPOp, FusedRMSNormGQAOp, FusedRMSNormLinearCrossEntropyOp
from module import QwenCoder

def make_tensor(shape, dtype, device, requires_grad, init_val):
    return Tensor(
        shape=shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        init_val=np.asarray(init_val, dtype=dtype).copy(),
    )

def reset_runtime(device):
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device)
    Runtime._default_runtime = rt
    return rt

def copy_to_host(rt, tensor):
    return np.asarray(rt.backend.copy_to_host(tensor)).copy()

def check(name, got, ref, *, fail_fast=False, atol=ATOL, rtol=RTOL):
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        raise AssertionError(f"{name}: shape {got.shape} != {ref.shape}")
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-30)
    max_abs = float(diff.max(initial=0.0))
    max_rel = float((diff / denom).max(initial=0.0))
    ok = bool(np.allclose(got, ref, atol=atol, rtol=rtol, equal_nan=False))
    if PRINT_PASSES or not ok:
        print(
            f"  {'PASS' if ok else 'FAIL'} {name}: max_abs={max_abs:.8e}, "
            f"max_rel={max_rel:.8e}, atol={atol:g}, rtol={rtol:g}"
        )
    if fail_fast and not ok:
        bad = np.argwhere(~np.isclose(got, ref, atol=atol, rtol=rtol, equal_nan=False))
        first = tuple(int(v) for v in bad[0]) if bad.size else ()
        raise AssertionError(
            f"{name} mismatch at {first}: got={got[first]!r}, ref={ref[first]!r}, "
            f"max_abs={max_abs:.8e}, max_rel={max_rel:.8e}"
        )
    if not ok:
        FAILURES.append((name, max_abs, max_rel))
    return ok

def check_qwen_value(
    name,
    got,
    ref,
    *,
    grad_dtype="fp32",
    parameter=None,
):
    if grad_dtype not in ("fp32", "bf16"):
        raise ValueError(f"unsupported grad_dtype={grad_dtype!r}")

    atol, rtol = qwen_check_tolerance(name)

    if grad_dtype == "bf16" and parameter is not None:
        bf16_rtol = (
            8e-3 if parameter == "tok_emb.weight" else 4e-3
        )
        atol = max(atol, 2e-5)
        rtol = max(rtol, bf16_rtol)

    return check(
        f"grad_dtype={grad_dtype}/{name}",
        got,
        ref,
        atol=atol,
        rtol=rtol,
    )

def qwen_check_tolerance(name):
    """
    CPU 参考按数学结构计算，Metal 使用并行归约。
    Qwen 的 weighted RMSNorm 反向误差会沿残差和线性层传播。
    """

    # 新增的 weighted RMSNorm dgamma 归约。
    if (
        "post_norm_weight.grad" in name
        or "input_norm_weight.grad" in name
    ):
        return 1e-4, 1e-5

    # MLP 本身的输入和权重梯度。
    if "FusedRMSNormMLP" in name and "backward" in name:
        return 1e-5, 1e-5

    # Add 的输出梯度来自上游 MLP 反向。
    if name.startswith("Add[") and "backward" in name:
        return 1e-5, 1e-5

    # FusedLinearAdd 的输入梯度误差仍处于微小范围。
    if (
        "FusedLinearAdd" in name
        and "backward" in name
        and "proj.weight.grad" not in name
    ):
        return 1e-5, 1e-5

    # dW = act.T @ d_out，会放大上游 d_out 的归约误差。
    if (
        "FusedLinearAdd" in name
        and "proj.weight.grad" in name
    ):
        return 5e-4, 1e-5

    if (
        "FusedRMSNormGQA" in name
        and "v_proj.bias.grad" in name
    ):
        return 1e-4, 1e-5

    # GQA 的上游梯度以及后续 Gather 梯度。
    if (
        "FusedRMSNormGQA" in name
        or "Gather" in name
    ) and "backward" in name:
        return 1e-5, 1e-5

    # 整图参数梯度中的误差传播。
    if "parameter.grad" in name:
        if (
            "mlp.fc1.weight" in name
            or "mlp.fc2.weight" in name
            or "post_norm_weight" in name
            or "input_norm_weight" in name
            or "proj.weight" in name
            or "v_proj" in name
            or "tok_emb.weight" in name
        ):
            return 5e-4, 1e-5

    return ATOL, RTOL

def op_name(node):
    return node.op.op_name

def _dispatch(rt, callback):
    rt.backend.begin_pass()
    callback()
    rt.backend.end_pass()
    rt.backend.synchronize()

def _zero(rt, tensor):
    rt.backend.fill(tensor, 0.0)

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

    gamma = np.asarray(params["norm_weight"], dtype=np.float32)
    XN = (x_norm * gamma).reshape(N, C)

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

    dgamma = np.sum(dx_norm * x_norm, axis=(0, 1)).astype(np.float32)
    dx_norm = dx_norm * gamma

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
        "dgamma": dgamma,
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

def probe_mlp_internal(
    cfg,
    x_np,
    arrays,
    upstream,
    residual_grad,
):
    print(
        "\n=== FusedRMSNormMLP SwiGLU validation ==="
    )

    upstream = np.asarray(
        upstream,
        dtype=np.float32,
    ).copy()

    residual_grad = np.asarray(
        residual_grad,
        dtype=np.float32,
    ).copy()

    x_after_ref = np.asarray(x_np, dtype=np.float32).copy()
    mp = arrays

    # 只验证 Qwen MLP，不构造 attention 或整模型图。
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

        t = {
            "x_after": make_tensor(x_after_ref.shape, np.float32, device, True, x_after_ref),
            "norm_weight": make_tensor(mp["norm_weight"].shape, np.float32, device, True, mp["norm_weight"]),
            "w1": make_tensor(mp["w1"].shape, np.float32, device, True, mp["w1"]),
            "w2": make_tensor(mp["w2"].shape, np.float32, device, True, mp["w2"]),
        }
        mlp = FusedRMSNormMLPOp(eps=1e-6)(
            t["x_after"], t["norm_weight"], t["w1"], t["w2"]
        )
        t["mlp"] = mlp

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
            expected_mps_bytes = ((N * 2 * H * itemsize + 63) // 64 * 64
                                  + (N * H * itemsize + 63) // 64 * 64)

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

            if new_peak_bytes != expected_mps_bytes:
                raise AssertionError(
                    "MPS SwiGLU scratch memory increased: "
                    f"actual={new_peak_bytes}, expected={expected_mps_bytes}"
                )

        rt.forward()
        rt.backend.synchronize()

        # 先验证 MLP 收到的输入正确。
        check(
            f"{device}/QwenMLP/input_vs_oracle",
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
            atol=2e-5,
            rtol=1e-5,
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
            atol=2e-5,
            rtol=1e-5,
        )
        check(
            f"{device}/FusedRMSNormMLP/backward/wgu.grad_vs_oracle",
            got_dwgu,
            oracle["dw1"],
            fail_fast=True,
            atol=1e-5,
            rtol=1e-5,
        )
        check(
            f"{device}/FusedRMSNormMLP/backward/wdown.grad_vs_oracle",
            got_dwdown,
            oracle["dw2"],
            fail_fast=True,
            atol=1e-5,
            rtol=1e-5,
        )

        got_dgamma = copy_to_host(rt, t["norm_weight"].grad)
        check(
            f"{device}/FusedRMSNormMLP/backward/norm_weight.grad_vs_oracle",
            got_dgamma, oracle["dgamma"], fail_fast=True,
            atol=1e-5, rtol=1e-5,
        )
        results[device] = {
            "dgamma": got_dgamma,
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
        atol=1e-5,
        rtol=1e-5,
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/x.grad",
        results["mps"]["dx"],
        results["cpu"]["dx"],
        fail_fast=True,
        atol=3e-5,
        rtol=1e-5,
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/wgu.grad",
        results["mps"]["dwgu"],
        results["cpu"]["dwgu"],
        fail_fast=True,
        atol=1e-5,
        rtol=1e-5,
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/wdown.grad",
        results["mps"]["dwdown"],
        results["cpu"]["dwdown"],
        fail_fast=True,
        atol=1e-5,
        rtol=1e-5,
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/norm_weight.grad",
        results["mps"]["dgamma"], results["cpu"]["dgamma"],
        fail_fast=True,
        atol=1e-5,
        rtol=1e-5,
    )

def gqa_oracle(x, params, upstream, residual_grad, rope_cos, rope_sin,
               n_head=12, n_kv_head=2, eps=1e-6, is_causal=True, rope_sign=1):
    x = np.asarray(x, dtype=np.float32)
    upstream = np.asarray(upstream, dtype=np.float32)
    residual_grad = np.asarray(residual_grad, dtype=np.float32)
    B, T, C = x.shape
    D = C // n_head
    G = n_head // n_kv_head
    CKV = n_kv_head * D
    _, inv_rms, x_hat = rmsnorm_values(x, eps)
    gamma = np.asarray(params["norm_weight"], dtype=np.float32)
    z = (x_hat * gamma).reshape(B*T, C)

    q = (z @ params["wq"] + params["bq"]).reshape(B, T, n_head, D)
    k = (z @ params["wk"] + params["bk"]).reshape(B, T, n_kv_head, D)
    v = (z @ params["wv"] + params["bv"]).reshape(B, T, n_kv_head, D)
    cos = np.asarray(rope_cos[:T], dtype=np.float32)[None, :, None, :]
    sin = np.asarray(rope_sin[:T], dtype=np.float32)[None, :, None, :] * np.float32(rope_sign)
    q0, q1 = q[..., :D//2], q[..., D//2:]
    k0, k1 = k[..., :D//2], k[..., D//2:]
    q = np.concatenate((q0*cos-q1*sin, q0*sin+q1*cos), axis=-1)
    k = np.concatenate((k0*cos-k1*sin, k0*sin+k1*cos), axis=-1)

    # 显式分组维度 G，通过广播 K/V 表达 GQA，不复刻 Metal 的地址映射。
    Q = q.transpose(0, 2, 1, 3).reshape(B, n_kv_head, G, T, D)
    K = k.transpose(0, 2, 1, 3)[:, :, None, :, :]
    V = v.transpose(0, 2, 1, 3)[:, :, None, :, :]
    scale = np.float32(D ** -0.5)
    probs = np.matmul(Q, K.swapaxes(-1, -2)) * scale
    if is_causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        probs[..., mask] = -np.inf
    probs -= np.max(probs, axis=-1, keepdims=True)
    np.exp(probs, out=probs)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    context = np.matmul(probs, V)
    out = context.reshape(B, n_head, T, D).transpose(0, 2, 1, 3).reshape(B, T, C)

    dO = upstream.reshape(B, T, n_head, D).transpose(0, 2, 1, 3)
    dO = dO.reshape(B, n_kv_head, G, T, D)
    dV = np.matmul(probs.swapaxes(-1, -2), dO).sum(axis=2)
    dP = np.matmul(dO, V.swapaxes(-1, -2))
    dS = probs * (dP - np.sum(probs*dP, axis=-1, keepdims=True)) * scale
    dQ = np.matmul(dS, K)
    dK = np.matmul(dS.swapaxes(-1, -2), Q).sum(axis=2)
    dq = dQ.reshape(B, n_head, T, D).transpose(0, 2, 1, 3)
    dk = dK.transpose(0, 2, 1, 3)
    dv = dV.transpose(0, 2, 1, 3).reshape(B*T, CKV)

    dq0, dq1 = dq[..., :D//2], dq[..., D//2:]
    dk0, dk1 = dk[..., :D//2], dk[..., D//2:]
    dq = np.concatenate((dq0*cos+dq1*sin, dq1*cos-dq0*sin), axis=-1).reshape(B*T, C)
    dk = np.concatenate((dk0*cos+dk1*sin, dk1*cos-dk0*sin), axis=-1).reshape(B*T, CKV)
    dz = (dq @ params["wq"].T + dk @ params["wk"].T + dv @ params["wv"].T).reshape(B, T, C)
    dgamma = np.sum(dz*x_hat, axis=(0, 1))
    d_hat = dz*gamma
    dx_local = inv_rms*(d_hat-x_hat*np.mean(d_hat*x_hat, axis=-1, keepdims=True))
    return {
        "out": out.astype(np.float32),
        "dx": (dx_local+residual_grad).astype(np.float32),
        "dx_local": dx_local.astype(np.float32),
        "dgamma": dgamma.astype(np.float32),
        "dwq": (z.T @ dq).astype(np.float32),
        "dbq": dq.sum(axis=0).astype(np.float32),
        "dwk": (z.T @ dk).astype(np.float32),
        "dbk": dk.sum(axis=0).astype(np.float32),
        "dwv": (z.T @ dv).astype(np.float32),
        "dbv": dv.sum(axis=0).astype(np.float32),
    }

def probe_gqa_internal(cfg, x_np, arrays, upstream, residual_grad, rope_cos, rope_sin):
    print("\n=== FusedRMSNormGQA validation ===")
    upstream = np.asarray(upstream, dtype=np.float32).copy()
    residual_grad = np.asarray(residual_grad, dtype=np.float32).copy()
    x_ref = np.asarray(x_np, dtype=np.float32).copy()
    oracle = gqa_oracle(
        x_ref, arrays, upstream, residual_grad, rope_cos, rope_sin,
        n_head=cfg.n_head, n_kv_head=cfg.n_kv_head,
        eps=1e-6, is_causal=True, rope_sign=1,
    )
    results = {}
    for device in ("cpu", "mps"):
        print(f"\n=== FusedRMSNormGQA {device.upper()} validation ===")
        rt = reset_runtime(device)
        t = {"x": make_tensor(x_ref.shape, np.float32, device, True, x_ref)}
        for name in ("norm_weight", "wq", "bq", "wk", "bk", "wv", "bv"):
            t[name] = make_tensor(arrays[name].shape, np.float32, device, True, arrays[name])
        t["rope_cos"] = make_tensor(rope_cos.shape, np.float32, device, False, rope_cos)
        t["rope_sin"] = make_tensor(rope_sin.shape, np.float32, device, False, rope_sin)
        gqa = FusedRMSNormGQAOp(
            n_head=cfg.n_head, n_kv_head=cfg.n_kv_head,
            eps=1e-6, is_causal=True, rope_sign=1,
        )(t["x"], t["norm_weight"], t["wq"], t["bq"], t["wk"], t["bk"],
          t["wv"], t["bv"], t["rope_cos"], t["rope_sin"])
        t["gqa"] = gqa
        rt.loss = gqa
        rt.build_topo(gqa)
        rt.bind_memory()
        node = next(n for n in rt.topo if op_name(n) == "FusedRMSNormGQA")

        C, D = cfg.C, cfg.C // cfg.n_head
        CKV = cfg.n_kv_head * D
        for name, shape in (("wq", (C,C)), ("wk", (C,CKV)), ("wv", (C,CKV))):
            if t[name].shape != shape:
                raise AssertionError(f"{device}: invalid {name} shape {t[name].shape}")
        if device == "mps":
            N = cfg.B * cfg.T
            itemsize = np.dtype(np.float32).itemsize
            # 与现有 MLP probe 一样检查 scratch 大小；只替换为该算子的槽位大小。
            fwd_elements = (N*C, N*CKV, N*CKV)
            bwd_elements = (N*CKV, N*CKV, N*cfg.n_head, N*cfg.n_head,
                            N*C, N*C, N*CKV, N*CKV)
            expected_fwd = sum((n*itemsize+63)//64*64 for n in fwd_elements)
            expected_bwd = sum((n*itemsize+63)//64*64 for n in bwd_elements)
            actual_fwd = node.scratch_fwd_layout.nbytes if node.scratch_fwd_layout else 0
            actual_bwd = node.scratch_bwd_layout.nbytes if node.scratch_bwd_layout else 0
            if actual_fwd != expected_fwd or actual_bwd != expected_bwd:
                raise AssertionError(
                    "MPS GQA scratch memory mismatch: "
                    f"actual=({actual_fwd},{actual_bwd}), expected=({expected_fwd},{expected_bwd})"
                )
        rt.forward()
        rt.backend.synchronize()
        check(f"{device}/FusedRMSNormGQA/input_vs_oracle",
              copy_to_host(rt, t["x"]), x_ref, fail_fast=True)
        got_out = copy_to_host(rt, gqa)
        check(f"{device}/FusedRMSNormGQA/forward_vs_oracle",
              got_out, oracle["out"], fail_fast=True, atol=2e-5, rtol=1e-5)
        for tensor in t.values():
            if isinstance(tensor, Tensor) and tensor.grad is not None:
                _zero(rt, tensor.grad)
        rt.backend.copy_from_host(gqa.grad, upstream)
        rt.backend.copy_from_host(t["x"].grad, residual_grad)
        _dispatch(rt, lambda: node.backward(node.inputs, node.outputs))
        results[device] = {"out": got_out}
        for name, key in (("x", "dx"), ("norm_weight", "dgamma"),
                          ("wq", "dwq"), ("bq", "dbq"),
                          ("wk", "dwk"), ("bk", "dbk"),
                          ("wv", "dwv"), ("bv", "dbv")):
            got = copy_to_host(rt, t[name].grad)
            check(f"{device}/FusedRMSNormGQA/backward/{name}.grad_vs_oracle",
                  got, oracle[key], fail_fast=True,
                  atol=2e-5 if name == "x" else 1e-5, rtol=1e-5)
            results[device][key] = got
    check("FusedRMSNormGQA/CPU_vs_Metal/forward",
          results["mps"]["out"], results["cpu"]["out"],
          fail_fast=True, atol=1e-5, rtol=1e-5)
    for name, key in (("x", "dx"), ("norm_weight", "dgamma"),
                      ("wq", "dwq"), ("bq", "dbq"),
                      ("wk", "dwk"), ("bk", "dbk"),
                      ("wv", "dwv"), ("bv", "dbv")):
        check(f"FusedRMSNormGQA/CPU_vs_Metal/{name}.grad",
              results["mps"][key], results["cpu"][key], fail_fast=True,
              atol=3e-5 if name == "x" else 1e-5, rtol=1e-5)

def fused_rmsnorm_linear_ce_oracle(
    x,
    weight,
    norm_weight,
    targets,
    row_scale,
    eps=1e-6,
):
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    norm_weight = np.asarray(norm_weight, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int32)
    row_scale = np.asarray(row_scale, dtype=np.float32)

    B, T, C = x.shape
    N = B * T
    V = weight.shape[0]

    X = x.reshape(N, C)
    Y = targets.reshape(N)
    S = row_scale.reshape(N)

    # RMSNorm
    mean_square = np.mean(
        X * X,
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )
    inv_rms = (
        np.float32(1.0)
        / np.sqrt(mean_square + np.float32(eps))
    ).astype(np.float32)

    x_hat = (X * inv_rms).astype(np.float32)
    Z = (x_hat * norm_weight.reshape(1, C)).astype(np.float32)

    # Linear
    logits = (Z @ weight.T).astype(np.float32)

    # Stable softmax
    row_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - row_max).astype(np.float32)
    exp_sum = np.sum(
        exp_logits,
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )
    probs = (exp_logits / exp_sum).astype(np.float32)

    valid = (Y >= 0) & (Y < V)
    valid_rows = np.flatnonzero(valid)

    row_loss = np.zeros((N,), dtype=np.float32)
    row_loss[valid_rows] = (
        row_max[valid_rows, 0]
        + np.log(exp_sum[valid_rows, 0])
        - logits[valid_rows, Y[valid_rows]]
    )

    # Qwen training path uses row_scale.
    loss = np.sum(
        row_loss * S,
        dtype=np.float32,
    )

    dlogits = probs.copy()
    dlogits[valid_rows, Y[valid_rows]] -= np.float32(1.0)
    dlogits *= S.reshape(N, 1)
    dlogits[~valid] = np.float32(0.0)

    dweight = (dlogits.T @ Z).astype(np.float32)
    dZ = (dlogits @ weight).astype(np.float32)

    dnorm_weight = np.sum(
        dZ * x_hat,
        axis=0,
        dtype=np.float32,
    ).astype(np.float32)

    dY = dZ * norm_weight.reshape(1, C)
    mean_dot = np.mean(
        dY * x_hat,
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )

    dx = (
        inv_rms * (dY - x_hat * mean_dot)
    ).astype(np.float32)

    return {
        "loss": np.asarray([loss], dtype=np.float32),
        "dx": dx.reshape(B, T, C),
        "dweight": dweight,
        "dnorm_weight": dnorm_weight,
    }

def probe_fused_rmsnorm_linear_ce(cfg, rng):
    print(
        "\n=== Qwen FusedRMSNormLinearCrossEntropy "
        "CPU/Metal validation ==="
    )

    B = cfg.B
    T = cfg.T
    C = 1536
    V = 151936
    N = B * T
    eps = 1e-6

    x_np = (
        rng.standard_normal((B, T, C))
        .astype(np.float32)
        * 0.02
    )

    # Qwen tied embedding / lm_head 权重布局：(V, C)
    weight_np = (
        rng.standard_normal((V, C))
        .astype(np.float32)
        * 0.02
    )

    norm_weight_np = (
        1.0
        + rng.standard_normal(C).astype(np.float32) * 0.02
    )

    targets_np = rng.integers(
        low=0,
        high=V,
        size=(B, T),
        dtype=np.int32,
    )

    # 与当前 CE kernel 的无效 target 语义一致：
    # target < 0 或 target >= V 时 loss 和梯度为零。
    targets_np.reshape(-1)[0] = -1
    if N > 1:
        targets_np.reshape(-1)[1] = V

    row_scale_np = (
        0.5
        + rng.random((N, 1)).astype(np.float32)
    )

    oracle = fused_rmsnorm_linear_ce_oracle(
        x_np,
        weight_np,
        norm_weight_np,
        targets_np,
        row_scale_np,
        eps=eps,
    )

    results = {}

    for device in ("cpu", "mps"):
        print(
            f"\n=== Qwen FusedRMSNormLinearCrossEntropy "
            f"{device.upper()} ==="
        )

        rt = reset_runtime(device)

        t = {
            "x": make_tensor(
                x_np.shape,
                np.float32,
                device,
                True,
                x_np,
            ),
            "weight": make_tensor(
                weight_np.shape,
                np.float32,
                device,
                True,
                weight_np,
            ),
            "norm_weight": make_tensor(
                norm_weight_np.shape,
                np.float32,
                device,
                True,
                norm_weight_np,
            ),
            "targets": make_tensor(
                targets_np.shape,
                np.int32,
                device,
                False,
                targets_np,
            ),
            "row_scale": make_tensor(
                row_scale_np.shape,
                np.float32,
                device,
                False,
                row_scale_np,
            ),
        }

        loss = FusedRMSNormLinearCrossEntropyOp(
            eps=eps,
        )(
            t["x"],
            t["weight"],
            t["norm_weight"],
            t["targets"],
            t["row_scale"],
        )

        rt.loss = loss
        rt.build_topo(loss)
        rt.bind_memory()

        node = next(
            n
            for n in rt.topo
            if op_name(n) == "FusedRMSNormLinearCrossEntropy"
        )

        if t["x"].shape != (B, T, C):
            raise AssertionError(
                f"{device}: invalid x shape {t['x'].shape}"
            )
        if t["weight"].shape != (V, C):
            raise AssertionError(
                f"{device}: invalid weight shape {t['weight'].shape}"
            )
        if t["norm_weight"].shape != (C,):
            raise AssertionError(
                f"{device}: invalid norm_weight shape "
                f"{t['norm_weight'].shape}"
            )
        if t["targets"].shape != (B, T):
            raise AssertionError(
                f"{device}: invalid targets shape "
                f"{t['targets'].shape}"
            )
        if t["row_scale"].shape != (N, 1):
            raise AssertionError(
                f"{device}: invalid row_scale shape "
                f"{t['row_scale'].shape}"
            )

        rt.forward()
        rt.backend.synchronize()

        got_loss = copy_to_host(rt, loss)

        check(
            f"{device}/QwenFusedRMSNormLinearCE/forward",
            got_loss,
            oracle["loss"],
            fail_fast=True,
            atol=3e-5,
            rtol=1e-5,
        )

        if device == "cpu":
            for tensor in t.values():
                if (
                    isinstance(tensor, Tensor)
                    and tensor.grad is not None
                ):
                    _zero(rt, tensor.grad)

            rt.backend.copy_from_host(
                loss.grad,
                np.ones((1,), dtype=np.float32),
            )

            _dispatch(
                rt,
                lambda: node.backward(
                    node.inputs,
                    node.outputs,
                ),
            )

        got_dx = copy_to_host(rt, t["x"].grad)
        got_dweight = copy_to_host(rt, t["weight"].grad)
        got_dnorm_weight = copy_to_host(
            rt,
            t["norm_weight"].grad,
        )

        check(
            f"{device}/QwenFusedRMSNormLinearCE/x.grad",
            got_dx,
            oracle["dx"],
            fail_fast=True,
            atol=5e-5,
            rtol=2e-5,
        )

        check(
            f"{device}/QwenFusedRMSNormLinearCE/weight.grad",
            got_dweight,
            oracle["dweight"],
            fail_fast=True,
            atol=5e-5,
            rtol=2e-5,
        )

        check(
            f"{device}/QwenFusedRMSNormLinearCE/norm_weight.grad",
            got_dnorm_weight,
            oracle["dnorm_weight"],
            fail_fast=True,
            atol=5e-5,
            rtol=2e-5,
        )

        results[device] = {
            "loss": got_loss,
            "dx": got_dx,
            "dweight": got_dweight,
            "dnorm_weight": got_dnorm_weight,
        }

    check(
        "QwenFusedRMSNormLinearCE/CPU_vs_Metal/forward",
        results["mps"]["loss"],
        results["cpu"]["loss"],
        fail_fast=True,
        atol=3e-5,
        rtol=1e-5,
    )

    check(
        "QwenFusedRMSNormLinearCE/CPU_vs_Metal/x.grad",
        results["mps"]["dx"],
        results["cpu"]["dx"],
        fail_fast=True,
        atol=5e-5,
        rtol=2e-5,
    )

    check(
        "QwenFusedRMSNormLinearCE/CPU_vs_Metal/weight.grad",
        results["mps"]["dweight"],
        results["cpu"]["dweight"],
        fail_fast=True,
        atol=5e-5,
        rtol=2e-5,
    )

    check(
        "QwenFusedRMSNormLinearCE/CPU_vs_Metal/norm_weight.grad",
        results["mps"]["dnorm_weight"],
        results["cpu"]["dnorm_weight"],
        fail_fast=True,
        atol=5e-5,
        rtol=2e-5,
    )

def probe_existing_qwen_ops(cfg, seed):
    rng = np.random.default_rng(seed)

    x = (
        rng.standard_normal((cfg.B, cfg.T, cfg.C)).astype(np.float32)
        * 0.02
    )

    arrays = {
        "norm_weight": (
            1.0 + rng.standard_normal(cfg.C).astype(np.float32) * 0.02
        ),
        "w1": (
            rng.standard_normal((cfg.C, 2 * cfg.H)).astype(np.float32)
            * 0.02
        ),
        "w2": (
            rng.standard_normal((cfg.H, cfg.C)).astype(np.float32)
            * 0.02
        ),
    }
    arrays["norm_weight"][0] = 0.0

    upstream = rng.standard_normal(x.shape).astype(np.float32) * 0.02
    residual_grad = upstream.copy()

    probe_mlp_internal(cfg, x, arrays, upstream, residual_grad)

    D = cfg.C // cfg.n_head
    CKV = cfg.n_kv_head * D

    gqa_arrays = {
        "norm_weight": (
            1.0 + rng.standard_normal(cfg.C).astype(np.float32) * 0.02
        ),
        "wq": (
            rng.standard_normal((cfg.C, cfg.C)).astype(np.float32) * 0.02
        ),
        "bq": rng.standard_normal(cfg.C).astype(np.float32) * 0.02,
        "wk": (
            rng.standard_normal((cfg.C, CKV)).astype(np.float32) * 0.02
        ),
        "bk": rng.standard_normal(CKV).astype(np.float32) * 0.02,
        "wv": (
            rng.standard_normal((cfg.C, CKV)).astype(np.float32) * 0.02
        ),
        "bv": rng.standard_normal(CKV).astype(np.float32) * 0.02,
    }
    gqa_arrays["norm_weight"][0] = 0.0

    positions = np.arange(cfg.T, dtype=np.float32)[:, None]
    channels = np.arange(0, D, 2, dtype=np.float32)[None, :]
    freqs = positions * (
        np.float32(cfg.rope_theta) ** (-channels / D)
    )
    rope_cos = np.cos(freqs).astype(np.float32)
    rope_sin = np.sin(freqs).astype(np.float32)

    probe_gqa_internal(
        cfg, x, gqa_arrays, upstream, residual_grad,
        rope_cos, rope_sin,
    )
    probe_fused_rmsnorm_linear_ce(cfg, rng)

@dataclass(frozen=True)
class Config:
    B: int
    T: int
    L: int = 2

    V: int = 151936
    C: int = 1536
    H: int = 8960
    n_head: int = 12
    n_kv_head: int = 2
    max_T: int = 32768
    eps: float = 1e-6
    rope_theta: float = 1000000.0
    dtype: object = np.float32


def make_qwen_model(cfg, device):
    return QwenCoder(
        V=cfg.V,
        C=cfg.C,
        max_T=cfg.max_T,
        n_head=cfg.n_head,
        n_layer=cfg.L,
        dtype=cfg.dtype,
        device=device,
        H=cfg.H,
        n_kv_head=cfg.n_kv_head,
        eps=cfg.eps,
        rope_theta=cfg.rope_theta,
    )


def iter_qwen_named_tensors(model):
    # 【参数替换】对应当前 module.py 的实际成员。
    # 共享 embedding / LM head 只枚举一次。
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

    yield "final_norm_weight", model.final_norm_weight


def build_parameter_arrays(cfg, seed):
    # 与 GPT2 脚本相同：固定 seed，按参数形状生成相同的随机权重。
    rng = np.random.default_rng(seed)
    model = make_qwen_model(cfg, "cpu")

    arrays = {}
    for name, tensor in iter_qwen_named_tensors(model):
        arrays[name] = (
            rng.standard_normal(tensor.shape).astype(np.float32) * 0.02
        ).copy()

    return arrays


def initialize_model(model, arrays):
    names = []
    for name, tensor in iter_qwen_named_tensors(model):
        if name not in arrays:
            raise KeyError(f"missing parameter {name}")
        if tuple(arrays[name].shape) != tuple(tensor.shape):
            raise AssertionError(
                f"{name}: array shape {arrays[name].shape} "
                f"!= parameter shape {tensor.shape}"
            )
        tensor.init_val = arrays[name].copy()
        names.append(name)

    if set(names) != set(arrays):
        raise AssertionError("parameter array set does not match QwenCoder")


def build_inputs(cfg, seed):
    # 与 GPT2 脚本相同：生成合法 token id 和独立 labels。
    rng = np.random.default_rng(seed + 1)
    input_ids = rng.integers(
        0, cfg.V, (cfg.B, cfg.T), dtype=np.int32
    )
    labels = rng.integers(
        0, cfg.V, (cfg.B, cfg.T), dtype=np.int32
    )
    return input_ids, labels


def topology_labels(topo):
    # 与 GPT2 脚本相同。
    counts = defaultdict(int)
    labels = []
    for node in topo:
        name = op_name(node)
        labels.append(f"{name}[{counts[name]}]")
        counts[name] += 1
    return labels


def verify_training_topology(topo, cfg):
    # 【算子替换】仅替换 Qwen 使用的 attention 和最终 CE。
    expected = Counter({
        "Gather": 1,
        "Add": cfg.L,
        "FusedRMSNormGQA": cfg.L,
        "FusedLinearAdd": cfg.L,
        "FusedRMSNormMLP": cfg.L,
        "FusedRMSNormLinearCrossEntropy": 1,
    })

    actual = Counter(op_name(node) for node in topo)

    if actual != expected:
        raise AssertionError(
            "QwenCoder.forward topology differs from expected training graph:\n"
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

    print("Training topology:", " -> ".join(topology_labels(topo)))


# ============================================================
# 【新增】以下内存隔离逻辑与 GPT2 参考脚本一致
# ============================================================

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
            tensor.grad.storage = rt.allocator.alloc(
                tensor.grad.shape, tensor.grad.dtype
            )
            rt.backend.fill(tensor.grad, 0.0)

    for tensor in tensors:
        if tensor.op is None or op_name(tensor.op) not in view_names:
            continue

        base = tensor.op.inputs[0]
        tensor.storage = rt.allocator.view(
            base.storage, 0, tensor.shape, tensor.dtype
        )

        if tensor.grad is not None and base.grad is not None:
            tensor.grad.storage = rt.allocator.view(
                base.grad.storage,
                0,
                tensor.grad.shape,
                tensor.grad.dtype,
            )

    for node in rt.topo:
        for phase in ("forward", "backward"):
            layout = getattr(
                node,
                f"scratch_{'fwd' if phase == 'forward' else 'bwd'}_layout",
                None,
            )
            if layout is None:
                continue

            private_storage = rt.allocator.alloc(
                (layout.nbytes,), np.uint8
            )
            private_scratch = rt.scratch_bind(private_storage, layout)
            bound = getattr(node, phase)

            if isinstance(bound, functools.partial):
                keywords = dict(bound.keywords or {})
                keywords["scratch"] = private_scratch
                bound = functools.partial(
                    bound.func, *bound.args, **keywords
                )
            else:
                bound = functools.partial(
                    bound, scratch=private_scratch
                )

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


def build_graph(device, cfg, arrays, input_ids, labels, grad_dtype="fp32"):
    rt = reset_runtime(device)
    model = make_qwen_model(cfg, device)
    parameter_grad_dtype = {"fp32": np.float32, "bf16": "bf16"}[grad_dtype]
    for parameter in model.parameters():
        parameter.grad_dtype = parameter_grad_dtype

    initialize_model(model, arrays)

    input_t = make_tensor(
        input_ids.shape, np.int32, device, False, input_ids
    )
    labels_t = make_tensor(
        labels.shape, np.int32, device, False, labels
    )

    # 与参考脚本一致：调用模型 forward，不在检查脚本里手工拼图。
    loss = model(input_t, targets=labels_t)

    rt.loss = loss
    rt.build_topo(loss)
    verify_training_topology(rt.topo, cfg)
    rt.bind_memory()

    if device == "cpu":
        isolate_cpu_reference_memory(rt)

    rt.is_compiled = True

    param_by_name = dict(iter_qwen_named_tensors(model))

    # 【Qwen 共享参数检查】
    parameter_ids = {id(t) for t in param_by_name.values()}
    if len(parameter_ids) != len(param_by_name):
        raise AssertionError("shared parameters were enumerated twice")
    if parameter_ids != {id(t) for t in model.parameters()}:
        raise AssertionError("parameter enumeration misses model parameters")

    return GraphState(
        rt=rt,
        model=model,
        loss=loss,
        input_ids=input_t,
        labels=labels_t,
        param_by_name=param_by_name,
        param_name_by_id={
            id(t): name for name, t in param_by_name.items()
        },
        labels_by_node={
            id(node): label
            for node, label in zip(
                rt.topo, topology_labels(rt.topo)
            )
        },
    )


# ============================================================
# 【新增】完整执行与比较：保持 GPT2 参考脚本的验证逻辑
# ============================================================

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


def compare_exact_graph(cpu, mps, grad_dtype="fp32"):
    print("\n=== Exact Runtime.execute() training path ===")

    if len(cpu) != len(mps):
        raise AssertionError("CPU/MPS repeat count differs")

    for repeat, (cpu_result, mps_result) in enumerate(zip(cpu, mps), 1):
        print(f"  [training execution {repeat}]")

        check_qwen_value(
            f"repeat[{repeat}]/loss",
            mps_result["loss"],
            cpu_result["loss"],
            grad_dtype=grad_dtype,
        )

        if set(cpu_result["parameter_grads"]) != set(mps_result["parameter_grads"]):
            raise AssertionError("CPU/MPS parameter gradient names differ")

        for name in sorted(cpu_result["parameter_grads"]):
            name_label = f"repeat[{repeat}]/parameter.grad/{name}"
            atol, rtol = qwen_check_tolerance(name_label)

            check_qwen_value(
                name_label,
                mps_result["parameter_grads"][name],
                cpu_result["parameter_grads"][name],
                grad_dtype=grad_dtype,
                parameter=name,
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
                raise AssertionError(f"{label}: input count differs")

            for tensor, value in zip(node.inputs, expected_inputs):
                rt.backend.copy_from_host(
                    tensor,
                    np.asarray(value, dtype=tensor.dtype),
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

    # 保留参考脚本的时序：
    # CPU forward 不生成 CE 梯度；Metal 融合 CE forward 已写入梯度。
    # 因此这里仅清零 CPU，不能额外清零 Metal。
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
                "parameter": state.param_name_by_id.get(id(tensor)),
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


def compare_staged_graph(cpu, mps, grad_dtype="fp32"):
    print("\n=== Per-operator exact training path ===")

    if list(cpu["forward"]) != list(mps["forward"]):
        raise AssertionError("CPU/MPS forward topology labels differ")

    for label in cpu["forward"]:
        cpu_outputs = cpu["forward"][label]
        mps_outputs = mps["forward"][label]

        if len(cpu_outputs) != len(mps_outputs):
            raise AssertionError(f"{label}: forward output count differs")

        for i, (mps_value, cpu_value) in enumerate(zip(mps_outputs, cpu_outputs)):
            check_qwen_value(f"{label}/forward/output[{i}]", mps_value, cpu_value, grad_dtype=grad_dtype)

    for label in reversed(list(cpu["forward"])):
        cpu_bwd = cpu["backward"][label]
        mps_bwd = mps["backward"][label]

        if len(cpu_bwd["output_grads"]) != len(mps_bwd["output_grads"]):
            raise AssertionError(f"{label}: output gradient count differs")

        for i, (mps_value, cpu_value) in enumerate(zip(mps_bwd["output_grads"], cpu_bwd["output_grads"])):
            if (cpu_value is None) != (mps_value is None):
                raise AssertionError(f"{label}: output[{i}] gradient presence differs")

            if cpu_value is not None:
                check_qwen_value(f"{label}/backward/output[{i}].grad", mps_value, cpu_value, grad_dtype=grad_dtype)

        if len(cpu_bwd["inputs"]) != len(mps_bwd["inputs"]):
            raise AssertionError(f"{label}: different differentiable input count")

        for cpu_item, mps_item in zip(cpu_bwd["inputs"], mps_bwd["inputs"]):
            if (cpu_item["index"], cpu_item["parameter"]) != (mps_item["index"], mps_item["parameter"]):
                raise AssertionError(f"{label}: input mapping differs")

            suffix = (cpu_item["parameter"] or f"input[{cpu_item['index']}]")
            check_qwen_value(f"{label}/backward/{suffix}.grad", mps_item["grad"], cpu_item["grad"], 
                grad_dtype=grad_dtype, parameter=cpu_item["parameter"])

def main():
    parser = argparse.ArgumentParser(description=("Verify the QwenCoder training graph on CPU and Metal"))
    parser.add_argument(
        "--max-seq-len", "--T",
        dest="max_seq_len", type=int, default=7,
    )
    parser.add_argument(
        "--device-batch-size", "--B",
        dest="device_batch_size", type=int, default=2,
    )
    parser.add_argument(
        "--depth", "--L",
        dest="depth", type=int, default=2,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--skip-internal-kernels",
        action="store_true",
        help="run exact graph and per-operator checks only",
    )
    parser.add_argument("--grad_dtype", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()

    if min(
        args.max_seq_len,
        args.device_batch_size,
        args.depth,
    ) <= 0:
        parser.error("all model dimensions must be positive")

    cfg = Config(
        B=args.device_batch_size,
        T=args.max_seq_len,
        L=args.depth,
    )
    if cfg.T > cfg.max_T:
        parser.error(f"--T must be <= {cfg.max_T}")
    if cfg.L > 28:
        parser.error("--L must be <= 28 for this Qwen configuration")

    global PRINT_PASSES
    PRINT_PASSES = not args.compact
    FAILURES.clear()

    print(f"project_root={PROJECT_ROOT}")
    print(f"runtime={sys.modules[Runtime.__module__].__file__}")
    print(
        "config: "
        f"B={cfg.B}, T={cfg.T}, V={cfg.V}, C={cfg.C}, "
        f"H={cfg.H}, n_head={cfg.n_head}, "
        f"n_kv_head={cfg.n_kv_head}, L={cfg.L}, "
        f"max_T={cfg.max_T}, eps={cfg.eps:g}, "
        f"rope_theta={cfg.rope_theta:g}, "
        f"atol={ATOL:g}, rtol={RTOL:g}"
    )

    arrays = build_parameter_arrays(cfg, args.seed)
    input_ids, labels = build_inputs(cfg, args.seed)

    # 与 GPT2 main 相同：先初始化 Metal，再执行 CPU 参考。
    mps_exact = run_exact_training_graph(build_graph("mps", cfg, arrays, input_ids, labels, grad_dtype=args.grad_dtype))
    cpu_exact = run_exact_training_graph(build_graph("cpu", cfg, arrays, input_ids, labels, grad_dtype="fp32"))
    compare_exact_graph(cpu_exact, mps_exact, grad_dtype=args.grad_dtype)

    # 比较完释放大参数梯度快照，不影响验证结果。
    del mps_exact, cpu_exact
    gc.collect()

    # 与 GPT2 main 相同：CPU staged -> Metal 注入 CPU 输入。
    cpu_trace = run_staged_graph(build_graph("cpu", cfg, arrays, input_ids, labels, grad_dtype="fp32"))
    mps_trace = run_staged_graph(build_graph("mps", cfg, arrays, input_ids, labels, grad_dtype=args.grad_dtype), reference_inputs=cpu_trace["forward_inputs"])
    compare_staged_graph(cpu_trace, mps_trace, grad_dtype=args.grad_dtype)

    del cpu_trace, mps_trace, arrays
    gc.collect()

    # 保留当前 Qwen 脚本已有的三个独立算子检查。
    if not args.skip_internal_kernels:
        probe_existing_qwen_ops(cfg, args.seed)

    if FAILURES:
        print(f"\n{len(FAILURES)} correctness check(s) failed:")
        for name, max_abs, max_rel in FAILURES:
            print(
                f"  {name}: "
                f"max_abs={max_abs:.8e}, max_rel={max_rel:.8e}"
            )
        raise AssertionError(
            "QwenCoder training-path correctness validation failed"
        )

    print("\nAll QwenCoder training-path correctness checks passed.")

if __name__ == "__main__":
    main()
