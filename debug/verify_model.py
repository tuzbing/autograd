#!/usr/bin/env python3
import gc
import os
import sys
import time
from dataclasses import dataclass
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
SEED = 1234
FLOAT_ATOL = 1e-4
FLOAT_RTOL = 1e-4
GRAD_ATOL = 2e-2
GRAD_RTOL = 2e-2
# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
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
def copy_to_host(rt, tensor):
    return rt.backend.copy_to_host(tensor).copy()
def assert_close(name, got, ref, atol=FLOAT_ATOL, rtol=FLOAT_RTOL):
    got = np.asarray(got)
    ref = np.asarray(ref)
    diff = np.abs(got - ref)
    rel = diff / (np.abs(ref) + 1e-12)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    max_rel = float(np.max(rel)) if rel.size else 0.0
    if not np.allclose(got, ref, atol=atol, rtol=rtol, equal_nan=True):
        raise AssertionError(
            f"{name} mismatch: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
            f"atol={atol}, rtol={rtol}"
        )
    # print(f"  PASS {name}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")

def sample_flat_indices(shape, count, seed):
    rng = np.random.default_rng(seed)
    size = int(prod(shape))
    count = min(count, size)
    return rng.choice(size, size=count, replace=False)
def finite_diff_scalar(fn, arrays, name, idxs, eps=1e-3):
    arr = arrays[name]
    flat = arr.reshape(-1)
    out = []
    for idx in idxs:
        orig = flat[idx]
        flat[idx] = orig + eps
        plus = float(fn(arrays))
        flat[idx] = orig - eps
        minus = float(fn(arrays))
        flat[idx] = orig
        out.append((plus - minus) / (2.0 * eps))
    return np.asarray(out, dtype=np.float32)

def run_backend_case(rt, loss, fetch_map, grad_map=None):
    rt.loss = loss
    rt.build_topo(loss)
    rt.bind_memory()
    t0 = time.perf_counter()
    rt.forward()
    rt.backend.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000.0
    fetched = {k: copy_to_host(rt, v) for k, v in fetch_map.items()}
    rt.zero_grad()
    rt.backend.fill(loss.grad, 1.0)
    t1 = time.perf_counter()
    rt.backward_graph()
    rt.backend.synchronize()
    bwd_ms = (time.perf_counter() - t1) * 1000.0
    grads = {}
    if grad_map is None:
        grad_map = fetch_map
    for k, tensor in grad_map.items():
        grads[k] = None if tensor.grad is None else copy_to_host(rt, tensor.grad)
    return fetched, grads, fwd_ms, bwd_ms

def timed_repeat(rt, loss, warmup=1, iters=3):
    # For a coarse timing report only.
    for _ in range(warmup):
        rt.zero_grad()
        rt.backend.fill(loss.grad, 1.0)
        rt.forward()
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
# ------------------------------------------------------------
# NumPy oracles
# ------------------------------------------------------------
def gelu_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))
def layernorm_np(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    y = x_hat * gamma + beta
    return y, mean, inv_std, x_hat
def softmax_np(x, dim=-1):
    x = x - np.max(x, axis=dim, keepdims=True)
    y = np.exp(x)
    y = y / np.sum(y, axis=dim, keepdims=True)
    return y
def fused_linear_add_np(x, act, w, b):
    return x + act @ w + b
def fused_ln_mlp_np(x, ln_w, ln_b, w1, b1, w2, b2, eps=1e-5):
    x_mid, mean, inv_std, x_hat = layernorm_np(x, ln_w, ln_b, eps=eps)
    h1 = x_mid @ w1 + b1
    h2 = gelu_np(h1)
    out = h2 @ w2 + b2
    cache = {
        "x_mid": x_mid,
        "mean": mean,
        "inv_std": inv_std,
        "x_hat": x_hat,
        "h1": h1,
        "h2": h2,
    }
    return out, cache
def fused_ln_fa_np(x, ln_w, ln_b, wq, bq, wk, bk, wv, bv, n_head, eps=1e-5, is_causal=True):
    B, T, C = x.shape
    H = n_head
    D = C // H
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm = (x - mean) * inv_std * ln_w + ln_b
    q = x_norm @ wq + bq
    k = x_norm @ wk + bk
    v = x_norm @ wv + bv
    Q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    K = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    V = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(D)
    if is_causal:
        mask = np.triu(np.ones((T, T), dtype=bool), 1)
        scores = np.where(mask[None, None, :, :], -np.inf, scores)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    out = np.matmul(probs, V).transpose(0, 2, 1, 3).reshape(B, T, C)
    cache = {
        "mean": mean,
        "inv_std": inv_std,
        "x_norm": x_norm,
        "q": q,
        "k": k,
        "v": v,
        "probs": probs,
    }
    return out, cache
def cross_entropy_np(logits, targets, ignore_index=-1):
    x = logits.reshape(-1, logits.shape[-1])
    y = targets.reshape(-1)
    x = x - np.max(x, axis=-1, keepdims=True)
    probs = np.exp(x)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    n = probs.shape[0]
    loss = 0.0
    for i in range(n):
        if y[i] != ignore_index:
            loss -= np.log(probs[i, y[i]] + 1e-10)
    return loss / n
# ------------------------------------------------------------
# Generic backend-vs-numpy checker
# ------------------------------------------------------------
@dataclass
class CaseResult:
    name: str
    cpu_fwd_ms: float
    cpu_bwd_ms: float
    metal_fwd_ms: float
    metal_bwd_ms: float

def check_case(name, build_numpy, build_backend, sampled_inputs, scalar_loss_fn=None, out_atol=1e-4, out_rtol=1e-4, probe_count=3, device_tolerances=None):
    print(f"\n[{name}]")
    rng = np.random.default_rng(SEED)
    ref = build_numpy(rng)
    ref_out = ref["out"]
    results = []
    for device in ("cpu", "mps"):
        rt = reset_runtime(device)
        built = build_backend(rt, rng, ref)
        loss = built["loss"]
        fetch_map = built["fetch_map"]
        grad_map = built["grad_map"]
        fetched, grads, fwd_ms, bwd_ms = run_backend_case(rt, loss, fetch_map, grad_map)
        print(f"  {device:4s} fwd={fwd_ms:.3f}ms bwd={bwd_ms:.3f}ms")
        atol, rtol = out_atol, out_rtol
        if device_tolerances is not None and device in device_tolerances:
            atol, rtol = device_tolerances[device]
        assert_close(f"{device} forward", fetched["out"], ref_out, atol=atol, rtol=rtol)
        for inp_name in sampled_inputs:
            if inp_name not in built["arrays"]:
                continue
            arrs = built["arrays"]
            idxs = sample_flat_indices(
                arrs[inp_name].shape,
                probe_count,
                seed=SEED + hash((name, device, inp_name)) % 10000,
            )
            def loss_fn(local_arrs):
                return scalar_loss_fn(local_arrs)
            fd = finite_diff_scalar(loss_fn, arrs, inp_name, idxs, eps=1e-3)
            got = grads[inp_name].reshape(-1)[idxs]
            assert_close(f"{device} {inp_name} sampled grad", got, fd, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        results.append((device, fwd_ms, bwd_ms))
    return results

# ------------------------------------------------------------
# GPT2 operator cases
# ------------------------------------------------------------
def run_add_case():
    name = "Add"
    B, T, C = 2, 3, 4
    def build_numpy(rng):
        x = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
        y = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
        out = x + y
        return {"x": x, "y": y, "out": out}

    def build_backend(rt, rng, ref):
        arrays = {"x": ref["x"].copy(), "y": ref["y"].copy()}
        x = make_tensor((B, T, C), np.float32, rt.allocator.device, True, arrays["x"])
        y = make_tensor((B, T, C), np.float32, rt.allocator.device, True, arrays["y"])
        out = AddOp()(x, y)
        return {
            "arrays": arrays,
            "loss": out,
            "fetch_map": {"out": out},
            "grad_map": {"x": x, "y": y},
        }

    def scalar_loss_fn(arrs):
        return float(np.sum(arrs["x"] + arrs["y"]))

    return check_case(name, build_numpy, build_backend, sampled_inputs=["x", "y"], scalar_loss_fn=scalar_loss_fn)

def run_mul_case():
    name = "Mul"
    B, T, C = 2, 3, 4
    def build_numpy(rng):
        x = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
        y = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
        out = x * y
        return {"x": x, "y": y, "out": out}
    def build_backend(rt, rng, ref):
        arrays = {"x": ref["x"].copy(), "y": ref["y"].copy()}
        x = make_tensor((B, T, C), np.float32, rt.allocator.device, True, arrays["x"])
        y = make_tensor((B, T, C), np.float32, rt.allocator.device, True, arrays["y"])
        out = MulOp()(x, y)
        return {"arrays": arrays, 
                "loss": out, 
                "fetch_map": {"out": out}, 
                "grad_map": {"x": x, "y": y},
        }
    def scalar_loss_fn(arrs):
        return float(np.sum(arrs["x"] * arrs["y"]))
    return check_case(name, build_numpy, build_backend, sampled_inputs=["x", "y"], scalar_loss_fn=scalar_loss_fn)

def run_gather_case():
    name = "Gather"
    B, T, V, C = 2, 3, 11, 4
    def build_numpy(rng):
        idx = rng.integers(0, V, size=(B, T), dtype=np.int32)
        w = (rng.standard_normal((V, C)) * 0.1).astype(np.float32)
        out = w[idx]
        return {"idx": idx, "w": w, "out": out}
    def build_backend(rt, rng, ref):
        arrays = {"idx": ref["idx"].copy(), "w": ref["w"].copy()}
        idx = make_tensor((B, T), np.int32, rt.allocator.device, False, arrays["idx"])
        w = make_tensor((V, C), np.float32, rt.allocator.device, True, arrays["w"])
        out = GatherOp()(idx, w)
        return {"arrays": arrays, 
            "loss": out, 
            "fetch_map": {"out": out},
            "grad_map": {"w": w},
        }
    def scalar_loss_fn(arrs):
        return float(np.sum(arrs["w"][arrs["idx"]]))
    return check_case(name, build_numpy, build_backend, sampled_inputs=["w"], scalar_loss_fn=scalar_loss_fn)

def run_fused_linear_add_case():
    name = "FusedLinearAdd"
    B, T, C = 2, 3, 4
    def build_numpy(rng):
        x = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
        act = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
        w = (rng.standard_normal((C, C)) * 0.1).astype(np.float32)
        b = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
        out = fused_linear_add_np(x, act, w, b)
        return {"x": x, "act": act, "w": w, "b": b, "out": out}
    def build_backend(rt, rng, ref):
        arrays = {k: ref[k].copy() for k in ("x", "act", "w", "b")}
        x = make_tensor((B, T, C), np.float32, rt.allocator.device, True, arrays["x"])
        act = make_tensor((B, T, C), np.float32, rt.allocator.device, True, arrays["act"])
        w = make_tensor((C, C), np.float32, rt.allocator.device, True, arrays["w"])
        b = make_tensor((C,), np.float32, rt.allocator.device, True, arrays["b"])
        out = FusedLinearAddOp()(x, act, w, b)
        return {"arrays": arrays, 
            "loss": out, 
            "fetch_map": {"out": out}, 
            "grad_map": {"x": x, "act": act, "w": w, "b": b},
        }
    def scalar_loss_fn(arrs):
        return float(np.sum(fused_linear_add_np(arrs["x"], arrs["act"], arrs["w"], arrs["b"])))
    return check_case(name, build_numpy, build_backend, sampled_inputs=["x", "act", "w", "b"], scalar_loss_fn=scalar_loss_fn, out_atol=1e-4, out_rtol=1e-4)

def run_fused_ln_mlp_case():
    name = "FusedLayerNormMLP"
    B, T, C = 1, 3, 4
    C4 = 16
    def build_numpy(rng):
        x = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
        ln_w = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
        ln_b = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
        w1 = (rng.standard_normal((C, C4)) * 0.1).astype(np.float32)
        b1 = (rng.standard_normal((C4,)) * 0.1).astype(np.float32)
        w2 = (rng.standard_normal((C4, C)) * 0.1).astype(np.float32)
        b2 = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
        out, _ = fused_ln_mlp_np(x, ln_w, ln_b, w1, b1, w2, b2, eps=1e-5)
        return {"x": x, "ln_w": ln_w, "ln_b": ln_b, "w1": w1, "b1": b1, "w2": w2, "b2": b2, "out": out}
    def build_backend(rt, rng, ref):
        arrays = {k: ref[k].copy() for k in ("x", "ln_w", "ln_b", "w1", "b1", "w2", "b2")}
        x = make_tensor((B, T, C), np.float32, rt.allocator.device, True, arrays["x"])
        ln_w = make_tensor((C,), np.float32, rt.allocator.device, True, arrays["ln_w"])
        ln_b = make_tensor((C,), np.float32, rt.allocator.device, True, arrays["ln_b"])
        w1 = make_tensor((C, C4), np.float32, rt.allocator.device, True, arrays["w1"])
        b1 = make_tensor((C4,), np.float32, rt.allocator.device, True, arrays["b1"])
        w2 = make_tensor((C4, C), np.float32, rt.allocator.device, True, arrays["w2"])
        b2 = make_tensor((C,), np.float32, rt.allocator.device, True, arrays["b2"])
        out = FusedLayerNormMLPOp(eps=1e-5)(x, ln_w, ln_b, w1, b1, w2, b2)
        return {"arrays": arrays, 
            "loss": out, 
            "fetch_map": {"out": out}, 
            "grad_map": {"x": x, "ln_w": ln_w, "ln_b": ln_b, "w1": w1, "b1": b1, "w2": w2, "b2": b2},
        }
    def scalar_loss_fn(arrs):
        out, _ = fused_ln_mlp_np(arrs["x"], arrs["ln_w"], arrs["ln_b"], arrs["w1"], arrs["b1"], arrs["w2"], arrs["b2"], eps=1e-5)
        return float(np.sum(out))
    return check_case(name, build_numpy, build_backend, sampled_inputs=["x", "ln_w", "ln_b", "w1", "b1", "w2", "b2"],
        scalar_loss_fn=scalar_loss_fn, out_atol=2e-4, out_rtol=2e-4, device_tolerances={"mps": (3e-3, 1e-2),},)

def run_fused_ln_fa_case():
    name = "FusedLayerNormFlashAttention"
    B, T, C, H = 1, 3, 4, 2
    def build_numpy(rng):
        x = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
        ln_w = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
        ln_b = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
        wq = (rng.standard_normal((C, C)) * 0.1).astype(np.float32)
        bq = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
        wk = (rng.standard_normal((C, C)) * 0.1).astype(np.float32)
        bk = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
        wv = (rng.standard_normal((C, C)) * 0.1).astype(np.float32)
        bv = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
        out, _ = fused_ln_fa_np(x, ln_w, ln_b, wq, bq, wk, bk, wv, bv, n_head=H, eps=1e-5, is_causal=True)
        return {
            "x": x, "ln_w": ln_w, "ln_b": ln_b,
            "wq": wq, "bq": bq, "wk": wk, "bk": bk, "wv": wv, "bv": bv,
            "out": out,
        }
    def build_backend(rt, rng, ref):
        arrays = {k: ref[k].copy() for k in ("x", "ln_w", "ln_b", "wq", "bq", "wk", "bk", "wv", "bv")}
        x = make_tensor((B, T, C), np.float32, rt.allocator.device, True, arrays["x"])
        ln_w = make_tensor((C,), np.float32, rt.allocator.device, True, arrays["ln_w"])
        ln_b = make_tensor((C,), np.float32, rt.allocator.device, True, arrays["ln_b"])
        wq = make_tensor((C, C), np.float32, rt.allocator.device, True, arrays["wq"])
        bq = make_tensor((C,), np.float32, rt.allocator.device, True, arrays["bq"])
        wk = make_tensor((C, C), np.float32, rt.allocator.device, True, arrays["wk"])
        bk = make_tensor((C,), np.float32, rt.allocator.device, True, arrays["bk"])
        wv = make_tensor((C, C), np.float32, rt.allocator.device, True, arrays["wv"])
        bv = make_tensor((C,), np.float32, rt.allocator.device, True, arrays["bv"])
        out = FusedLayerNormFlashAttentionOp(n_head=H, eps=1e-5, is_causal=True)(
            x, ln_w, ln_b, wq, bq, wk, bk, wv, bv
        )
        return {"arrays": arrays, 
            "loss": out, 
            "fetch_map": {"out": out}, 
            "grad_map": {"x": x, "ln_w": ln_w, "ln_b": ln_b, "wq": wq, "bq": bq, "wk": wk, "bk": bk, "wv": wv, "bv": bv},
        }
    def scalar_loss_fn(arrs):
        out, _ = fused_ln_fa_np(arrs["x"], arrs["ln_w"], arrs["ln_b"], arrs["wq"], arrs["bq"], arrs["wk"], arrs["bk"], arrs["wv"], arrs["bv"], n_head=H, eps=1e-5, is_causal=True)
        return float(np.sum(out))
    return check_case(name, build_numpy, build_backend, sampled_inputs=["x", "ln_w", "ln_b", "wq", "bq", "wk", "bk", "wv", "bv"],
        scalar_loss_fn=scalar_loss_fn, out_atol=3e-4, out_rtol=3e-4,)

def run_fused_linear_ce_case():
    name = "FusedLinearCrossEntropy"
    B, T, V, C = 2, 3, 11, 4
    def build_numpy(rng):
        x = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
        w = (rng.standard_normal((V, C)) * 0.1).astype(np.float32)
        targets = rng.integers(0, V, size=(B, T), dtype=np.int32)
        out = np.asarray([cross_entropy_np(x @ w.T, targets)], dtype=np.float32)
        return {"x": x, "w": w, "targets": targets, "out": out}
    def build_backend(rt, rng, ref):
        arrays = {k: ref[k].copy() for k in ("x", "w", "targets")}
        x = make_tensor((B, T, C), np.float32, rt.allocator.device, True, arrays["x"])
        w = make_tensor((V, C), np.float32, rt.allocator.device, True, arrays["w"])
        targets = make_tensor((B, T), np.int32, rt.allocator.device, False, arrays["targets"])
        out = FusedLinearCrossEntropyOp()(x, w, targets)
        return {"arrays": arrays, 
                "loss": out, 
                "fetch_map": {"out": out}, 
                "grad_map": {"x": x, "w": w, "targets": targets},
        }
    def scalar_loss_fn(arrs):
        logits = arrs["x"] @ arrs["w"].T
        return float(cross_entropy_np(logits, arrs["targets"]))
    return check_case(name, build_numpy, build_backend, sampled_inputs=["x", "w"],
        scalar_loss_fn=scalar_loss_fn, out_atol=3e-4, out_rtol=3e-4,)

# ------------------------------------------------------------
# Tiny GPT2 end-to-end
# ------------------------------------------------------------
def iter_gpt2_named_tensors(model):
    yield "tok_emb.weight", model.tok_emb.weight
    yield "pos_emb.weight", model.pos_emb.weight
    for i, blk in enumerate(model.blocks):
        p = f"blocks[{i}]"
        yield f"{p}.ln1.gamma", blk.ln1.gamma
        yield f"{p}.ln1.beta", blk.ln1.beta
        yield f"{p}.attn.q_proj.weight", blk.attn.q_proj.weight
        yield f"{p}.attn.q_proj.bias", blk.attn.q_proj.bias
        yield f"{p}.attn.k_proj.weight", blk.attn.k_proj.weight
        yield f"{p}.attn.k_proj.bias", blk.attn.k_proj.bias
        yield f"{p}.attn.v_proj.weight", blk.attn.v_proj.weight
        yield f"{p}.attn.v_proj.bias", blk.attn.v_proj.bias
        yield f"{p}.attn.proj.weight", blk.attn.proj.weight
        yield f"{p}.attn.proj.bias", blk.attn.proj.bias
        yield f"{p}.ln2.gamma", blk.ln2.gamma
        yield f"{p}.ln2.beta", blk.ln2.beta
        yield f"{p}.mlp.fc1.weight", blk.mlp.fc1.weight
        yield f"{p}.mlp.fc1.bias", blk.mlp.fc1.bias
        yield f"{p}.mlp.fc2.weight", blk.mlp.fc2.weight
        yield f"{p}.mlp.fc2.bias", blk.mlp.fc2.bias
    yield "lm_head.weight", model.lm_head.weight

def run_tiny_gpt2_case():
    print("\n[Tiny GPT2]")
    cfg = dict(B=2, T=32, V=1024, C=512, n_head=8, n_layer=2, dtype=np.float32)
    # cfg = dict(B=1, T=2, V=11, C=128, n_head=2, n_layer=1, dtype=np.float32)
    rng = np.random.default_rng(SEED + 99)
    # Build a CPU reference model just to get exact tensor shapes.
    ref_model = GPT2(cfg["V"], cfg["C"], cfg["T"], cfg["n_head"], cfg["B"], cfg["n_layer"], cfg["dtype"], "cpu")
    param_arrays = {}
    for name, tensor in iter_gpt2_named_tensors(ref_model):
        param_arrays[name] = (rng.standard_normal(tensor.shape).astype(np.float32) * 0.02).copy()
    input_ids = rng.integers(0, cfg["V"], size=(cfg["B"], cfg["T"]), dtype=np.int32)
    pos_ids = np.tile(np.arange(cfg["T"], dtype=np.int32), (cfg["B"], 1))
    labels = rng.integers(0, cfg["V"], size=(cfg["B"], cfg["T"]), dtype=np.int32)
    def numpy_gpt2_loss(local):
        x = local["tok_emb.weight"][input_ids] + local["pos_emb.weight"][pos_ids]
        for i in range(cfg["n_layer"]):
            p = f"blocks[{i}]"
            attn_out, _ = fused_ln_fa_np(
                x,
                local[f"{p}.ln1.gamma"],
                local[f"{p}.ln1.beta"],
                local[f"{p}.attn.q_proj.weight"],
                local[f"{p}.attn.q_proj.bias"],
                local[f"{p}.attn.k_proj.weight"],
                local[f"{p}.attn.k_proj.bias"],
                local[f"{p}.attn.v_proj.weight"],
                local[f"{p}.attn.v_proj.bias"],
                n_head=cfg["n_head"],
                eps=1e-6,
                is_causal=True,
            )
            x = fused_linear_add_np(x, attn_out, local[f"{p}.attn.proj.weight"], local[f"{p}.attn.proj.bias"])
            mlp_out, _ = fused_ln_mlp_np(
                x,
                local[f"{p}.ln2.gamma"],
                local[f"{p}.ln2.beta"],
                local[f"{p}.mlp.fc1.weight"],
                local[f"{p}.mlp.fc1.bias"],
                local[f"{p}.mlp.fc2.weight"],
                local[f"{p}.mlp.fc2.bias"],
                eps=1e-6,
            )
            x = x + mlp_out
        return cross_entropy_np(x @ local["lm_head.weight"].T, labels)
    # Sample a few gradients only, to keep runtime reasonable.
    probe_names = [
        "tok_emb.weight",
        "pos_emb.weight",
        "blocks[0].attn.q_proj.weight",
        "blocks[0].attn.proj.weight",
        "blocks[0].mlp.fc1.weight",
        "blocks[0].mlp.fc2.weight",
        "lm_head.weight",
    ]
    for device in ("cpu", "mps"):
        rt = reset_runtime(device)
        model = GPT2(cfg["V"], cfg["C"], cfg["T"], cfg["n_head"], cfg["B"], cfg["n_layer"], cfg["dtype"], device)
        # Force identical weights.
        for name, tensor in iter_gpt2_named_tensors(model):
            tensor.init_val = param_arrays[name].copy()
        input_ids_t = make_tensor((cfg["B"], cfg["T"]), np.int32, device, False, input_ids)
        pos_ids_t = make_tensor((cfg["B"], cfg["T"]), np.int32, device, False, pos_ids)
        labels_t = make_tensor((cfg["B"], cfg["T"]), np.int32, device, False, labels)
        loss = model(input_ids_t, pos_ids_t, targets=labels_t)
        fetched, grads, fwd_ms, bwd_ms = run_backend_case(rt, loss, {"loss": loss})
        print(f"  {device:4s} fwd={fwd_ms:.3f}ms bwd={bwd_ms:.3f}ms")
        ref_loss = numpy_gpt2_loss(param_arrays)
        assert_close(f"{device} loss", fetched["loss"], np.asarray([ref_loss], dtype=np.float32), atol=3e-4, rtol=3e-4)
        # Numerical gradient sampling on selected parameters.
        backend_grads = {name: copy_to_host(rt, tensor.grad) for name, tensor in iter_gpt2_named_tensors(model) if tensor.grad is not None}
        for pname in probe_names:
            idxs = sample_flat_indices(param_arrays[pname].shape, 3, seed=SEED + hash((device, pname)) % 10000)
            def loss_fn(local):
                return numpy_gpt2_loss(local)
            fd = finite_diff_scalar(loss_fn, param_arrays, pname, idxs, eps=1e-3)
            got = backend_grads[pname].reshape(-1)[idxs]
            assert_close(f"{device} {pname} sampled grad", got, fd, atol=GRAD_ATOL, rtol=GRAD_RTOL)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # print("NumPy-oracle correctness and timing suite")
    run_add_case()
    run_mul_case()
    run_gather_case()
    run_fused_linear_add_case()
    run_fused_ln_mlp_case()
    run_fused_ln_fa_case()
    run_fused_linear_ce_case()
    run_tiny_gpt2_case()
    print("\nAll checks passed.")
if __name__ == "__main__":
    main()