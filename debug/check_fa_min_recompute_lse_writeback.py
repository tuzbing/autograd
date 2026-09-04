#!/usr/bin/env python3
import gc
import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parent))

from op import FusedLayerNormFlashAttentionOp
import runtime
import metal_ext
from runtime import Runtime
from tensor import Tensor
from verify_model import fused_ln_fa_np

print(runtime.__file__)
print(metal_ext.__file__)

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


def read_float32(rt, st):
    return np.array(rt.backend.ctx.numpy_view_float32(st.ptr.buffer_id, st.ptr.offset, st.shape)).copy()


def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


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


def print_section(title):
    print(f"\n=== {title} ===")


def dump_lse_rows(lse_got, cpu_lse):
    lse_got = np.asarray(lse_got)
    cpu_lse = np.asarray(cpu_lse)
    B, H, T = lse_got.shape
    for b in range(B):
        for h in range(H):
            print(f"lse[b={b}, h={h}] got={lse_got[b, h]}")
            print(f"lse[b={b}, h={h}] cpu={cpu_lse[b, h]}")
            print(f"lse[b={b}, h={h}] diff={lse_got[b, h] - cpu_lse[b, h]}")
            print(f"lse[b={b}, h={h}] max_abs={max_abs_diff(lse_got[b, h], cpu_lse[b, h]):.6e}")
            for sg in range(min(4, T)):
                q_row0 = sg
                q_row1 = sg + 16
                print(
                    f"  sg={sg} q_row0={q_row0} valid_q0={q_row0 < T} "
                    f"q_row1={q_row1} valid_q1={q_row1 < T} "
                    f"got0={lse_got[b, h, q_row0] if q_row0 < T else 'NA'} "
                    f"cpu0={cpu_lse[b, h, q_row0] if q_row0 < T else 'NA'}"
                )


def main():
    seed = 1234
    # B, T, C, H = 1, 65, 65, 1
    # B, T, C, H = 1, 3, 4, 2
    # B, T, C, H = 1, 3, 12, 1
    B, T, C, H = 1, 5, 128, 2
    D = C // H
    eps = 1e-5
    eps_bits = struct.unpack("<i", struct.pack("<f", np.float32(eps)))[0]

    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)
    ln_w = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
    ln_b = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
    wq = (rng.standard_normal((C, C)) * 0.1).astype(np.float32)
    bq = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
    wk = (rng.standard_normal((C, C)) * 0.1).astype(np.float32)
    bk = (rng.standard_normal((C,)) * 0.1).astype(np.float32)
    wv = (rng.standard_normal((C, C)) * 0.1).astype(np.float32)
    bv = (rng.standard_normal((C,)) * 0.1).astype(np.float32)

    cpu_out, cpu_cache = fused_ln_fa_np(x, ln_w, ln_b, wq, bq, wk, bk, wv, bv, n_head=H, eps=eps, is_causal=True)
    cpu_x_norm = cpu_cache["x_norm"].reshape(B, T, C)
    q_cpu = cpu_cache["q"].reshape(B, T, H, D).transpose(0, 2, 1, 3)
    k_cpu = cpu_cache["k"].reshape(B, T, H, D).transpose(0, 2, 1, 3)
    v_cpu = cpu_cache["v"].reshape(B, T, H, D).transpose(0, 2, 1, 3)
    scores = np.matmul(q_cpu, k_cpu.transpose(0, 1, 3, 2)) / np.sqrt(D)
    scores = scores.copy()
    scores[..., np.triu(np.ones((T, T), dtype=bool), 1)] = -np.inf
    row_max = np.max(scores, axis=-1, keepdims=True)
    cpu_lse = (row_max + np.log(np.sum(np.exp(scores - row_max), axis=-1, keepdims=True))).squeeze(-1)
    probs_cpu = np.exp(scores - row_max)
    probs_cpu /= np.sum(probs_cpu, axis=-1, keepdims=True)
    out_grad_cpu = np.ones_like(cpu_out).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    work_cpu = np.matmul(out_grad_cpu, v_cpu.transpose(0, 1, 3, 2))
    tmp_cpu = probs_cpu * work_cpu
    row_sum_cpu = np.sum(tmp_cpu, axis=-1, keepdims=True)
    work_cpu = (work_cpu - row_sum_cpu) * probs_cpu * (1.0 / np.sqrt(D))
    dq_cpu = np.matmul(work_cpu, k_cpu).transpose(0, 2, 1, 3).reshape(B, T, C)
    dk_cpu = np.matmul(work_cpu.transpose(0, 1, 3, 2), q_cpu).transpose(0, 2, 1, 3).reshape(B, T, C)
    dv_cpu = np.matmul(probs_cpu.transpose(0, 1, 3, 2), out_grad_cpu).transpose(0, 2, 1, 3).reshape(B, T, C)
    d_val_cpu = np.sum(cpu_out.reshape(B, T, H, D).transpose(0, 2, 1, 3), axis=-1)
    cpu_mean = np.mean(x, axis=-1)
    cpu_inv_std = 1.0 / np.sqrt(np.var(x, axis=-1) + eps)
    dx_norm_cpu = (
        np.matmul(dq_cpu.reshape(B * T, C), wq.T)
        + np.matmul(dk_cpu.reshape(B * T, C), wk.T)
        + np.matmul(dv_cpu.reshape(B * T, C), wv.T)
    ).reshape(B, T, C)
    cpu_x_hat = (x - cpu_mean[..., None]) * cpu_inv_std[..., None]
    dx_norm_scaled_cpu = dx_norm_cpu * ln_w[None, None, :]
    cpu_dx = (
        (
            np.float32(C) * dx_norm_scaled_cpu
            - np.sum(dx_norm_scaled_cpu, axis=-1, keepdims=True)
            - cpu_x_hat * np.sum(dx_norm_scaled_cpu * cpu_x_hat, axis=-1, keepdims=True)
        )
        * cpu_inv_std[..., None]
        / np.float32(C)
    ).astype(np.float32)
    cpu_dln_w = np.sum(dx_norm_cpu * cpu_x_hat, axis=(0, 1))
    cpu_dln_b = np.sum(dx_norm_cpu, axis=(0, 1))

    rt = reset_runtime("mps")

    def alloc_tensor(shape, arr, requires_grad=True):
        return make_tensor(shape, np.float32, "mps", requires_grad, np.asarray(arr, dtype=np.float32))

    xt = alloc_tensor((B, T, C), x)
    ln_wt = alloc_tensor((C,), ln_w)
    ln_bt = alloc_tensor((C,), ln_b)
    wqt = alloc_tensor((C, C), wq)
    bqt = alloc_tensor((C,), bq)
    wkt = alloc_tensor((C, C), wk)
    bkt = alloc_tensor((C,), bk)
    wvt = alloc_tensor((C, C), wv)
    bvt = alloc_tensor((C,), bv)
    out = FusedLayerNormFlashAttentionOp(n_head=H, eps=eps, is_causal=True)(xt, ln_wt, ln_bt, wqt, bqt, wkt, bkt, wvt, bvt)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()

    node = next(n for n in rt.topo if type(n.op).__name__ == "FusedLayerNormFlashAttentionOp")
    scratch = node.backward.keywords["scratch"]
    q_st = scratch["q"]
    k_st = scratch["k"]
    v_st = scratch["v"]
    lse_st = scratch["lse"]
    dq_st = scratch["dq"]
    dk_st = scratch["dk"]
    dv_st = scratch["dv"]
    dval_st = scratch["d_val"]

    rt.forward()
    rt.zero_grad()
    rt.backend.fill(out.grad, 1.0)

    mean_offset = lse_st.ptr.offset
    inv_std_offset = lse_st.ptr.offset + (B * T * 4)

    backend = rt.backend
    print_section("forward / layernorm_compute_stats_kernel")
    backend.begin_pass()
    backend.ctx.dispatch_with_args(
        "layernorm_compute_stats_kernel",
        [xt.storage.ptr.buffer_id, lse_st.ptr.buffer_id, lse_st.ptr.buffer_id],
        [xt.storage.ptr.offset, mean_offset, inv_std_offset],
        [B * T, C, eps_bits],
        B * T * 1024,
    )
    backend.end_pass()
    backend.synchronize()

    fwd_x_got = read_float32(rt, xt.storage).reshape(B, T, C)
    fwd_mean_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, mean_offset, (B * T,))).copy().reshape(B, T)
    fwd_inv_std_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, inv_std_offset, (B * T,))).copy().reshape(B, T)
    print_max_abs("forward/layernorm_compute_stats/input x", fwd_x_got, x)
    print_max_abs("forward/layernorm_compute_stats/output mean", fwd_mean_got, cpu_mean)
    print_max_abs("forward/layernorm_compute_stats/output inv_std", fwd_inv_std_got, cpu_inv_std)

    print_section("forward / fa_opt_project_kv_kernel")
    backend.begin_pass()
    backend.ctx.dispatch_sdpa(
        "fa_opt_project_kv_kernel",
        [
            xt.storage.ptr.buffer_id,
            ln_wt.storage.ptr.buffer_id,
            ln_bt.storage.ptr.buffer_id,
            wkt.storage.ptr.buffer_id,
            bkt.storage.ptr.buffer_id,
            wvt.storage.ptr.buffer_id,
            bvt.storage.ptr.buffer_id,
            k_st.ptr.buffer_id,
            v_st.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            out.storage.ptr.buffer_id,
        ],
        [
            xt.storage.ptr.offset,
            ln_wt.storage.ptr.offset,
            ln_bt.storage.ptr.offset,
            wkt.storage.ptr.offset,
            bkt.storage.ptr.offset,
            wvt.storage.ptr.offset,
            bvt.storage.ptr.offset,
            k_st.ptr.offset,
            v_st.ptr.offset,
            mean_offset,
            inv_std_offset,
            out.storage.ptr.offset,
        ],
        [B, T, C, H, eps_bits],
        B * H * T,
    )
    backend.end_pass()
    backend.synchronize()

    fwd_x_got = read_float32(rt, xt.storage).reshape(B, T, C)
    fwd_mean_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, mean_offset, (B * T,))).copy().reshape(B, T)
    fwd_inv_std_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, inv_std_offset, (B * T,))).copy().reshape(B, T)
    fwd_k_got = read_float32(rt, k_st).reshape(B, H, T, D)
    fwd_v_got = read_float32(rt, v_st).reshape(B, H, T, D)
    print_max_abs("forward/fa_opt_project_kv/input x", fwd_x_got, x)
    print_max_abs("forward/fa_opt_project_kv/input ln_w", read_float32(rt, ln_wt.storage).reshape(C), ln_w)
    print_max_abs("forward/fa_opt_project_kv/input ln_b", read_float32(rt, ln_bt.storage).reshape(C), ln_b)
    print_max_abs("forward/fa_opt_project_kv/input mean", fwd_mean_got, cpu_mean)
    print_max_abs("forward/fa_opt_project_kv/input inv_std", fwd_inv_std_got, cpu_inv_std)
    print_max_abs("forward/fa_opt_project_kv/input wk", read_float32(rt, wkt.storage).reshape(C, C), wk)
    print_max_abs("forward/fa_opt_project_kv/input bk", read_float32(rt, bkt.storage).reshape(C), bk)
    print_max_abs("forward/fa_opt_project_kv/input wv", read_float32(rt, wvt.storage).reshape(C, C), wv)
    print_max_abs("forward/fa_opt_project_kv/input bv", read_float32(rt, bvt.storage).reshape(C), bv)
    print_max_abs("forward/fa_opt_project_kv/output k", fwd_k_got, k_cpu)
    print_max_abs("forward/fa_opt_project_kv/output v", fwd_v_got, v_cpu)

    print_section("forward / fa_opt_fwd_kernel")
    backend.begin_pass()
    fwd_q_blocks = (T + 32 - 1) // 32
    backend.ctx.dispatch_flash_attn_bwd(
        "fa_opt_fwd_kernel",
        [
            xt.storage.ptr.buffer_id,
            ln_wt.storage.ptr.buffer_id,
            ln_bt.storage.ptr.buffer_id,
            wqt.storage.ptr.buffer_id,
            bqt.storage.ptr.buffer_id,
            k_st.ptr.buffer_id,
            v_st.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            out.storage.ptr.buffer_id,
        ],
        [
            xt.storage.ptr.offset,
            ln_wt.storage.ptr.offset,
            ln_bt.storage.ptr.offset,
            wqt.storage.ptr.offset,
            bqt.storage.ptr.offset,
            k_st.ptr.offset,
            v_st.ptr.offset,
            mean_offset,
            inv_std_offset,
            out.storage.ptr.offset,
        ],
        [B, H, T, C, int(True), eps_bits],
        B * H * fwd_q_blocks,
    )
    backend.end_pass()
    backend.synchronize()

    fwd_out_got = read_float32(rt, out.storage).reshape(B, T, C)
    print_max_abs("forward/fa_opt_fwd_d64/input x", fwd_x_got, x)
    print_max_abs("forward/fa_opt_fwd_d64/input ln_w", read_float32(rt, ln_wt.storage).reshape(C), ln_w)
    print_max_abs("forward/fa_opt_fwd_d64/input ln_b", read_float32(rt, ln_bt.storage).reshape(C), ln_b)
    print_max_abs("forward/fa_opt_fwd_d64/input wq", read_float32(rt, wqt.storage).reshape(C, C), wq)
    print_max_abs("forward/fa_opt_fwd_d64/input bq", read_float32(rt, bqt.storage).reshape(C), bq)
    print_max_abs("forward/fa_opt_fwd_d64/input k", fwd_k_got, k_cpu)
    print_max_abs("forward/fa_opt_fwd_d64/input v", fwd_v_got, v_cpu)
    print_max_abs("forward/fa_opt_fwd_d64/input mean", fwd_mean_got, cpu_mean)
    print_max_abs("forward/fa_opt_fwd_d64/input inv_std", fwd_inv_std_got, cpu_inv_std)
    print_max_abs("forward/fa_opt_fwd_d64/output out", fwd_out_got, cpu_out)

    print_section("backward / fused_ln_fa_project_kv_d64_kernel")
    backend.begin_pass()
    backend.ctx.dispatch_sdpa(
        "fused_ln_fa_project_kv_kernel",
        [
            xt.storage.ptr.buffer_id,
            ln_wt.storage.ptr.buffer_id,
            ln_bt.storage.ptr.buffer_id,
            wkt.storage.ptr.buffer_id,
            bkt.storage.ptr.buffer_id,
            wvt.storage.ptr.buffer_id,
            bvt.storage.ptr.buffer_id,
            k_st.ptr.buffer_id,
            v_st.ptr.buffer_id,
        ],
        [
            xt.storage.ptr.offset,
            ln_wt.storage.ptr.offset,
            ln_bt.storage.ptr.offset,
            wkt.storage.ptr.offset,
            bkt.storage.ptr.offset,
            wvt.storage.ptr.offset,
            bvt.storage.ptr.offset,
            k_st.ptr.offset,
            v_st.ptr.offset,
        ],
        [B, T, C, H, eps_bits],
        B * H * T,
    )
    backend.end_pass()
    backend.synchronize()

    kv_x_got = read_float32(rt, xt.storage).reshape(B, T, C)
    kv_ln_w_got = read_float32(rt, ln_wt.storage).reshape(C)
    kv_ln_b_got = read_float32(rt, ln_bt.storage).reshape(C)
    kv_mean_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, mean_offset, (B * T,))).copy().reshape(B, T)
    kv_inv_std_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, inv_std_offset, (B * T,))).copy().reshape(B, T)
    kv_k_got = read_float32(rt, k_st).reshape(B, H, T, D)
    kv_v_got = read_float32(rt, v_st).reshape(B, H, T, D)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/input x", kv_x_got, x)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/input ln_w", kv_ln_w_got, ln_w)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/input ln_b", kv_ln_b_got, ln_b)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/output mean", kv_mean_got, cpu_mean)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/output inv_std", kv_inv_std_got, cpu_inv_std)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/input wk", read_float32(rt, wkt.storage).reshape(C, C), wk)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/input bk", read_float32(rt, bkt.storage).reshape(C), bk)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/input wv", read_float32(rt, wvt.storage).reshape(C, C), wv)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/input bv", read_float32(rt, bvt.storage).reshape(C), bv)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/output k", kv_k_got, k_cpu)
    print_max_abs("backward/fused_ln_fa_project_kv_d64/output v", kv_v_got, v_cpu)

    print_section("backward / fused_ln_fa_recompute_q_kernel")
    backend.begin_pass()
    backend.ctx.dispatch_sdpa(
        "fused_ln_fa_recompute_q_kernel",
        [
            xt.storage.ptr.buffer_id,
            ln_wt.storage.ptr.buffer_id,
            ln_bt.storage.ptr.buffer_id,
            wqt.storage.ptr.buffer_id,
            bqt.storage.ptr.buffer_id,
            q_st.ptr.buffer_id,
        ],
        [
            xt.storage.ptr.offset,
            ln_wt.storage.ptr.offset,
            ln_bt.storage.ptr.offset,
            wqt.storage.ptr.offset,
            bqt.storage.ptr.offset,
            q_st.ptr.offset,
        ],
        [B, T, C, H, D, eps_bits],
        B * H * T,
    )
    backend.end_pass()
    backend.synchronize()

    q_in_got = read_float32(rt, q_st).reshape(B, H, T, D)
    print_max_abs("backward/fused_ln_fa_recompute_q_kernel/input x", kv_x_got, x)
    print_max_abs("backward/fused_ln_fa_recompute_q_kernel/input ln_w", kv_ln_w_got, ln_w)
    print_max_abs("backward/fused_ln_fa_recompute_q_kernel/input ln_b", kv_ln_b_got, ln_b)
    print_max_abs("backward/fused_ln_fa_recompute_q_kernel/input wq", read_float32(rt, wqt.storage).reshape(C, C), wq)
    print_max_abs("backward/fused_ln_fa_recompute_q_kernel/input bq", read_float32(rt, bqt.storage).reshape(C), bq)
    print_max_abs("backward/fused_ln_fa_recompute_q_kernel/output q", q_in_got, q_cpu)

    print_section("backward / layernorm_compute_stats_kernel")
    backend.begin_pass()
    backend.ctx.dispatch_with_args(
        "layernorm_compute_stats_kernel",
        [xt.storage.ptr.buffer_id, lse_st.ptr.buffer_id, lse_st.ptr.buffer_id],
        [xt.storage.ptr.offset, mean_offset, inv_std_offset],
        [B * T, C, eps_bits],
        B * T * 1024,
    )
    backend.end_pass()
    backend.synchronize()

    mean_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, mean_offset, (B * T,))).copy().reshape(B, T)
    inv_std_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, inv_std_offset, (B * T,))).copy().reshape(B, T)
    print_max_abs("backward/layernorm_compute_stats/input x", read_float32(rt, xt.storage).reshape(B, T, C), x)
    print_max_abs("backward/layernorm_compute_stats/output mean", mean_got, cpu_mean)
    print_max_abs("backward/layernorm_compute_stats/output inv_std", inv_std_got, cpu_inv_std)

    print_section("backward / layernorm_forward_kernel_no_stats")
    backend.begin_pass()
    backend.ctx.dispatch_with_args(
        "layernorm_forward_kernel_no_stats",
        [
            xt.storage.ptr.buffer_id,
            ln_wt.storage.ptr.buffer_id,
            ln_bt.storage.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            dq_st.ptr.buffer_id,
        ],
        [
            xt.storage.ptr.offset,
            ln_wt.storage.ptr.offset,
            ln_bt.storage.ptr.offset,
            mean_offset,
            inv_std_offset,
            dq_st.ptr.offset,
        ],
        [B * T, C],
        B * T * 1024,
    )
    backend.end_pass()
    backend.synchronize()

    x_norm_got = read_float32(rt, dq_st).reshape(B, T, C)
    print_max_abs("backward/layernorm_forward_kernel_no_stats/input x", read_float32(rt, xt.storage).reshape(B, T, C), x)
    print_max_abs("backward/layernorm_forward_kernel_no_stats/input ln_w", read_float32(rt, ln_wt.storage).reshape(C), ln_w)
    print_max_abs("backward/layernorm_forward_kernel_no_stats/input ln_b", read_float32(rt, ln_bt.storage).reshape(C), ln_b)
    print_max_abs("backward/layernorm_forward_kernel_no_stats/input mean", mean_got, cpu_mean)
    print_max_abs("backward/layernorm_forward_kernel_no_stats/input inv_std", inv_std_got, cpu_inv_std)
    print_max_abs("backward/layernorm_forward_kernel_no_stats/output x_norm_tmp", x_norm_got, cpu_x_norm)

    print_section("backward / fa_bwd_calc_d_kernel")
    backend.begin_pass()
    backend.ctx.dispatch_with_args(
        "fa_bwd_calc_d_kernel",
        [out.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, dval_st.ptr.buffer_id],
        [out.storage.ptr.offset, out.grad.storage.ptr.offset, dval_st.ptr.offset],
        [B, H, T, C],
        B * H * T,
    )
    backend.end_pass()
    backend.synchronize()

    out_got = read_float32(rt, out.storage).reshape(B, T, C)
    grad_out_got = read_float32(rt, out.grad.storage).reshape(B, T, C)
    dval_got = read_float32(rt, dval_st).reshape(B, H, T)
    print_max_abs("backward/fa_bwd_calc_d_kernel/input out", out_got, cpu_out)
    print_max_abs("backward/fa_bwd_calc_d_kernel/input out.grad", grad_out_got, np.ones_like(cpu_out))
    print_max_abs("backward/fa_bwd_calc_d_kernel/output d_val", dval_got, d_val_cpu)

    print_section("backward / fa_min_recompute_lse_kernel")
    fa_min_x_in_got = read_float32(rt, dq_st).reshape(B, T, C)
    fa_min_wq_in_got = read_float32(rt, wqt.storage).reshape(C, C)
    fa_min_bq_in_got = read_float32(rt, bqt.storage).reshape(C)
    fa_min_k_in_got = read_float32(rt, k_st).reshape(B, H, T, D)
    fa_min_lse_in_got = read_float32(rt, lse_st).reshape(B, H, T)
    print_max_abs("backward/fa_min_recompute_lse/input x_norm_tmp", fa_min_x_in_got, cpu_x_norm)
    print_max_abs("backward/fa_min_recompute_lse/input wq", fa_min_wq_in_got, wq)
    print_max_abs("backward/fa_min_recompute_lse/input bq", fa_min_bq_in_got, bq)
    print_max_abs("backward/fa_min_recompute_lse/input k", fa_min_k_in_got, k_cpu)
    print_max_abs("backward/fa_min_recompute_lse/input lse", fa_min_lse_in_got, cpu_lse)
    backend.begin_pass()
    q_blocks = ((T + 32 - 1) // 32) * B * H
    backend.ctx.dispatch_flash_attn(
        "fa_min_recompute_lse_kernel",
        [dq_st.ptr.buffer_id, wqt.storage.ptr.buffer_id, bqt.storage.ptr.buffer_id, k_st.ptr.buffer_id, lse_st.ptr.buffer_id],
        [dq_st.ptr.offset, wqt.storage.ptr.offset, bqt.storage.ptr.offset, k_st.ptr.offset, lse_st.ptr.offset],
        [B, H, T, C, int(True)],
        q_blocks,
    )
    backend.end_pass()
    backend.synchronize()

    x_norm_got = read_float32(rt, dq_st).reshape(B, T, C)
    k_got = read_float32(rt, k_st).reshape(B, H, T, D)
    lse_got = read_float32(rt, lse_st).reshape(B, H, T)
    print_max_abs("backward/fa_min_recompute_lse/output x_norm_tmp", x_norm_got, cpu_x_norm)
    print_max_abs("backward/fa_min_recompute_lse/output k", k_got, k_cpu)
    print_max_abs("backward/fa_min_recompute_lse/output lse", lse_got, cpu_lse)
    # print_array_block("stage2 lse full", lse_got, cpu_lse)
    # dump_lse_rows(lse_got, cpu_lse)

    print_section("backward / fa_pure_bwd_dq_kernel")
    dq_in_q_got = read_float32(rt, q_st).reshape(B, H, T, D)
    dq_in_k_got = read_float32(rt, k_st).reshape(B, H, T, D)
    dq_in_v_got = read_float32(rt, v_st).reshape(B, H, T, D)
    dq_in_out_got = read_float32(rt, out.storage).reshape(B, T, C)
    dq_in_grad_got = read_float32(rt, out.grad.storage).reshape(B, T, C)
    dq_in_lse_got = read_float32(rt, lse_st).reshape(B, H, T)
    dq_in_dval_got = read_float32(rt, dval_st).reshape(B, H, T)
    print_max_abs("backward/fa_pure_bwd_dq/input q", dq_in_q_got, q_cpu)
    print_max_abs("backward/fa_pure_bwd_dq/input k", dq_in_k_got, k_cpu)
    print_max_abs("backward/fa_pure_bwd_dq/input v", dq_in_v_got, v_cpu)
    print_max_abs("backward/fa_pure_bwd_dq/input out", dq_in_out_got, cpu_out)
    print_max_abs("backward/fa_pure_bwd_dq/input out.grad", dq_in_grad_got, np.ones_like(cpu_out))
    print_max_abs("backward/fa_pure_bwd_dq/input lse", dq_in_lse_got, cpu_lse)
    print_max_abs("backward/fa_pure_bwd_dq/input d_val", dq_in_dval_got, d_val_cpu)
    # print("q")
    # print(np.asarray(dq_in_q_got))
    # print("k")
    # print(np.asarray(dq_in_k_got))
    # print("v")
    # print(np.asarray(dq_in_v_got))
    # print("out")
    # print(np.asarray(dq_in_out_got))
    # print("out.grad")
    # print(np.asarray(dq_in_grad_got))
    # print("lse")
    # print(np.asarray(dq_in_lse_got))
    # print("d_val")
    # print(np.asarray(dq_in_dval_got))

    backend.begin_pass()
    backend.ctx.dispatch_flash_attn_bwd(
        "fa_pure_bwd_dq_kernel",
        [
            q_st.ptr.buffer_id,
            k_st.ptr.buffer_id,
            v_st.ptr.buffer_id,
            out.storage.ptr.buffer_id,
            out.grad.storage.ptr.buffer_id,
            dq_st.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            dval_st.ptr.buffer_id,
        ],
        [
            q_st.ptr.offset,
            k_st.ptr.offset,
            v_st.ptr.offset,
            out.storage.ptr.offset,
            out.grad.storage.ptr.offset,
            dq_st.ptr.offset,
            lse_st.ptr.offset,
            dval_st.ptr.offset,
        ],
        [B, H, T, C, int(True)],
        q_blocks,
    )
    backend.end_pass()
    backend.synchronize()

    dq_got = read_float32(rt, dq_st).reshape(B, T, C)
    # dq_got = read_float32(rt, dq_st).reshape(B, H, T, D).transpose(0, 2, 1, 3).reshape(B, T, C)
    # print_array_block("dq full", dq_got, dq_cpu)
    print_max_abs("backward/fa_pure_bwd_dq/output dq", dq_got, dq_cpu)

    print_section("backward / fa_pure_bwd_dkv_kernel")
    backend.begin_pass()
    backend.ctx.dispatch_flash_attn_bwd(
        "fa_pure_bwd_dkv_kernel",
        [
            q_st.ptr.buffer_id,
            k_st.ptr.buffer_id,
            v_st.ptr.buffer_id,
            out.storage.ptr.buffer_id,
            out.grad.storage.ptr.buffer_id,
            dk_st.ptr.buffer_id,
            dv_st.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            dval_st.ptr.buffer_id,
        ],
        [
            q_st.ptr.offset,
            k_st.ptr.offset,
            v_st.ptr.offset,
            out.storage.ptr.offset,
            out.grad.storage.ptr.offset,
            dk_st.ptr.offset,
            dv_st.ptr.offset,
            lse_st.ptr.offset,
            dval_st.ptr.offset,
        ],
        [B, H, T, C, int(True)],
        q_blocks,
    )
    backend.end_pass()
    backend.synchronize()

    q_in_got = read_float32(rt, q_st).reshape(B, H, T, D)
    k_in_got = read_float32(rt, k_st).reshape(B, H, T, D)
    v_in_got = read_float32(rt, v_st).reshape(B, H, T, D)
    lse_in_got = read_float32(rt, lse_st).reshape(B, H, T)
    dval_in_got = read_float32(rt, dval_st).reshape(B, H, T)
    print_max_abs("backward/fa_pure_bwd_dkv/input q", q_in_got, q_cpu)
    print_max_abs("backward/fa_pure_bwd_dkv/input k", k_in_got, k_cpu)
    print_max_abs("backward/fa_pure_bwd_dkv/input v", v_in_got, v_cpu)
    print_max_abs("backward/fa_pure_bwd_dkv/input lse", lse_in_got, cpu_lse)
    print_max_abs("backward/fa_pure_bwd_dkv/input d_val", dval_in_got, d_val_cpu)
    # print("q")
    # print(np.asarray(dq_in_q_got))
    # print("k")
    # print(np.asarray(dq_in_k_got))
    # print("v")
    # print(np.asarray(dq_in_v_got))
    # print("out")
    # print(np.asarray(dq_in_out_got))
    # print("lse")
    # print(np.asarray(dq_in_lse_got))
    # print("d_val")
    # print(np.asarray(dq_in_dval_got))

    dk_got = read_float32(rt, dk_st).reshape(B, T, C)
    dv_got = read_float32(rt, dv_st).reshape(B, T, C)
    # print_array_block("dk full", dk_got, dk_cpu)
    # print_array_block("dv full", dv_got, dv_cpu)
    print(f"backward/fa_pure_bwd_dkv/output dk max_abs={float(np.max(np.abs(dk_got - dk_cpu))):.6e}")
    print(f"backward/fa_pure_bwd_dkv/output dv max_abs={float(np.max(np.abs(dv_got - dv_cpu))):.6e}")

    print_section("backward / layernorm_compute_stats_kernel")
    backend.begin_pass()
    backend.ctx.dispatch_with_args(
        "layernorm_compute_stats_kernel",
        [xt.storage.ptr.buffer_id, lse_st.ptr.buffer_id, lse_st.ptr.buffer_id],
        [xt.storage.ptr.offset, mean_offset, inv_std_offset],
        [B * T, C, eps_bits],
        B * T * 1024,
    )
    backend.ctx.dispatch_with_args(
        "layernorm_forward_kernel_no_stats",
        [
            xt.storage.ptr.buffer_id,
            ln_wt.storage.ptr.buffer_id,
            ln_bt.storage.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            q_st.ptr.buffer_id,
        ],
        [
            xt.storage.ptr.offset,
            ln_wt.storage.ptr.offset,
            ln_bt.storage.ptr.offset,
            mean_offset,
            inv_std_offset,
            q_st.ptr.offset,
        ],
        [B * T, C],
        B * T * 1024,
    )
    backend.end_pass()
    backend.synchronize()

    ln_x_got = read_float32(rt, xt.storage).reshape(B, T, C)
    ln_w_got = read_float32(rt, ln_wt.storage).reshape(C)
    ln_b_got = read_float32(rt, ln_bt.storage).reshape(C)
    mean2_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, mean_offset, (B * T,))).copy().reshape(B, T)
    inv_std2_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, inv_std_offset, (B * T,))).copy().reshape(B, T)
    x_norm2_got = read_float32(rt, q_st).reshape(B, T, C)
    print_max_abs("backward/layernorm_compute_stats/input x", ln_x_got, x)
    print_max_abs("backward/layernorm_compute_stats/input ln_w", ln_w_got, ln_w)
    print_max_abs("backward/layernorm_compute_stats/input ln_b", ln_b_got, ln_b)
    print_max_abs("backward/layernorm_compute_stats/output mean", mean2_got, cpu_mean)
    print_max_abs("backward/layernorm_compute_stats/output inv_std", inv_std2_got, cpu_inv_std)
    print_max_abs("backward/layernorm_forward_kernel_no_stats/output x_norm", x_norm2_got, cpu_x_norm)

    print_section("backward / dx accumulation with gemm_backward_dx_batched_mps")
    backend.fill(xt.grad, 0.0)
    backend.fill_storage(k_st, 0.0)
    dq_in_got = read_float32(rt, dq_st).reshape(B, T, C)
    dk_in_got = read_float32(rt, dk_st).reshape(B, T, C)
    dv_in_got = read_float32(rt, dv_st).reshape(B, T, C)
    wq_in_got = read_float32(rt, wqt.storage).reshape(C, C)
    wk_in_got = read_float32(rt, wkt.storage).reshape(C, C)
    wv_in_got = read_float32(rt, wvt.storage).reshape(C, C)
    k_zero_got = read_float32(rt, k_st).reshape(B, H, T, D)
    x_grad_zero_got = read_float32(rt, xt.grad.storage).reshape(B, T, C)
    print_max_abs("backward/gemm_backward_dx_batched_mps/input dq", dq_in_got, dq_cpu)
    print_max_abs("backward/gemm_backward_dx_batched_mps/input dk", dk_in_got, dk_cpu)
    print_max_abs("backward/gemm_backward_dx_batched_mps/input dv", dv_in_got, dv_cpu)
    print_max_abs("backward/gemm_backward_dx_batched_mps/input wq", wq_in_got, wq)
    print_max_abs("backward/gemm_backward_dx_batched_mps/input wk", wk_in_got, wk)
    print_max_abs("backward/gemm_backward_dx_batched_mps/input wv", wv_in_got, wv)
    print_max_abs("backward/gemm_backward_dx_batched_mps/output k cleared", k_zero_got, np.zeros_like(k_zero_got))
    print_max_abs("backward/gemm_backward_dx_batched_mps/output x.grad cleared", x_grad_zero_got, np.zeros_like(x_grad_zero_got))

    backend.begin_pass()
    backend.ctx.gemm_backward_dx_batched_mps(
        dq_st.ptr.buffer_id,
        dq_st.ptr.offset,
        wqt.storage.ptr.buffer_id,
        wqt.storage.ptr.offset,
        k_st.ptr.buffer_id,
        k_st.ptr.offset,
        1,
        B * T,
        C,
        C,
        0,
    )
    backend.ctx.gemm_backward_dx_batched_mps(
        dk_st.ptr.buffer_id,
        dk_st.ptr.offset,
        wkt.storage.ptr.buffer_id,
        wkt.storage.ptr.offset,
        k_st.ptr.buffer_id,
        k_st.ptr.offset,
        1,
        B * T,
        C,
        C,
        0,
    )
    backend.ctx.gemm_backward_dx_batched_mps(
        dv_st.ptr.buffer_id,
        dv_st.ptr.offset,
        wvt.storage.ptr.buffer_id,
        wvt.storage.ptr.offset,
        k_st.ptr.buffer_id,
        k_st.ptr.offset,
        1,
        B * T,
        C,
        C,
        0,
    )
    backend.end_pass()
    backend.synchronize()

    dx_norm_got = read_float32(rt, k_st).reshape(B, T, C)
    print_max_abs("backward/gemm_backward_dx_batched_mps/output dx_norm accum", dx_norm_got, dx_norm_cpu)

    print_section("backward / layernorm_backward_kernel_no_stats")
    backend.fill(xt.grad, 0.0)
    backend.fill(ln_wt.grad, 0.0)
    backend.fill(ln_bt.grad, 0.0)
    backend.begin_pass()
    backend.ctx.dispatch_with_args(
        "layernorm_backward_kernel_no_stats",
        [
            xt.storage.ptr.buffer_id,
            k_st.ptr.buffer_id,
            ln_wt.storage.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            lse_st.ptr.buffer_id,
            xt.grad.storage.ptr.buffer_id,
            ln_wt.grad.storage.ptr.buffer_id if ln_wt.requires_grad else 0,
            ln_bt.grad.storage.ptr.buffer_id if ln_bt.requires_grad else 0,
        ],
        [
            xt.storage.ptr.offset,
            k_st.ptr.offset,
            ln_wt.storage.ptr.offset,
            mean_offset,
            inv_std_offset,
            xt.grad.storage.ptr.offset,
            ln_wt.grad.storage.ptr.offset if ln_wt.requires_grad else 0,
            ln_bt.grad.storage.ptr.offset if ln_bt.requires_grad else 0,
        ],
        [B * T, C],
        B * T * 1024,
    )
    backend.end_pass()
    backend.synchronize()

    ln_x_got = read_float32(rt, xt.storage).reshape(B, T, C)
    dx_norm_in_got = read_float32(rt, k_st).reshape(B, T, C)
    mean_dx_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, mean_offset, (B * T,))).copy().reshape(B, T)
    inv_std_dx_got = np.array(rt.backend.ctx.numpy_view_float32(lse_st.ptr.buffer_id, inv_std_offset, (B * T,))).copy().reshape(B, T)
    dx_got = read_float32(rt, xt.grad.storage).reshape(B, T, C)
    dln_w_got = read_float32(rt, ln_wt.grad.storage).reshape(C)
    dln_b_got = read_float32(rt, ln_bt.grad.storage).reshape(C)
    print_max_abs("backward/layernorm_backward_kernel_no_stats/input x", ln_x_got, x)
    print_max_abs("backward/layernorm_backward_kernel_no_stats/input dx_norm", dx_norm_in_got, dx_norm_cpu)
    print_max_abs("backward/layernorm_backward_kernel_no_stats/input ln_w", read_float32(rt, ln_wt.storage).reshape(C), ln_w)
    print_max_abs("backward/layernorm_backward_kernel_no_stats/output mean", mean_dx_got, cpu_mean)
    print_max_abs("backward/layernorm_backward_kernel_no_stats/output inv_std", inv_std_dx_got, cpu_inv_std)
    print_max_abs("backward/layernorm_backward_kernel_no_stats/output x.grad", dx_got, cpu_dx)
    print_max_abs("backward/layernorm_backward_kernel_no_stats/output ln_w.grad", dln_w_got, cpu_dln_w)
    print_max_abs("backward/layernorm_backward_kernel_no_stats/output ln_b.grad", dln_b_got, cpu_dln_b)
    print(f"script_path={Path(__file__).resolve()}")


if __name__ == "__main__":
    main()
