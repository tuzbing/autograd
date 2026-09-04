#!/usr/bin/env python3
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parent))

from op import FusedLayerNormFlashAttentionOp
from verify_model import fused_ln_fa_np, make_tensor, reset_runtime


def read_float32(rt, st):
    return np.array(rt.backend.ctx.numpy_view_float32(st.ptr.buffer_id, st.ptr.offset, st.shape)).copy()


def read_flat_float32(rt, st, length):
    return np.array(rt.backend.ctx.numpy_view_float32(st.ptr.buffer_id, st.ptr.offset, (length,))).copy()


def main():
    seed = 1234
    B, T, C, H = 1, 3, 4, 2
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
    cpu_x_norm = cpu_cache["x_norm"]
    cpu_k = cpu_cache["k"].reshape(B, T, H, D).transpose(0, 2, 1, 3)

    scores = np.matmul(
        cpu_cache["q"].reshape(B, T, H, D).transpose(0, 2, 1, 3),
        cpu_k.transpose(0, 1, 3, 2),
    ) / np.sqrt(D)
    scores = scores.copy()
    scores[..., np.triu(np.ones((T, T), dtype=bool), 1)] = -np.inf
    row_max = np.max(scores, axis=-1, keepdims=True)
    cpu_lse = (row_max + np.log(np.sum(np.exp(scores - row_max), axis=-1, keepdims=True))).squeeze(-1)

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
    dval_st = scratch["d_val"]

    rt.forward()
    rt.zero_grad()
    rt.backend.fill(out.grad, 1.0)

    backend = rt.backend
    backend.begin_pass()

    backend.ctx.dispatch_sdpa(
        "fused_ln_fa_project_kv_d64_kernel",
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

    mean_offset = lse_st.ptr.offset
    inv_std_offset = lse_st.ptr.offset + (B * T * 4)
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
    backend.ctx.dispatch_with_args(
        "fa_bwd_calc_d_kernel",
        [out.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, dval_st.ptr.buffer_id],
        [out.storage.ptr.offset, out.grad.storage.ptr.offset, dval_st.ptr.offset],
        [B, H, T, C],
        B * H * T,
    )
    backend.ctx.dispatch_flash_attn(
        "fa_min_recompute_lse_d64_kernel",
        [dq_st.ptr.buffer_id, wqt.storage.ptr.buffer_id, bqt.storage.ptr.buffer_id, k_st.ptr.buffer_id, lse_st.ptr.buffer_id],
        [dq_st.ptr.offset, wqt.storage.ptr.offset, bqt.storage.ptr.offset, k_st.ptr.offset, lse_st.ptr.offset],
        [B, H, T, C, int(True)],
        ((T + 32 - 1) // 32) * B * H,
    )
    backend.end_pass()
    backend.synchronize()

    x_norm_got = read_flat_float32(rt, dq_st, B * T * C).reshape(B, T, C)
    wq_got = read_float32(rt, wqt.storage)
    bq_got = read_float32(rt, bqt.storage)
    k_got = read_float32(rt, k_st)
    lse_got = read_float32(rt, lse_st)

    print(f"x_norm_tmp max_abs={float(np.max(np.abs(x_norm_got - cpu_x_norm))):.6e}")
    print(f"wq max_abs={float(np.max(np.abs(wq_got - wq))):.6e}")
    print(f"bq max_abs={float(np.max(np.abs(bq_got - bq))):.6e}")
    print(f"k max_abs={float(np.max(np.abs(k_got - cpu_k))):.6e}")
    print(f"lse max_abs={float(np.max(np.abs(lse_got - cpu_lse))):.6e}")
    print(f"script_path={Path(__file__).resolve()}")


if __name__ == "__main__":
    main()
