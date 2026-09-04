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
from module import GPT2
from op import FusedLinearAddOp, FusedLayerNormFlashAttentionOp, FusedLayerNormMLPOp
from runtime import Runtime
from verify_model import make_tensor, iter_gpt2_named_tensors
from check_tiny_gpt2_kernel import CFG, build_param_arrays, build_inputs
FLOAT_ATOL = 3e-4
FLOAT_RTOL = 3e-4
GRAD_ATOL = 2e-2
GRAD_RTOL = 2e-2
def reset_runtime(device):
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device)
    Runtime._default_runtime = rt
    return rt
def copy_to_host(rt, tensor):
    return rt.backend.copy_to_host(tensor).copy()
def compare(name, got, ref, atol=FLOAT_ATOL, rtol=FLOAT_RTOL, fail_fast=True):
    got = np.asarray(got)
    ref = np.asarray(ref)
    ok = np.allclose(got, ref, atol=atol, rtol=rtol, equal_nan=True)
    max_abs = float(np.max(np.abs(got - ref))) if got.size else 0.0
    max_rel = float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-12))) if got.size else 0.0
    print(
        f"  {'PASS' if ok else 'FAIL'} {name}: "
        f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
        f"atol={atol}, rtol={rtol}"
    )
    if fail_fast and not ok:
        print(f"    got={got}")
        print(f"    ref={ref}")
        raise AssertionError(f"{name} mismatch")
    return ok
def init_model_params(model, param_arrays):
    for name, tensor in iter_gpt2_named_tensors(model):
        tensor.init_val = param_arrays[name].copy()
def build_live_graph(device, param_arrays, input_ids, pos_ids, labels):
    """
    Build a live graph and retain tensor refs needed for tracing.
    No stale post-hoc snapshots are used.
    """
    rt = reset_runtime(device)
    model = GPT2(
        CFG["V"], CFG["C"], CFG["T"], CFG["n_head"], CFG["B"], CFG["n_layer"], CFG["dtype"], device
    )
    init_model_params(model, param_arrays)
    input_ids_t = make_tensor((CFG["B"], CFG["T"]), np.int32, device, False, input_ids)
    pos_ids_t = make_tensor((CFG["B"], CFG["T"]), np.int32, device, False, pos_ids)
    labels_t = make_tensor((CFG["B"], CFG["T"]), np.int32, device, False, labels)
    blocks = []
    x = model.tok_emb(input_ids_t) + model.pos_emb(pos_ids_t)
    for i, blk in enumerate(model.blocks):
        rec = {"block_idx": i}
        rec["x"] = x
        attn_out = FusedLayerNormFlashAttentionOp(
            n_head=blk.attn.n_head,
            eps=blk.ln1.eps,
            is_causal=True,
        )(
            x,
            blk.ln1.gamma,
            blk.ln1.beta,
            blk.attn.q_proj.weight,
            blk.attn.q_proj.bias,
            blk.attn.k_proj.weight,
            blk.attn.k_proj.bias,
            blk.attn.v_proj.weight,
            blk.attn.v_proj.bias,
        )
        rec["act"] = attn_out
        rec["w"] = blk.attn.proj.weight
        rec["b"] = blk.attn.proj.bias
        x_after_attn = FusedLinearAddOp()(
            x,
            attn_out,
            blk.attn.proj.weight,
            blk.attn.proj.bias,
        )
        rec["out"] = x_after_attn
        mlp_out = FusedLayerNormMLPOp(eps=blk.ln2.eps)(
            x_after_attn,
            blk.ln2.gamma,
            blk.ln2.beta,
            blk.mlp.fc1.weight,
            blk.mlp.fc1.bias,
            blk.mlp.fc2.weight,
            blk.mlp.fc2.bias,
        )
        x = x_after_attn + mlp_out
        blocks.append(rec)
    loss = x.fused_linear_cross_entropy(model.lm_head.weight, labels_t)
    rt.loss = loss
    rt.build_topo(loss)
    rt.bind_memory()
    return rt, loss, blocks
def cpu_fused_linear_add_oracle(x_np, act_np, w_np, b_np, out_grad_np=None):
    x_np = np.asarray(x_np, dtype=np.float32)
    act_np = np.asarray(act_np, dtype=np.float32)
    w_np = np.asarray(w_np, dtype=np.float32)
    b_np = np.asarray(b_np, dtype=np.float32)
    M = int(np.prod(x_np.shape[:-1]))
    N = x_np.shape[-1]
    K = act_np.shape[-1]
    X = x_np.reshape(M, N)
    A = act_np.reshape(M, K)
    B = b_np.reshape(1, N)
    out = (A @ w_np + X + B).reshape(x_np.shape).astype(np.float32)
    result = {"out": out}
    if out_grad_np is not None:
        dout = np.asarray(out_grad_np, dtype=np.float32).reshape(M, N)
        result["dx"] = dout.reshape(x_np.shape).astype(np.float32)
        result["dact"] = (dout @ w_np.T).reshape(act_np.shape).astype(np.float32)
        result["dw"] = (A.T @ dout).astype(np.float32)
        result["db"] = np.sum(dout, axis=0).astype(np.float32)
    return result
def capture_and_compare(rt, loss, blocks, trace=False):
    """
    Forward:
      - sync before FusedLinearAddOp inputs are read
      - snapshot x/act/w/b
      - run kernel
      - sync and snapshot out
      - compare immediately against CPU oracle
    Backward:
      - snapshot out.grad before backward
      - run backward kernel
      - sync and snapshot x.grad/act.grad/w.grad/b.grad
      - compare immediately against CPU oracle
    """
    # Forward pass
    rt.backend.begin_pass()
    fla_idx = 0
    for idx, node in enumerate(rt.topo):
        if trace:
            print(f"  [FWD {idx:02d}] {type(node.op).__name__}", flush=True)
        if type(node.op).__name__ == "FusedLinearAddOp":
            # Flush all prior kernels so inputs are stable.
            rt.backend.end_pass()
            rt.backend.synchronize()
            rec = blocks[fla_idx]
            print("--------FusedLinearAddOp begin----------")
            rec["x_snap"] = copy_to_host(rt, node.inputs[0]).copy()
            rec["act_snap"] = copy_to_host(rt, node.inputs[1]).copy()
            rec["w_snap"] = copy_to_host(rt, node.inputs[2]).copy()
            rec["b_snap"] = copy_to_host(rt, node.inputs[3]).copy()
            print("  [snapshot] x_in =", rec["x_snap"])
            print("  [snapshot] act_in =", rec["act_snap"])
            # Re-open a fresh pass for the op itself.
            rt.backend.begin_pass()
            node.forward(node.inputs, node.outputs)
            rt.backend.end_pass()
            rt.backend.synchronize()
            rec["out_snap"] = copy_to_host(rt, node.outputs[0]).copy()
            print("--------FusedLinearAddOp end----------")
            print("  [snapshot] out =", rec["out_snap"])
            oracle = cpu_fused_linear_add_oracle(
                rec["x_snap"], rec["act_snap"], rec["w_snap"], rec["b_snap"]
            )
            compare(f"block {fla_idx} forward", rec["out_snap"], oracle["out"], atol=FLOAT_ATOL, rtol=FLOAT_RTOL)
            fla_idx += 1
            rt.backend.begin_pass()
        else:
            node.forward(node.inputs, node.outputs)
    rt.backend.end_pass()
    rt.backend.synchronize()
    # Backward pass
    rt.zero_grad()
    rt.backend.fill(loss.grad, 1.0)
    rt.backend.begin_pass()
    fla_idx = len(blocks) - 1
    for idx, node in enumerate(list(reversed(rt.topo))):
        if trace:
            print(f"  [BWD {idx:02d}] {type(node.op).__name__}", flush=True)
        for t in rt.node_to_clear_grads.get(node, []):
            if t.grad is not None:
                rt.backend.fill(t.grad, 0)
        if type(node.op).__name__ == "FusedLinearAddOp":
            rt.backend.end_pass()
            rt.backend.synchronize()
            rec = blocks[fla_idx]
            rec["out_grad_snap"] = copy_to_host(rt, node.outputs[0].grad).copy()
            print("--------FusedLinearAddOp backward begin----------")
            print("  [snapshot] out_grad =", rec["out_grad_snap"])
            rt.backend.begin_pass()
            node.backward(node.inputs, node.outputs)
            rt.backend.end_pass()
            rt.backend.synchronize()
            rec["x_grad_snap"] = copy_to_host(rt, node.inputs[0].grad).copy()
            rec["act_grad_snap"] = copy_to_host(rt, node.inputs[1].grad).copy()
            rec["w_grad_snap"] = copy_to_host(rt, node.inputs[2].grad).copy()
            rec["b_grad_snap"] = copy_to_host(rt, node.inputs[3].grad).copy()
            oracle = cpu_fused_linear_add_oracle(
                rec["x_snap"],
                rec["act_snap"],
                rec["w_snap"],
                rec["b_snap"],
                rec["out_grad_snap"],
            )
            compare(f"block {fla_idx} x.grad", rec["x_grad_snap"], oracle["dx"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
            compare(f"block {fla_idx} act.grad", rec["act_grad_snap"], oracle["dact"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
            compare(f"block {fla_idx} w.grad", rec["w_grad_snap"], oracle["dw"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
            compare(f"block {fla_idx} b.grad", rec["b_grad_snap"], oracle["db"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
            print("--------FusedLinearAddOp backward end----------")
            fla_idx -= 1
            rt.backend.begin_pass()
        else:
            node.backward(node.inputs, node.outputs)
    rt.backend.end_pass()
    rt.backend.synchronize()
    return blocks
def main():
    param_arrays = build_param_arrays()
    input_ids, pos_ids, labels = build_inputs()
    # Use MPS live trace only; CPU oracle is computed from immediate snapshots.
    rt, loss, blocks = build_live_graph("mps", param_arrays, input_ids, pos_ids, labels)
    capture_and_compare(rt, loss, blocks, trace=False)
    print("\nAll FusedLinearAdd live traces finished.")
if __name__ == "__main__":
    main()