import gc
import os
import struct
import sys
import re
import math
from pathlib import Path
import numpy as np
# ROOT = Path(__file__).resolve().parent
# os.chdir(ROOT)
# sys.path.insert(0, str(ROOT))
# sys.path.insert(1, str(ROOT.parent))
from tensor import Tensor
from math import prod

import runtime
import metal_ext

from runtime import Runtime
from module import GPT2
from op import (
    AddOp,
    MulOp,
    MatMulOp,
    ContiguousOp,
    TransposeOp,
    LayerNormOp,
    GELUOp,
    GatherOp,
    FusedLinearAddOp,
    FusedLayerNormMLPOp,
    FusedLayerNormFlashAttentionOp,
    FusedLinearCrossEntropyOp,
)
from verify_model import (
    make_tensor,
    fused_ln_fa_np,
    fused_ln_mlp_np,
    fused_linear_add_np,
    cross_entropy_np,
    iter_gpt2_named_tensors,
)

print(runtime.__file__)
print(metal_ext.__file__)

SEED = 1234 + 99
# CFG = dict(B=2, T=32, V=1024, C=1024, n_head=2, n_layer=2, dtype=np.float32)
CFG = dict(B=2, T=32, V=2048, C=2048, n_head=4, n_layer=2, dtype=np.float32)
# CFG = dict(B=2, T=32, V=2048, C=4096, n_head=8, n_layer=2, dtype=np.float32)
FLOAT_ATOL = 1e-4
FLOAT_RTOL = 1e-4
GRAD_ATOL = 1e-6
GRAD_RTOL = 1e-6

def reset_runtime(device):
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device)
    Runtime._default_runtime = rt
    return rt
def copy_to_host(rt, tensor):
    return rt.backend.copy_to_host(tensor).copy()
def read_f32(rt, storage, shape):
    return np.array(
        rt.backend.ctx.numpy_view_float32(
            storage.ptr.buffer_id,
            storage.ptr.offset,
            shape,
        )
    ).copy()

def compare(name, got, ref, atol=FLOAT_ATOL, rtol=FLOAT_RTOL, fail_fast=False):
    got = np.asarray(got)
    ref = np.asarray(ref)
    diff = np.abs(got - ref)
    rel = diff / (np.abs(ref) + 1e-12)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    max_rel = float(np.max(rel)) if rel.size else 0.0
    ok = np.allclose(got, ref, atol=atol, rtol=rtol, equal_nan=True)
    print(
        f"  {'PASS' if ok else 'FAIL'} {name}: "
        f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
        f"atol={atol}, rtol={rtol}"
    )
    if fail_fast and not ok:
        raise AssertionError(
            f"{name} mismatch: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, "
            f"atol={atol}, rtol={rtol}"
        )
    return ok


def load_metal_kernels():
    metal_path = ROOT / "metal_ops.metal"
    source = metal_path.read_text()
    return set(re.findall(r"kernel void ([A-Za-z0-9_]+)\s*\(", source))

def snapshot_tensor(rt, tensor):
    return copy_to_host(rt, tensor).copy()

def register_snapshot_target(stages, tensor, container_key, field_name):
    """
    stages["_snapshot_targets"][id(tensor)] = [(container_key, field_name), ...]
    支持同一个 tensor 对应多个 snapshot 字段。
    """
    stages.setdefault("_snapshot_targets", {})
    stages.setdefault(container_key, {})
    stages[container_key][field_name] = None
    stages["_snapshot_targets"].setdefault(id(tensor), []).append((container_key, field_name))
def register_grad_snapshot_target(stages, tensor, container_key, field_name):
    stages.setdefault("_grad_snapshot_targets", {})
    stages.setdefault(container_key, {})
    stages[container_key][field_name] = None
    stages["_grad_snapshot_targets"].setdefault(id(tensor), []).append((container_key, field_name))
def store_snapshot_targets(stages, rt, tensor):
    targets = stages.get("_snapshot_targets", {}).get(id(tensor), None)
    if not targets:
        return False
    snap = snapshot_tensor(rt, tensor)
    for container_key, field_name in targets:
        stages[container_key][field_name] = snap.copy()
    return True
def store_grad_snapshot_targets(stages, rt, tensor):
    if tensor.grad is None:
        return False
    targets = stages.get("_grad_snapshot_targets", {}).get(id(tensor), None)
    if not targets:
        return False
    snap = snapshot_tensor(rt, tensor.grad)
    for container_key, field_name in targets:
        stages[container_key][field_name] = snap.copy()
    return True


def trace_step(label):
    print(f"\n>>> {label}", flush=True)


def _flush(rt):
    rt.backend.end_pass()
    rt.backend.synchronize()

def run_graph_forward(rt, stages=None, trace=False):
    rt.backend.begin_pass()
    fused_linear_add_idx = 0
    fused_ln_fa_idx = 0
    for idx, node in enumerate(rt.topo):
        if trace:
            print(f"  [FWD {idx:02d}] {type(node.op).__name__}", flush=True)
        if stages is not None:
            # 先把前面已经提交的命令真正完成，避免读到被复用后的旧值
            rt.backend.end_pass()
            rt.backend.synchronize()

            op_name = type(node.op).__name__
            if op_name == "FusedLayerNormFlashAttentionOp":
                blk = stages[f"blocks[{fused_ln_fa_idx}]"]
                blk["x_in_snapshot"] = copy_to_host(rt, node.inputs[0]).copy()
            elif op_name == "FusedLinearAddOp":
                blk = stages[f"blocks[{fused_linear_add_idx}]"]
                blk["x_in_snapshot"] = copy_to_host(rt, node.inputs[0]).copy()
                blk["attn_out_snapshot"] = copy_to_host(rt, node.inputs[1]).copy()
            else:
                for t in node.inputs:
                    store_snapshot_targets(stages, rt, t)
            # 开一个新 pass 只跑这个 node
            rt.backend.begin_pass()

        node.forward(node.inputs, node.outputs)

        if stages is not None:
            # 这个 node 跑完，立即同步并快照输出
            rt.backend.end_pass()
            rt.backend.synchronize()

            op_name = type(node.op).__name__
            if op_name == "FusedLayerNormFlashAttentionOp":
                blk = stages[f"blocks[{fused_ln_fa_idx}]"]
                blk["attn_out_snapshot"] = copy_to_host(rt, node.outputs[0]).copy()
                fused_ln_fa_idx += 1
            elif op_name == "FusedLinearAddOp":
                blk = stages[f"blocks[{fused_linear_add_idx}]"]
                blk["x_after_attn_snapshot"] = copy_to_host(rt, node.outputs[0]).copy()
                fused_linear_add_idx += 1
            else:
                for t in node.outputs:
                    store_snapshot_targets(stages, rt, t)
            # 为下一个 node 重新开 pass
            rt.backend.begin_pass()
    rt.backend.end_pass()
    rt.backend.synchronize()

def run_graph_backward(rt, stages=None, trace=False):
    rt.backend.begin_pass()
    residual_add_bwd_idx = CFG['n_layer'] - 1
    fused_ln_mlp_bwd_idx = CFG['n_layer'] - 1
    fused_linear_add_bwd_idx = CFG["n_layer"] - 1
    fused_ln_fa_bwd_idx = CFG["n_layer"] - 1
    for idx, node in enumerate(list(reversed(rt.topo))):
        if trace:
            print(f"  [BWD {idx:02d}] {type(node.op).__name__}", flush=True)
        if stages is not None:
            # 先完成前面的 backward 命令，避免梯度被后续复用覆盖
            rt.backend.end_pass()
            rt.backend.synchronize()

            op_name = type(node.op).__name__
            if op_name == 'FusedLayerNormFlashAttentionOp':
                blk = stages[f"blocks[{fused_ln_fa_bwd_idx}]"]
                blk['x_in_grad_before_fa_snapshot'] = copy_to_host(rt, node.inputs[0].grad).copy()


            # 先快照 backward 需要的输入梯度，尤其是 outputs[i].grad
            for t in node.outputs:
                store_grad_snapshot_targets(stages, rt, t)
            rt.backend.begin_pass()

        for t in rt.node_to_clear_grads.get(node, []):
            if t.grad is not None:
                # if type(node.op).__name__ == "FusedLayerNormFlashAttentionOp" and t is node.inputs[0]:
                #     print("skip FusedLayerNormFlashAttentionOp x.grad clear")
                #     continue
                rt.backend.fill(t.grad, 0)
        node.backward(node.inputs, node.outputs)

        if stages is not None:
            # backward 刚结束，立即同步并快照输入梯度/参数梯度
            rt.backend.end_pass()
            rt.backend.synchronize()

            op_name = type(node.op).__name__
            if op_name == 'FusedLinearCrossEntropyOp':
                blk = stages[f"blocks[{CFG['n_layer']-1}]"]
                blk['x_out_grad_after_ce_snapshot'] = copy_to_host(rt, node.inputs[0].grad).copy()
            elif op_name == 'AddOp' and residual_add_bwd_idx >= 0:
                blk = stages[f"blocks[{residual_add_bwd_idx}]"]
                if node.outputs[0] is blk['x_out']:
                    x_out_grad = copy_to_host(rt, node.outputs[0].grad).copy()
                    blk['x_out_grad_before_add_snapshot'] = x_out_grad
                    blk['x_out_grad_after_ce_snapshot'] = x_out_grad
                    blk['x_after_attn_grad_after_add_snapshot'] = copy_to_host(rt, node.inputs[0].grad).copy()
                    blk['mlp_out_grad_after_add_snapshot'] = copy_to_host(rt, node.inputs[1].grad).copy()
                    residual_add_bwd_idx -= 1
            elif op_name == 'FusedLayerNormMLPOp':
                blk = stages[f"blocks[{fused_ln_mlp_bwd_idx}]"]
                blk['x_after_attn_grad_after_mlp_snapshot'] = copy_to_host(rt, node.inputs[0].grad).copy()
                fused_ln_mlp_bwd_idx -= 1
            elif op_name == "FusedLinearAddOp":
                blk = stages[f"blocks[{fused_linear_add_bwd_idx}]"]
                blk["x_in_grad_after_linear_add_snapshot"] = copy_to_host(rt, node.inputs[0].grad).copy()
                blk["attn_out_grad_after_linear_add_snapshot"] = copy_to_host(rt, node.inputs[1].grad).copy()
                fused_linear_add_bwd_idx -= 1
            elif op_name == "FusedLayerNormFlashAttentionOp":
                blk = stages[f"blocks[{fused_ln_fa_bwd_idx}]"]
                blk['x_in_grad_after_fa_snapshot'] = copy_to_host(rt, node.inputs[0].grad).copy()
                fused_ln_fa_bwd_idx -= 1

            for t in node.inputs + node.outputs:
                store_grad_snapshot_targets(stages, rt, t)
            rt.backend.begin_pass()
    rt.backend.end_pass()
    rt.backend.synchronize()

def build_param_arrays():
    rng = np.random.default_rng(SEED)
    ref_model = GPT2(
        CFG["V"], CFG["C"], CFG["T"], CFG["n_head"], CFG["B"], CFG["n_layer"], CFG["dtype"], "cpu"
    )
    param_arrays = {}
    for name, tensor in iter_gpt2_named_tensors(ref_model):
        param_arrays[name] = (rng.standard_normal(tensor.shape).astype(np.float32) * 0.02).copy()
    return param_arrays

def init_model_params(model, param_arrays):
    for name, tensor in iter_gpt2_named_tensors(model):
        tensor.init_val = param_arrays[name].copy()

def build_inputs():
    rng = np.random.default_rng(SEED)
    input_ids = rng.integers(0, CFG["V"], size=(CFG["B"], CFG["T"]), dtype=np.int32)
    pos_ids = np.tile(np.arange(CFG["T"], dtype=np.int32), (CFG["B"], 1))
    labels = rng.integers(0, CFG["V"], size=(CFG["B"], CFG["T"]), dtype=np.int32)
    return input_ids, pos_ids, labels

def build_manual_tiny_gpt2_state(device, param_arrays, input_ids, pos_ids, labels, trace=False):
    print(f"------------------ {device} --------------------------")
    rt = reset_runtime(device)
    model = GPT2(
        CFG["V"], CFG["C"], CFG["T"], CFG["n_head"], CFG["B"], CFG["n_layer"], CFG["dtype"], device
    )
    init_model_params(model, param_arrays)
    input_ids_t = make_tensor((CFG["B"], CFG["T"]), np.int32, device, False, input_ids)
    pos_ids_t = make_tensor((CFG["B"], CFG["T"]), np.int32, device, False, pos_ids)
    labels_t = make_tensor((CFG["B"], CFG["T"]), np.int32, device, False, labels)
    stages = {
        "_snapshot_targets": {},
        "_grad_snapshot_targets": {},
        "snapshots": {},
    }
    x = model.tok_emb(input_ids_t) + model.pos_emb(pos_ids_t)
    stages["x0"] = x
    register_snapshot_target(stages, x, "snapshots", "x0_snapshot")
    for i, blk in enumerate(model.blocks):
        block_key = f"blocks[{i}]"
        block = {}
        block["x_in"] = x
        block["x_in_snapshot"] = None
        block["attn_out_snapshot"] = None
        block["x_after_attn_snapshot"] = None
        block["mlp_out_snapshot"] = None
        block["x_out_snapshot"] = None
        block['x_out_grad_before_add_snapshot'] = None
        block['x_out_grad_after_ce_snapshot'] = None
        register_snapshot_target(stages, x, block_key, "x_in_snapshot")
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
        block["attn_out"] = attn_out
        block["attn_out_snapshot"] = None
        register_snapshot_target(stages, attn_out, block_key, "attn_out_snapshot")
        x_after_attn = FusedLinearAddOp()(
            x,
            attn_out,
            blk.attn.proj.weight,
            blk.attn.proj.bias,
        )
        block["x_after_attn"] = x_after_attn
        block["x_after_attn_snapshot"] = None
        register_snapshot_target(stages, x_after_attn, block_key, "x_after_attn_snapshot")
        mlp_out = FusedLayerNormMLPOp(eps=blk.ln2.eps)(
            x_after_attn,
            blk.ln2.gamma,
            blk.ln2.beta,
            blk.mlp.fc1.weight,
            blk.mlp.fc1.bias,
            blk.mlp.fc2.weight,
            blk.mlp.fc2.bias,
        )
        block["mlp_out"] = mlp_out
        block["mlp_out_snapshot"] = None
        register_snapshot_target(stages, mlp_out, block_key, "mlp_out_snapshot")
        x = x_after_attn + mlp_out
        block["x_out"] = x
        block["x_out_snapshot"] = None
        register_snapshot_target(stages, x, block_key, "x_out_snapshot")
        stages[block_key] = block
    loss = x.fused_linear_cross_entropy(model.lm_head.weight, labels_t)
    stages["loss"] = loss
    stages["loss_snapshot"] = None
    register_snapshot_target(stages, loss, "snapshots", "loss_snapshot")
    rt.loss = loss
    rt.build_topo(loss)
    rt.bind_memory()
    if device == "cpu":
        view_op_names = {"View", "Reshape", "Permute", "Transpose"}
        graph_tensors = []
        visited_tensor_ids = set()
        for node in rt.topo:
            for tensor in node.inputs + node.outputs:
                tensor_id = id(tensor)
                if tensor_id in visited_tensor_ids:
                    continue
                visited_tensor_ids.add(tensor_id)
                graph_tensors.append(tensor)
        for tensor in graph_tensors:
            if tensor.op is None:
                continue
            if tensor.op.op.op_name in view_op_names:
                continue
            tensor.storage = rt.allocator.alloc(tensor.shape, tensor.dtype)
            if tensor.grad is not None:
                tensor.grad.storage = rt.allocator.alloc(tensor.grad.shape, tensor.grad.dtype)
        for tensor in graph_tensors:
            if tensor.op is None:
                continue
            if tensor.op.op.op_name not in view_op_names:
                continue
            base = tensor.op.inputs[0]
            tensor.storage = rt.allocator.view(base.storage, 0, tensor.shape, tensor.dtype)
            if tensor.grad is not None and base.grad is not None:
                tensor.grad.storage = rt.allocator.view(base.grad.storage, 0, tensor.grad.shape, tensor.grad.dtype)

    # grad snapshot targets: 在 bind_memory 之后，grad tensor 已存在
    register_grad_snapshot_target(stages, stages["x0"], "snapshots", "x0_grad_snapshot")
    for i in range(CFG["n_layer"]):
        blk = stages[f"blocks[{i}]"]
        block_key = f"blocks[{i}]"
        register_grad_snapshot_target(stages, blk["x_in"], block_key, "x_in_grad_snapshot")
        register_grad_snapshot_target(stages, blk["attn_out"], block_key, "attn_out_grad_snapshot")
        register_grad_snapshot_target(stages, blk["x_after_attn"], block_key, "x_after_attn_grad_snapshot")
        register_grad_snapshot_target(stages, blk["mlp_out"], block_key, "mlp_out_grad_snapshot")
        register_grad_snapshot_target(stages, blk["x_out"], block_key, "x_out_grad_snapshot")
    if trace:
        trace_step(f"{device} forward trace")
    run_graph_forward(rt, stages=stages, trace=trace)
    rt.zero_grad()
    rt.backend.fill(loss.grad, 1.0)
    if trace:
        trace_step(f"{device} backward trace")
    run_graph_backward(rt, stages=stages, trace=trace)

    parameter_grads = {}
    for name, tensor in iter_gpt2_named_tensors(model):
        if tensor.grad is None:
            parameter_grads[
                name
            ] = None
        else:
            parameter_grads[name] = copy_to_host(
                rt,
                tensor.grad,
            )

    result = {
        "loss": stages["snapshots"]["loss_snapshot"],
        "x0": stages["snapshots"]["x0_snapshot"],
        "x0_grad": stages["snapshots"]["x0_grad_snapshot"],
        "tok_emb_weight_grad": copy_to_host(rt, model.tok_emb.weight.grad),
        "pos_emb_weight_grad": copy_to_host(rt, model.pos_emb.weight.grad),
        "lm_head_weight_grad": copy_to_host(rt, model.lm_head.weight.grad),
        "parameter_grads": parameter_grads,
        "blocks": [],
    }
    # print(f" {device} x0_grad = ", stages["snapshots"]["x0_grad_snapshot"])
    for i in range(CFG["n_layer"]):
        blk = stages[f"blocks[{i}]"]
        blk_fwd = stages[f"blocks[{CFG['n_layer'] - i - 1}]"]
        blk_res = {
            "x_in": blk["x_in_snapshot"],
            "attn_out": blk["attn_out_snapshot"],
            "x_after_attn": blk["x_after_attn_snapshot"],
            "mlp_out": blk["mlp_out_snapshot"],
            "x_out": blk["x_out_snapshot"],

            "x_out_grad_before_add": blk["x_out_grad_before_add_snapshot"],
            "x_out_grad_after_ce": blk["x_out_grad_before_add_snapshot"],

            # "attn_out_before_fa_backward": blk["attn_out_before_fa_backward_snapshot"],

            "x_out_grad_after_ce": blk["x_out_grad_after_ce_snapshot"],
            "x_after_attn_grad_after_add": blk["x_after_attn_grad_after_add_snapshot"],
            "mlp_out_grad_after_add": blk['mlp_out_grad_after_add_snapshot'],
            "x_after_attn_grad_after_mlp": blk["x_after_attn_grad_after_mlp_snapshot"],
            "x_in_grad_after_linear_add": blk["x_in_grad_after_linear_add_snapshot"],
            "attn_out_grad_after_linear_add": blk["attn_out_grad_after_linear_add_snapshot"],
            "x_in_grad_before_fa": blk["x_in_grad_before_fa_snapshot"],
            "x_in_grad_after_fa": blk["x_in_grad_after_fa_snapshot"],
        }
        result["blocks"].append(blk_res)

        # print("x_out_grad_snapshot =", blk["x_out_grad_snapshot"])
        # print("x_after_attn_grad_snapshot =", blk["x_after_attn_grad_snapshot"])
        # print("mlp_out_grad_snapshot =", blk["mlp_out_grad_snapshot"])

        # print("attn_out_grad_snapshot =", stages[f"blocks[{i}]"]["attn_out_grad_snapshot"])
        # print("current attn_out.grad  =", copy_to_host(rt, stages[f"blocks[{i}]"]["attn_out"].grad))

    return result

def run_single_op(device, build_fn, upstream_grad=None):
    rt = reset_runtime(device)
    out, tracked = build_fn(rt)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()
    rt.forward()
    rt.backend.synchronize()
    if upstream_grad is not None:
        rt.zero_grad()
        rt.backend.copy_from_host(out.grad, np.asarray(upstream_grad, dtype=np.float32))
        rt.backward_graph()
        rt.backend.synchronize()
    out_host = copy_to_host(rt, out)
    grads = {}
    for name, tensor in tracked.items():
        grads[name] = None if tensor.grad is None else copy_to_host(rt, tensor.grad)
    return out_host, grads

def full_model_stage_summary(cpu_state, mps_state):
    print("\n=== Full Tiny GPT2 ===")
    compare("loss", mps_state["loss"], cpu_state["loss"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("x0", mps_state["x0"], cpu_state["x0"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("x0.grad", mps_state["x0_grad"], cpu_state["x0_grad"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # print("mps x0.grad =", mps_state["x0_grad"])
    # print("cpu x0.grad =", cpu_state["x0_grad"])
    compare(
        "tok_emb.weight.grad",
        mps_state["tok_emb_weight_grad"],
        cpu_state["tok_emb_weight_grad"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "pos_emb.weight.grad",
        mps_state["pos_emb_weight_grad"],
        cpu_state["pos_emb_weight_grad"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "lm_head.weight.grad",
        mps_state["lm_head_weight_grad"],
        cpu_state["lm_head_weight_grad"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    forward_stage_keys = ('x_in', "attn_out", "x_after_attn", "mlp_out", "x_out")
    grad_stage_keys = ("x_out_grad_after_ce", "x_after_attn_grad_after_add",
        "mlp_out_grad_after_add", "x_after_attn_grad_after_mlp", "x_in_grad_after_linear_add",
        "attn_out_grad_after_linear_add", "x_in_grad_before_fa", "x_in_grad_after_fa")

    for i in range(CFG["n_layer"]):
        print(f"\n  [block {i}]")
        cpu_blk = cpu_state['blocks'][i]
        mps_blk = mps_state['blocks'][i]
        for key in forward_stage_keys:
            compare(f"block {i}/{key}", mps_blk[key], cpu_blk[key], atol=GRAD_ATOL, rtol=GRAD_RTOL, fail_fast=True)

        for key in grad_stage_keys:
            compare(f"block {i}/{key}", mps_blk[key], cpu_blk[key], atol=GRAD_ATOL, rtol=GRAD_RTOL, fail_fast=True)

            compare('Add backward/ x_after_attn.grad', mps_blk['x_after_attn_grad_after_add'], mps_blk['x_out_grad_after_ce'],
                atol=GRAD_ATOL, rtol=GRAD_RTOL)
            compare("Add backward / mlp_out.grad",
                mps_blk['mlp_out_grad_after_add'], mps_blk['x_out_grad_after_ce'], atol=GRAD_ATOL, rtol=GRAD_RTOL)

        compare(f"block {i}/LinearAdd residual", mps_blk["x_in_grad_after_linear_add"], mps_blk["x_after_attn_grad_after_mlp"],
            atol=GRAD_ATOL, rtol=GRAD_RTOL, fail_fast=True)

        compare(f"block {i}/LinearAdd -> FA boundary", mps_blk['x_in_grad_before_fa'], mps_blk['x_in_grad_after_linear_add'],
            atol=GRAD_ATOL, rtol=GRAD_RTOL, fail_fast=True)

    print(
        "\n  Parameter gradients"
    )

    cpu_parameter_grads = (
        cpu_state[
            "parameter_grads"
        ]
    )

    mps_parameter_grads = (
        mps_state[
            "parameter_grads"
        ]
    )

    cpu_names = set(
        cpu_parameter_grads.keys()
    )

    mps_names = set(
        mps_parameter_grads.keys()
    )

    if cpu_names != mps_names:
        missing_on_cpu = sorted(
            mps_names -
            cpu_names
        )

        missing_on_mps = sorted(
            cpu_names -
            mps_names
        )

        raise AssertionError(
            "CPU/MPS parameter集合不一致："
            f"missing_on_cpu={missing_on_cpu}, "
            f"missing_on_mps={missing_on_mps}"
        )

    all_parameter_grads_ok = True

    for name in sorted(
        cpu_names
    ):
        cpu_grad = (
            cpu_parameter_grads[
                name
            ]
        )

        mps_grad = (
            mps_parameter_grads[
                name
            ]
        )

        if (
            cpu_grad is None or
            mps_grad is None
        ):
            same_none_state = (
                cpu_grad is None and
                mps_grad is None
            )

            print(
                "  "
                f"{'PASS' if same_none_state else 'FAIL'} "
                f"parameter.grad / {name}: "
                f"cpu_none={cpu_grad is None}, "
                f"mps_none={mps_grad is None}"
            )

            all_parameter_grads_ok = (
                all_parameter_grads_ok and
                same_none_state
            )

            continue

        parameter_ok = compare(
            (
                "parameter.grad / "
                f"{name}"
            ),
            mps_grad,
            cpu_grad,
            atol=1e-4,
            rtol=1e-4,
            fail_fast=False,
        )

        all_parameter_grads_ok = (
            all_parameter_grads_ok and
            parameter_ok
        )

    if not all_parameter_grads_ok:
        raise AssertionError(
            "Tiny GPT2存在parameter gradient错误"
        )


def _probe_cpu_mps_binary(title, build_fn, x_np, y_np, upstream=None, tracked=("x", "y")):
    print(f"\n=== {title} ===")
    if upstream is None:
        # 与 out shape 一致的上游梯度
        out_shape = np.broadcast_shapes(x_np.shape, y_np.shape)
        upstream = np.random.default_rng(SEED).standard_normal(out_shape).astype(np.float32)
    cpu_out, cpu_grads = run_single_op(
        "cpu", lambda rt: build_fn("cpu", x_np, y_np), upstream_grad=upstream
    )
    mps_out, mps_grads = run_single_op(
        "mps", lambda rt: build_fn("mps", x_np, y_np), upstream_grad=upstream
    )
    compare("forward", mps_out, cpu_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    for name in tracked:
        if cpu_grads.get(name) is None:
            continue
        compare(
            f"{name}.grad",
            mps_grads[name],
            cpu_grads[name],
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
        )
def build_add_case(device, x_np, y_np):
    x = make_tensor(x_np.shape, np.float32, device, True, x_np)
    y = make_tensor(y_np.shape, np.float32, device, True, y_np)
    return AddOp()(x, y), {"x": x, "y": y}
def probe_add():
    rng = np.random.default_rng(SEED)
    B, T, C = 2, 32, 64
    _probe_cpu_mps_binary(
        "AddOp / residual [B,T,C]+[B,T,C]",
        build_add_case,
        rng.standard_normal((B, T, C), dtype=np.float32),
        rng.standard_normal((B, T, C), dtype=np.float32),
    )
    _probe_cpu_mps_binary(
        "AddOp / pos-broadcast [B,T,C]+[T,C]",
        build_add_case,
        rng.standard_normal((B, T, C), dtype=np.float32),
        rng.standard_normal((T, C), dtype=np.float32),
    )
    _probe_cpu_mps_binary(
        "AddOp / bias-broadcast [B,T,C]+[C]",
        build_add_case,
        rng.standard_normal((B, T, C), dtype=np.float32),
        rng.standard_normal((C,), dtype=np.float32),
    )
def build_mul_case(device, x_np, y_np):
    x = make_tensor(x_np.shape, np.float32, device, True, x_np)
    y = make_tensor(y_np.shape, np.float32, device, True, y_np)
    return MulOp()(x, y), {"x": x, "y": y}
def probe_mul():
    rng = np.random.default_rng(SEED + 1)
    B, T, C = 2, 16, 32
    _probe_cpu_mps_binary(
        "MulOp / same",
        build_mul_case,
        rng.standard_normal((B, T, C), dtype=np.float32),
        rng.standard_normal((B, T, C), dtype=np.float32),
    )
    _probe_cpu_mps_binary(
        "MulOp / scalar",
        build_mul_case,
        rng.standard_normal((B, T, C), dtype=np.float32),
        np.array([0.37], dtype=np.float32),
    )
    _probe_cpu_mps_binary(
        "MulOp / row-broadcast [B,T,C]*[C]",
        build_mul_case,
        rng.standard_normal((B, T, C), dtype=np.float32),
        rng.standard_normal((C,), dtype=np.float32),
    )
def build_matmul_case(device, x_np, w_np):
    x = make_tensor(x_np.shape, np.float32, device, True, x_np)
    w = make_tensor(w_np.shape, np.float32, device, True, w_np)
    return MatMulOp()(x, w), {"x": x, "w": w}
def probe_matmul():
    rng = np.random.default_rng(SEED + 2)
    B, T, C, V = 2, 8, 32, 64
    x_np = (rng.standard_normal((B, T, C)) * 0.02).astype(np.float32)
    w_np = (rng.standard_normal((C, V)) * 0.02).astype(np.float32)
    upstream = rng.standard_normal((B, T, V), dtype=np.float32)
    print("\n=== MatMulOp / [B,T,C]@[C,V] ===")
    cpu_out, cpu_grads = run_single_op(
        "cpu", lambda rt: build_matmul_case("cpu", x_np, w_np), upstream_grad=upstream
    )
    mps_out, mps_grads = run_single_op(
        "mps", lambda rt: build_matmul_case("mps", x_np, w_np), upstream_grad=upstream
    )
    compare("forward", mps_out, cpu_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("x.grad", mps_grads["x"], cpu_grads["x"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("w.grad", mps_grads["w"], cpu_grads["w"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
def probe_transpose_contiguous():
    rng = np.random.default_rng(SEED + 3)
    x_np = rng.standard_normal((2, 8, 16), dtype=np.float32)
    upstream = rng.standard_normal((2, 16, 8), dtype=np.float32)
    print("\n=== TransposeOp + ContiguousOp ===")
    def build(device):
        x = make_tensor(x_np.shape, np.float32, device, True, x_np)
        y = TransposeOp(1, 2)(x)
        out = ContiguousOp()(y)
        return out, {"x": x}
    cpu_out, cpu_grads = run_single_op("cpu", lambda rt: build("cpu"), upstream_grad=upstream)
    mps_out, mps_grads = run_single_op("mps", lambda rt: build("mps"), upstream_grad=upstream)
    compare("forward", mps_out, cpu_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("x.grad", mps_grads["x"], cpu_grads["x"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
def probe_layernorm():
    rng = np.random.default_rng(SEED + 4)
    B, T, C = 2, 16, 64
    x_np = rng.standard_normal((B, T, C), dtype=np.float32)
    g_np = rng.standard_normal((C,), dtype=np.float32)
    b_np = rng.standard_normal((C,), dtype=np.float32)
    upstream = rng.standard_normal((B, T, C), dtype=np.float32)
    print("\n=== LayerNormOp ===")
    def build(device):
        x = make_tensor(x_np.shape, np.float32, device, True, x_np)
        g = make_tensor(g_np.shape, np.float32, device, True, g_np)
        b = make_tensor(b_np.shape, np.float32, device, True, b_np)
        out = LayerNormOp(eps=1e-5)(x, g, b)
        return out, {"x": x, "gamma": g, "beta": b}
    cpu_out, cpu_grads = run_single_op("cpu", lambda rt: build("cpu"), upstream_grad=upstream)
    mps_out, mps_grads = run_single_op("mps", lambda rt: build("mps"), upstream_grad=upstream)
    compare("forward", mps_out, cpu_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    for name in ("x", "gamma", "beta"):
        compare(
            f"{name}.grad",
            mps_grads[name],
            cpu_grads[name],
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
        )
def probe_gelu():
    rng = np.random.default_rng(SEED + 5)
    x_np = rng.standard_normal((2, 16, 64), dtype=np.float32)
    upstream = rng.standard_normal(x_np.shape, dtype=np.float32)
    print("\n=== GELUOp ===")
    def build(device):
        x = make_tensor(x_np.shape, np.float32, device, True, x_np)
        return GELUOp()(x), {"x": x}
    cpu_out, cpu_grads = run_single_op("cpu", lambda rt: build("cpu"), upstream_grad=upstream)
    mps_out, mps_grads = run_single_op("mps", lambda rt: build("mps"), upstream_grad=upstream)
    compare("forward", mps_out, cpu_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("x.grad", mps_grads["x"], cpu_grads["x"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
def probe_fill():
    print("\n=== fill_storage ===")
    rt = reset_runtime("mps")
    t = Tensor(shape=(8,), dtype=np.float32, requires_grad=False, device="mps")
    t.storage = rt.allocator.alloc(t.shape, t.dtype)
    rt.backend.copy_from_host(t, np.full((8,), 9.0, dtype=np.float32))
    rt.backend.begin_pass()
    rt.backend.fill(t, 0.0)
    rt.backend.end_pass()
    rt.backend.synchronize()
    compare("fill(0)", copy_to_host(rt, t), np.zeros((8,), np.float32), atol=GRAD_ATOL, rtol=GRAD_RTOL)
    rt.backend.begin_pass()
    rt.backend.fill(t, 1.0)
    rt.backend.end_pass()
    rt.backend.synchronize()
    compare("fill(1)", copy_to_host(rt, t), np.ones((8,), np.float32), atol=GRAD_ATOL, rtol=GRAD_RTOL)

def _cpu_adamw_ref(p, g, m, v, lr, lr_t, b1, b2, eps, wd):
    p, g, m, v = p.copy(), g.copy(), m.copy(), v.copy()
    if wd != 0:
        p = p * (1.0 - lr * wd)
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * (g * g)
    p = p - lr_t * m / (np.sqrt(v) + eps)
    return p.astype(np.float32), m.astype(np.float32), v.astype(np.float32)

def probe_adamw():
    print("\n=== AdamW step ===")
    rng = np.random.default_rng(SEED + 6)
    shape = (128, 64)
    p0 = (rng.standard_normal(shape) * 0.02).astype(np.float32)
    g0 = (rng.standard_normal(shape) * 0.01).astype(np.float32)
    m0 = np.zeros(shape, dtype=np.float32)
    v0 = np.zeros(shape, dtype=np.float32)
    lr, b1, b2, eps, wd = 0.02, 0.9, 0.95, 1e-8, 0.28
    t_step = 1
    lr_t = lr * math.sqrt(1.0 - b2**t_step) / (1.0 - b1**t_step)
    pref, mref, vref = _cpu_adamw_ref(p0, g0, m0, v0, lr, lr_t, b1, b2, eps, wd)
    rt = reset_runtime("mps")
    p = Tensor(shape=shape, dtype=np.float32, requires_grad=True, device="mps")
    p.storage = rt.allocator.alloc(shape, np.float32)
    rt.backend.init(p)
    rt.backend.copy_from_host(p, p0)

    # make_tensor 一般会 alloc；再显式确保 grad
    p.grad = Tensor(shape=shape, dtype=np.float32, requires_grad=False, device="mps")
    p.grad.storage = rt.allocator.alloc(shape, np.float32)
    rt.backend.copy_from_host(p.grad, g0)
    
    m_st = rt.allocator.zeros(shape, np.float32)
    v_st = rt.allocator.zeros(shape, np.float32)
    rt.backend.begin_pass()
    rt.backend.adamw_step(
        p.storage, p.grad.storage, m_st, v_st, None,
        lr, lr_t, b1, b2, eps, wd,
    )
    rt.backend.end_pass()
    rt.backend.synchronize()
    compare("param", copy_to_host(rt, p), pref, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    m_host = read_f32(rt, m_st, shape)
    v_host = read_f32(rt, v_st, shape)
    compare("m", m_host, mref, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("v", v_host, vref, atol=GRAD_ATOL, rtol=GRAD_RTOL)
def probe_primitive_ops():
    """check_tiny 未覆盖的基础算子 / 训练辅助。"""
    probe_add()
    probe_mul()
    probe_matmul()
    probe_transpose_contiguous()
    probe_layernorm()
    probe_gelu()
    probe_fill()
    probe_adamw()

def build_gather_op_case(device, idx_np, w_np):
    idx = make_tensor(idx_np.shape, np.int32, device, False, idx_np)
    w = make_tensor(w_np.shape, np.float32, device, True, w_np)
    out = GatherOp()(idx, w)
    return out, {"w": w}

def probe_gather(cpu_state, param_arrays, input_ids, pos_ids):
    print("\n=== GatherOp / pos_emb.weight ===")
    pos_w = param_arrays["pos_emb.weight"]
    cpu_upstream = cpu_state["blocks"][0]["x_in_grad_after_fa"]
    cpu_out, cpu_grads = run_single_op(
        "cpu",
        lambda rt: build_gather_op_case("cpu", pos_ids, pos_w),
        upstream_grad=cpu_upstream,
    )
    mps_out, mps_grads = run_single_op(
        "mps",
        lambda rt: build_gather_op_case("mps", pos_ids, pos_w),
        upstream_grad=cpu_upstream,
    )
    compare("forward", mps_out, cpu_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("w.grad", mps_grads["w"], cpu_grads["w"], atol=GRAD_ATOL, rtol=GRAD_RTOL)

def build_ce_case(device, x2_np, w_np, labels_np):
    x = make_tensor(x2_np.shape, np.float32, device, True, x2_np)
    w = make_tensor(w_np.shape, np.float32, device, True, w_np)
    labels = make_tensor(labels_np.shape, np.int32, device, False, labels_np)
    out = FusedLinearCrossEntropyOp()(x, w, labels)
    return out, {"x": x, "w": w}

def build_ce_oracle(x2_np, w_np, labels_np):
    x2_np = np.asarray(x2_np, dtype=np.float32)
    w_np = np.asarray(w_np, dtype=np.float32)
    labels_np = np.asarray(labels_np, dtype=np.int32)

    B, T, C = x2_np.shape
    V = w_np.shape[0]
    X = x2_np.reshape(B * T, C)
    Y = labels_np.reshape(B * T)

    logits = X @ w_np.T
    loss = np.asarray([cross_entropy_np(logits, labels_np)], dtype=np.float32)

    row_max = np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits - row_max)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    for i in range(B * T):
        y = int(Y[i])
        if 0 <= y < V:
            probs[i, y] -= 1.0

    grad_out = np.float32(1.0 / (B * T))
    dlogits = probs * grad_out
    dx = (dlogits @ w_np).reshape(B, T, C).astype(np.float32)
    dw = (dlogits.T @ X).astype(np.float32)

    return {
        "loss": loss,
        "x_grad": dx,
        "w_grad": dw,
    }

def probe_fused_linear_ce(cpu_state, param_arrays, labels):
    print("\n=== FusedLinearCrossEntropyOp ===")
    x2_np = cpu_state["blocks"][-1]["x_out"]
    w_np = param_arrays["lm_head.weight"]

    oracle = build_ce_oracle(x2_np, w_np, labels)
    cpu_out, cpu_grads = run_single_op(
        "cpu",
        lambda rt: build_ce_case("cpu", x2_np, w_np, labels),
        upstream_grad=np.asarray([1.0], dtype=np.float32),
    )
    mps_out, mps_grads = run_single_op(
        "mps",
        lambda rt: build_ce_case("mps", x2_np, w_np, labels),
        upstream_grad=np.asarray([1.0], dtype=np.float32),
    )
    print("\n=== FusedLinearCrossEntropyOp / three-way ===")
    compare("forward cpu_vs_oracle", cpu_out, oracle["loss"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("forward mps_vs_oracle", mps_out, oracle["loss"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("forward mps_vs_cpu", mps_out, cpu_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("x.grad cpu_vs_oracle", cpu_grads["x"], oracle["x_grad"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("x.grad mps_vs_oracle", mps_grads["x"], oracle["x_grad"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("x.grad mps_vs_cpu", mps_grads["x"], cpu_grads["x"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("w.grad cpu_vs_oracle", cpu_grads["w"], oracle["w_grad"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("w.grad mps_vs_oracle", mps_grads["w"], oracle["w_grad"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("w.grad mps_vs_cpu", mps_grads["w"], cpu_grads["w"], atol=GRAD_ATOL, rtol=GRAD_RTOL)

    cpu_full_dx = cpu_state['blocks'][-1]['x_out_grad_after_ce']
    compare('full cpu x_out.grad vs oracle', cpu_full_dx, oracle['x_grad'], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare('single mps x_out.grad vs oracle', mps_grads['x'], oracle['x_grad'], atol=GRAD_ATOL, rtol=GRAD_RTOL)

def build_linear_add_case(device, x_np, act_np, w_np, b_np):
    x = make_tensor(x_np.shape, np.float32, device, True, x_np)
    act = make_tensor(act_np.shape, np.float32, device, True, act_np)
    w = make_tensor(w_np.shape, np.float32, device, True, w_np)
    b = make_tensor(b_np.shape, np.float32, device, True, b_np)
    out = FusedLinearAddOp()(x, act, w, b)
    return out, {"x": x, "act": act, "w": w, "b": b}

def probe_fused_linear_add(cpu_state, param_arrays):
    print("\n=== FusedLinearAddOp ===")
    for i, blk in enumerate(cpu_state["blocks"]):
        print(f"\n  [block {i}]")
        x_np = blk["x_in"]
        act_np = blk["attn_out"]
        w_np = param_arrays[f"blocks[{i}].attn.proj.weight"]
        b_np = param_arrays[f"blocks[{i}].attn.proj.bias"]
        upstream = blk["x_after_attn_grad_after_mlp"]
        cpu_out, cpu_grads = run_single_op(
            "cpu",
            lambda rt: build_linear_add_case("cpu", x_np, act_np, w_np, b_np),
            upstream_grad=upstream,
        )
        mps_out, mps_grads = run_single_op(
            "mps",
            lambda rt: build_linear_add_case("mps", x_np, act_np, w_np, b_np),
            upstream_grad=upstream,
        )
        compare("forward", mps_out, cpu_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        compare("x.grad", mps_grads["x"], cpu_grads["x"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
        compare("act.grad", mps_grads["act"], cpu_grads["act"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
        compare("w.grad", mps_grads["w"], cpu_grads["w"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
        compare("b.grad", mps_grads["b"], cpu_grads["b"], atol=GRAD_ATOL, rtol=GRAD_RTOL)

def build_linear_add_case_from_fa(device, x_np, param_arrays, block_idx):
    x = make_tensor(x_np.shape, np.float32, device, True, x_np)
    ln_w = make_tensor(
        param_arrays[f"blocks[{block_idx}].ln1.gamma"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].ln1.gamma"],
    )
    ln_b = make_tensor(
        param_arrays[f"blocks[{block_idx}].ln1.beta"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].ln1.beta"],
    )
    wq = make_tensor(
        param_arrays[f"blocks[{block_idx}].attn.q_proj.weight"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].attn.q_proj.weight"],
    )
    bq = make_tensor(
        param_arrays[f"blocks[{block_idx}].attn.q_proj.bias"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].attn.q_proj.bias"],
    )
    wk = make_tensor(
        param_arrays[f"blocks[{block_idx}].attn.k_proj.weight"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].attn.k_proj.weight"],
    )
    bk = make_tensor(
        param_arrays[f"blocks[{block_idx}].attn.k_proj.bias"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].attn.k_proj.bias"],
    )
    wv = make_tensor(
        param_arrays[f"blocks[{block_idx}].attn.v_proj.weight"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].attn.v_proj.weight"],
    )
    bv = make_tensor(
        param_arrays[f"blocks[{block_idx}].attn.v_proj.bias"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].attn.v_proj.bias"],
    )
    act = FusedLayerNormFlashAttentionOp(
        n_head=CFG["n_head"],
        eps=1e-6,
        is_causal=True,
    )(x, ln_w, ln_b, wq, bq, wk, bk, wv, bv)
    w = make_tensor(
        param_arrays[f"blocks[{block_idx}].attn.proj.weight"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].attn.proj.weight"],
    )
    b = make_tensor(
        param_arrays[f"blocks[{block_idx}].attn.proj.bias"].shape,
        np.float32,
        device,
        True,
        param_arrays[f"blocks[{block_idx}].attn.proj.bias"],
    )
    out = FusedLinearAddOp()(x, act, w, b)
    return out, {
        "x": x,
        "act": act,
        "w": w,
        "b": b,
        "ln_w": ln_w,
        "ln_b": ln_b,
        "wq": wq,
        "bq": bq,
        "wk": wk,
        "bk": bk,
        "wv": wv,
        "bv": bv,
    }

def probe_fused_linear_add_from_fa(cpu_state, param_arrays):
    print("\n=== FusedLinearAddOp / tiny GPT2 path ===")
    for block_idx, blk in enumerate(cpu_state["blocks"]):
        print(f"\n  [block {block_idx}]")
        x_np = blk["x_in"]
        upstream = blk["x_after_attn_grad_after_mlp"]
        attn_params = {
            "ln_w": param_arrays[f"blocks[{block_idx}].ln1.gamma"],
            "ln_b": param_arrays[f"blocks[{block_idx}].ln1.beta"],
            "wq": param_arrays[f"blocks[{block_idx}].attn.q_proj.weight"],
            "bq": param_arrays[f"blocks[{block_idx}].attn.q_proj.bias"],
            "wk": param_arrays[f"blocks[{block_idx}].attn.k_proj.weight"],
            "bk": param_arrays[f"blocks[{block_idx}].attn.k_proj.bias"],
            "wv": param_arrays[f"blocks[{block_idx}].attn.v_proj.weight"],
            "bv": param_arrays[f"blocks[{block_idx}].attn.v_proj.bias"],
        }
        cpu_fa = attention_oracle(
            x_np,
            attn_params,
            blk["attn_out_grad_after_linear_add"],
            n_head=CFG["n_head"],
            eps=1e-6,
            is_causal=True,
            x_grad_before = blk['x_in_grad_before_fa']
        )
        cpu_add_out = blk["x_after_attn"]
        cpu_dw = np.matmul(
            cpu_fa["out"].reshape(CFG["B"] * CFG["T"], CFG["C"]).T,
            upstream.reshape(CFG["B"] * CFG["T"], CFG["C"]),
        ).astype(np.float32)
        cpu_dact = np.matmul(
            upstream.reshape(CFG["B"] * CFG["T"], CFG["C"]),
            param_arrays[f"blocks[{block_idx}].attn.proj.weight"].T,
        ).reshape(CFG["B"], CFG["T"], CFG["C"]).astype(np.float32)
        cpu_db = np.sum(upstream.reshape(CFG["B"] * CFG["T"], CFG["C"]), axis=0).astype(np.float32)
        rt = reset_runtime("mps")
        out, tracked = build_linear_add_case_from_fa("mps", x_np, param_arrays, block_idx)
        rt.loss = out
        rt.build_topo(out)
        rt.bind_memory()
        node = next(n for n in rt.topo if type(n.op).__name__ == "FusedLinearAddOp")
        scratch = node.backward.keywords["scratch"]
        x = tracked["x"]
        act = tracked["act"]
        w = tracked["w"]
        b = tracked["b"]
        ln_w = tracked["ln_w"]
        ln_b = tracked["ln_b"]
        wq = tracked["wq"]
        bq = tracked["bq"]
        wk = tracked["wk"]
        bk = tracked["bk"]
        wv = tracked["wv"]
        bv = tracked["bv"]
        fa_mean = scratch["fa_mean"]
        fa_inv_std = scratch["fa_inv_std"]
        fa_k = scratch["fa_k"]
        fa_v = scratch["fa_v"]
        rt.forward()
        rt.backend.synchronize()
        compare("forward", copy_to_host(rt, out), cpu_add_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        rt.zero_grad()
        rt.backend.copy_from_host(out.grad, np.asarray(upstream, dtype=np.float32))
        B, T, C = x_np.shape
        H = CFG["n_head"]
        D = C // H
        eps_bits = struct.unpack("<i", struct.pack("<f", np.float32(1e-6)))[0]
        mean_offset = fa_mean.ptr.offset
        inv_std_offset = fa_inv_std.ptr.offset

        print("  backward / metal_recompute_fused_ln_fa")
        # 最好在调用前清空这些buffer，避免旧结果让测试假通过。
        zero_tensor(rt, act.grad)
        rt.backend.begin_pass()
        rt.backend.metal_recompute_fused_ln_fa(
            act,
            scratch,
            act.grad.storage,
        )
        rt.backend.end_pass()
        rt.backend.synchronize()

        mean_got = read_f32(
            rt,
            fa_mean,
            fa_mean.shape,
        ).reshape(
            B,
            T,
        )

        inv_std_got = read_f32(
            rt,
            fa_inv_std,
            fa_inv_std.shape,
        ).reshape(
            B,
            T,
        )

        k_got = read_f32(
            rt,
            fa_k,
            (
                B,
                H,
                T,
                D,
            ),
        )

        v_got = read_f32(
            rt,
            fa_v,
            (
                B,
                H,
                T,
                D,
            ),
        )

        recomputed_out_got = copy_to_host(
            rt,
            act.grad,
        ).reshape(
            B,
            T,
            C,
        )

        compare(
            "recompute / mean",
            mean_got,
            cpu_fa["mean"],
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
            fail_fast=True,
        )

        compare(
            "recompute / inv_std",
            inv_std_got,
            cpu_fa["inv_std"],
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
            fail_fast=True,
        )

        compare(
            "recompute / K",
            k_got,
            cpu_fa["k"],
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
            fail_fast=True,
        )

        compare(
            "recompute / V",
            v_got,
            cpu_fa["v"],
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
            fail_fast=True,
        )

        compare(
            "recompute / attention output",
            recomputed_out_got,
            cpu_fa["out"],
            atol=GRAD_ATOL,
            rtol=GRAD_RTOL,
            fail_fast=True,
        )

        # print("  backward / layernorm_compute_stats_kernel")
        # rt.backend.begin_pass()
        # rt.backend.ctx.dispatch_with_args(
        #     "layernorm_compute_stats_kernel",
        #     [x.storage.ptr.buffer_id, fa_mean.ptr.buffer_id, fa_inv_std.ptr.buffer_id],
        #     [x.storage.ptr.offset, fa_mean.ptr.offset, fa_inv_std.ptr.offset],
        #     [B * T, C, eps_bits],
        #     B * T * 1024,
        # )
        # rt.backend.end_pass()
        # rt.backend.synchronize()
        # compare("mean", read_f32(rt, fa_mean, fa_mean.shape).reshape(B, T), cpu_fa["mean"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
        # compare("inv_std", read_f32(rt, fa_inv_std, fa_inv_std.shape).reshape(B, T), cpu_fa["inv_std"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
        # print("  backward / fused_ln_fa_project_kv_kernel")
        # rt.backend.begin_pass()
        # rt.backend.ctx.dispatch_sdpa(
        #     "fa_opt_project_kv_kernel",
        #     [
        #         x.storage.ptr.buffer_id,
        #         ln_w.storage.ptr.buffer_id,
        #         ln_b.storage.ptr.buffer_id,
        #         wk.storage.ptr.buffer_id,
        #         bk.storage.ptr.buffer_id,
        #         wv.storage.ptr.buffer_id,
        #         bv.storage.ptr.buffer_id,
        #         fa_k.ptr.buffer_id,
        #         fa_v.ptr.buffer_id,
        #         fa_mean.ptr.buffer_id,
        #         fa_inv_std.ptr.buffer_id,
        #         out.storage.ptr.buffer_id,
        #     ],
        #     [
        #         x.storage.ptr.offset,
        #         ln_w.storage.ptr.offset,
        #         ln_b.storage.ptr.offset,
        #         wk.storage.ptr.offset,
        #         bk.storage.ptr.offset,
        #         wv.storage.ptr.offset,
        #         bv.storage.ptr.offset,
        #         fa_k.ptr.offset,
        #         fa_v.ptr.offset,
        #         mean_offset,
        #         inv_std_offset,
        #         out.storage.ptr.offset,
        #     ],
        #     [B, T, C, H, eps_bits],
        #     B * H * T,
        # )
        # rt.backend.end_pass()
        # rt.backend.synchronize()
        # compare("k", read_f32(rt, fa_k, (B, H, T, D)), cpu_fa["k"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
        # compare("v", read_f32(rt, fa_v, (B, H, T, D)), cpu_fa["v"], atol=GRAD_ATOL, rtol=GRAD_RTOL)

        # print("  backward / fused_ln_fa_min_forward_kernel")
        # rt.backend.begin_pass()
        # rt.backend.ctx.dispatch_flash_attn_bwd(
        #     "fused_ln_fa_min_forward_kernel",
        #     [
        #         x.storage.ptr.buffer_id,
        #         ln_w.storage.ptr.buffer_id,
        #         ln_b.storage.ptr.buffer_id,
        #         wq.storage.ptr.buffer_id,
        #         bq.storage.ptr.buffer_id,
        #         fa_k.ptr.buffer_id,
        #         fa_v.ptr.buffer_id,
        #         fa_mean.ptr.buffer_id,
        #         fa_inv_std.ptr.buffer_id,
        #         act.grad.storage.ptr.buffer_id,
        #     ],
        #     [
        #         x.storage.ptr.offset,
        #         ln_w.storage.ptr.offset,
        #         ln_b.storage.ptr.offset,
        #         wq.storage.ptr.offset,
        #         bq.storage.ptr.offset,
        #         fa_k.ptr.offset,
        #         fa_v.ptr.offset,
        #         mean_offset,
        #         inv_std_offset,
        #         act.grad.storage.ptr.offset,
        #     ],
        #     [B, H, T, C, int(True), eps_bits],
        #     B * H * ((T + 32 - 1) // 32),
        # )
        # rt.backend.end_pass()
        # rt.backend.synchronize()
        # compare("act.recompute", copy_to_host(rt, act.grad), cpu_fa["out"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
        # print("  backward / gemm_backward_dw_batched_mps")
        # print("  [debug] mps act.grad =", copy_to_host(rt, act.grad))
        # print("  [debug] mps w =", copy_to_host(rt, w))
        # print("  [debug] act.op.op_name =", act.op.op.op_name if act.op is not None else None)

        # zero_tensor(rt, w.grad)
        # rt.backend.begin_pass()
        # rt.backend.ctx.gemm_backward_dw_batched_mps(
        #     act.grad.storage.ptr.buffer_id,
        #     act.grad.storage.ptr.offset,
        #     out.grad.storage.ptr.buffer_id,
        #     out.grad.storage.ptr.offset,
        #     w.grad.storage.ptr.buffer_id,
        #     w.grad.storage.ptr.offset,
        #     B,
        #     T,
        #     C,
        #     C,
        #     0,
        # )
        # rt.backend.end_pass()
        # rt.backend.synchronize()
        # # print("  [debug] w.grad after gemm =", copy_to_host(rt, w.grad))
        # # print("  [debug] w.grad reference =", cpu_dw)
        # compare("w.grad", copy_to_host(rt, w.grad), cpu_dw, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        print("  backward / strided_add_backward_kernel")
        zero_tensor(rt, x.grad)
        rt.backend.begin_pass()
        rt.backend.ctx.dispatch_strided(
            "strided_add_backward_kernel",
            out.grad.storage.ptr.buffer_id,
            out.grad.storage.ptr.offset,
            x.grad.storage.ptr.buffer_id,
            x.grad.storage.ptr.offset,
            list(x.shape),
            x.strides,
            prod(x.shape),
        )
        rt.backend.end_pass()
        rt.backend.synchronize()
        compare("x.grad", copy_to_host(rt, x.grad), upstream, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        # print("  backward / gemm_backward_dx_batched_mps")
        # zero_tensor(rt, act.grad)
        # rt.backend.begin_pass()
        # rt.backend.ctx.dispatch_with_args(
        #     "fill_scalar_kernel",
        #     [act.grad.storage.ptr.buffer_id],
        #     [act.grad.storage.ptr.offset],
        #     [0],
        #     prod(act.shape),
        # )
        # rt.backend.ctx.gemm_backward_dx_batched_mps(
        #     out.grad.storage.ptr.buffer_id,
        #     out.grad.storage.ptr.offset,
        #     w.storage.ptr.buffer_id,
        #     w.storage.ptr.offset,
        #     act.grad.storage.ptr.buffer_id,
        #     act.grad.storage.ptr.offset,
        #     B,
        #     T,
        #     C,
        #     C,
        #     0,
        # )
        # rt.backend.end_pass()
        # rt.backend.synchronize()
        # compare("act.grad", copy_to_host(rt, act.grad), cpu_dact, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        # print("  backward / bias_grad_reduce")
        # zero_tensor(rt, b.grad)
        # rt.backend.begin_pass()
        # rt.backend.ctx.dispatch_with_args(
        #     "bias_grad_reduce",
        #     [out.grad.storage.ptr.buffer_id, b.grad.storage.ptr.buffer_id],
        #     [out.grad.storage.ptr.offset, b.grad.storage.ptr.offset],
        #     [B * T, C],
        #     C,
        # )
        # rt.backend.end_pass()
        # rt.backend.synchronize()
        # compare("b.grad", copy_to_host(rt, b.grad), cpu_db, atol=GRAD_ATOL, rtol=GRAD_RTOL)

def build_mlp_case(device, x_np, blk, param_arrays, block_idx):
    x = make_tensor(x_np.shape, np.float32, device, True, x_np)
    ln_w = make_tensor(blk["ln2_gamma"].shape, np.float32, device, True, blk["ln2_gamma"])
    ln_b = make_tensor(blk["ln2_beta"].shape, np.float32, device, True, blk["ln2_beta"])
    w1 = make_tensor(param_arrays[f"blocks[{block_idx}].mlp.fc1.weight"].shape, np.float32, device, True, param_arrays[f"blocks[{block_idx}].mlp.fc1.weight"])
    b1 = make_tensor(param_arrays[f"blocks[{block_idx}].mlp.fc1.bias"].shape, np.float32, device, True, param_arrays[f"blocks[{block_idx}].mlp.fc1.bias"])
    w2 = make_tensor(param_arrays[f"blocks[{block_idx}].mlp.fc2.weight"].shape, np.float32, device, True, param_arrays[f"blocks[{block_idx}].mlp.fc2.weight"])
    b2 = make_tensor(param_arrays[f"blocks[{block_idx}].mlp.fc2.bias"].shape, np.float32, device, True, param_arrays[f"blocks[{block_idx}].mlp.fc2.bias"])
    # out = FusedLayerNormMLPOp(eps=1e-6)(x, ln_w, ln_b, w1, b1, w2, b2)
    mlp_out = FusedLayerNormMLPOp(eps=1e-6)(x, ln_w, ln_b, w1, b1, w2, b2)
    out = x + mlp_out
    return out, {"x": x, "ln_w": ln_w, "ln_b": ln_b, "w1": w1, "b1": b1, "w2": w2, "b2": b2}

def probe_fused_layernorm_mlp(cpu_state, param_arrays):
    print("\n=== FusedLayerNormMLPOp ===")
    for i, blk in enumerate(cpu_state["blocks"]):
        print(f"\n  [block {i}]")
        x_np = blk["x_after_attn"]
        blk_seed = {
            "ln2_gamma": param_arrays[f"blocks[{i}].ln2.gamma"],
            "ln2_beta": param_arrays[f"blocks[{i}].ln2.beta"],
        }
        mlp_upstream = blk["mlp_out_grad_after_add"]
        # x_grad_before_mlp = blk["x_after_attn_grad_after_add"]
        def run_mlp_case(device):
            rt = reset_runtime(device)
            x = make_tensor(x_np.shape, np.float32, device, True, x_np)
            ln_w = make_tensor(blk_seed["ln2_gamma"].shape, np.float32, device, True, blk_seed["ln2_gamma"])
            ln_b = make_tensor(blk_seed["ln2_beta"].shape, np.float32, device, True, blk_seed["ln2_beta"])
            w1 = make_tensor(param_arrays[f"blocks[{i}].mlp.fc1.weight"].shape, np.float32, device, True, param_arrays[f"blocks[{i}].mlp.fc1.weight"])
            b1 = make_tensor(param_arrays[f"blocks[{i}].mlp.fc1.bias"].shape, np.float32, device, True, param_arrays[f"blocks[{i}].mlp.fc1.bias"])
            w2 = make_tensor(param_arrays[f"blocks[{i}].mlp.fc2.weight"].shape, np.float32, device, True, param_arrays[f"blocks[{i}].mlp.fc2.weight"])
            b2 = make_tensor(param_arrays[f"blocks[{i}].mlp.fc2.bias"].shape, np.float32, device, True, param_arrays[f"blocks[{i}].mlp.fc2.bias"])
            # out = FusedLayerNormMLPOp(eps=1e-6)(x, ln_w, ln_b, w1, b1, w2, b2)
            mlp_out = FusedLayerNormMLPOp(eps=1e-6)(x, ln_w, ln_b, w1, b1, w2, b2)
            out = x + mlp_out
            rt.loss = out
            rt.build_topo(out)
            rt.bind_memory()
            rt.forward()
            rt.backend.synchronize()
            rt.zero_grad()
            # grad = np.asarray(upstream, dtype=np.float32)
            rt.backend.copy_from_host(out.grad, mlp_upstream)
            # rt.backend.copy_from_host(x.grad, x_grad_before_mlp)
            rt.backend.synchronize()
            # print("before backward out.grad =", copy_to_host(rt, out.grad))
            # print("before backward x.grad =", copy_to_host(rt, x.grad))
            rt.backward_graph()
            rt.backend.synchronize()
            # if device == "mps":
            #     print("out.grad buf/off =", out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset)
            #     print("x.grad buf/off =", x.grad.storage.ptr.buffer_id, x.grad.storage.ptr.offset)
            grads = {
                "x": copy_to_host(rt, x.grad),
                "ln_w": copy_to_host(rt, ln_w.grad),
                "ln_b": copy_to_host(rt, ln_b.grad),
                "w1": copy_to_host(rt, w1.grad),
                "b1": copy_to_host(rt, b1.grad),
                "w2": copy_to_host(rt, w2.grad),
                "b2": copy_to_host(rt, b2.grad),
            }
            out_host = copy_to_host(rt, out)
            return out_host, grads
        cpu_out, cpu_grads = run_mlp_case("cpu")
        mps_out, mps_grads = run_mlp_case("mps")
        compare("forward", mps_out, cpu_out, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        for key in ("x", "ln_w", "ln_b", "w1", "b1", "w2", "b2"):
            compare(f"{key}.grad", mps_grads[key], cpu_grads[key], atol=GRAD_ATOL, rtol=GRAD_RTOL)

def probe_fused_layernorm_mlp_block(cpu_state, param_arrays, block_idx):
    blk = cpu_state["blocks"][block_idx]
    x_np = np.asarray(blk["x_after_attn"], dtype=np.float32)
    mlp_upstream = np.asarray(blk["mlp_out_grad_after_add"], dtype=np.float32)
    x_grad_before_mlp = np.asarray(blk['x_after_attn_grad_after_add'], dtype=np.float32)
    params = {
        "ln_w": param_arrays[f"blocks[{block_idx}].ln2.gamma"].astype(np.float32),
        "ln_b": param_arrays[f"blocks[{block_idx}].ln2.beta"].astype(np.float32),
        "w1": param_arrays[f"blocks[{block_idx}].mlp.fc1.weight"].astype(np.float32),
        "b1": param_arrays[f"blocks[{block_idx}].mlp.fc1.bias"].astype(np.float32),
        "w2": param_arrays[f"blocks[{block_idx}].mlp.fc2.weight"].astype(np.float32),
        "b2": param_arrays[f"blocks[{block_idx}].mlp.fc2.bias"].astype(np.float32),
    }
    def mlp_oracle(x, upstream, x_grad_before, params, eps=1e-6):
        x = np.asarray(x, dtype=np.float32)
        upstream = np.asarray(upstream, dtype=np.float32)
        B, T, C = x.shape
        N = B * T
        C4 = params["w1"].shape[-1]
        X = x.reshape(N, C)
        dO = upstream.reshape(N, C)
        LN_W = params["ln_w"]
        LN_B = params["ln_b"]
        W1 = params["w1"]
        B1 = params["b1"]
        W2 = params["w2"]
        B2 = params["b2"]
        # forward
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (X - mean) * inv_std
        x_norm = x_hat * LN_W + LN_B
        h1 = x_norm @ W1
        h1_b = h1 + B1
        inner = np.sqrt(2.0 / np.pi) * (h1_b + 0.044715 * h1_b * h1_b * h1_b)
        inner = np.clip(inner, -20.0, 20.0)
        h2 = 0.5 * h1_b * (1.0 + np.tanh(inner))
        out = h2 @ W2 + B2
        # backward
        db2 = np.sum(dO, axis=0)
        dw2 = h2.T @ dO
        dh = dO @ W2.T
        v = h1_b
        inner = np.sqrt(2.0 / np.pi) * (v + 0.044715 * v * v * v)
        inner = np.clip(inner, -20.0, 20.0)
        t = np.tanh(inner)
        left_deriv = 0.5 * (1.0 + t)
        tanh_deriv = 1.0 - t * t
        inner_deriv = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * v * v)
        gelu_grad = left_deriv + 0.5 * v * tanh_deriv * inner_deriv
        dh_act = dh * gelu_grad
        db1 = np.sum(dh_act, axis=0)
        dw1 = x_norm.T @ dh_act
        tmp_dx_mid = dh_act @ W1.T
        dln_w = np.sum(tmp_dx_mid * x_hat, axis=0)
        dln_b = np.sum(tmp_dx_mid, axis=0)
        dx_norm_scaled = tmp_dx_mid * LN_W
        sum_dy = np.sum(dx_norm_scaled, axis=-1, keepdims=True)
        sum_dy_xhat = np.sum(dx_norm_scaled * x_hat, axis=-1, keepdims=True)
        dx_local = (
            (C * dx_norm_scaled - sum_dy - x_hat * sum_dy_xhat)
            * inv_std
            / np.float32(C)
        ).astype(np.float32)
        # dx = upstream.reshape(B, T, C).astype(np.float32) + dx
        dx = np.asarray(x_grad_before, dtype=np.float32).reshape(B, T, C) + dx_local.reshape(B, T, C)
        return {
            # forward
            "out": out.reshape(B, T, C).astype(np.float32),
            "mean": mean.reshape(B, T, 1).astype(np.float32),
            "inv_std": inv_std.reshape(B, T, 1).astype(np.float32),
            "x_hat": x_hat.reshape(B, T, C).astype(np.float32),
            "x_norm": x_norm.reshape(B, T, C).astype(np.float32),
            # MLP hidden
            "h1": h1.reshape(B, T, C4).astype(np.float32),
            "h1_b": h1_b.reshape(B, T, C4).astype(np.float32),
            "h2": h2.reshape(B, T, C4).astype(np.float32),
            "dh": dh.reshape(B, T, C4).astype(np.float32),
            "dh_act": dh_act.reshape(B, T, C4).astype(np.float32),
            # LayerNorm backward
            "tmp_dx_mid": tmp_dx_mid.reshape(B, T, C).astype(np.float32),
            "dx_local": dx_local.reshape(B, T, C).astype(np.float32),
            "dx": dx.reshape(B, T, C).astype(np.float32),
            # parameter gradients
            "dln_w": dln_w.astype(np.float32),
            "dln_b": dln_b.astype(np.float32),
            "dw1": dw1.astype(np.float32),
            "db1": db1.astype(np.float32),
            "dw2": dw2.astype(np.float32),
            "db2": db2.astype(np.float32),
        }
    # mlp_upstream = blk["mlp_out_grad_after_add"]
    # x_grad_before_mlp = blk["x_after_attn_grad_after_add"]
    cpu_ref = mlp_oracle(x_np, mlp_upstream, x_grad_before_mlp, params, eps=1e-6)
    print(f"\n=== FusedLayerNormMLPOp / block {block_idx} ===")
    rt = reset_runtime("mps")
    model = GPT2(
        CFG["V"], CFG["C"], CFG["T"], CFG["n_head"], CFG["B"], CFG["n_layer"], CFG["dtype"], "mps"
    )
    init_model_params(model, param_arrays)
    blk_m = model.blocks[block_idx]
    x = make_tensor(x_np.shape, np.float32, "mps", True, x_np)
    mlp_out = FusedLayerNormMLPOp(eps=blk_m.ln2.eps)(
        x,
        blk_m.ln2.gamma,
        blk_m.ln2.beta,
        blk_m.mlp.fc1.weight,
        blk_m.mlp.fc1.bias,
        blk_m.mlp.fc2.weight,
        blk_m.mlp.fc2.bias,
    )
    out = x + mlp_out
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()
    node = next(n for n in rt.topo if type(n.op).__name__ == "FusedLayerNormMLPOp")
    add_node = out.op
    scratch = node.backward.keywords["scratch"]
    h = scratch["h"]
    dh = scratch["dh"]
    tmp_dx = x.grad.storage
    grad_out = mlp_out.grad
    rt.forward()
    rt.backend.synchronize()
    rt.zero_grad()
    # grad = np.asarray(upstream_np, dtype=np.float32)
    rt.backend.fill(x.grad, 0.0)
    rt.backend.fill(mlp_out.grad, 0.0)
    rt.backend.copy_from_host(out.grad, np.asarray(mlp_upstream, dtype=np.float32))
    rt.backend.begin_pass()
    add_node.backward(add_node.inputs, add_node.outputs)
    rt.backend.end_pass()
    rt.backend.synchronize()
    compare("Add backward / x.grad",
        copy_to_host(rt, x.grad),
        np.asarray(x_grad_before_mlp, dtype=np.float32),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
        fail_fast=True,
    )
    compare("Add backward / mlp_out.grad",
        copy_to_host(rt, mlp_out.grad),
        np.asarray(mlp_upstream, dtype=np.float32),
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
        fail_fast=True,
    )
    # rt.backend.copy_from_host(x.grad, np.asarray(x_grad_before_mlp, dtype=np.float32))
    # rt.backend.synchronize()
    B, T, C = x_np.shape
    N = B * T
    C4 = params["w1"].shape[-1]
    eps_bits = struct.unpack("<i", struct.pack("<f", np.float32(blk_m.ln2.eps)),)[0]
    # print("before backward out.grad =", copy_to_host(rt, out.grad))
    # print("before backward x.grad =", copy_to_host(rt, x.grad))
    # print("before backward out.grad buf/off =", out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset)
    # print("before backward x.grad buf/off =", x.grad.storage.ptr.buffer_id, x.grad.storage.ptr.offset)
    # 1. x_norm -> tmp_dx
    print("\n  [1] layernorm_forward_kernel_v2")
    # print("  before x =", copy_to_host(rt, x))
    # print("  before tmp_dx =", read_f32(rt, tmp_dx, tmp_dx.shape))
    # print("  before gamma =", copy_to_host(rt, blk_m.ln2.gamma))
    # print("  before beta =", copy_to_host(rt, blk_m.ln2.beta))
    rt.backend.begin_pass()
    rt.backend.ctx.dispatch_with_args_3d("layernorm_forward_kernel_v2",
        [x.storage.ptr.buffer_id, blk_m.ln2.gamma.storage.ptr.buffer_id, blk_m.ln2.beta.storage.ptr.buffer_id, tmp_dx.ptr.buffer_id],
        [x.storage.ptr.offset, blk_m.ln2.gamma.storage.ptr.offset, blk_m.ln2.beta.storage.ptr.offset, tmp_dx.ptr.offset,],
        [C], N * 256, 1, 1, 256)
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after tmp_dx =", read_f32(rt, tmp_dx, tmp_dx.shape))
    compare("1 tmp_dx", read_f32(rt, tmp_dx, tmp_dx.shape), cpu_ref["x_norm"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 2. h1_pre = x_norm * W1 -> h
    print("\n  [2] gemm_forward_batched")
    # print("  before tmp_dx =", read_f32(rt, tmp_dx, tmp_dx.shape))
    # print("  before w1 =", copy_to_host(rt, blk_m.mlp.fc1.weight))
    # print("  before h =", read_f32(rt, h, h.shape))
    rt.backend.begin_pass()
    rt.backend.ctx.dispatch_gemm_mps([tmp_dx.ptr.buffer_id, blk_m.mlp.fc1.weight.storage.ptr.buffer_id, h.ptr.buffer_id],
        [tmp_dx.ptr.offset, blk_m.mlp.fc1.weight.storage.ptr.offset, h.ptr.offset],
        [1, N, C4, C, 0, 0, 0, 0], 1.0, 0.0)
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after h =", read_f32(rt, h, h.shape))
    compare("2 h", read_f32(rt, h, h.shape), cpu_ref["h1"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 3. h1_act = GeLU(h1_pre + b1) -> dh
    print("\n  [3] gelu_add_bias_kernel")
    # print("  before h =", read_f32(rt, h, h.shape))
    # print("  before b1 =", copy_to_host(rt, blk_m.mlp.fc1.bias))
    # print("  before dh =", read_f32(rt, dh, dh.shape))
    rt.backend.begin_pass()
    rt.backend.ctx.dispatch_with_args_3d("gelu_add_bias_kernel",
        [h.ptr.buffer_id, blk_m.mlp.fc1.bias.storage.ptr.buffer_id, dh.ptr.buffer_id],
        [h.ptr.offset, blk_m.mlp.fc1.bias.storage.ptr.offset, dh.ptr.offset],
        [N * C4, C4], N * C4, 1, 1, min(N * C4, 256))
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after dh =", read_f32(rt, dh, dh.shape))
    compare("3 dh", read_f32(rt, dh, dh.shape), cpu_ref["h2"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 4. db2 = sum(grad_out, axis=0)
    print("\n  [4] bias_grad_reduce (b2)")
    # print("  before out.grad =", copy_to_host(rt, out.grad))
    # print("  before b2.grad =", copy_to_host(rt, blk_m.mlp.fc2.bias.grad))
    rt.backend.begin_pass()
    rt.backend.ctx.dispatch_with_args_3d("bias_grad_reduce", 
        [grad_out.storage.ptr.buffer_id, blk_m.mlp.fc2.bias.grad.storage.ptr.buffer_id],
        [grad_out.storage.ptr.offset, blk_m.mlp.fc2.bias.grad.storage.ptr.offset],
        [N, C], C, 1, 1, min(C, 256))
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after b2.grad =", copy_to_host(rt, blk_m.mlp.fc2.bias.grad))
    compare("4 b2.grad", copy_to_host(rt, blk_m.mlp.fc2.bias.grad), cpu_ref["db2"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 5. dW2 = h1_act^T * grad_out
    print("\n  [5] gemm_backward_dw_batched_mps (w2)")
    # print("  before dh =", read_f32(rt, dh, dh.shape))
    # print("  before out.grad =", copy_to_host(rt, out.grad))
    # print("  before w2.grad =", copy_to_host(rt, blk_m.mlp.fc2.weight.grad))
    rt.backend.begin_pass()
    rt.backend.ctx.dispatch_gemm_mps([dh.ptr.buffer_id, grad_out.storage.ptr.buffer_id, blk_m.mlp.fc2.weight.grad.storage.ptr.buffer_id],
        [dh.ptr.offset, grad_out.storage.ptr.offset, blk_m.mlp.fc2.weight.grad.storage.ptr.offset],
        [1, C4, C, N, 1, 0, 0, 0], 1.0, 1.0)
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after w2.grad =", copy_to_host(rt, blk_m.mlp.fc2.weight.grad))
    compare("5 w2.grad", copy_to_host(rt, blk_m.mlp.fc2.weight.grad), cpu_ref["dw2"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 6. dh1 = grad_out * W2^T -> dh
    print("\n  [6] gemm_backward_dx_batched_mps (w2)")
    # print("  before out.grad =", copy_to_host(rt, out.grad))
    # print("  before w2 =", copy_to_host(rt, blk_m.mlp.fc2.weight))
    # print("  before dh =", read_f32(rt, dh, dh.shape))
    rt.backend.begin_pass()
    rt.backend.fill_storage(dh, 0.0)
    rt.backend.ctx.dispatch_gemm_mps([grad_out.storage.ptr.buffer_id, blk_m.mlp.fc2.weight.storage.ptr.buffer_id, dh.ptr.buffer_id],
        [grad_out.storage.ptr.offset, blk_m.mlp.fc2.weight.storage.ptr.offset, dh.ptr.offset],
        [1, N, C4, C, 0, 1, 0, 0], 1.0, 1.0)
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after dh =", read_f32(rt, dh, dh.shape))
    compare("6 dh", read_f32(rt, dh, dh.shape), cpu_ref["dh"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 7. dh1_act = dh1 * GeLU_grad(h1_pre + b1) -> dh
    print("\n  [7] gelu_backward_bias_kernel")
    # print("  before dh =", read_f32(rt, dh, dh.shape))
    # print("  before h =", read_f32(rt, h, h.shape))
    # print("  before b1 =", copy_to_host(rt, blk_m.mlp.fc1.bias))
    rt.backend.begin_pass()
    rt.backend.ctx.dispatch_with_args_3d('gelu_backward_bias_kernel',
        [dh.ptr.buffer_id, h.ptr.buffer_id, blk_m.mlp.fc1.bias.storage.ptr.buffer_id, dh.ptr.buffer_id],
        [dh.ptr.offset, h.ptr.offset, blk_m.mlp.fc1.bias.storage.ptr.offset, dh.ptr.offset],
        [N*C4, C4], N*C4, 1, 1, min(N*C4, 256))
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after dh =", read_f32(rt, dh, dh.shape))
    compare("7 dh", read_f32(rt, dh, dh.shape), cpu_ref["dh_act"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 8. db1 = sum(dh1_act, axis=0)
    print("\n  [8] bias_grad_reduce (b1)")
    # print("  before dh =", read_f32(rt, dh, dh.shape))
    # print("  before b1.grad =", copy_to_host(rt, blk_m.mlp.fc1.bias.grad))
    rt.backend.begin_pass()
    rt.backend.ctx.dispatch_with_args_3d('bias_grad_reduce', 
        [dh.ptr.buffer_id, blk_m.mlp.fc1.bias.grad.storage.ptr.buffer_id],
        [dh.ptr.offset, blk_m.mlp.fc1.bias.grad.storage.ptr.offset], 
        [N, C4], C4, 1, 1, min(C4, 256))
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after b1.grad =", copy_to_host(rt, blk_m.mlp.fc1.bias.grad))
    compare("8 b1.grad", copy_to_host(rt, blk_m.mlp.fc1.bias.grad), cpu_ref["db1"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 9. dW1 = x_norm^T * dh1_act
    print("\n  [9] gemm_backward_dw_batched_mps (w1)")
    # print("  before tmp_dx =", read_f32(rt, tmp_dx, tmp_dx.shape))
    # print("  before dh =", read_f32(rt, dh, dh.shape))
    # print("  before w1.grad =", copy_to_host(rt, blk_m.mlp.fc1.weight.grad))
    rt.backend.begin_pass()
    rt.backend.ctx.dispatch_gemm_mps([tmp_dx.ptr.buffer_id, dh.ptr.buffer_id, blk_m.mlp.fc1.weight.grad.storage.ptr.buffer_id],
        [tmp_dx.ptr.offset, dh.ptr.offset, blk_m.mlp.fc1.weight.grad.storage.ptr.offset],
        [1, C, C4, N, 1, 0, 0, 0], 1.0, 1.0)
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after w1.grad =", copy_to_host(rt, blk_m.mlp.fc1.weight.grad))
    compare("9 w1.grad", copy_to_host(rt, blk_m.mlp.fc1.weight.grad), cpu_ref["dw1"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 10. dx_norm = dh1_act * W1^T -> tmp_dx
    print("\n  [10] gemm_backward_dx_batched_mps (w1)")
    # print("  before dh =", read_f32(rt, dh, dh.shape))
    # print("  before w1 =", copy_to_host(rt, blk_m.mlp.fc1.weight))
    # print("  before tmp_dx =", read_f32(rt, tmp_dx, tmp_dx.shape))
    rt.backend.begin_pass()
    rt.backend.fill_storage(tmp_dx, 0.0)
    rt.backend.ctx.dispatch_gemm_mps([dh.ptr.buffer_id, blk_m.mlp.fc1.weight.storage.ptr.buffer_id, tmp_dx.ptr.buffer_id],
        [dh.ptr.offset, blk_m.mlp.fc1.weight.storage.ptr.offset, tmp_dx.ptr.offset],
        [1, N, C, C4, 0, 1, 0, 0], 1.0, 1.0)
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after tmp_dx =", read_f32(rt, tmp_dx, tmp_dx.shape))
    compare("10 tmp_dx", read_f32(rt, tmp_dx, tmp_dx.shape), cpu_ref["tmp_dx_mid"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # 11. LayerNorm Backward
    print("\n  [11] layernorm_backward_kernel_no_stats")
    # print("  before x =", copy_to_host(rt, x))
    # print("  before tmp_dx =", read_f32(rt, tmp_dx, tmp_dx.shape))
    # print("  before gamma =", copy_to_host(rt, blk_m.ln2.gamma))
    # print("  before x.grad =", copy_to_host(rt, x.grad))
    # print("  before ln_w.grad =", copy_to_host(rt, blk_m.ln2.gamma.grad))
    # print("  before ln_b.grad =", copy_to_host(rt, blk_m.ln2.beta.grad))
    rt.backend.begin_pass()
    rt.backend.ctx.dispatch_with_args_3d(
        "layernorm_backward_kernel_no_stats",
        [
            x.storage.ptr.buffer_id,
            tmp_dx.ptr.buffer_id,
            blk_m.ln2.gamma.storage.ptr.buffer_id,
            grad_out.storage.ptr.buffer_id,
            blk_m.ln2.gamma.grad.storage.ptr.buffer_id if blk_m.ln2.gamma.requires_grad else 0,
            blk_m.ln2.beta.grad.storage.ptr.buffer_id if blk_m.ln2.beta.requires_grad else 0,
        ],
        [
            x.storage.ptr.offset,
            tmp_dx.ptr.offset,
            blk_m.ln2.gamma.storage.ptr.offset,
            grad_out.storage.ptr.offset,
            blk_m.ln2.gamma.grad.storage.ptr.offset if blk_m.ln2.gamma.requires_grad else 0,
            blk_m.ln2.beta.grad.storage.ptr.offset if blk_m.ln2.beta.requires_grad else 0,
        ],
        [N, C, eps_bits], 256, (N+7)//8, 1, 256)
    rt.backend.end_pass()
    rt.backend.synchronize()
    # print("  after x.grad =", copy_to_host(rt, x.grad))
    # print("  after ln_w.grad =", copy_to_host(rt, blk_m.ln2.gamma.grad))
    # print("  after ln_b.grad =", copy_to_host(rt, blk_m.ln2.beta.grad))
    compare("11 x.grad", copy_to_host(rt, x.grad), cpu_ref["dx"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("11 ln_w.grad", copy_to_host(rt, blk_m.ln2.gamma.grad), cpu_ref["dln_w"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    compare("11 ln_b.grad", copy_to_host(rt, blk_m.ln2.beta.grad), cpu_ref["dln_b"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # compare("forward", copy_to_host(rt, out), cpu_ref["out"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    residual_out_ref = (
        np.asarray(x_np, dtype=np.float32) +
        np.asarray(cpu_ref["out"], dtype=np.float32)
    )
    compare(
        "forward",
        copy_to_host(rt, out),
        residual_out_ref,
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

def attention_oracle(x_np, params, upstream_grad_np, n_head, eps, is_causal=True, x_grad_before=None):
    x = np.asarray(x_np, dtype=np.float32)
    upstream = np.asarray(upstream_grad_np, dtype=np.float32)
    B, T, C = x.shape
    H = n_head
    D = C // H
    out, cache = fused_ln_fa_np(
        x,
        params["ln_w"],
        params["ln_b"],
        params["wq"],
        params["bq"],
        params["wk"],
        params["bk"],
        params["wv"],
        params["bv"],
        n_head=H,
        eps=eps,
        is_causal=is_causal,
    )
    x_norm = cache["x_norm"].reshape(B, T, C)
    q = cache["q"].reshape(B, T, H, D).transpose(0, 2, 1, 3)
    k = cache["k"].reshape(B, T, H, D).transpose(0, 2, 1, 3)
    v = cache["v"].reshape(B, T, H, D).transpose(0, 2, 1, 3)
    out_grad = upstream.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(D)
    if is_causal:
        scores = scores.copy()
        scores[..., np.triu(np.ones((T, T), dtype=bool), 1)] = -np.inf
    row_max = np.max(scores, axis=-1, keepdims=True)
    lse = (row_max + np.log(np.sum(np.exp(scores - row_max), axis=-1, keepdims=True))).squeeze(-1)
    probs = np.exp(scores - row_max)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    d_val = np.sum(out.reshape(B, T, H, D).transpose(0, 2, 1, 3) * out_grad, axis=-1)
    work = np.matmul(out_grad, v.transpose(0, 1, 3, 2))
    tmp = probs * work
    row_sum = np.sum(tmp, axis=-1, keepdims=True)
    softmax_backward = (work - row_sum) * probs * (1.0 / np.sqrt(D))
    dq_proj = np.matmul(softmax_backward, k).transpose(0, 2, 1, 3).reshape(B, T, C)
    dk_proj = np.matmul(softmax_backward.transpose(0, 1, 3, 2), q).transpose(0, 2, 1, 3).reshape(B, T, C)
    dv_proj = np.matmul(probs.transpose(0, 1, 3, 2), out_grad).transpose(0, 2, 1, 3).reshape(B, T, C)
    dx_norm = (
        np.matmul(dq_proj.reshape(B * T, C), params["wq"].T)
        + np.matmul(dk_proj.reshape(B * T, C), params["wk"].T)
        + np.matmul(dv_proj.reshape(B * T, C), params["wv"].T)
    ).reshape(B, T, C)
    cpu_mean = np.mean(x, axis=-1)
    cpu_inv_std = 1.0 / np.sqrt(np.var(x, axis=-1) + eps)
    x_hat = (x - cpu_mean[..., None]) * cpu_inv_std[..., None]
    dx_norm_scaled = dx_norm * params["ln_w"][None, None, :]
    dx_local = (
        (
            np.float32(C) * dx_norm_scaled
            - np.sum(dx_norm_scaled, axis=-1, keepdims=True)
            - x_hat * np.sum(dx_norm_scaled * x_hat, axis=-1, keepdims=True)
        )
        * cpu_inv_std[..., None]
        / np.float32(C)
    ).astype(np.float32)
    if x_grad_before is None:
        dx = dx_local
    else:
        dx = (np.asarray(x_grad_before, dtype=np.float32) + dx_local).astype(np.float32)
    dln_w = np.sum(dx_norm * x_hat, axis=(0, 1))
    dln_b = np.sum(dx_norm, axis=(0, 1))
    dwq = np.matmul(x_norm.reshape(B * T, C).T, dq_proj.reshape(B * T, C))
    dbq = np.sum(dq_proj.reshape(B * T, C), axis=0)
    dwk = np.matmul(x_norm.reshape(B * T, C).T, dk_proj.reshape(B * T, C))
    dbk = np.sum(dk_proj.reshape(B * T, C), axis=0)
    dwv = np.matmul(x_norm.reshape(B * T, C).T, dv_proj.reshape(B * T, C))
    dbv = np.sum(dv_proj.reshape(B * T, C), axis=0)
    dx_q = np.matmul(dq_proj.reshape(B * T, C), params["wq"].T).reshape(B, T, C)
    dx_k = np.matmul(dk_proj.reshape(B * T, C), params["wk"].T).reshape(B, T, C)
    dx_v = np.matmul(dv_proj.reshape(B * T, C), params["wv"].T).reshape(B, T, C)
    return {
        "out": out,
        "x_norm": x_norm,
        "q": q,
        "k": k,
        "v": v,
        "mean": cpu_mean,
        "inv_std": cpu_inv_std,
        "lse": lse,
        "d_val": d_val,
        "dq_proj": dq_proj,
        "dk_proj": dk_proj,
        "dv_proj": dv_proj,
        "dx_norm": dx_norm,
        "dx_local": dx_local,
        "dx": dx,
        "dln_w": dln_w,
        "dln_b": dln_b,
        "dwq": dwq,
        "dbq": dbq,
        "dwk": dwk,
        "dbk": dbk,
        "dwv": dwv,
        "dbv": dbv,
        "dx_q": dx_q,
        "dx_k": dx_k,
        "dx_v": dx_v,
    }

def _alias_check_close(
    name,
    got,
    ref,
    failures,
    atol=2e-5,
    rtol=2e-5,
):
    got = np.asarray(got)
    ref = np.asarray(ref)

    if got.shape != ref.shape:
        message = (
            f"{name}: shape mismatch: "
            f"got={got.shape}, ref={ref.shape}"
        )
        print(f"  FAIL {message}")
        failures.append(message)
        return False

    if not np.all(np.isfinite(got)):
        message = f"{name}: got contains NaN/Inf"
        print(f"  FAIL {message}")
        failures.append(message)
        return False

    if not np.all(np.isfinite(ref)):
        message = f"{name}: reference contains NaN/Inf"
        print(f"  FAIL {message}")
        failures.append(message)
        return False

    diff = got.astype(np.float64) - ref.astype(np.float64)
    max_abs = (
        float(np.max(np.abs(diff)))
        if diff.size
        else 0.0
    )

    ref_norm = float(
        np.linalg.norm(ref.astype(np.float64).reshape(-1))
    )
    diff_norm = float(
        np.linalg.norm(diff.reshape(-1))
    )
    rel_l2 = diff_norm / max(ref_norm, 1e-30)

    ok = np.allclose(
        got,
        ref,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )

    print(
        f"  {'PASS' if ok else 'FAIL'} {name}: "
        f"max_abs={max_abs:.6e}, "
        f"rel_l2={rel_l2:.6e}, "
        f"atol={atol}, rtol={rtol}"
    )

    if not ok:
        failures.append(
            f"{name}: "
            f"max_abs={max_abs:.6e}, "
            f"rel_l2={rel_l2:.6e}"
        )

    return ok

def _mps_same_address(lhs_storage, rhs_storage):
    return (
        lhs_storage.ptr.buffer_id
        == rhs_storage.ptr.buffer_id
        and
        lhs_storage.ptr.offset
        == rhs_storage.ptr.offset
    )

def _mps_storage_interval(storage):
    nbytes = (
        prod(storage.shape)
        * np.dtype(storage.dtype).itemsize
    )
    return (
        storage.ptr.buffer_id,
        int(storage.ptr.offset),
        int(storage.ptr.offset) + int(nbytes),
    )

def _mps_overlap_bytes(lhs_storage, rhs_storage):
    lhs_buffer, lhs_begin, lhs_end = (
        _mps_storage_interval(lhs_storage)
    )
    rhs_buffer, rhs_begin, rhs_end = (
        _mps_storage_interval(rhs_storage)
    )

    if lhs_buffer != rhs_buffer:
        return 0

    return max(
        0,
        min(lhs_end, rhs_end)
        - max(lhs_begin, rhs_begin),
    )

def probe_fused_linear_add_residual_alias_correctness():
    print(
        "\n=== Alias correctness: "
        "FusedLinearAdd output -> residual x ==="
    )

    rng = np.random.default_rng(
        20260727
    )

    B = 2
    T = 16
    C = 64
    H = 4

    eps = np.float32(1e-6)

    a_np = rng.standard_normal(
        (B, T, C)
    ).astype(np.float32)

    b_np = rng.standard_normal(
        (B, T, C)
    ).astype(np.float32)

    x0_np = (
        a_np + b_np
    ).astype(np.float32)

    ln_w_np = (
        1.0
        + 0.05
        * rng.standard_normal((C,))
    ).astype(np.float32)

    ln_b_np = (
        0.05
        * rng.standard_normal((C,))
    ).astype(np.float32)

    qkv_scale = np.float32(
        0.40 / np.sqrt(C)
    )

    proj_scale = np.float32(
        0.50 / np.sqrt(C)
    )

    wq_np = (
        rng.standard_normal((C, C))
        * qkv_scale
    ).astype(np.float32)

    bq_np = (
        0.03
        * rng.standard_normal((C,))
    ).astype(np.float32)

    wk_np = (
        rng.standard_normal((C, C))
        * qkv_scale
    ).astype(np.float32)

    bk_np = (
        0.03
        * rng.standard_normal((C,))
    ).astype(np.float32)

    wv_np = (
        rng.standard_normal((C, C))
        * qkv_scale
    ).astype(np.float32)

    bv_np = (
        0.03
        * rng.standard_normal((C,))
    ).astype(np.float32)

    proj_w_np = (
        rng.standard_normal((C, C))
        * proj_scale
    ).astype(np.float32)

    proj_b_np = (
        0.03
        * rng.standard_normal((C,))
    ).astype(np.float32)

    upstream_np = (
        0.25
        * rng.standard_normal(
            (B, T, C)
        )
    ).astype(np.float32)

    params_np = {
        "ln_w": ln_w_np,
        "ln_b": ln_b_np,
        "wq": wq_np,
        "bq": bq_np,
        "wk": wk_np,
        "bk": bk_np,
        "wv": wv_np,
        "bv": bv_np,
    }

    rt = reset_runtime("cpu")

    def make_cpu_tensor(array):
        return make_tensor(
            array.shape,
            np.float32,
            "cpu",
            True,
            array,
        )

    a = make_cpu_tensor(a_np)
    b = make_cpu_tensor(b_np)

    parameter_tensors = {
        name: make_cpu_tensor(array)
        for name, array
        in params_np.items()
    }

    proj_w = make_cpu_tensor(
        proj_w_np
    )

    proj_b = make_cpu_tensor(
        proj_b_np
    )

    # x 必须是上游算子的输出，以覆盖 unified arena 场景。
    x = AddOp()(a, b)

    act = FusedLayerNormFlashAttentionOp(
        n_head=H,
        eps=eps,
        is_causal=True,
    )(
        x,
        parameter_tensors["ln_w"],
        parameter_tensors["ln_b"],
        parameter_tensors["wq"],
        parameter_tensors["bq"],
        parameter_tensors["wk"],
        parameter_tensors["bk"],
        parameter_tensors["wv"],
        parameter_tensors["bv"],
    )

    out = FusedLinearAddOp()(
        x,
        act,
        proj_w,
        proj_b,
    )

    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()

    alias_enabled = np.shares_memory(
        x.storage.data,
        out.storage.data,
    )

    print(
        "  residual/output alias:",
        alias_enabled,
    )

    # FA backward 需要原始 residual，所以这个拓扑不允许 alias。
    if alias_enabled:
        raise AssertionError(
            "Illegal FusedLinearAdd residual alias: "
            "the original residual is still required "
            "by FA backward/recompute"
        )

    # --------------------------------------------------------
    # Forward oracle
    # --------------------------------------------------------

    forward_oracle = attention_oracle(
        x0_np,
        params_np,
        np.zeros_like(x0_np),
        n_head=H,
        eps=eps,
        is_causal=True,
        x_grad_before=None,
    )

    attn_out_ref = np.asarray(
        forward_oracle["out"],
        dtype=np.float32,
    )

    out_ref = (
        x0_np
        + np.matmul(
            attn_out_ref,
            proj_w_np,
        )
        + proj_b_np
    ).astype(np.float32)

    # 确认测试参数产生了明显的 residual update。
    if np.allclose(
        out_ref,
        x0_np,
        atol=1e-7,
        rtol=1e-7,
    ):
        raise AssertionError(
            "FusedLinearAdd output is too close "
            "to the original residual; "
            "test parameters are too weak"
        )

    rt.forward()
    rt.backend.synchronize()

    failures = []

    out_got = copy_to_host(
        rt,
        out,
    )

    residual_after_forward = (
        copy_to_host(
            rt,
            x,
        )
    )

    _alias_check_close(
        "forward / FusedLinearAdd output",
        out_got,
        out_ref,
        failures,
    )

    # alias 被禁止，所以 x 必须继续保存原始 residual。
    _alias_check_close(
        "forward / original residual is preserved",
        residual_after_forward,
        x0_np,
        failures,
    )

    # --------------------------------------------------------
    # Backward oracle
    # --------------------------------------------------------

    upstream_2d = upstream_np.reshape(
        B * T,
        C,
    )

    attn_out_ref_2d = (
        attn_out_ref.reshape(
            B * T,
            C,
        )
    )

    dact_ref = np.matmul(
        upstream_2d,
        proj_w_np.T,
    ).reshape(
        B,
        T,
        C,
    ).astype(np.float32)

    dproj_w_ref = np.matmul(
        attn_out_ref_2d.T,
        upstream_2d,
    ).astype(np.float32)

    dproj_b_ref = np.sum(
        upstream_2d,
        axis=0,
    ).astype(np.float32)

    attention_backward_ref = (
        attention_oracle(
            x0_np,
            params_np,
            dact_ref,
            n_head=H,
            eps=eps,
            is_causal=True,
            # residual 分支已经向 x.grad 加入 upstream。
            x_grad_before=upstream_np,
        )
    )

    rt.zero_grad()

    rt.backend.copy_from_host(
        out.grad,
        upstream_np,
    )

    rt.backward_graph()
    rt.backend.synchronize()

    _alias_check_close(
        "backward / projection weight.grad",
        copy_to_host(
            rt,
            proj_w.grad,
        ),
        dproj_w_ref,
        failures,
    )

    _alias_check_close(
        "backward / projection bias.grad",
        copy_to_host(
            rt,
            proj_b.grad,
        ),
        dproj_b_ref,
        failures,
    )

    attention_gradient_checks = [
        (
            "wq.grad",
            parameter_tensors["wq"].grad,
            attention_backward_ref["dwq"],
        ),
        (
            "bq.grad",
            parameter_tensors["bq"].grad,
            attention_backward_ref["dbq"],
        ),
        (
            "wk.grad",
            parameter_tensors["wk"].grad,
            attention_backward_ref["dwk"],
        ),
        (
            "bk.grad",
            parameter_tensors["bk"].grad,
            attention_backward_ref["dbk"],
        ),
        (
            "wv.grad",
            parameter_tensors["wv"].grad,
            attention_backward_ref["dwv"],
        ),
        (
            "bv.grad",
            parameter_tensors["bv"].grad,
            attention_backward_ref["dbv"],
        ),
        (
            "ln_w.grad",
            parameter_tensors["ln_w"].grad,
            attention_backward_ref["dln_w"],
        ),
        (
            "ln_b.grad",
            parameter_tensors["ln_b"].grad,
            attention_backward_ref["dln_b"],
        ),
    ]

    for (
        name,
        tensor_grad,
        reference_grad,
    ) in attention_gradient_checks:
        _alias_check_close(
            f"backward / {name}",
            copy_to_host(
                rt,
                tensor_grad,
            ),
            reference_grad,
            failures,
        )

    # x = a + b，所以两个 leaf 获得相同完整梯度。
    _alias_check_close(
        "backward / a.grad",
        copy_to_host(
            rt,
            a.grad,
        ),
        attention_backward_ref["dx"],
        failures,
    )

    _alias_check_close(
        "backward / b.grad",
        copy_to_host(
            rt,
            b.grad,
        ),
        attention_backward_ref["dx"],
        failures,
    )

    if failures:
        raise AssertionError(
            "FusedLinearAdd residual safety "
            "correctness failed:\n"
            + "\n".join(
                f"  - {item}"
                for item in failures
            )
        )

    print(
        "  PASS illegal FusedLinearAdd residual alias "
        "was rejected and forward/backward are correct"
    )

def probe_attention_output_grad_alias_correctness():
    print(
        "\n=== Alias correctness: "
        "attention output -> attention output.grad ==="
    )

    B = 2
    T = 128
    V = 256
    C = 128
    H = 2
    L = 3

    eps = np.float32(1e-6)

    rng = np.random.default_rng(
        20260728
    )

    rt = reset_runtime("mps")

    model = GPT2(
        V,
        C,
        T,
        H,
        B,
        L,
        np.float32,
        "mps",
    )

    parameter_arrays = {}

    for name, tensor in iter_gpt2_named_tensors(
        model
    ):
        array = (
            rng.standard_normal(
                tensor.shape
            )
            * 0.02
        ).astype(np.float32)

        parameter_arrays[name] = (
            array.copy()
        )

        tensor.init_val = array.copy()

    input_ids_np = rng.integers(
        0,
        V,
        size=(B, T),
        dtype=np.int32,
    )

    pos_ids_np = np.tile(
        np.arange(
            T,
            dtype=np.int32,
        ),
        (B, 1),
    )

    labels_np = rng.integers(
        0,
        V,
        size=(B, T),
        dtype=np.int32,
    )

    input_ids = make_tensor(
        input_ids_np.shape,
        np.int32,
        "mps",
        False,
        input_ids_np,
    )

    pos_ids = make_tensor(
        pos_ids_np.shape,
        np.int32,
        "mps",
        False,
        pos_ids_np,
    )

    labels = make_tensor(
        labels_np.shape,
        np.int32,
        "mps",
        False,
        labels_np,
    )

    x = (
        model.tok_emb(input_ids)
        + model.pos_emb(pos_ids)
    )

    records = []

    for (
        layer_idx,
        block,
    ) in enumerate(model.blocks):
        residual_x = x

        attn_out = (
            FusedLayerNormFlashAttentionOp(
                n_head=block.attn.n_head,
                eps=block.ln1.eps,
                is_causal=True,
            )(
                residual_x,
                block.ln1.gamma,
                block.ln1.beta,
                block.attn.q_proj.weight,
                block.attn.q_proj.bias,
                block.attn.k_proj.weight,
                block.attn.k_proj.bias,
                block.attn.v_proj.weight,
                block.attn.v_proj.bias,
            )
        )

        x_after_attn = FusedLinearAddOp()(
            residual_x,
            attn_out,
            block.attn.proj.weight,
            block.attn.proj.bias,
        )

        mlp_out = FusedLayerNormMLPOp(
            eps=block.ln2.eps,
        )(
            x_after_attn,
            block.ln2.gamma,
            block.ln2.beta,
            block.mlp.fc1.weight,
            block.mlp.fc1.bias,
            block.mlp.fc2.weight,
            block.mlp.fc2.bias,
        )

        x = x_after_attn + mlp_out

        records.append(
            {
                "layer_idx": layer_idx,
                "residual_x": residual_x,
                "attn_out": attn_out,
                "x_after_attn": x_after_attn,
                "fa_node": attn_out.op,
                "params": {
                    "ln_w": parameter_arrays[
                        f"blocks[{layer_idx}].ln1.gamma"
                    ],
                    "ln_b": parameter_arrays[
                        f"blocks[{layer_idx}].ln1.beta"
                    ],
                    "wq": parameter_arrays[
                        f"blocks[{layer_idx}].attn.q_proj.weight"
                    ],
                    "bq": parameter_arrays[
                        f"blocks[{layer_idx}].attn.q_proj.bias"
                    ],
                    "wk": parameter_arrays[
                        f"blocks[{layer_idx}].attn.k_proj.weight"
                    ],
                    "bk": parameter_arrays[
                        f"blocks[{layer_idx}].attn.k_proj.bias"
                    ],
                    "wv": parameter_arrays[
                        f"blocks[{layer_idx}].attn.v_proj.weight"
                    ],
                    "bv": parameter_arrays[
                        f"blocks[{layer_idx}].attn.v_proj.bias"
                    ],
                },
            }
        )

    # 最终 CE 会影响 planner 的完整地址布局，不能省略。
    loss = x.fused_linear_cross_entropy(
        model.lm_head.weight,
        labels,
    )

    rt.loss = loss
    rt.build_topo(loss)
    rt.bind_memory()

    # --------------------------------------------------------
    # Layout correctness
    # --------------------------------------------------------

    layout_failures = []

    for record in records:
        layer_idx = record["layer_idx"]
        residual_x = record["residual_x"]
        attn_out = record["attn_out"]
        x_after_attn = record[
            "x_after_attn"
        ]
        fa_node = record["fa_node"]

        linear_add_alias = (
            _mps_same_address(
                residual_x.storage,
                x_after_attn.storage,
            )
        )

        attention_grad_alias = (
            _mps_same_address(
                attn_out.storage,
                attn_out.grad.storage,
            )
        )

        print(
            f"\n  [layer {layer_idx}]"
        )

        print(
            "    FusedLinearAdd "
            "output/residual alias:",
            linear_add_alias,
        )

        print(
            "    attention "
            "output/output.grad alias:",
            attention_grad_alias,
        )

        # FA backward 需要原始 residual，禁止 FLA 覆盖。
        if linear_add_alias:
            layout_failures.append(
                f"layer {layer_idx}: illegal "
                "FusedLinearAdd output/residual "
                "alias is enabled"
            )

        # attention output/output.grad 策略应该继续生效。
        if not attention_grad_alias:
            layout_failures.append(
                f"layer {layer_idx}: "
                "attention output does not alias "
                "output.grad"
            )

        forward_keywords = getattr(
            fa_node.forward,
            "keywords",
            None,
        )

        if not forward_keywords:
            layout_failures.append(
                f"layer {layer_idx}: "
                "FA forward has no bound scratch"
            )
            continue

        forward_scratch = (
            forward_keywords.get(
                "scratch"
            )
        )

        if forward_scratch is None:
            layout_failures.append(
                f"layer {layer_idx}: "
                "FA forward scratch is None"
            )
            continue

        overlaps = {
            name: _mps_overlap_bytes(
                attn_out.storage,
                forward_scratch[name],
            )
            for name in (
                "q",
                "k",
                "v",
            )
        }

        print(
            "    attention output interval:",
            _mps_storage_interval(
                attn_out.storage
            ),
        )

        for name in (
            "q",
            "k",
            "v",
        ):
            print(
                f"    {name} scratch interval:",
                _mps_storage_interval(
                    forward_scratch[name]
                ),
                f"overlap_bytes="
                f"{overlaps[name]}",
            )

        illegal_overlaps = {
            name: overlap_nbytes
            for (
                name,
                overlap_nbytes,
            ) in overlaps.items()
            if overlap_nbytes > 0
        }

        if illegal_overlaps:
            layout_failures.append(
                f"layer {layer_idx}: "
                "attention output/output.grad "
                "overlaps forward scratch: "
                f"{illegal_overlaps}"
            )

    if layout_failures:
        raise AssertionError(
            "Attention alias safety layout failed:\n"
            + "\n".join(
                f"  - {item}"
                for item in layout_failures
            )
        )

    # --------------------------------------------------------
    # Forward numerical correctness
    # --------------------------------------------------------

    records_by_node_id = {
        id(record["fa_node"]): record
        for record in records
    }

    numerical_failures = []

    backend = rt.backend

    for (
        node_idx,
        node,
    ) in enumerate(rt.topo):
        record = records_by_node_id.get(
            id(node)
        )

        x_before_fa = None

        if record is not None:
            # 前面的节点已经同步完成，此时复制真实 FA input。
            x_before_fa = copy_to_host(
                rt,
                node.inputs[0],
            )

        backend.begin_pass()

        node.forward(
            node.inputs,
            node.outputs,
        )

        backend.end_pass()
        backend.synchronize()

        if record is None:
            continue

        layer_idx = record[
            "layer_idx"
        ]

        attn_got = copy_to_host(
            rt,
            record["attn_out"],
        )

        oracle = attention_oracle(
            x_before_fa,
            record["params"],
            np.zeros_like(
                x_before_fa
            ),
            n_head=H,
            eps=eps,
            is_causal=True,
            x_grad_before=None,
        )

        _alias_check_close(
            (
                f"layer {layer_idx} / "
                "attention forward under "
                "safe output-grad alias"
            ),
            attn_got,
            oracle["out"],
            numerical_failures,
            atol=2e-5,
            rtol=2e-5,
        )

    if numerical_failures:
        raise AssertionError(
            "Attention output -> output.grad "
            "alias produced incorrect forward values:\n"
            + "\n".join(
                f"  - {item}"
                for item
                in numerical_failures
            )
        )

    print(
        "  PASS attention output/output.grad alias "
        "is enabled without Q/K/V scratch overlap"
    )

def run_alias_correctness_probes():
    probes = [
        (
            "FusedLinearAdd residual safety",
            probe_fused_linear_add_residual_alias_correctness,
        ),
        (
            "attention output -> output.grad safety",
            probe_attention_output_grad_alias_correctness,
        ),
    ]

    failures = []

    for (
        name,
        probe,
    ) in probes:
        try:
            probe()
        except Exception as exc:
            print(
                f"\nFAIL alias probe: {name}\n"
                f"  {type(exc).__name__}: {exc}"
            )

            failures.append(
                f"{name}: "
                f"{type(exc).__name__}: {exc}"
            )

    if failures:
        raise AssertionError(
            "Alias correctness probes failed:\n"
            + "\n".join(
                f"  - {item}"
                for item in failures
            )
        )

    print(
        "\nPASS all alias correctness probes"
    )


def build_attention_case(device, x_np, params, n_head, eps):
    x = make_tensor(x_np.shape, np.float32, device, True, x_np)
    ln_w = make_tensor(params["ln_w"].shape, np.float32, device, True, params["ln_w"])
    ln_b = make_tensor(params["ln_b"].shape, np.float32, device, True, params["ln_b"])
    wq = make_tensor(params["wq"].shape, np.float32, device, True, params["wq"])
    bq = make_tensor(params["bq"].shape, np.float32, device, True, params["bq"])
    wk = make_tensor(params["wk"].shape, np.float32, device, True, params["wk"])
    bk = make_tensor(params["bk"].shape, np.float32, device, True, params["bk"])
    wv = make_tensor(params["wv"].shape, np.float32, device, True, params["wv"])
    bv = make_tensor(params["bv"].shape, np.float32, device, True, params["bv"])
    out = FusedLayerNormFlashAttentionOp(n_head=n_head, eps=eps, is_causal=True)(
        x, ln_w, ln_b, wq, bq, wk, bk, wv, bv
    )
    return out, {
        "x": x,
        "ln_w": ln_w,
        "ln_b": ln_b,
        "wq": wq,
        "bq": bq,
        "wk": wk,
        "bk": bk,
        "wv": wv,
        "bv": bv,
    }

def zero_tensor(rt, tensor):
    rt.backend.fill(tensor, 0.0)

def read_lse_stats(rt, lse_st, B, T):
    mean_offset = lse_st.ptr.offset
    inv_std_offset = lse_st.ptr.offset + (B * T * 4)
    mean = np.array(
        rt.backend.ctx.numpy_view_float32(
            lse_st.ptr.buffer_id,
            mean_offset,
            (B * T,),
        )
    ).copy().reshape(B, T)
    inv_std = np.array(
        rt.backend.ctx.numpy_view_float32(
            lse_st.ptr.buffer_id,
            inv_std_offset,
            (B * T,),
        )
    ).copy().reshape(B, T)
    return mean, inv_std

def probe_attention_block(cpu_state, param_arrays, block_idx):
    blk = cpu_state["blocks"][block_idx]

    x_np = np.asarray(
        blk["x_in"],
        dtype=np.float32,
    )
    upstream_np = np.asarray(
        blk["attn_out_grad_after_linear_add"],
        dtype=np.float32,
    )
    x_grad_before_fa_np = np.asarray(
        blk["x_in_grad_before_fa"],
        dtype=np.float32,
    )

    params = {
        "ln_w": param_arrays[
            f"blocks[{block_idx}].ln1.gamma"
        ].astype(np.float32),
        "ln_b": param_arrays[
            f"blocks[{block_idx}].ln1.beta"
        ].astype(np.float32),
        "wq": param_arrays[
            f"blocks[{block_idx}].attn.q_proj.weight"
        ].astype(np.float32),
        "bq": param_arrays[
            f"blocks[{block_idx}].attn.q_proj.bias"
        ].astype(np.float32),
        "wk": param_arrays[
            f"blocks[{block_idx}].attn.k_proj.weight"
        ].astype(np.float32),
        "bk": param_arrays[
            f"blocks[{block_idx}].attn.k_proj.bias"
        ].astype(np.float32),
        "wv": param_arrays[
            f"blocks[{block_idx}].attn.v_proj.weight"
        ].astype(np.float32),
        "bv": param_arrays[
            f"blocks[{block_idx}].attn.v_proj.bias"
        ].astype(np.float32),
    }

    B, T, C = x_np.shape
    H = CFG["n_head"]
    D = C // H
    eps = np.float32(1e-6)

    oracle = attention_oracle(
        x_np,
        params,
        upstream_np,
        n_head=H,
        eps=eps,
        is_causal=True,
        x_grad_before=x_grad_before_fa_np,
    )

    print(
        f"\n=== FusedLayerNormFlashAttentionOp / "
        f"block {block_idx} ==="
    )

    # ------------------------------------------------------------
    # 构建 MPS case
    # ------------------------------------------------------------
    rt = reset_runtime("mps")

    model = GPT2(
        CFG["V"],
        CFG["C"],
        CFG["T"],
        CFG["n_head"],
        CFG["B"],
        CFG["n_layer"],
        CFG["dtype"],
        "mps",
    )
    init_model_params(model, param_arrays)

    blk_m = model.blocks[block_idx]

    x = make_tensor(
        x_np.shape,
        np.float32,
        "mps",
        True,
        x_np,
    )

    out = FusedLayerNormFlashAttentionOp(
        n_head=blk_m.attn.n_head,
        eps=blk_m.ln1.eps,
        is_causal=True,
    )(
        x,
        blk_m.ln1.gamma,
        blk_m.ln1.beta,
        blk_m.attn.q_proj.weight,
        blk_m.attn.q_proj.bias,
        blk_m.attn.k_proj.weight,
        blk_m.attn.k_proj.bias,
        blk_m.attn.v_proj.weight,
        blk_m.attn.v_proj.bias,
    )

    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()

    node = next(
        node
        for node in rt.topo
        if isinstance(
            node.op,
            FusedLayerNormFlashAttentionOp,
        )
    )

    backend = rt.backend
    ctx = backend.ctx

    ln_w = blk_m.ln1.gamma
    ln_b = blk_m.ln1.beta
    wq = blk_m.attn.q_proj.weight
    bq = blk_m.attn.q_proj.bias
    wk = blk_m.attn.k_proj.weight
    bk = blk_m.attn.k_proj.bias
    wv = blk_m.attn.v_proj.weight
    bv = blk_m.attn.v_proj.bias

    eps_bits = struct.unpack(
        "<i",
        struct.pack("<f", float(node.op.eps)),
    )[0]
    is_causal = int(node.op.is_causal)

    # forward 和 backward 使用不同的 scratch storage。
    forward_scratch = node.forward.keywords["scratch"]
    backward_scratch = node.backward.keywords["scratch"]

    fq_st = forward_scratch["q"]
    fk_st = forward_scratch["k"]
    fv_st = forward_scratch["v"]

    q_st = backward_scratch["q"]
    k_st = backward_scratch["k"]
    v_st = backward_scratch["v"]
    dq_st = backward_scratch["dq"]
    dk_st = backward_scratch["dk"]
    dv_st = backward_scratch["dv"]
    lse_st = backward_scratch["lse"]
    dval_st = backward_scratch["d_val"]

    mean_offset = lse_st.ptr.offset
    inv_std_offset = (
        lse_st.ptr.offset + B * T * 4
    )

    def run_stage(name, encode):
        print(f"\n  {name}")
        backend.begin_pass()
        encode()
        backend.end_pass()
        backend.synchronize()

    # ============================================================
    # Forward 1:
    # fused_ln_fa_qkv
    # ============================================================
    def encode_forward_qkv():
        ctx.dispatch_with_args_3d(
            "fused_ln_fa_qkv",
            [
                x.storage.ptr.buffer_id,
                ln_w.storage.ptr.buffer_id,
                ln_b.storage.ptr.buffer_id,
                wq.storage.ptr.buffer_id,
                bq.storage.ptr.buffer_id,
                wk.storage.ptr.buffer_id,
                bk.storage.ptr.buffer_id,
                wv.storage.ptr.buffer_id,
                bv.storage.ptr.buffer_id,
                fq_st.ptr.buffer_id,
                fk_st.ptr.buffer_id,
                fv_st.ptr.buffer_id,
            ],
            [
                x.storage.ptr.offset,
                ln_w.storage.ptr.offset,
                ln_b.storage.ptr.offset,
                wq.storage.ptr.offset,
                bq.storage.ptr.offset,
                wk.storage.ptr.offset,
                bk.storage.ptr.offset,
                wv.storage.ptr.offset,
                bv.storage.ptr.offset,
                fq_st.ptr.offset,
                fk_st.ptr.offset,
                fv_st.ptr.offset,
            ],
            # [B, T, C, H, eps_bits], 32, (B * T + 1) // 2, 1, 32,)
            [B, T, C, H, eps_bits], 256, (B * T + 15) // 16, 1, 256)

    run_stage(
        "forward / fused_ln_fa_qkv",
        encode_forward_qkv,
    )

    compare(
        "q",
        read_f32(rt, fq_st, (B, H, T, D)),
        oracle["q"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "k",
        read_f32(rt, fk_st, (B, H, T, D)),
        oracle["k"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "v",
        read_f32(rt, fv_st, (B, H, T, D)),
        oracle["v"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Forward 2:
    # fused_fa_from_qkv
    # ============================================================
    def encode_forward_attention():
        ctx.dispatch_with_args_3d(
            "fused_fa_from_qkv",
            [
                fq_st.ptr.buffer_id,
                fk_st.ptr.buffer_id,
                fv_st.ptr.buffer_id,
                out.storage.ptr.buffer_id,
            ],
            [
                fq_st.ptr.offset,
                fk_st.ptr.offset,
                fv_st.ptr.offset,
                out.storage.ptr.offset,
            ],
            [B, H, T, C, is_causal],
            T,
            H,
            B,
            1,
        )

    run_stage(
        "forward / fused_fa_from_qkv",
        encode_forward_attention,
    )

    compare(
        "forward",
        copy_to_host(rt, out),
        oracle["out"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # 初始化 backward
    # ============================================================
    rt.zero_grad()

    backend.copy_from_host(
        out.grad,
        upstream_np,
    )
    backend.copy_from_host(
        x.grad,
        x_grad_before_fa_np,
    )
    backend.synchronize()

    # ============================================================
    # Backward 1:
    # recompute Q/K/V
    # ============================================================
    def encode_backward_qkv():
        ctx.dispatch_with_args_3d(
            "fused_ln_fa_qkv",
            [
                x.storage.ptr.buffer_id,
                ln_w.storage.ptr.buffer_id,
                ln_b.storage.ptr.buffer_id,
                wq.storage.ptr.buffer_id,
                bq.storage.ptr.buffer_id,
                wk.storage.ptr.buffer_id,
                bk.storage.ptr.buffer_id,
                wv.storage.ptr.buffer_id,
                bv.storage.ptr.buffer_id,
                q_st.ptr.buffer_id,
                k_st.ptr.buffer_id,
                v_st.ptr.buffer_id,
            ],
            [
                x.storage.ptr.offset,
                ln_w.storage.ptr.offset,
                ln_b.storage.ptr.offset,
                wq.storage.ptr.offset,
                bq.storage.ptr.offset,
                wk.storage.ptr.offset,
                bk.storage.ptr.offset,
                wv.storage.ptr.offset,
                bv.storage.ptr.offset,
                q_st.ptr.offset,
                k_st.ptr.offset,
                v_st.ptr.offset,
            ],
            # [B, T, C, H, eps_bits], 32, (B * T + 1) // 2, 1, 32,)
            [B, T, C, H, eps_bits], 256, (B * T + 15) // 16, 1, 256)

    run_stage(
        "backward / fused_ln_fa_qkv recompute",
        encode_backward_qkv,
    )

    compare(
        "q recompute",
        read_f32(rt, q_st, (B, H, T, D)),
        oracle["q"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "k recompute",
        read_f32(rt, k_st, (B, H, T, D)),
        oracle["k"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "v recompute",
        read_f32(rt, v_st, (B, H, T, D)),
        oracle["v"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 2:
    # recompute attention output, reuse dk_st
    # ============================================================
    def encode_attention_recompute():
        ctx.dispatch_with_args_3d(
            "fused_fa_from_qkv",
            [
                q_st.ptr.buffer_id,
                k_st.ptr.buffer_id,
                v_st.ptr.buffer_id,
                dk_st.ptr.buffer_id,
            ],
            [
                q_st.ptr.offset,
                k_st.ptr.offset,
                v_st.ptr.offset,
                dk_st.ptr.offset,
            ],
            [B, H, T, C, is_causal],
            T,
            H,
            B,
            1,
        )

    run_stage(
        "backward / fused_fa_from_qkv recompute",
        encode_attention_recompute,
    )

    compare(
        "attention output recompute",
        read_f32(rt, dk_st, (B, T, C)),
        oracle["out"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 3:
    # fused_fa_bwd_prepare -> attention LSE + D
    # ============================================================
    def encode_fa_prepare():
        ctx.dispatch_with_args_3d(
            "fused_fa_bwd_prepare",
            [
                q_st.ptr.buffer_id,
                k_st.ptr.buffer_id,
                dk_st.ptr.buffer_id,
                out.grad.storage.ptr.buffer_id,
                lse_st.ptr.buffer_id,
                dval_st.ptr.buffer_id,
            ],
            [
                q_st.ptr.offset,
                k_st.ptr.offset,
                dk_st.ptr.offset,
                out.grad.storage.ptr.offset,
                lse_st.ptr.offset,
                dval_st.ptr.offset,
            ],
            [B, H, T, C, is_causal],
            T,
            H,
            B,
            1,
        )

    run_stage(
        "backward / fused_fa_bwd_prepare",
        encode_fa_prepare,
    )

    compare(
        "attention lse",
        read_f32(rt, lse_st, (B, H, T)),
        oracle["lse"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "d_val",
        read_f32(rt, dval_st, (B, H, T)),
        oracle["d_val"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 4:
    # fused_fa_bwd_dq
    #
    # 注意：kernel 将 dQ 按连续 BTC 写入 dq_st。
    # ============================================================
    def encode_fa_dq():
        ctx.dispatch_with_args_3d(
            "fused_fa_bwd_dq",
            [
                q_st.ptr.buffer_id,
                k_st.ptr.buffer_id,
                v_st.ptr.buffer_id,
                out.grad.storage.ptr.buffer_id,
                dq_st.ptr.buffer_id,
                lse_st.ptr.buffer_id,
                dval_st.ptr.buffer_id,
            ],
            [
                q_st.ptr.offset,
                k_st.ptr.offset,
                v_st.ptr.offset,
                out.grad.storage.ptr.offset,
                dq_st.ptr.offset,
                lse_st.ptr.offset,
                dval_st.ptr.offset,
            ],
            [B, H, T, C, is_causal],
            128,
            (T + 7) // 8,
            B * H,
            128,
        )

    run_stage(
        "backward / fused_fa_bwd_dq",
        encode_fa_dq,
    )

    compare(
        "dq_proj",
        read_f32(rt, dq_st, (B, T, C)),
        oracle["dq_proj"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 5:
    # fused_fa_bwd_dkv
    #
    # dk_st 原来存放 recomputed output，现在被覆盖为 dK。
    # dK/dV 均按连续 BTC 写回。
    # ============================================================
    def encode_fa_dkv():
        ctx.dispatch_with_args_3d(
            "fused_fa_bwd_dkv",
            [
                q_st.ptr.buffer_id,
                k_st.ptr.buffer_id,
                v_st.ptr.buffer_id,
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
                out.grad.storage.ptr.offset,
                dk_st.ptr.offset,
                dv_st.ptr.offset,
                lse_st.ptr.offset,
                dval_st.ptr.offset,
            ],
            [B, H, T, C, is_causal],
            128,
            (T + 7) // 8,
            B * H,
            128,
        )

    run_stage(
        "backward / fused_fa_bwd_dkv",
        encode_fa_dkv,
    )

    compare(
        "dk_proj",
        read_f32(rt, dk_st, (B, T, C)),
        oracle["dk_proj"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "dv_proj",
        read_f32(rt, dv_st, (B, T, C)),
        oracle["dv_proj"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 6:
    # fused_ln_recompute_affine_stats
    #
    # q_st 被覆盖为 LayerNorm affine 输出 Z。
    # lse_st 被覆盖为 mean + inv_std。
    # ============================================================
    def encode_ln_recompute():
        ctx.dispatch_with_args_3d(
            "fused_ln_recompute_affine_stats",
            [
                x.storage.ptr.buffer_id,
                ln_w.storage.ptr.buffer_id,
                ln_b.storage.ptr.buffer_id,
                q_st.ptr.buffer_id,
                lse_st.ptr.buffer_id,
                lse_st.ptr.buffer_id,
            ],
            [
                x.storage.ptr.offset,
                ln_w.storage.ptr.offset,
                ln_b.storage.ptr.offset,
                q_st.ptr.offset,
                mean_offset,
                inv_std_offset,
            ],
            [B, T, C, eps_bits],
            B * T,
            1,
            1,
            1,
        )

    run_stage(
        "backward / fused_ln_recompute_affine_stats",
        encode_ln_recompute,
    )

    compare(
        "x_norm",
        read_f32(rt, q_st, (B, T, C)),
        oracle["x_norm"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    mean_got, inv_std_got = read_lse_stats(
        rt,
        lse_st,
        B,
        T,
    )

    compare(
        "mean",
        mean_got,
        oracle["mean"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "inv_std",
        inv_std_got,
        oracle["inv_std"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 7:
    # dWq / dWk / dWv
    # ============================================================
    def encode_projection_weight_grads():
        ctx.dispatch_gemm_mps(
            [
                q_st.ptr.buffer_id,
                dq_st.ptr.buffer_id,
                wq.grad.storage.ptr.buffer_id,
            ],
            [
                q_st.ptr.offset,
                dq_st.ptr.offset,
                wq.grad.storage.ptr.offset,
            ],
            [1, C, C, B * T, 1, 0, 0, 0],
            1.0,
            0.0,
        )

        ctx.dispatch_gemm_mps(
            [
                q_st.ptr.buffer_id,
                dk_st.ptr.buffer_id,
                wk.grad.storage.ptr.buffer_id,
            ],
            [
                q_st.ptr.offset,
                dk_st.ptr.offset,
                wk.grad.storage.ptr.offset,
            ],
            [1, C, C, B * T, 1, 0, 0, 0],
            1.0,
            0.0,
        )

        ctx.dispatch_gemm_mps(
            [
                q_st.ptr.buffer_id,
                dv_st.ptr.buffer_id,
                wv.grad.storage.ptr.buffer_id,
            ],
            [
                q_st.ptr.offset,
                dv_st.ptr.offset,
                wv.grad.storage.ptr.offset,
            ],
            [1, C, C, B * T, 1, 0, 0, 0],
            1.0,
            0.0,
        )

    run_stage(
        "backward / projection weight GEMMs",
        encode_projection_weight_grads,
    )

    compare(
        "dwq",
        copy_to_host(rt, wq.grad),
        oracle["dwq"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "dwk",
        copy_to_host(rt, wk.grad),
        oracle["dwk"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "dwv",
        copy_to_host(rt, wv.grad),
        oracle["dwv"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 8:
    # fused_qkv_bias_grad
    # ============================================================
    def encode_projection_bias_grads():
        ctx.dispatch_with_args_3d(
            "fused_qkv_bias_grad",
            [
                dq_st.ptr.buffer_id,
                dk_st.ptr.buffer_id,
                dv_st.ptr.buffer_id,
                bq.grad.storage.ptr.buffer_id,
                bk.grad.storage.ptr.buffer_id,
                bv.grad.storage.ptr.buffer_id,
            ],
            [
                dq_st.ptr.offset,
                dk_st.ptr.offset,
                dv_st.ptr.offset,
                bq.grad.storage.ptr.offset,
                bk.grad.storage.ptr.offset,
                bv.grad.storage.ptr.offset,
            ],
            [B * T, C],
            128,
            (C + 3) // 4,
            1,
            128,
        )

    run_stage(
        "backward / fused_qkv_bias_grad",
        encode_projection_bias_grads,
    )

    compare(
        "dbq",
        copy_to_host(rt, bq.grad),
        oracle["dbq"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "dbk",
        copy_to_host(rt, bk.grad),
        oracle["dbk"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "dbv",
        copy_to_host(rt, bv.grad),
        oracle["dbv"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 9a:
    # dZ_q = dQ @ Wq.T
    #
    # k_st 开始被复用为 dZ。
    # ============================================================
    def encode_dx_q():
        ctx.dispatch_gemm_mps(
            [
                dq_st.ptr.buffer_id,
                wq.storage.ptr.buffer_id,
                k_st.ptr.buffer_id,
            ],
            [
                dq_st.ptr.offset,
                wq.storage.ptr.offset,
                k_st.ptr.offset,
            ],
            [1, B * T, C, C, 0, 1, 0, 0],
            1.0,
            0.0,
        )

    run_stage(
        "backward / dQ @ Wq.T",
        encode_dx_q,
    )

    compare(
        "dx_q",
        read_f32(rt, k_st, (B, T, C)),
        oracle["dx_q"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 9b:
    # dZ += dK @ Wk.T
    # ============================================================
    def encode_dx_k():
        ctx.dispatch_gemm_mps(
            [
                dk_st.ptr.buffer_id,
                wk.storage.ptr.buffer_id,
                k_st.ptr.buffer_id,
            ],
            [
                dk_st.ptr.offset,
                wk.storage.ptr.offset,
                k_st.ptr.offset,
            ],
            [1, B * T, C, C, 0, 1, 0, 0],
            1.0,
            1.0,
        )

    run_stage(
        "backward / dK @ Wk.T",
        encode_dx_k,
    )

    compare(
        "dx_q + dx_k",
        read_f32(rt, k_st, (B, T, C)),
        oracle["dx_q"] + oracle["dx_k"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 9c:
    # dZ += dV @ Wv.T
    # ============================================================
    def encode_dx_v():
        ctx.dispatch_gemm_mps(
            [
                dv_st.ptr.buffer_id,
                wv.storage.ptr.buffer_id,
                k_st.ptr.buffer_id,
            ],
            [
                dv_st.ptr.offset,
                wv.storage.ptr.offset,
                k_st.ptr.offset,
            ],
            [1, B * T, C, C, 0, 1, 0, 0],
            1.0,
            1.0,
        )

    run_stage(
        "backward / dV @ Wv.T",
        encode_dx_v,
    )

    compare(
        "dx_norm",
        read_f32(rt, k_st, (B, T, C)),
        oracle["dx_norm"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 10:
    # fused_layernorm_bwd_dx
    #
    # 该 kernel 对 x.grad 使用 +=，因此保留之前写入的
    # x_grad_before_fa_np。
    # ============================================================
    def encode_layernorm_dx():
        ctx.dispatch_with_args_3d(
            "fused_layernorm_bwd_dx",
            [
                x.storage.ptr.buffer_id,
                k_st.ptr.buffer_id,
                ln_w.storage.ptr.buffer_id,
                lse_st.ptr.buffer_id,
                lse_st.ptr.buffer_id,
                x.grad.storage.ptr.buffer_id,
            ],
            [
                x.storage.ptr.offset,
                k_st.ptr.offset,
                ln_w.storage.ptr.offset,
                mean_offset,
                inv_std_offset,
                x.grad.storage.ptr.offset,
            ],
            [B * T, C],
            32,
            B * T,
            1,
            32,
        )

    run_stage(
        "backward / fused_layernorm_bwd_dx",
        encode_layernorm_dx,
    )

    compare(
        "x.grad",
        copy_to_host(rt, x.grad),
        oracle["dx"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

    # ============================================================
    # Backward 11:
    # fused_layernorm_bwd_dgamma_dbeta
    # ============================================================
    def encode_layernorm_parameter_grads():
        ctx.dispatch_with_args_3d(
            "fused_layernorm_bwd_dgamma_dbeta",
            [
                x.storage.ptr.buffer_id,
                k_st.ptr.buffer_id,
                lse_st.ptr.buffer_id,
                lse_st.ptr.buffer_id,
                ln_w.grad.storage.ptr.buffer_id,
                ln_b.grad.storage.ptr.buffer_id,
            ],
            [
                x.storage.ptr.offset,
                k_st.ptr.offset,
                mean_offset,
                inv_std_offset,
                ln_w.grad.storage.ptr.offset,
                ln_b.grad.storage.ptr.offset,
            ],
            [B * T, C],
            128,
            (C + 3) // 4,
            1,
            128,
        )

    run_stage(
        "backward / fused_layernorm_bwd_dgamma_dbeta",
        encode_layernorm_parameter_grads,
    )

    compare(
        "ln_w.grad",
        copy_to_host(rt, ln_w.grad),
        oracle["dln_w"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    compare(
        "ln_b.grad",
        copy_to_host(rt, ln_b.grad),
        oracle["dln_b"],
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )

def main():
    param_arrays = build_param_arrays()
    input_ids, pos_ids, labels = build_inputs()
    cpu_state = build_manual_tiny_gpt2_state("cpu", param_arrays, input_ids, pos_ids, labels, trace=False)
    probe_gather(cpu_state, param_arrays, input_ids, pos_ids)
    probe_fused_linear_ce(cpu_state, param_arrays, labels)
    probe_fused_linear_add(cpu_state, param_arrays)
    probe_fused_linear_add_from_fa(cpu_state, param_arrays)
    probe_fused_layernorm_mlp(cpu_state, param_arrays)
    probe_fused_layernorm_mlp_block(cpu_state, param_arrays, 0)
    for block_idx in range(CFG["n_layer"]):
        probe_attention_block(cpu_state, param_arrays, block_idx)
    mps_state = build_manual_tiny_gpt2_state("mps", param_arrays, input_ids, pos_ids, labels, trace=False)
    full_model_stage_summary(cpu_state, mps_state)
    print("\nAll tiny GPT2 probes finished.")

if __name__ == "__main__":
    main()