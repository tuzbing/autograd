#!/usr/bin/env python3
"""Correctness checker for GatherOp and FusedLinearAddOp on CUDA against the FP32 CPU GPT2 oracle."""

import math
import argparse
import functools
import gc
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ATOL = 1e-6
RTOL = 1e-6
SEED = 1333
FAILURES = []
PRINT_PASSES = True


def _locate_project_root():
    candidates = []
    env_root = os.environ.get("AUTOGRAD_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve().parent
    candidates.extend((Path.cwd(), Path.cwd().parent, here, here.parent))
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "runtime.py").is_file() and (candidate / "module.py").is_file():
            return candidate
    raise RuntimeError(
        "Cannot locate autograd project. Run from the repository directory "
        "or set AUTOGRAD_ROOT=/absolute/path/to/autograd."
    )


PROJECT_ROOT = _locate_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runtime  # noqa: E402
from module import GPT2  # noqa: E402
from op import (
    FusedLinearAddOp, FusedRMSNormMLPOp, FusedRMSNormFlashAttentionOp, FusedLinearCrossEntropyOp
)
from optimizer import AdamW
from tensor import Tensor  # noqa: E402
from utils import f32_bits

Runtime = runtime.Runtime


def make_tensor(shape, dtype, device, requires_grad, init_val, grad_dtype=None):
    return Tensor(
        shape=shape,
        dtype=dtype,
        grad_dtype=grad_dtype,
        device=device,
        requires_grad=requires_grad,
        init_val=np.asarray(init_val, dtype=dtype).copy(),
    )


def grad_dtype_value(value):
    if value == "fp32":
        return np.float32
    if value == "bf16":
        return "bf16"
    raise ValueError(f"unsupported grad_dtype={value!r}")


def iter_gpt2_named_tensors(model):
    yield "tok_emb.weight", model.tok_emb.weight
    for index, block in enumerate(model.blocks):
        p = f"blocks[{index}]"
        yield f"{p}.attn.q_proj.weight", block.attn.q_proj.weight
        yield f"{p}.attn.q_proj.bias", block.attn.q_proj.bias
        yield f"{p}.attn.k_proj.weight", block.attn.k_proj.weight
        yield f"{p}.attn.k_proj.bias", block.attn.k_proj.bias
        yield f"{p}.attn.v_proj.weight", block.attn.v_proj.weight
        yield f"{p}.attn.v_proj.bias", block.attn.v_proj.bias
        yield f"{p}.attn.proj.weight", block.attn.proj.weight
        yield f"{p}.attn.proj.bias", block.attn.proj.bias
        yield f"{p}.mlp.fc1.weight", block.mlp.fc1.weight
        yield f"{p}.mlp.fc2.weight", block.mlp.fc2.weight
    yield "lm_head.weight", model.lm_head.weight


@dataclass(frozen=True)
class Config:
    B: int
    T: int
    V: int
    C: int
    H: int
    D: int
    L: int
    dtype: object = np.float32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify GatherOp and FusedLinearAddOp on CUDA against the FP32 CPU GPT2 oracle"
    )
    parser.add_argument("--depth", "--L", dest="depth", type=int, default=2)
    parser.add_argument("--aspect-ratio", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-seq-len", "--T", dest="max_seq_len", type=int, default=32)
    parser.add_argument("--device-batch-size", "--B", dest="device_batch_size", type=int, default=2)
    parser.add_argument("--vocab-size", "--V", dest="vocab_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--kernel-block", type=int, default=0)
    parser.add_argument("--skip-internal-kernels", action="store_true")
    parser.add_argument("--skip-gradient-accumulation", action="store_true")
    parser.add_argument("--skip-optimizer", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--grad-dtype", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()

    if min(args.depth, args.aspect_ratio, args.head_dim, args.max_seq_len, args.device_batch_size, args.vocab_size) <= 0:
        parser.error("all model dimensions must be positive")

    base_dim = args.depth * args.aspect_ratio
    model_dim = (base_dim + args.head_dim - 1) // args.head_dim * args.head_dim
    if model_dim % 4 != 0:
        parser.error(f"model dimension C={model_dim} must be divisible by 4")

    cfg = Config(
        B=args.device_batch_size,
        T=args.max_seq_len,
        V=args.vocab_size,
        C=model_dim,
        H=model_dim // args.head_dim,
        D=args.head_dim,
        L=args.depth,
    )
    if not 0 <= args.kernel_block < cfg.L:
        parser.error(f"--kernel-block must be in [0, {cfg.L})")
    return args, cfg


def reset_runtime(device):
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device, use_fp16_gemm=False)
    rt.backend.accumulate_parameter_grads = False
    rt.backend.grad_scale = 1.0
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


def tolerances_for_grad_dtype(grad_dtype, value_kind):
    if value_kind in ("loss", "activation", "activation_grad"):
        return 1e-6, 1e-6
    if value_kind == "parameter_grad":
        if str(grad_dtype).lower() in ("bf16", "bfloat16"):
            return 2e-5, 4e-3
        return 1e-6, 1e-6
    raise ValueError(f"unknown value_kind={value_kind!r}")


def check_parameter_gradient(name, got, reference, grad_dtype, fail_fast=False):
    atol, rtol = tolerances_for_grad_dtype(grad_dtype, "parameter_grad")
    return check(name, got, reference, fail_fast=fail_fast, atol=atol, rtol=rtol)


def op_name(node):
    return node.op.op_name


def topology_labels(topo):
    counts = defaultdict(int)
    labels = []
    for node in topo:
        name = op_name(node)
        labels.append(f"{name}[{counts[name]}]")
        counts[name] += 1
    return labels


def verify_training_topology(topo, cfg):
    expected = Counter({
        "Gather": 1,
        "Add": cfg.L,
        "FusedRMSNormFlashAttention": cfg.L,
        "FusedLinearAdd": cfg.L,
        "FusedRMSNormMLP": cfg.L,
        "FusedLinearCrossEntropy": 1,
    })
    actual = Counter(op_name(node) for node in topo)
    if actual != expected:
        raise AssertionError(
            f"GPT2.forward topology differs from expected training graph:\n"
            f"  expected={dict(expected)}\n  actual={dict(actual)}"
        )
    print("Training topology:", " -> ".join(topology_labels(topo)))


def build_parameter_arrays(cfg, seed):
    rng = np.random.default_rng(seed)
    model = GPT2(cfg.V, cfg.C, cfg.T, cfg.H, cfg.B, cfg.L, cfg.dtype, "cpu")
    return {
        name: (rng.standard_normal(tensor.shape).astype(np.float32) * 0.02).copy()
        for name, tensor in iter_gpt2_named_tensors(model)
    }


def initialize_model(model, arrays):
    for name, tensor in iter_gpt2_named_tensors(model):
        tensor.init_val = arrays[name].copy()


def build_inputs(cfg, seed):
    rng = np.random.default_rng(seed + 1)
    input_ids = rng.integers(0, cfg.V, (cfg.B, cfg.T), dtype=np.int32)
    labels = rng.integers(0, cfg.V, (cfg.B, cfg.T), dtype=np.int32)
    return input_ids, labels


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
    view_names = {"View", "Reshape", "Permute", "Transpose"}
    tensors = all_graph_tensors(rt)
    for tensor in tensors:
        if tensor.op is None or op_name(tensor.op) in view_names:
            continue
        tensor.storage = rt.allocator.alloc(tensor.shape, tensor.dtype)
        if tensor.grad is not None:
            tensor.grad.storage = rt.allocator.alloc(tensor.grad.shape, tensor.grad.dtype)
            rt.backend.fill(tensor.grad, 0.0)
    for tensor in tensors:
        if tensor.op is None or op_name(tensor.op) not in view_names:
            continue
        base = tensor.op.inputs[0]
        tensor.storage = rt.allocator.view(base.storage, 0, tensor.shape, tensor.dtype)
        if tensor.grad is not None and base.grad is not None:
            tensor.grad.storage = rt.allocator.view(base.grad.storage, 0, tensor.grad.shape, tensor.grad.dtype)
    for node in rt.topo:
        for phase in ("forward", "backward"):
            layout = getattr(node, f"scratch_{'fwd' if phase == 'forward' else 'bwd'}_layout", None)
            if layout is None:
                continue
            private_storage = rt.allocator.alloc((layout.nbytes,), np.uint8)
            private_scratch = rt.scratch_bind(private_storage, layout)
            bound = getattr(node, phase)
            if isinstance(bound, functools.partial):
                keywords = dict(bound.keywords or {})
                keywords["scratch"] = private_scratch
                bound = functools.partial(bound.func, *bound.args, **keywords)
            else:
                bound = functools.partial(bound, scratch=private_scratch)
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


def build_gpt2_graph(device, cfg, arrays, input_ids, labels, grad_dtype=np.float32):
    rt = reset_runtime(device)
    model = GPT2(cfg.V, cfg.C, cfg.T, cfg.H, cfg.B, cfg.L, cfg.dtype, device)
    for parameter in model.parameters():
        parameter.grad_dtype = grad_dtype if device == "cuda" else np.float32
    initialize_model(model, arrays)
    input_t = make_tensor(input_ids.shape, np.int32, device, False, input_ids)
    labels_t = make_tensor(labels.shape, np.int32, device, False, labels)
    loss = model(input_t, targets=labels_t)
    rt.loss = loss
    rt.build_topo(loss)
    verify_training_topology(rt.topo, cfg)
    rt.bind_memory()
    if device == "cpu":
        isolate_cpu_reference_memory(rt)
    rt.is_compiled = True
    param_by_name = dict(iter_gpt2_named_tensors(model))
    return GraphState(
        rt=rt,
        model=model,
        loss=loss,
        input_ids=input_t,
        labels=labels_t,
        param_by_name=param_by_name,
        param_name_by_id={id(t): n for n, t in param_by_name.items()},
        labels_by_node={id(node): label for node, label in zip(rt.topo, topology_labels(rt.topo))},
    )

def build_cpu_gpt2_graph(cfg, arrays, input_ids, labels):
    return build_gpt2_graph("cpu", cfg, arrays, input_ids, labels)

def run_exact_training_graph(state, repeats=2):
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


def compare_exact_graph(cpu, cuda, cuda_grad_dtype=np.float32):
    print("\n=== Exact Runtime.execute() training path (Full Multi-Layer GPT2) ===")
    loss_atol, loss_rtol = tolerances_for_grad_dtype(cuda_grad_dtype, "loss")
    for repeat, (cpu_result, cuda_result) in enumerate(zip(cpu, cuda), 1):
        print(f"  [training execution {repeat}]")
        check(f"repeat[{repeat}]/loss", cuda_result["loss"], cpu_result["loss"], atol=loss_atol, rtol=loss_rtol)
        for name in sorted(cpu_result["parameter_grads"]):
            check_parameter_gradient(
                f"repeat[{repeat}]/parameter.grad/{name}",
                cuda_result["parameter_grads"][name],
                cpu_result["parameter_grads"][name],
                cuda_grad_dtype,
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
            for tensor, value in zip(node.inputs, reference_inputs[label]):
                rt.backend.copy_from_host(tensor, np.asarray(value, dtype=tensor.dtype))
        forward_inputs[label] = [copy_to_host(rt, tensor) for tensor in node.inputs]
        node.forward(node.inputs, node.outputs)
        rt.backend.end_pass()
        rt.backend.synchronize()
        forward[label] = [copy_to_host(rt, out) for out in node.outputs]
        rt.backend.begin_pass()
    rt.backend.end_pass()

    if rt.allocator.device == "cpu":  # <--- 修改：CUDA 的 FusedLinearCE 在前向已完成梯度写出，不能在此清零
        for tensor in all_graph_tensors(rt):
            if tensor.grad is not None:
                rt.backend.fill(tensor.grad, 0.0)

    rt.backend.fill(state.loss.grad, 1.0)
    rt.backend.begin_pass()
    for node in reversed(rt.topo):
        label = state.labels_by_node[id(node)]
        output_grads = [None if out.grad is None else copy_to_host(rt, out.grad) for out in node.outputs]
        input_grads_before = [None if tensor.grad is None else copy_to_host(rt, tensor.grad) for tensor in node.inputs]
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
    return {"forward": forward, "forward_inputs": forward_inputs, "backward": backward}

def run_staged_cpu_graph(state):
    return run_staged_graph(state)

def compare_staged_graph(cpu, cuda, cuda_grad_dtype=np.float32):
    print("\n=== Per-operator exact training path (Full Multi-Layer GPT2) ===")
    if list(cpu["forward"]) != list(cuda["forward"]):
        raise AssertionError("CPU/CUDA forward topology labels differ")

    act_atol, act_rtol = tolerances_for_grad_dtype(cuda_grad_dtype, "activation")
    loss_atol, loss_rtol = tolerances_for_grad_dtype(cuda_grad_dtype, "loss")
    act_grad_atol, act_grad_rtol = tolerances_for_grad_dtype(cuda_grad_dtype, "activation_grad")

    for label in cpu["forward"]:
        fwd_atol, fwd_rtol = (loss_atol, loss_rtol) if label.startswith("FusedLinearCrossEntropy") else (act_atol, act_rtol)
        for index, (cuda_val, cpu_val) in enumerate(zip(cuda["forward"][label], cpu["forward"][label])):
            check(f"{label}/forward/output[{index}]", cuda_val, cpu_val, atol=fwd_atol, rtol=fwd_rtol)

    for label in reversed(list(cpu["forward"])):
        cpu_bwd = cpu["backward"][label]
        cuda_bwd = cuda["backward"][label]
        for index, (cuda_val, cpu_val) in enumerate(zip(cuda_bwd["output_grads"], cpu_bwd["output_grads"])):
            if cpu_val is not None:
                check(f"{label}/backward/output[{index}].grad", cuda_val, cpu_val, atol=act_grad_atol, rtol=act_grad_rtol)
        for cpu_item, cuda_item in zip(cpu_bwd["inputs"], cuda_bwd["inputs"]):
            pname = cpu_item["parameter"]
            if pname is not None:
                check_parameter_gradient(f"{label}/backward/{pname}.grad", cuda_item["grad"], cpu_item["grad"], cuda_grad_dtype)
            else:
                check(f"{label}/backward/input[{cpu_item['index']}].grad", cuda_item["grad"], cpu_item["grad"], atol=act_grad_atol, rtol=act_grad_rtol)

def _run_op_graph(device, build_fn, upstream_np, grad_dtype=np.float32, accum_steps=1, seed=0):
    rt = reset_runtime(device)
    out, tracked = build_fn(device, grad_dtype if device == "cuda" else np.float32)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()
    if device == "cpu":
        isolate_cpu_reference_memory(rt)
    rt.is_compiled = True

    rng = np.random.default_rng(seed)
    scaled_upstream = np.asarray(upstream_np, dtype=np.float32) / np.float32(accum_steps)
    scalar_grad_scale = float(scaled_upstream.reshape(-1)[0]) if out.numel == 1 else 1.0

    for step in range(accum_steps):
        if step > 0:
            if "ids" in tracked:
                v_size = tracked["tok_emb.weight"].shape[0]
                new_ids = rng.integers(0, v_size, tracked["ids"].shape, dtype=np.int32)
                rt.backend.copy_from_host(tracked["ids"], new_ids)
            if "act" in tracked:
                new_act = (rng.standard_normal(tracked["act"].shape).astype(np.float32) * 0.02).copy()
                rt.backend.copy_from_host(tracked["act"], new_act)
            if "mlp_x" in tracked:
                new_x = (rng.standard_normal(tracked["mlp_x"].shape).astype(np.float32) * 0.02).copy()
                rt.backend.copy_from_host(tracked["mlp_x"], new_x)
            if "fa_x" in tracked:
                new_fa_x = (rng.standard_normal(tracked["fa_x"].shape).astype(np.float32) * 0.02).copy()
                rt.backend.copy_from_host(tracked["fa_x"], new_fa_x)
            if "ce_x" in tracked:
                new_ce_x = (rng.standard_normal(tracked["ce_x"].shape).astype(np.float32) * 0.02).copy()
                rt.backend.copy_from_host(tracked["ce_x"], new_ce_x)
            if "ce_labels" in tracked:
                v_size = tracked["lm_head.weight"].shape[0]
                new_labels = rng.integers(0, v_size, tracked["ce_labels"].shape, dtype=np.int32)
                rt.backend.copy_from_host(tracked["ce_labels"], new_labels)

        rt.accumulate_parameter_grads = step > 0
        rt.backend.accumulate_parameter_grads = step > 0
        rt.grad_scale = scalar_grad_scale
        rt.backend.grad_scale = scalar_grad_scale
        if device == "cpu" and step == 0:
            rt.zero_grad()

        rt.backend.begin_pass()
        for node in rt.topo:
            node.forward(node.inputs, node.outputs)
        rt.backend.end_pass()
        rt.backend.synchronize()

        rt.backend.copy_from_host(out.grad, upstream_np / np.float32(accum_steps))
        if device == "cpu":
            rt.backward_graph()
        else:
            rt.backend.begin_pass()
            for node in reversed(rt.topo):
                node.backward(node.inputs, node.outputs)
            rt.backend.end_pass()
        rt.backend.synchronize()

    return {
        "out": copy_to_host(rt, out),
        "grads": {
            name: copy_to_host(rt, t.grad)
            for name, t in tracked.items()
            if t.requires_grad and t.grad is not None
        },
    }


def _dispatch(rt, callback):
    rt.backend.begin_pass()
    callback()
    rt.backend.end_pass()
    rt.backend.synchronize()


def probe_gather_internal(cfg, ids_np, tok_w_np, expected_out, upstream, expected_dw, grad_dtype=np.float32):
    """对应 check_tiny_gpt2_kernel.py 内部内核级验证: 直接 dispatch GatherOp 的 CUDA 内核。"""
    print("\n=== Internal kernels: Gather ===")
    rt = reset_runtime("cuda")
    ids = make_tensor(ids_np.shape, np.int32, "cuda", False, ids_np)
    w = make_tensor(tok_w_np.shape, np.float32, "cuda", True, tok_w_np, grad_dtype=grad_dtype)
    out = ids.gather(w)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()

    rows = cfg.B * cfg.T
    vec4 = int(cfg.C % 4 == 0 and w.storage.ptr.offset % 16 == 0 and out.storage.ptr.offset % 16 == 0)
    cols_work = (cfg.C >> 2) if vec4 else cfg.C
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch(
            "gather_forward_kernel",
            [ids.storage.ptr.buffer_id, w.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
            [ids.storage.ptr.offset, w.storage.ptr.offset, out.storage.ptr.offset],
            [rows, cfg.C, vec4],
            (cols_work + 255) // 256, min(rows, 65535), 1, 256,
        ),
    )
    check("gather_forward_kernel/output", copy_to_host(rt, out), expected_out)

    rt.backend.fill(w.grad, 0.0)
    rt.backend.copy_from_host(out.grad, np.asarray(upstream, dtype=np.float32))
    out_bf16 = int(np.dtype(w.grad.storage.dtype) == np.dtype(np.uint16))
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch(
            "gather_backward_kernel",
            [ids.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
            [ids.storage.ptr.offset, out.grad.storage.ptr.offset, w.grad.storage.ptr.offset],
            [rows, cfg.C, out_bf16],
            (cfg.C + 255) // 256, min(rows, 65535), 1, 256,
        ),
    )
    check_parameter_gradient("gather_backward_kernel/tok_emb.weight.grad", copy_to_host(rt, w.grad), expected_dw, grad_dtype)


def probe_linear_add_internal(cfg, x_np, act_np, arrays, block, upstream, grad_dtype=np.float32):
    """严格对齐 check_tiny_gpt2_kernel.py 的 probe_linear_add_internal, 替换为对应的 CUDA 内核。"""
    print("\n=== Internal kernels: FusedLinearAdd ===")
    p = f"blocks[{block}]"
    fa_params = {
        "wq": arrays[f"{p}.attn.q_proj.weight"],
        "bq": arrays[f"{p}.attn.q_proj.bias"],
        "wk": arrays[f"{p}.attn.k_proj.weight"],
        "bk": arrays[f"{p}.attn.k_proj.bias"],
        "wv": arrays[f"{p}.attn.v_proj.weight"],
        "bv": arrays[f"{p}.attn.v_proj.bias"],
    }
    fa_ref = attention_oracle(x_np, fa_params, np.zeros_like(x_np), cfg.H)
    proj_w_np = arrays[f"{p}.attn.proj.weight"]
    proj_b_np = arrays[f"{p}.attn.proj.bias"]
    expected_out = (
        x_np.reshape(-1, cfg.C)
        + fa_ref["out"].reshape(-1, cfg.C) @ proj_w_np
        + proj_b_np
    ).reshape(x_np.shape).astype(np.float32)
    dO = np.asarray(upstream, dtype=np.float32)
    expected_dw = (fa_ref["out"].reshape(-1, cfg.C).T @ dO.reshape(-1, cfg.C)).astype(np.float32)
    expected_dact = (dO.reshape(-1, cfg.C) @ proj_w_np.T).reshape(x_np.shape).astype(np.float32)
    expected_db = np.sum(dO.reshape(-1, cfg.C), axis=0).astype(np.float32)

    rt = reset_runtime("cuda")
    x = make_tensor(x_np.shape, np.float32, "cuda", True, x_np)
    names = ("wq", "bq", "wk", "bk", "wv", "bv")
    fa_tensors = {n: make_tensor(fa_params[n].shape, np.float32, "cuda", True, fa_params[n], grad_dtype=grad_dtype) for n in names}
    rope_cos = make_tensor(fa_ref["rope_cos"].shape, np.float32, "cuda", False, fa_ref["rope_cos"])
    rope_sin = make_tensor(fa_ref["rope_sin"].shape, np.float32, "cuda", False, fa_ref["rope_sin"])
    act = FusedRMSNormFlashAttentionOp(n_head=cfg.H, eps=1e-6, is_causal=True)(
        x, *(fa_tensors[n] for n in names), rope_cos, rope_sin
    )
    proj_w = make_tensor(proj_w_np.shape, np.float32, "cuda", True, proj_w_np, grad_dtype=grad_dtype)
    proj_b = make_tensor(proj_b_np.shape, np.float32, "cuda", True, proj_b_np, grad_dtype=grad_dtype)
    out = FusedLinearAddOp()(x, act, proj_w, proj_b)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()

    fla_node = rt.topo[-1]
    sb = fla_node.backward.keywords["scratch"]
    M = cfg.B * cfg.T
    total_pairs = M * cfg.H * (cfg.D // 2)
    eps_bits = f32_bits(1e-6)

    rt.forward()
    rt.backend.synchronize()
    check("fused_linear_add_forward_kernel/output", copy_to_host(rt, out), expected_out)

    for tensor in (x, act, proj_w, proj_b, out, *fa_tensors.values()):
        if tensor.grad is not None:
            rt.backend.fill(tensor.grad, 0.0)
    rt.backend.copy_from_host(out.grad, dO)

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "fused_rms_fa_qkv_kernel",
        [x.storage.ptr.buffer_id] + [fa_tensors[n].storage.ptr.buffer_id for n in names]
        + [act.grad.storage.ptr.buffer_id, sb["fa_k"].ptr.buffer_id, sb["fa_v"].ptr.buffer_id],
        [x.storage.ptr.offset] + [fa_tensors[n].storage.ptr.offset for n in names]
        + [act.grad.storage.ptr.offset, sb["fa_k"].ptr.offset, sb["fa_v"].ptr.offset],
        [cfg.B, cfg.T, cfg.C, cfg.H, eps_bits, 1],
        (cfg.C + 63) // 64, (M + 63) // 64, 3, 256,
    ))
    check("recompute/fused_rms_fa_qkv_kernel/Q", read_storage_f32(rt, act.grad.storage, (cfg.B, cfg.T, cfg.C)), heads_to_btc(fa_ref["q_raw"]))
    check("recompute/fused_rms_fa_qkv_kernel/K", read_storage_f32(rt, sb["fa_k"]), fa_ref["k_raw"])
    check("recompute/fused_rms_fa_qkv_kernel/V", read_storage_f32(rt, sb["fa_v"]), fa_ref["v"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "rope_qk_inplace_kernel",
        [act.grad.storage.ptr.buffer_id, sb["fa_k"].ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
        [act.grad.storage.ptr.offset, sb["fa_k"].ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
        [cfg.B, cfg.H, cfg.T, cfg.D, 2],
        (total_pairs + 255) // 256, 1, 1, 256,
    ))
    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "flash_attn_fwd_kernel",
        [act.grad.storage.ptr.buffer_id, sb["fa_k"].ptr.buffer_id, sb["fa_v"].ptr.buffer_id, act.grad.storage.ptr.buffer_id, 0],
        [act.grad.storage.ptr.offset, sb["fa_k"].ptr.offset, sb["fa_v"].ptr.offset, act.grad.storage.ptr.offset, 0],
        [cfg.B, cfg.H, cfg.T, cfg.C, 1, 0, 1],
        (cfg.T + 31) // 32, cfg.B * cfg.H, 1, 256,
    ))
    check("recompute/flash_attn_fwd_kernel/attention_output", copy_to_host(rt, act.grad), fa_ref["out"])

    dw_bf16 = int(np.dtype(proj_w.grad.storage.dtype) == np.dtype(np.uint16))
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch(
            "cuda_sgemm_tn_kernel",
            [act.grad.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, proj_w.grad.storage.ptr.buffer_id],
            [act.grad.storage.ptr.offset, out.grad.storage.ptr.offset, proj_w.grad.storage.ptr.offset],
            [cfg.C, cfg.C, M, 0, dw_bf16],
            (cfg.C + 63) // 64, (cfg.C + 63) // 64, 1, 256,
        ),
    )
    check_parameter_gradient("cuda_sgemm_tn_kernel/proj_w.grad", copy_to_host(rt, proj_w.grad), expected_dw, grad_dtype)

    numel = M * cfg.C
    vec4 = int(numel % 4 == 0 and out.grad.storage.ptr.offset % 16 == 0 and x.grad.storage.ptr.offset % 16 == 0)
    n_work = (numel >> 2) if vec4 else numel
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch(
            "linear_add_dx_backward_kernel",
            [out.grad.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
            [out.grad.storage.ptr.offset, x.grad.storage.ptr.offset],
            [numel, 1, vec4, 0, cfg.B, cfg.T, cfg.C, 0, cfg.T * cfg.C, cfg.C, 1],
            (n_work + 255) // 256, 1, 1, 256,
        ),
    )
    check("linear_add_dx_backward_kernel/residual.grad", copy_to_host(rt, x.grad), dO)

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch(
            "cuda_sgemm_nt_kernel",
            [out.grad.storage.ptr.buffer_id, proj_w.storage.ptr.buffer_id, act.grad.storage.ptr.buffer_id],
            [out.grad.storage.ptr.offset, proj_w.storage.ptr.offset, act.grad.storage.ptr.offset],
            [M, cfg.C, cfg.C, 0],
            (cfg.C + 63) // 64, (M + 63) // 64, 1, 256,
        ),
    )
    check("cuda_sgemm_nt_kernel/attention.grad", copy_to_host(rt, act.grad), expected_dact)

    db_bf16 = int(np.dtype(proj_b.grad.storage.dtype) == np.dtype(np.uint16))
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch(
            "bias_grad_reduce_kernel",
            [out.grad.storage.ptr.buffer_id, proj_b.grad.storage.ptr.buffer_id],
            [out.grad.storage.ptr.offset, proj_b.grad.storage.ptr.offset],
            [M, cfg.C, 0, db_bf16],
            (cfg.C + 7) // 8, 1, 1, 256,
        ),
    )
    check_parameter_gradient("bias_grad_reduce_kernel/proj_b.grad", copy_to_host(rt, proj_b.grad), expected_db, grad_dtype)

    for tensor in (x, act, proj_w, proj_b):
        rt.backend.fill(tensor.grad, 0.0)
    rt.backend.copy_from_host(out.grad, dO)
    _dispatch(rt, lambda: fla_node.backward(fla_node.inputs, fla_node.outputs))
    check_parameter_gradient("FusedLinearAdd/recompute_bwd/proj_w.grad", copy_to_host(rt, proj_w.grad), expected_dw, grad_dtype)
    check("FusedLinearAdd/recompute_bwd/residual.grad", copy_to_host(rt, x.grad), dO)
    check("FusedLinearAdd/recompute_bwd/attention.grad", copy_to_host(rt, act.grad), expected_dact)
    check_parameter_gradient("FusedLinearAdd/recompute_bwd/proj_b.grad", copy_to_host(rt, proj_b.grad), expected_db, grad_dtype)

def read_storage_f32(rt, storage, shape=None):
    shape = tuple(shape if shape is not None else storage.shape)
    raw = rt.backend.ctx.copy_bytes_to_host(storage.ptr.buffer_id, storage.ptr.offset, int(np.prod(shape)) * 4)
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def rmsnorm_values(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    mean_square = np.mean(x * x, axis=-1, keepdims=True).astype(np.float32)
    inv_rms = (np.float32(1.0) / np.sqrt(mean_square + np.float32(eps))).astype(np.float32)
    x_norm = (x * inv_rms).astype(np.float32)
    return mean_square, inv_rms, x_norm


def mlp_oracle(x, wgu, wdown, upstream, residual_grad, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    wgu = np.asarray(wgu, dtype=np.float32)
    wdown = np.asarray(wdown, dtype=np.float32)
    upstream = np.asarray(upstream, dtype=np.float32)
    residual_grad = np.asarray(residual_grad, dtype=np.float32)

    B, T, C = x.shape
    N = B * T
    H = wgu.shape[1] // 2

    _, inv_rms, x_norm = rmsnorm_values(x, eps)
    XN = x_norm.reshape(N, C)
    GU = XN @ wgu
    G, U = GU[:, :H], GU[:, H:]
    sigmoid_g = 1.0 / (1.0 + np.exp(-G))
    silu_g = G * sigmoid_g
    HIDDEN = silu_g * U
    out = (HIDDEN @ wdown).reshape(B, T, C).astype(np.float32)

    DOUT = upstream.reshape(N, C)
    dwdown = (HIDDEN.T @ DOUT).astype(np.float32)
    d_hidden = DOUT @ wdown.T
    dsilu_dg = sigmoid_g * (1.0 + G * (1.0 - sigmoid_g))
    d_gate = d_hidden * U * dsilu_dg
    d_up = d_hidden * silu_g
    dGU = np.concatenate((d_gate, d_up), axis=-1).astype(np.float32)
    dwgu = (XN.T @ dGU).astype(np.float32)
    dx_norm = (dGU @ wgu.T).reshape(B, T, C).astype(np.float32)

    mean_dy_xnorm = np.mean(dx_norm * x_norm, axis=-1, keepdims=True)
    dx_local = (inv_rms * (dx_norm - x_norm * mean_dy_xnorm)).astype(np.float32)
    dx = (dx_local + residual_grad).astype(np.float32)

    return {
        "x_norm": x_norm,
        "out": out,
        "dx": dx,
        "dx_local": dx_local,
        "dw1": dwgu,
        "dw2": dwdown,
        "gu": GU.reshape(B, T, 2 * H).astype(np.float32),
        "hidden": HIDDEN.reshape(B, T, H).astype(np.float32),
        "dgu": dGU.reshape(B, T, 2 * H).astype(np.float32),
        "dhidden": d_hidden.reshape(B, T, H).astype(np.float32),
        "dx_norm": dx_norm,
    }


def probe_mlp_internal(cfg, x_np, act_np, arrays, block, upstream, residual_grad, grad_dtype=np.float32):
    """对齐 check_tiny_gpt2_kernel.py 的 probe_mlp_internal，并增加内部 CUDA 内核逐级验证。"""
    print("\n=== FusedRMSNormMLP SwiGLU validation ===")
    p = f"blocks[{block}]"
    upstream = np.asarray(upstream, dtype=np.float32).copy()
    residual_grad = np.asarray(residual_grad, dtype=np.float32).copy()

    proj_w_np = arrays[f"{p}.attn.proj.weight"]
    proj_b_np = arrays[f"{p}.attn.proj.bias"]
    wgu_np = arrays[f"{p}.mlp.fc1.weight"]
    wdown_np = arrays[f"{p}.mlp.fc2.weight"]

    x_after_ref = (
        np.asarray(x_np, dtype=np.float32).reshape(-1, cfg.C)
        + np.asarray(act_np, dtype=np.float32).reshape(-1, cfg.C) @ proj_w_np
        + proj_b_np
    ).reshape(x_np.shape).astype(np.float32)

    oracle = mlp_oracle(x_after_ref, wgu_np, wdown_np, upstream, residual_grad, eps=1e-6)
    results = {}

    for device in ("cpu", "cuda"):
        print(f"\n=== FusedRMSNormMLP SwiGLU {device.upper()} validation ===")
        rt = reset_runtime(device)
        gdtype = grad_dtype if device == "cuda" else np.float32

        x = make_tensor(x_np.shape, np.float32, device, True, x_np)
        attn = make_tensor(act_np.shape, np.float32, device, True, act_np)
        proj_w = make_tensor(proj_w_np.shape, np.float32, device, True, proj_w_np, grad_dtype=gdtype)
        proj_b = make_tensor(proj_b_np.shape, np.float32, device, True, proj_b_np, grad_dtype=gdtype)
        x_after = FusedLinearAddOp()(x, attn, proj_w, proj_b)

        wgu = make_tensor(wgu_np.shape, np.float32, device, True, wgu_np, grad_dtype=gdtype)
        wdown = make_tensor(wdown_np.shape, np.float32, device, True, wdown_np, grad_dtype=gdtype)
        mlp = FusedRMSNormMLPOp(eps=1e-6)(x_after, wgu, wdown)

        rt.loss = mlp
        rt.build_topo(mlp)
        rt.bind_memory()
        if device == "cpu":
            isolate_cpu_reference_memory(rt)

        node = next(n for n in rt.topo if op_name(n) == "FusedRMSNormMLP")
        if device == "cuda":
            N = cfg.B * cfg.T
            max_scratch_bytes = 8 * N * cfg.C * 4 + 128
            fwd_bytes = node.scratch_fwd_layout.nbytes if node.scratch_fwd_layout is not None else 0
            bwd_bytes = node.scratch_bwd_layout.nbytes if node.scratch_bwd_layout is not None else 0
            if max(fwd_bytes, bwd_bytes) > max_scratch_bytes:
                raise AssertionError(
                    f"CUDA SwiGLU scratch memory increased: new={max(fwd_bytes, bwd_bytes)}, max={max_scratch_bytes}"
                )

        rt.forward()
        rt.backend.synchronize()
        check(f"{device}/FusedLinearAdd/x_after_vs_oracle", copy_to_host(rt, x_after), x_after_ref, fail_fast=True)
        got_out = copy_to_host(rt, mlp)
        check(f"{device}/FusedRMSNormMLP/forward_vs_oracle", got_out, oracle["out"], fail_fast=True)

        for tensor in (x, attn, proj_w, proj_b, x_after, wgu, wdown, mlp):
            if tensor.grad is not None:
                rt.backend.fill(tensor.grad, 0.0)

        rt.backend.copy_from_host(mlp.grad, upstream)
        rt.backend.copy_from_host(x_after.grad, residual_grad)
        _dispatch(rt, lambda: node.backward(node.inputs, node.outputs))

        got_dx = copy_to_host(rt, x_after.grad)
        got_dwgu = copy_to_host(rt, wgu.grad)
        got_dwdown = copy_to_host(rt, wdown.grad)

        check(f"{device}/FusedRMSNormMLP/backward/x.grad_vs_oracle", got_dx, oracle["dx"], fail_fast=True)
        check_parameter_gradient(
            f"{device}/FusedRMSNormMLP/backward/wgu.grad_vs_oracle",
            got_dwgu, oracle["dw1"], gdtype, fail_fast=True,
        )
        check_parameter_gradient(
            f"{device}/FusedRMSNormMLP/backward/wdown.grad_vs_oracle",
            got_dwdown, oracle["dw2"], gdtype, fail_fast=True,
        )
        results[device] = {"out": got_out, "dx": got_dx, "dwgu": got_dwgu, "dwdown": got_dwdown}

    check("FusedRMSNormMLP/CPU_vs_CUDA/forward", results["cuda"]["out"], results["cpu"]["out"], fail_fast=True)
    check("FusedRMSNormMLP/CPU_vs_CUDA/x.grad", results["cuda"]["dx"], results["cpu"]["dx"], fail_fast=True)
    check_parameter_gradient("FusedRMSNormMLP/CPU_vs_CUDA/wgu.grad", results["cuda"]["dwgu"], results["cpu"]["dwgu"], grad_dtype, fail_fast=True)
    check_parameter_gradient("FusedRMSNormMLP/CPU_vs_CUDA/wdown.grad", results["cuda"]["dwdown"], results["cpu"]["dwdown"], grad_dtype, fail_fast=True)

    print("\n=== Internal kernels: FusedRMSNormMLP ===")
    rt = reset_runtime("cuda")
    x_t = make_tensor(x_after_ref.shape, np.float32, "cuda", True, x_after_ref)
    wgu_t = make_tensor(wgu_np.shape, np.float32, "cuda", True, wgu_np, grad_dtype=grad_dtype)
    wdown_t = make_tensor(wdown_np.shape, np.float32, "cuda", True, wdown_np, grad_dtype=grad_dtype)
    out_t = FusedRMSNormMLPOp(eps=1e-6)(x_t, wgu_t, wdown_t)
    rt.loss = out_t
    rt.build_topo(out_t)
    rt.bind_memory()

    node = rt.topo[-1]
    sf = node.forward.keywords["scratch"]
    sb = node.backward.keywords["scratch"]
    N = cfg.B * cfg.T
    H2 = wgu_np.shape[1]
    H = H2 // 2
    eps_bits = f32_bits(1e-6)

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "rmsnorm_forward_kernel",
        [x_t.storage.ptr.buffer_id, 0, out_t.storage.ptr.buffer_id],
        [x_t.storage.ptr.offset, 0, out_t.storage.ptr.offset],
        [N, cfg.C, eps_bits], (N + 3) // 4, 1, 1, 128,
    ))
    check("rmsnorm_forward_kernel/x_norm", copy_to_host(rt, out_t), oracle["x_norm"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "fused_wgu_swiglu_forward_kernel",
        [out_t.storage.ptr.buffer_id, wgu_t.storage.ptr.buffer_id, sf["hidden"].ptr.buffer_id],
        [out_t.storage.ptr.offset, wgu_t.storage.ptr.offset, sf["hidden"].ptr.offset],
        [N, H, cfg.C], (H + 31) // 32, (N + 63) // 64, 1, 256,
    ))
    check("fused_wgu_swiglu_forward_kernel/hidden", read_storage_f32(rt, sf["hidden"], (cfg.B, cfg.T, H)), oracle["hidden"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "cuda_sgemm_nn_kernel",
        [sf["hidden"].ptr.buffer_id, wdown_t.storage.ptr.buffer_id, out_t.storage.ptr.buffer_id],
        [sf["hidden"].ptr.offset, wdown_t.storage.ptr.offset, out_t.storage.ptr.offset],
        [N, cfg.C, H, 0], (cfg.C + 63) // 64, (N + 63) // 64, 1, 256,
    ))
    check("cuda_sgemm_nn_kernel/output", copy_to_host(rt, out_t), oracle["out"])

    for tensor in (x_t, wgu_t, wdown_t, out_t):
        if tensor.grad is not None:
            rt.backend.fill(tensor.grad, 0.0)
    rt.backend.copy_from_host(out_t.grad, upstream)
    rt.backend.copy_from_host(x_t.grad, residual_grad)

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "rmsnorm_forward_kernel",
        [x_t.storage.ptr.buffer_id, 0, sb["hidden"].ptr.buffer_id],
        [x_t.storage.ptr.offset, 0, sb["hidden"].ptr.offset],
        [N, cfg.C, eps_bits], (N + 3) // 4, 1, 1, 128,
    ))
    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "cuda_sgemm_nn_kernel",
        [sb["hidden"].ptr.buffer_id, wgu_t.storage.ptr.buffer_id, sb["gu"].ptr.buffer_id],
        [sb["hidden"].ptr.offset, wgu_t.storage.ptr.offset, sb["gu"].ptr.offset],
        [N, H2, cfg.C, 0], (H2 + 63) // 64, (N + 63) // 64, 1, 256,
    ))
    check("backward/cuda_sgemm_nn_kernel/gu", read_storage_f32(rt, sb["gu"], (cfg.B, cfg.T, H2)), oracle["gu"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "swiglu_forward_kernel",
        [sb["gu"].ptr.buffer_id, sb["hidden"].ptr.buffer_id],
        [sb["gu"].ptr.offset, sb["hidden"].ptr.offset],
        [N, H, H], (H + 255) // 256, min(N, 65535), 1, 256,
    ))
    check("swiglu_forward_kernel/hidden", read_storage_f32(rt, sb["hidden"], (cfg.B, cfg.T, H)), oracle["hidden"])

    dwdown_bf16 = int(np.dtype(wdown_t.grad.storage.dtype) == np.dtype(np.uint16))
    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "cuda_sgemm_tn_kernel",
        [sb["hidden"].ptr.buffer_id, out_t.grad.storage.ptr.buffer_id, wdown_t.grad.storage.ptr.buffer_id],
        [sb["hidden"].ptr.offset, out_t.grad.storage.ptr.offset, wdown_t.grad.storage.ptr.offset],
        [H, cfg.C, N, 0, dwdown_bf16], (cfg.C + 63) // 64, (H + 63) // 64, 1, 256,
    ))
    check_parameter_gradient("cuda_sgemm_tn_kernel/wdown.grad", copy_to_host(rt, wdown_t.grad), oracle["dw2"], grad_dtype)

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "cuda_sgemm_nt_kernel",
        [out_t.grad.storage.ptr.buffer_id, wdown_t.storage.ptr.buffer_id, sb["hidden"].ptr.buffer_id],
        [out_t.grad.storage.ptr.offset, wdown_t.storage.ptr.offset, sb["hidden"].ptr.offset],
        [N, H, cfg.C, 0], (H + 63) // 64, (N + 63) // 64, 1, 256,
    ))
    check("cuda_sgemm_nt_kernel/dhidden", read_storage_f32(rt, sb["hidden"], (cfg.B, cfg.T, H)), oracle["dhidden"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "swiglu_backward_inplace_kernel",
        [sb["gu"].ptr.buffer_id, sb["hidden"].ptr.buffer_id],
        [sb["gu"].ptr.offset, sb["hidden"].ptr.offset],
        [N, H], (H + 255) // 256, min(N, 65535), 1, 256,
    ))
    check("swiglu_backward_inplace_kernel/dgu", read_storage_f32(rt, sb["gu"], (cfg.B, cfg.T, H2)), oracle["dgu"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "rmsnorm_forward_kernel",
        [x_t.storage.ptr.buffer_id, 0, sb["hidden"].ptr.buffer_id],
        [x_t.storage.ptr.offset, 0, sb["hidden"].ptr.offset],
        [N, cfg.C, eps_bits], (N + 3) // 4, 1, 1, 128,
    ))
    dwgu_bf16 = int(np.dtype(wgu_t.grad.storage.dtype) == np.dtype(np.uint16))
    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "cuda_sgemm_tn_kernel",
        [sb["hidden"].ptr.buffer_id, sb["gu"].ptr.buffer_id, wgu_t.grad.storage.ptr.buffer_id],
        [sb["hidden"].ptr.offset, sb["gu"].ptr.offset, wgu_t.grad.storage.ptr.offset],
        [cfg.C, H2, N, 0, dwgu_bf16], (H2 + 63) // 64, (cfg.C + 63) // 64, 1, 256,
    ))
    check_parameter_gradient("cuda_sgemm_tn_kernel/wgu.grad", copy_to_host(rt, wgu_t.grad), oracle["dw1"], grad_dtype)

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "cuda_sgemm_nt_kernel",
        [sb["gu"].ptr.buffer_id, wgu_t.storage.ptr.buffer_id, sb["hidden"].ptr.buffer_id],
        [sb["gu"].ptr.offset, wgu_t.storage.ptr.offset, sb["hidden"].ptr.offset],
        [N, cfg.C, H2, 0], (cfg.C + 63) // 64, (N + 63) // 64, 1, 256,
    ))
    check("cuda_sgemm_nt_kernel/dx_norm", read_storage_f32(rt, sb["hidden"], (cfg.B, cfg.T, cfg.C)), oracle["dx_norm"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "rmsnorm_backward_kernel",
        [x_t.storage.ptr.buffer_id, sb["hidden"].ptr.buffer_id, 0, x_t.grad.storage.ptr.buffer_id, 0],
        [x_t.storage.ptr.offset, sb["hidden"].ptr.offset, 0, x_t.grad.storage.ptr.offset, 0],
        [N, cfg.C, eps_bits, 1], (N + 3) // 4, 1, 1, 128,
    ))
    check("rmsnorm_backward_kernel/x.grad", copy_to_host(rt, x_t.grad), oracle["dx"])

def probe_add_and_mlp_recompute(cfg, x_in_np, arrays, block, upstream, grad_dtype=np.float32):
    print("\n=== Internal kernels: AddOp & FusedRMSNormMLP x_after_attn recompute ===")
    rng = np.random.default_rng(SEED + 333)
    a_np = (rng.standard_normal((cfg.B, cfg.T, cfg.C)).astype(np.float32) * 0.02).copy()
    b_np = (rng.standard_normal((cfg.B, cfg.T, cfg.C)).astype(np.float32) * 0.02).copy()
    pos_np = (rng.standard_normal((cfg.T, cfg.C)).astype(np.float32) * 0.02).copy()
    bias_np = (rng.standard_normal((cfg.C,)).astype(np.float32) * 0.02).copy()
    d_up = np.asarray(upstream, dtype=np.float32).copy()

    # 1. AddOp 同形状残差（含脏激活梯度覆盖校验）+ [T, C] 广播 + [C] 广播
    for tag, y_val, exp_dy in (
        ("residual", b_np, d_up),
        ("pos_2d", pos_np, np.sum(d_up, axis=0, dtype=np.float32)),
        ("bias_1d", bias_np, np.sum(d_up.reshape(-1, cfg.C), axis=0, dtype=np.float32)),
    ):
        rt = reset_runtime("cuda")
        tx = make_tensor(a_np.shape, np.float32, "cuda", True, a_np)
        ty = make_tensor(y_val.shape, np.float32, "cuda", True, y_val, grad_dtype=(grad_dtype if y_val.ndim < 3 else None))
        tout = tx + ty
        rt.loss = tout
        rt.build_topo(tout)
        rt.bind_memory()

        rt.forward()
        rt.backend.synchronize()
        check(f"AddOp/{tag}/forward", copy_to_host(rt, tout), (a_np + y_val).astype(np.float32))

        # 预填非零脏值，确保激活梯度走覆盖写 (=) 而非错误累加
        rt.backend.fill_storage(tx.grad.storage, -7.0)
        rt.backend.fill_storage(ty.grad.storage, 0.0 if y_val.ndim < 3 else -7.0)
        rt.backend.copy_from_host(tout.grad, d_up)
        add_node = rt.topo[-1]
        _dispatch(rt, lambda add_node=add_node: add_node.backward(add_node.inputs, add_node.outputs))
        check(f"AddOp/{tag}/backward/x.grad", copy_to_host(rt, tx.grad), d_up)
        if y_val.ndim < 3:
            check_parameter_gradient(f"AddOp/{tag}/backward/y.grad", copy_to_host(rt, ty.grad), exp_dy, grad_dtype)
        else:
            check(f"AddOp/{tag}/backward/y.grad", copy_to_host(rt, ty.grad), exp_dy)

    # 2. 完整 GPT2Block 拓扑 FA -> FLA -> MLP -> Add：显式将 x_after.storage 覆写为垃圾值后验证 MLP 反向重算
    p = f"blocks[{block}]"
    fa_params = {
        "wq": arrays[f"{p}.attn.q_proj.weight"],
        "bq": arrays[f"{p}.attn.q_proj.bias"],
        "wk": arrays[f"{p}.attn.k_proj.weight"],
        "bk": arrays[f"{p}.attn.k_proj.bias"],
        "wv": arrays[f"{p}.attn.v_proj.weight"],
        "bv": arrays[f"{p}.attn.v_proj.bias"],
    }
    fa_ref = attention_oracle(x_in_np, fa_params, np.zeros_like(x_in_np), cfg.H)
    proj_w_np = arrays[f"{p}.attn.proj.weight"]
    proj_b_np = arrays[f"{p}.attn.proj.bias"]
    wgu_np = arrays[f"{p}.mlp.fc1.weight"]
    wdown_np = arrays[f"{p}.mlp.fc2.weight"]
    x_after_ref = (
        np.asarray(x_in_np, dtype=np.float32).reshape(-1, cfg.C)
        + fa_ref["out"].reshape(-1, cfg.C) @ proj_w_np
        + proj_b_np
    ).reshape(x_in_np.shape).astype(np.float32)
    oracle = mlp_oracle(x_after_ref, wgu_np, wdown_np, d_up, d_up, eps=1e-6)

    rt = reset_runtime("cuda")
    x_in = make_tensor(x_in_np.shape, np.float32, "cuda", True, x_in_np)
    names = ("wq", "bq", "wk", "bk", "wv", "bv")
    fa_t = {n: make_tensor(fa_params[n].shape, np.float32, "cuda", True, fa_params[n], grad_dtype=grad_dtype) for n in names}
    rcos = make_tensor(fa_ref["rope_cos"].shape, np.float32, "cuda", False, fa_ref["rope_cos"])
    rsin = make_tensor(fa_ref["rope_sin"].shape, np.float32, "cuda", False, fa_ref["rope_sin"])
    attn = FusedRMSNormFlashAttentionOp(n_head=cfg.H, eps=1e-6, is_causal=True)(
        x_in, *(fa_t[n] for n in names), rcos, rsin
    )
    proj_w = make_tensor(proj_w_np.shape, np.float32, "cuda", True, proj_w_np, grad_dtype=grad_dtype)
    proj_b = make_tensor(proj_b_np.shape, np.float32, "cuda", True, proj_b_np, grad_dtype=grad_dtype)
    x_after = FusedLinearAddOp()(x_in, attn, proj_w, proj_b)
    wgu = make_tensor(wgu_np.shape, np.float32, "cuda", True, wgu_np, grad_dtype=grad_dtype)
    wdown = make_tensor(wdown_np.shape, np.float32, "cuda", True, wdown_np, grad_dtype=grad_dtype)
    mlp_out = FusedRMSNormMLPOp(eps=1e-6)(x_after, wgu, wdown)
    block_out = x_after + mlp_out

    rt.loss = block_out
    rt.build_topo(block_out)
    rt.bind_memory()
    rt.forward()
    rt.backend.synchronize()
    check("GPT2Block/Add_inplace/forward", copy_to_host(rt, block_out), (x_after_ref + oracle["out"]).astype(np.float32), fail_fast=True)

    # 强制污染 x_after.storage，确保 cuda_fused_rmsnorm_mlp_backward 完全依赖重算而非残留激活
    rt.backend.fill_storage(x_after.storage, 999.0)
    rt.backend.copy_from_host(block_out.grad, d_up)
    add_node = next(n for n in rt.topo if op_name(n) == "Add")
    mlp_node = next(n for n in rt.topo if op_name(n) == "FusedRMSNormMLP")
    _dispatch(rt, lambda: (
        add_node.backward(add_node.inputs, add_node.outputs),
        mlp_node.backward(mlp_node.inputs, mlp_node.outputs),
    ))
    check("GPT2Block/recompute_fla/x_after.grad", copy_to_host(rt, x_after.grad), oracle["dx"], fail_fast=True)
    check_parameter_gradient("GPT2Block/recompute_fla/wgu.grad", copy_to_host(rt, wgu.grad), oracle["dw1"], grad_dtype, fail_fast=True)
    check_parameter_gradient("GPT2Block/recompute_fla/wdown.grad", copy_to_host(rt, wdown.grad), oracle["dw2"], grad_dtype, fail_fast=True)

def probe_gpt2_grad_accumulation(cfg, arrays, seed, grad_dtype=np.float32):
    print("\n=== Full Multi-Layer GPT2 no-extra-buffer gradient accumulation ===")
    for k_steps in (1, 2, 4):
        atol_g, rtol_g = tolerances_for_grad_dtype(grad_dtype, "parameter_grad")
        if str(grad_dtype).lower() in ("bf16", "bfloat16"):
            atol_g *= k_steps
            rtol_g *= k_steps

        micro_batches = [build_inputs(cfg, seed + 2000 + k_steps * 31 + s) for s in range(k_steps)]
        cpu_state = build_gpt2_graph("cpu", cfg, arrays, *micro_batches[0])
        cuda_state = build_gpt2_graph("cuda", cfg, arrays, *micro_batches[0], grad_dtype=grad_dtype)
        initial_layout = {
            n: (t.grad.storage.ptr.buffer_id, t.grad.storage.ptr.offset, t.grad.storage.nbytes)
            for n, t in cuda_state.param_by_name.items()
        }

        for step, (ids_mb, labels_mb) in enumerate(micro_batches):
            for st in (cpu_state, cuda_state):
                st.rt.backend.copy_from_host(st.input_ids, ids_mb)
                st.rt.backend.copy_from_host(st.labels, labels_mb)
                st.rt.backward(st.loss, grad_scale=1.0 / k_steps, accumulate_parameter_grads=(step > 0))
                st.rt.backend.synchronize()

        after_layout = {
            n: (t.grad.storage.ptr.buffer_id, t.grad.storage.ptr.offset, t.grad.storage.nbytes)
            for n, t in cuda_state.param_by_name.items()
        }
        if after_layout != initial_layout:
            raise AssertionError("CUDA GPT2 gradient accumulation reallocated parameter gradient storage")

        check(f"GPT2_grad_accum[{k_steps}]/last_loss", copy_to_host(cuda_state.rt, cuda_state.loss), copy_to_host(cpu_state.rt, cpu_state.loss))
        for name in sorted(cpu_state.param_by_name):
            check(
                f"GPT2_grad_accum[{k_steps}]/{name}.grad",
                copy_to_host(cuda_state.rt, cuda_state.param_by_name[name].grad),
                copy_to_host(cpu_state.rt, cpu_state.param_by_name[name].grad),
                atol=atol_g,
                rtol=rtol_g,
            )

def read_storage_u8(rt, storage):
    raw = rt.backend.ctx.copy_bytes_to_host(storage.ptr.buffer_id, storage.ptr.offset, int(np.prod(storage.shape)))
    return np.frombuffer(raw, dtype=np.uint8).copy().reshape(-1)

def _optimizer_tolerances(state_mode, value_kind, *, quantized=False, stochastic=False):
    if state_mode in ("fp32", "8bit") or not quantized:
        return (3e-5, 2e-7) if value_kind == "parameter" else (3e-5, 2e-10)
    if value_kind == "parameter":
        return (5e-3, 1e-4) if stochastic else (3e-3, 5e-5)
    if value_kind == "m":
        return (5e-3, 2e-5) if stochastic else (3e-3, 1e-5)
    return 3e-5, 2e-10

def _check_optimizer_value(name, got, ref, *, state_mode, value_kind, quantized=False, stochastic=False, fail_fast=True):
    rtol, atol = _optimizer_tolerances(state_mode, value_kind, quantized=quantized, stochastic=stochastic)
    return check(name, got, ref, fail_fast=fail_fast, atol=atol, rtol=rtol)

def _mini_group_ids_map(rt):
    stochastic_ids, column_ids, head_counts = set(), set(), {}
    for node in rt.topo:
        name = op_name(node)
        if name in ("Gather", "FusedLinearCrossEntropy"):
            stochastic_ids.add(id(node.inputs[1]))
        elif name == "FusedRMSNormMLP":
            wgu = node.inputs[1] if len(node.inputs) == 3 else node.inputs[2]
            wdn = node.inputs[2] if len(node.inputs) == 3 else node.inputs[3]
            column_ids.update((id(wgu), id(wdn)))
        elif name == "FusedLinearAdd":
            column_ids.add(id(node.inputs[2]))
        elif name == "FusedRMSNormFlashAttention":
            for p in (node.inputs[1], node.inputs[3], node.inputs[5]):
                head_counts[id(p)] = int(node.op.n_head)

    def get_ids(p):
        cols = int(p.shape[-1]) if p.ndim else 1
        idx = np.arange(p.numel, dtype=np.int64)
        if id(p) in head_counts:
            head_cols = cols // head_counts[id(p)]
            return (idx % cols) // head_cols
        if id(p) in column_ids:
            return idx % cols
        if id(p) in stochastic_ids:
            return idx // cols
        return idx

    return get_ids

def read_adamw_state(rt, optimizer, index):
    p = optimizer.params[index]
    if optimizer.state_mode == "mini":
        item = optimizer.dispatches[index]
        m_st, meta_st, v_st, quantized = item[3], item[4], item[5], bool(item[13])
        if quantized:
            codes = read_storage_u8(rt, m_st)
            meta = read_storage_f32(rt, meta_st).reshape(-1, 2)
            blocks = np.arange(p.numel, dtype=np.int64) // optimizer.BLOCK_SIZE
            decoded = (codes.astype(np.float32) - meta[blocks, 1]) * meta[blocks, 0]
            if bool(item[12]):
                cols = int(item[9])
                rows = p.numel // cols
                j = np.arange(p.numel, dtype=np.int64)
                logical = np.empty_like(decoded)
                logical[(j % rows) * cols + j // rows] = decoded
                m = logical
            else:
                m = decoded
        else:
            m = read_storage_f32(rt, m_st).reshape(-1)
        return m.reshape(p.shape), read_storage_f32(rt, v_st).reshape(-1)

    if optimizer.state_mode == "8bit" and optimizer.dispatches[index][0] == "q8":
        item = optimizer.dispatches[index]
        blocks = np.arange(p.numel, dtype=np.int64) // optimizer.BLOCK_SIZE
        qm = read_storage_u8(rt, item[3])
        qv = read_storage_u8(rt, item[4])
        sm = read_storage_f32(rt, item[5]).reshape(-1)
        sv = read_storage_f32(rt, item[6]).reshape(-1)
        tm = read_storage_f32(rt, optimizer.qmap_m, (256,))
        tv = read_storage_f32(rt, optimizer.qmap_v, (256,))
        return (tm[qm] * sm[blocks]).reshape(p.shape), (tv[qv] * sv[blocks]).reshape(p.shape)

    return read_storage_f32(rt, optimizer.m[index], p.shape), read_storage_f32(rt, optimizer.v[index], p.shape)

def _adamw_reference_step(p, g, m, v, step, optimizer, group_ids=None, grad_scale=1.0, apply_decay=True):
    b1, b2 = optimizer.betas
    gs = (np.asarray(g, dtype=np.float32).reshape(-1) * np.float32(grad_scale)).astype(np.float32)
    decay = np.float32(1.0 - optimizer.lr * (optimizer.weight_decay if apply_decay else 0.0))
    p_dec = (np.asarray(p, dtype=np.float32).reshape(-1) * decay).astype(np.float32)

    if group_ids is None:
        target_v = (gs * gs).astype(np.float32)
        v_indices = slice(None)
    else:
        counts = np.bincount(group_ids)
        target_v = (np.bincount(group_ids, weights=gs.astype(np.float64) ** 2) / counts).astype(np.float32)
        v_indices = group_ids

    old_m = np.asarray(m, dtype=np.float32).reshape(-1)
    old_v = np.asarray(v, dtype=np.float32).reshape(-1)
    new_m = (old_m + (gs - old_m) * np.float32(1.0 - b1)).astype(np.float32)
    new_v = (old_v + (target_v - old_v) * np.float32(1.0 - b2)).astype(np.float32)

    lr_bc1 = np.float32(optimizer.lr / (1.0 - b1 ** step))
    inv_sqrt_bc2 = np.float32(1.0 / math.sqrt(1.0 - b2 ** step))
    denom = np.sqrt(np.maximum(new_v[v_indices], np.float32(0.0))) * inv_sqrt_bc2 + np.float32(optimizer.eps)
    new_p = (p_dec - lr_bc1 * (new_m / denom)).astype(np.float32)
    return new_p, new_m, new_v

def _check_quant_encoding(rt, label, optimizer, index, ref_m, ref_v):
    if optimizer.state_mode == "fp32":
        return
    item = optimizer.dispatches[index]
    if optimizer.state_mode == "8bit" and item[0] == "q8":
        for tag, q_st, s_st, t_st, ref in (
            ("m", item[3], item[5], optimizer.qmap_m, ref_m),
            ("v", item[4], item[6], optimizer.qmap_v, ref_v),
        ):
            codes = read_storage_u8(rt, q_st)
            table = read_storage_f32(rt, t_st, (256,))
            exp_s = np.maximum(np.maximum.reduceat(np.abs(ref), np.arange(0, ref.size, 256)), np.float32(1e-30))
            got_s = read_storage_f32(rt, s_st).reshape(-1)
            np.testing.assert_allclose(got_s, exp_s, rtol=3e-5, atol=1e-30, err_msg=f"{label}/{tag}/scale")
            norm = np.clip(ref / exp_s[np.arange(ref.size) // 256], table[0], table[-1])
            up = np.clip(np.searchsorted(table, norm, side="left"), 0, 255)
            dn = np.maximum(up - 1, 0)
            best = np.minimum(np.abs(table[up] - norm), np.abs(table[dn] - norm))
            assert np.all(np.abs(table[codes] - norm) <= best + 3e-6), f"{label}/{tag}/nearest-code"
    elif optimizer.state_mode == "mini" and bool(item[13]):
        codes = read_storage_u8(rt, item[3])
        meta = read_storage_f32(rt, item[4]).reshape(-1, 2)
        if bool(item[12]):
            cols = int(item[9])
            rows = ref_m.size // cols
            j = np.arange(ref_m.size, dtype=np.int64)
            exp_q = ref_m[(j % rows) * cols + j // rows]
        else:
            exp_q = ref_m
        bmin = np.minimum.reduceat(exp_q, np.arange(0, exp_q.size, 256))
        bmax = np.maximum.reduceat(exp_q, np.arange(0, exp_q.size, 256))
        exp_s = np.maximum(bmax - bmin, np.float32(1e-5)) / np.float32(255.0)
        np.testing.assert_allclose(meta[:, 0], exp_s, rtol=3e-5, atol=1e-12, err_msg=f"{label}/mini/scale")
        blocks = np.arange(exp_q.size, dtype=np.int64) // 256
        decoded = (codes.astype(np.float32) - meta[blocks, 1]) * meta[blocks, 0]
        assert np.all(np.abs(decoded - exp_q) <= meta[blocks, 0] + 3e-6), f"{label}/mini/decoded"

def _make_optimizer_leaf(rt, array, grad_dtype=np.float32):
    t = Tensor(array.shape, np.float32, device="cuda", requires_grad=True, grad_dtype=grad_dtype, init_val=array.copy())
    t.storage = rt.allocator.alloc(t.shape, np.float32)
    rt.backend.init(t)
    t.grad = Tensor(t.shape, np.float32, device="cuda", requires_grad=False, grad_dtype=grad_dtype)
    g_st_dtype = np.uint16 if str(grad_dtype).lower() in ("bf16", "bfloat16") else np.float32
    t.grad.storage = rt.allocator.alloc(t.shape, g_st_dtype)
    rt.backend.fill_storage2(t.grad.storage, 0.0)
    return t

def probe_grad_clip_and_adamw(cfg, arrays, seed, grad_dtype=np.float32):
    # 1. 内核级校验：覆盖 fp32 / 8bit / mini 三种模式、非对齐尾部尺寸、grad_norm_sq 裁剪与多步状态延续
    opt_cfg = Config(B=1, T=8, V=128, C=128, H=2, D=64, L=1)
    opt_arrays = build_parameter_arrays(opt_cfg, seed + 701)
    opt_inputs = build_inputs(opt_cfg, seed + 702)

    for state_mode in ("fp32", "8bit", "mini"):
        print(f"\n=== grad_norm_sq & AdamW state_mode={state_mode} (CUDA vs NumPy) ===")
        state = build_gpt2_graph("cuda", opt_cfg, opt_arrays, *opt_inputs, grad_dtype=grad_dtype)
        rt = state.rt
        rng = np.random.default_rng(seed + 703)

        cases = list(state.param_by_name.items())
        cases.append(("extra_vector_4101", _make_optimizer_leaf(rt, (rng.standard_normal((4101,)) * 0.02).astype(np.float32), grad_dtype)))
        if state_mode != "mini":
            cases.append(("small_matrix_17x13", _make_optimizer_leaf(rt, (rng.standard_normal((17, 13)) * 0.02).astype(np.float32), grad_dtype)))

        optimizer = AdamW([p for _, p in cases], lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.2, state_mode=state_mode)
        clip_fn = optimizer.build_grad_clip()
        get_mini_ids = _mini_group_ids_map(rt) if state_mode == "mini" else None
        group_ids_list = [get_mini_ids(p) if state_mode == "mini" else None for _, p in cases]

        for step, (wd, max_norm) in enumerate(((0.2, 1.0), (0.0, 0.05), (0.07, 100.0), (0.13, 1.0)), 1):
            optimizer.lr = 0.002 / step
            optimizer.weight_decay = wd
            realized_grads = []
            prev_states = []

            for i, (_, p) in enumerate(cases):
                if step == 1:
                    ids = group_ids_list[i]
                    prev_states.append((
                        np.zeros(p.numel, np.float32),
                        np.zeros(p.numel if ids is None else int(ids.max()) + 1, np.float32),
                    ))
                else:
                    prev_states.append(read_adamw_state(rt, optimizer, i))

                g = (rng.standard_normal(p.shape) * 0.03).astype(np.float32)
                if p.ndim > 1:
                    g *= np.linspace(0.2, 2.0, p.shape[-1], dtype=np.float32)
                if step in (1, 4):
                    g.fill(0.0)
                rt.backend.copy_from_host(p.grad, g)
                realized_grads.append(copy_to_host(rt, p.grad))

            ref_norm = math.sqrt(sum(float(np.sum(rg.astype(np.float64) ** 2)) for rg in realized_grads))
            ref_coef = min(1.0, float(max_norm) / (ref_norm + 1e-6))
            got_norm, got_coef = clip_fn(max_norm)
            check(f"grad_norm_sq/{state_mode}/step{step}/norm", np.float32(got_norm), np.float32(ref_norm), atol=1e-5, rtol=1e-5, fail_fast=True)
            check(f"grad_norm_sq/{state_mode}/step{step}/clip_coef", np.float32(got_coef), np.float32(ref_coef), atol=1e-5, rtol=1e-5, fail_fast=True)

            expected = [
                _adamw_reference_step(
                    copy_to_host(rt, p), realized_grads[i], prev_states[i][0], prev_states[i][1],
                    step, optimizer, group_ids=group_ids_list[i], grad_scale=got_coef, apply_decay=(p.ndim > 1),
                )
                for i, (_, p) in enumerate(cases)
            ]

            optimizer.step(grad_scale=got_coef)
            rt.backend.synchronize()

            for i, (name, p) in enumerate(cases):
                ref_p, ref_m, ref_v = expected[i]
                label = f"AdamW/{state_mode}/{name}/step{step}"
                quantized = (
                    (state_mode == "mini" and bool(optimizer.dispatches[i][13]))
                    or (state_mode == "8bit" and optimizer.dispatches[i][0] == "q8")
                )
                stochastic = (state_mode == "mini" and bool(optimizer.dispatches[i][11]))
                _check_optimizer_value(label + "/p", copy_to_host(rt, p).reshape(-1), ref_p,
                                       state_mode=state_mode, value_kind="parameter",
                                       quantized=quantized, stochastic=stochastic, fail_fast=True)
                got_m, got_v = read_adamw_state(rt, optimizer, i)
                _check_quant_encoding(rt, label, optimizer, i, ref_m, ref_v)
                if not quantized:
                    _check_optimizer_value(label + "/m", got_m.reshape(-1), ref_m,
                                           state_mode=state_mode, value_kind="m", fail_fast=True)
                if not (state_mode == "8bit" and quantized):
                    _check_optimizer_value(label + "/v", got_v.reshape(-1), ref_v,
                                           state_mode=state_mode, value_kind="v",
                                           quantized=quantized, stochastic=stochastic, fail_fast=True)

    # 2. 完整多层 GPT2 端到端单步参数更新校验 (forward + backward + clip_grad_norm + AdamW.step + 更新后 loss 下降验证)
    print("\n=== Full Multi-Layer GPT2 Single-Step Parameter Update (grad_norm_sq + AdamW) ===")
    batch_ids, batch_labels = build_inputs(cfg, seed + 901)
    cpu_eval = build_gpt2_graph("cpu", cfg, arrays, batch_ids, batch_labels)

    for state_mode in ("fp32", "8bit", "mini"):
        cuda_state = build_gpt2_graph("cuda", cfg, arrays, batch_ids, batch_labels, grad_dtype=grad_dtype)
        rt = cuda_state.rt
        optimizer = AdamW(cuda_state.model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, state_mode=state_mode)
        clip_fn = optimizer.build_grad_clip()
        get_mini_ids = _mini_group_ids_map(rt) if state_mode == "mini" else None

        params_before = {n: copy_to_host(rt, p) for n, p in cuda_state.param_by_name.items()}
        cuda_state.loss.backward(grad_scale=1.0, accumulate_parameter_grads=False)
        rt.backend.synchronize()
        loss_before = float(copy_to_host(rt, cuda_state.loss).reshape(-1)[0])
        grads_realized = {n: copy_to_host(rt, p.grad) for n, p in cuda_state.param_by_name.items()}

        ref_norm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads_realized.values()))
        max_norm = max(1e-4, 0.5 * ref_norm)
        got_norm, got_coef = clip_fn(max_norm)
        ref_coef = min(1.0, max_norm / (ref_norm + 1e-6))
        check(f"GPT2_step/{state_mode}/grad_norm", np.float32(got_norm), np.float32(ref_norm), atol=1e-5, rtol=1e-5, fail_fast=True)
        check(f"GPT2_step/{state_mode}/clip_coef", np.float32(got_coef), np.float32(ref_coef), atol=1e-5, rtol=1e-5, fail_fast=True)

        optimizer.step(grad_scale=got_coef)
        rt.backend.synchronize()

        max_delta = 0.0
        for idx, p in enumerate(optimizer.params):
            name = cuda_state.param_name_by_id[id(p)]
            ids = get_mini_ids(p) if state_mode == "mini" else None
            m0 = np.zeros(p.numel, np.float32)
            v0 = np.zeros(p.numel if ids is None else int(ids.max()) + 1, np.float32)
            ref_p, ref_m, ref_v = _adamw_reference_step(
                params_before[name], grads_realized[name], m0, v0,
                1, optimizer, group_ids=ids, grad_scale=got_coef, apply_decay=(p.ndim > 1),
            )
            got_p = copy_to_host(rt, p)
            max_delta = max(max_delta, float(np.max(np.abs(got_p - params_before[name]))))
            quantized = (
                (state_mode == "mini" and bool(optimizer.dispatches[idx][13]))
                or (state_mode == "8bit" and optimizer.dispatches[idx][0] == "q8")
            )
            stochastic = (state_mode == "mini" and bool(optimizer.dispatches[idx][11]))
            _check_optimizer_value(
                f"GPT2_step/{state_mode}/{name}/p", got_p.reshape(-1), ref_p,
                state_mode=state_mode, value_kind="parameter",
                quantized=quantized, stochastic=stochastic, fail_fast=True,
            )
            _check_quant_encoding(rt, f"GPT2_step/{state_mode}/{name}", optimizer, idx, ref_m, ref_v)
            cpu_eval.rt.backend.copy_from_host(cpu_eval.param_by_name[name], ref_p.reshape(p.shape))

        assert max_delta > 0.0, f"GPT2_step/{state_mode}: parameters did not update"
        rt.forward()
        rt.backend.synchronize()
        cpu_eval.rt.forward()
        loss_after_cuda = copy_to_host(rt, cuda_state.loss)
        loss_after_cpu = copy_to_host(cpu_eval.rt, cpu_eval.loss)
        check(f"GPT2_step/{state_mode}/post_step_loss_vs_cpu_ref", loss_after_cuda, loss_after_cpu, atol=1e-5, rtol=1e-5, fail_fast=True)
        loss_after_val = float(loss_after_cuda.reshape(-1)[0])
        print(f"  PASS GPT2_step/{state_mode}: loss {loss_before:.6f} -> {loss_after_val:.6f} (max_dp={max_delta:.4e})")
        if not (loss_after_val < loss_before):
            raise AssertionError(f"GPT2_step/{state_mode}: loss did not decrease ({loss_before} -> {loss_after_val})")

def apply_rope_reference(q, k, cos, sin):
    d2 = q.shape[-1] // 2
    q1, q2 = q[..., :d2].copy(), q[..., d2:].copy()
    k1, k2 = k[..., :d2].copy(), k[..., d2:].copy()
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    q[..., :d2] = q1 * c + q2 * s
    q[..., d2:] = -q1 * s + q2 * c
    k[..., :d2] = k1 * c + k2 * s
    k[..., d2:] = -k1 * s + k2 * c


def apply_rope_backward_reference(dq, dk, cos, sin):
    d2 = dq.shape[-1] // 2
    dq1, dq2 = dq[..., :d2].copy(), dq[..., d2:].copy()
    dk1, dk2 = dk[..., :d2].copy(), dk[..., d2:].copy()
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    dq[..., :d2] = dq1 * c - dq2 * s
    dq[..., d2:] = dq1 * s + dq2 * c
    dk[..., :d2] = dk1 * c - dk2 * s
    dk[..., d2:] = dk1 * s + dk2 * c


def heads_to_btc(value):
    value = np.asarray(value)
    B, H, T, D = value.shape
    return value.transpose(0, 2, 1, 3).reshape(B, T, H * D).copy()


def attention_oracle(x, params, upstream, heads, eps=1e-6, causal=True, rope_theta=100000.0):
    x = np.asarray(x, dtype=np.float32)
    upstream = np.asarray(upstream, dtype=np.float32)
    B, T, C = x.shape
    H, D = heads, C // heads

    positions = np.arange(T, dtype=np.float32)[:, None]
    channels = np.arange(0, D, 2, dtype=np.float32)[None, :]
    freqs = positions * np.asarray(rope_theta ** (-channels / D), dtype=np.float32)
    rope_cos, rope_sin = np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

    _, inv_rms, x_norm = rmsnorm_values(x, eps=eps)
    x_flat = x_norm.reshape(B * T, C)
    q = (x_flat @ params["wq"] + params["bq"]).astype(np.float32).reshape(B, T, H, D).transpose(0, 2, 1, 3).copy()
    k = (x_flat @ params["wk"] + params["bk"]).astype(np.float32).reshape(B, T, H, D).transpose(0, 2, 1, 3).copy()
    v = (x_flat @ params["wv"] + params["bv"]).astype(np.float32).reshape(B, T, H, D).transpose(0, 2, 1, 3).copy()
    q_raw, k_raw = q.copy(), k.copy()
    apply_rope_reference(q, k, rope_cos, rope_sin)

    scale = np.float32(1.0 / math.sqrt(D))
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    if causal:
        scores[..., np.triu(np.ones((T, T), dtype=bool), 1)] = -np.inf
    row_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    row_sum = np.sum(exp_scores, axis=-1, keepdims=True)
    probs = exp_scores / row_sum
    out_h = np.matmul(probs, v)
    out = out_h.transpose(0, 2, 1, 3).reshape(B, T, C).astype(np.float32)

    dO = upstream.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    work = np.matmul(dO, v.transpose(0, 1, 3, 2))
    d_scores = (work - np.sum(probs * work, axis=-1, keepdims=True)) * probs * scale
    dq_rot = np.matmul(d_scores, k).astype(np.float32)
    dk_rot = np.matmul(d_scores.transpose(0, 1, 3, 2), q).astype(np.float32)
    dv = np.matmul(probs.transpose(0, 1, 3, 2), dO).astype(np.float32)

    dq, dk = dq_rot.copy(), dk_rot.copy()
    apply_rope_backward_reference(dq, dk, rope_cos, rope_sin)
    dq2 = dq.transpose(0, 2, 1, 3).reshape(B * T, C)
    dk2 = dk.transpose(0, 2, 1, 3).reshape(B * T, C)
    dv2 = dv.transpose(0, 2, 1, 3).reshape(B * T, C)

    dx_norm = (dq2 @ params["wq"].T + dk2 @ params["wk"].T + dv2 @ params["wv"].T).reshape(B, T, C).astype(np.float32)
    dx_local = (inv_rms * (dx_norm - x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True))).astype(np.float32)

    return {
        "rope_cos": rope_cos, "rope_sin": rope_sin,
        "x_norm": x_norm, "q": q, "k": k, "v": v, "q_raw": q_raw, "k_raw": k_raw,
        "out": out,
        "lse": (row_max[..., 0] + np.log(row_sum[..., 0])).astype(np.float32),
        "d_val": np.sum(out_h * dO, axis=-1).astype(np.float32),
        "dq": dq, "dk": dk, "dv": dv, "dq_rot": dq_rot, "dk_rot": dk_rot,
        "dx_norm": dx_norm, "dx_local": dx_local,
        "dwq": (x_flat.T @ dq2).astype(np.float32),
        "dwk": (x_flat.T @ dk2).astype(np.float32),
        "dwv": (x_flat.T @ dv2).astype(np.float32),
        "dbq": np.sum(dq2, axis=0).astype(np.float32),
        "dbk": np.sum(dk2, axis=0).astype(np.float32),
        "dbv": np.sum(dv2, axis=0).astype(np.float32),
    }


def probe_fa_internal(cfg, x_np, arrays, block, upstream, grad_dtype=np.float32):
    """对齐 check_tiny_gpt2_kernel.py 的 probe_fa_internal，逐级验证 FusedRMSNormFlashAttention 各 CUDA 内核。"""
    print("\n=== Internal kernels: FusedRMSNormFlashAttention ===")
    p = f"blocks[{block}]"
    params = {
        "wq": arrays[f"{p}.attn.q_proj.weight"],
        "bq": arrays[f"{p}.attn.q_proj.bias"],
        "wk": arrays[f"{p}.attn.k_proj.weight"],
        "bk": arrays[f"{p}.attn.k_proj.bias"],
        "wv": arrays[f"{p}.attn.v_proj.weight"],
        "bv": arrays[f"{p}.attn.v_proj.bias"],
    }
    oracle = attention_oracle(x_np, params, upstream, cfg.H)

    rt = reset_runtime("cuda")
    x = make_tensor(x_np.shape, np.float32, "cuda", True, x_np)
    names = ("wq", "bq", "wk", "bk", "wv", "bv")
    p_tensors = {n: make_tensor(params[n].shape, np.float32, "cuda", True, params[n], grad_dtype=grad_dtype) for n in names}
    rope_cos = make_tensor(oracle["rope_cos"].shape, np.float32, "cuda", False, oracle["rope_cos"])
    rope_sin = make_tensor(oracle["rope_sin"].shape, np.float32, "cuda", False, oracle["rope_sin"])
    out = FusedRMSNormFlashAttentionOp(n_head=cfg.H, eps=1e-6, is_causal=True)(
        x, *(p_tensors[n] for n in names), rope_cos, rope_sin
    )
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()

    node = rt.topo[-1]
    sf = node.forward.keywords["scratch"]
    sb = node.backward.keywords["scratch"]
    B, T, C, H, D = cfg.B, cfg.T, cfg.C, cfg.H, cfg.D
    N = B * T
    total_pairs = N * H * (D // 2)
    eps_bits = f32_bits(1e-6)

    def dispatch_qkv(scratch):
        rt.backend.ctx.dispatch(
            "fused_rms_fa_qkv_kernel",
            [x.storage.ptr.buffer_id] + [p_tensors[n].storage.ptr.buffer_id for n in names]
            + [scratch[n].ptr.buffer_id for n in ("q", "k", "v")],
            [x.storage.ptr.offset] + [p_tensors[n].storage.ptr.offset for n in names]
            + [scratch[n].ptr.offset for n in ("q", "k", "v")],
            [B, T, C, H, eps_bits, 0], (C + 63) // 64, (N + 63) // 64, 3, 256,
        )

    def dispatch_rope(q_st, k_st, mode):
        rt.backend.ctx.dispatch(
            "rope_qk_inplace_kernel",
            [q_st.ptr.buffer_id, k_st.ptr.buffer_id, rope_cos.storage.ptr.buffer_id, rope_sin.storage.ptr.buffer_id],
            [q_st.ptr.offset, k_st.ptr.offset, rope_cos.storage.ptr.offset, rope_sin.storage.ptr.offset],
            [B, H, T, D, mode], (total_pairs + 255) // 256, 1, 1, 256,
        )

    _dispatch(rt, lambda: dispatch_qkv(sf))
    check("fused_rms_fa_qkv_kernel/Q", read_storage_f32(rt, sf["q"]), oracle["q_raw"])
    check("fused_rms_fa_qkv_kernel/K", read_storage_f32(rt, sf["k"]), oracle["k_raw"])
    check("fused_rms_fa_qkv_kernel/V", read_storage_f32(rt, sf["v"]), oracle["v"])

    _dispatch(rt, lambda: dispatch_rope(sf["q"], sf["k"], 0))
    check("rope_qk_inplace_kernel/Q", read_storage_f32(rt, sf["q"]), oracle["q"])
    check("rope_qk_inplace_kernel/K", read_storage_f32(rt, sf["k"]), oracle["k"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "flash_attn_fwd_kernel",
        [sf["q"].ptr.buffer_id, sf["k"].ptr.buffer_id, sf["v"].ptr.buffer_id, out.storage.ptr.buffer_id, 0],
        [sf["q"].ptr.offset, sf["k"].ptr.offset, sf["v"].ptr.offset, out.storage.ptr.offset, 0],
        [B, H, T, C, 1, 0, 0], (T + 31) // 32, B * H, 1, 256,
    ))
    check("flash_attn_fwd_kernel/output", copy_to_host(rt, out), oracle["out"])

    for tensor in [x, out, *p_tensors.values()]:
        if tensor.grad is not None:
            rt.backend.fill(tensor.grad, 0.0)
    rt.backend.copy_from_host(out.grad, np.asarray(upstream, dtype=np.float32))

    _dispatch(rt, lambda: dispatch_qkv(sb))
    _dispatch(rt, lambda: dispatch_rope(sb["q"], sb["k"], 0))
    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "flash_attn_fwd_kernel",
        [sb["q"].ptr.buffer_id, sb["k"].ptr.buffer_id, sb["v"].ptr.buffer_id, sb["dk"].ptr.buffer_id, sb["lse"].ptr.buffer_id],
        [sb["q"].ptr.offset, sb["k"].ptr.offset, sb["v"].ptr.offset, sb["dk"].ptr.offset, sb["lse"].ptr.offset],
        [B, H, T, C, 1, 1, 0], (T + 31) // 32, B * H, 1, 256,
    ))
    check("backward/flash_attn_fwd_kernel/output", read_storage_f32(rt, sb["dk"], (B, T, C)), oracle["out"])
    check("backward/flash_attn_fwd_kernel/lse", read_storage_f32(rt, sb["lse"]), oracle["lse"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "flash_attn_bwd_prepare_kernel",
        [sb["dk"].ptr.buffer_id, out.grad.storage.ptr.buffer_id, sb["d_val"].ptr.buffer_id],
        [sb["dk"].ptr.offset, out.grad.storage.ptr.offset, sb["d_val"].ptr.offset],
        [B, H, T, C], (N * H + 3) // 4, 1, 1, 128,
    ))
    check("flash_attn_bwd_prepare_kernel/d_val", read_storage_f32(rt, sb["d_val"]), oracle["d_val"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "flash_attn_bwd_kernel",
        [sb["q"].ptr.buffer_id, sb["k"].ptr.buffer_id, sb["v"].ptr.buffer_id, out.grad.storage.ptr.buffer_id,
         sb["dq"].ptr.buffer_id, sb["dk"].ptr.buffer_id, sb["dv"].ptr.buffer_id,
         sb["lse"].ptr.buffer_id, sb["d_val"].ptr.buffer_id],
        [sb["q"].ptr.offset, sb["k"].ptr.offset, sb["v"].ptr.offset, out.grad.storage.ptr.offset,
         sb["dq"].ptr.offset, sb["dk"].ptr.offset, sb["dv"].ptr.offset,
         sb["lse"].ptr.offset, sb["d_val"].ptr.offset],
        [B, H, T, C, 1], (T + 15) // 16, B * H, 2, 256,
    ))
    check("flash_attn_bwd_kernel/dQ", read_storage_f32(rt, sb["dq"], (B, T, C)), heads_to_btc(oracle["dq_rot"]))
    check("flash_attn_bwd_kernel/dK", read_storage_f32(rt, sb["dk"], (B, T, C)), heads_to_btc(oracle["dk_rot"]))
    check("flash_attn_bwd_kernel/dV", read_storage_f32(rt, sb["dv"], (B, T, C)), heads_to_btc(oracle["dv"]))

    _dispatch(rt, lambda: dispatch_rope(sb["dq"], sb["dk"], 1))
    check("rope_qk_inplace_backward/dQ", read_storage_f32(rt, sb["dq"], (B, T, C)), heads_to_btc(oracle["dq"]))
    check("rope_qk_inplace_backward/dK", read_storage_f32(rt, sb["dk"], (B, T, C)), heads_to_btc(oracle["dk"]))

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "rmsnorm_forward_kernel",
        [x.storage.ptr.buffer_id, 0, sb["q"].ptr.buffer_id],
        [x.storage.ptr.offset, 0, sb["q"].ptr.offset],
        [N, C, eps_bits], (N + 3) // 4, 1, 1, 128,
    ))
    for wn, dn, exp_key in (("wq", "dq", "dwq"), ("wk", "dk", "dwk"), ("wv", "dv", "dwv")):
        dw_bf16 = int(np.dtype(p_tensors[wn].grad.storage.dtype) == np.dtype(np.uint16))
        _dispatch(rt, lambda wn=wn, dn=dn, dw_bf16=dw_bf16: rt.backend.ctx.dispatch(
            "cuda_sgemm_tn_kernel",
            [sb["q"].ptr.buffer_id, sb[dn].ptr.buffer_id, p_tensors[wn].grad.storage.ptr.buffer_id],
            [sb["q"].ptr.offset, sb[dn].ptr.offset, p_tensors[wn].grad.storage.ptr.offset],
            [C, C, N, 0, dw_bf16], (C + 63) // 64, (C + 63) // 64, 1, 256,
        ))
        check_parameter_gradient(f"cuda_sgemm_tn_kernel/{wn}.grad", copy_to_host(rt, p_tensors[wn].grad), oracle[exp_key], grad_dtype)

    for bn, dn, exp_key in (("bq", "dq", "dbq"), ("bk", "dk", "dbk"), ("bv", "dv", "dbv")):
        db_bf16 = int(np.dtype(p_tensors[bn].grad.storage.dtype) == np.dtype(np.uint16))
        _dispatch(rt, lambda bn=bn, dn=dn, db_bf16=db_bf16: rt.backend.ctx.dispatch(
            "bias_grad_reduce_kernel",
            [sb[dn].ptr.buffer_id, p_tensors[bn].grad.storage.ptr.buffer_id],
            [sb[dn].ptr.offset, p_tensors[bn].grad.storage.ptr.offset],
            [N, C, 0, db_bf16], (C + 7) // 8, 1, 1, 256,
        ))
        check_parameter_gradient(f"bias_grad_reduce_kernel/{bn}.grad", copy_to_host(rt, p_tensors[bn].grad), oracle[exp_key], grad_dtype)

    for accum_flag, (dn, wn) in enumerate((("dq", "wq"), ("dk", "wk"), ("dv", "wv"))):
        _dispatch(rt, lambda dn=dn, wn=wn, accum_flag=accum_flag: rt.backend.ctx.dispatch(
            "cuda_sgemm_nt_kernel",
            [sb[dn].ptr.buffer_id, p_tensors[wn].storage.ptr.buffer_id, sb["k"].ptr.buffer_id],
            [sb[dn].ptr.offset, p_tensors[wn].storage.ptr.offset, sb["k"].ptr.offset],
            [N, C, C, int(accum_flag > 0)], (C + 63) // 64, (N + 63) // 64, 1, 256,
        ))
    check("three_projection_gemms/dx_norm", read_storage_f32(rt, sb["k"], (B, T, C)), oracle["dx_norm"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "rmsnorm_backward_kernel",
        [x.storage.ptr.buffer_id, sb["k"].ptr.buffer_id, 0, x.grad.storage.ptr.buffer_id, 0],
        [x.storage.ptr.offset, sb["k"].ptr.offset, 0, x.grad.storage.ptr.offset, 0],
        [N, C, eps_bits, 0], (N + 3) // 4, 1, 1, 128,
    ))
    check("rmsnorm_backward_kernel/x.grad", copy_to_host(rt, x.grad), oracle["dx_local"])

def ce_oracle(x, w, labels):
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    B, T, C = x.shape
    N, V = B * T, w.shape[0]

    X = x.reshape(N, C)
    y = labels.reshape(N)
    logits = (X @ w.T).astype(np.float32)

    row_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - row_max)
    sums = np.sum(exp_logits, axis=1, keepdims=True)
    lse = (row_max + np.log(sums)).astype(np.float32)
    row_loss = (lse[:, 0] - logits[np.arange(N), y]).astype(np.float32)

    probs = exp_logits / sums
    probs[np.arange(N), y] -= 1.0
    dlogits = (probs / np.float32(N)).astype(np.float32)

    return {
        "logits": logits,
        "row_loss": row_loss.reshape(N, 1),
        "loss": np.asarray([np.mean(row_loss)], dtype=np.float32),
        "lse": lse,
        "dlogits": dlogits,
        "dx": (dlogits @ w).reshape(B, T, C).astype(np.float32),
        "dw": (dlogits.T @ X).astype(np.float32),
    }

def probe_ce_internal(cfg, x_np, w_np, labels_np, grad_dtype=np.float32):
    """对齐 check_tiny_gpt2_kernel.py 的 probe_ce_internal，逐级验证 FusedLinearCrossEntropy 各 CUDA 内核。"""
    print("\n=== Internal kernels: Optimized FusedLinearCrossEntropy ===")
    oracle = ce_oracle(x_np, w_np, labels_np)

    rt = reset_runtime("cuda")
    x = make_tensor(x_np.shape, np.float32, "cuda", True, x_np)
    w = make_tensor(w_np.shape, np.float32, "cuda", True, w_np, grad_dtype=grad_dtype)
    labels = make_tensor(labels_np.shape, np.int32, "cuda", False, labels_np)
    loss = FusedLinearCrossEntropyOp()(x, w, labels)

    rt.loss = loss
    rt.build_topo(loss)
    rt.bind_memory()

    node = rt.topo[-1]
    sf = node.forward.keywords["scratch"]
    assert not isinstance(node.backward, functools.partial), (
        "fused CE backward should not carry scratch anymore"
    )

    N, C, V = cfg.B * cfg.T, cfg.C, cfg.V
    itemsize = np.dtype(x.dtype).itemsize
    logits, row_loss, lse = sf["logits"], sf["row_loss"], sf["lse"]
    chunk_rows = min(logits.shape[0], N)
    scale_bits = f32_bits(1.0 / float(N))
    ignore_bits = (-1) & 0xFFFFFFFF
    dw_bf16 = np.dtype(w.grad.storage.dtype) == np.dtype(np.uint16)
    defer_dw_bf16 = dw_bf16 and (N > chunk_rows)

    rt.backend.fill(x.grad, 0.0)
    rt.backend.fill(w.grad, 0.0)

    for row_base in range(0, N, chunk_rows):
        rows = min(chunk_rows, N - row_base)
        x_off = x.storage.ptr.offset + row_base * C * itemsize
        dx_off = x.grad.storage.ptr.offset + row_base * C * itemsize

        _dispatch(rt, lambda rows=rows, x_off=x_off: rt.backend.ctx.dispatch(
            "cuda_sgemm_nt_kernel",
            [x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
            [x_off, w.storage.ptr.offset, logits.ptr.offset],
            [rows, V, C, 0], (V + 63) // 64, (rows + 63) // 64, 1, 256,
        ))
        check(
            f"fused_ce/logits[{row_base}:{row_base + rows}]",
            read_storage_f32(rt, logits, (rows, V)),
            oracle["logits"][row_base:row_base + rows],
        )

        def submit_grads(rows=rows, row_base=row_base, x_off=x_off, dx_off=dx_off):
            rt.backend.ctx.dispatch(
                "fused_linear_ce_logits_to_dlogits_kernel",
                [logits.ptr.buffer_id, labels.storage.ptr.buffer_id, row_loss.ptr.buffer_id, lse.ptr.buffer_id, 0],
                [logits.ptr.offset, labels.storage.ptr.offset, row_loss.ptr.offset, lse.ptr.offset, 0],
                [rows, V, row_base, N, scale_bits, 1, ignore_bits],
                rows, 1, 1, 256,
            )
            rt.backend.ctx.dispatch(
                "cuda_sgemm_nn_kernel",
                [logits.ptr.buffer_id, w.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
                [logits.ptr.offset, w.storage.ptr.offset, dx_off],
                [rows, C, V, 0], (C + 63) // 64, (rows + 63) // 64, 1, 256,
            )
            if not defer_dw_bf16:
                rt.backend.ctx.dispatch(
                    "cuda_sgemm_tn_kernel",
                    [logits.ptr.buffer_id, x.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
                    [logits.ptr.offset, x_off, w.grad.storage.ptr.offset],
                    [V, C, rows, int(row_base != 0), int(dw_bf16)],
                    (C + 63) // 64, (V + 63) // 64, 1, 256,
                )

        _dispatch(rt, submit_grads)
        rl_view = rt.allocator.view(row_loss, row_base * itemsize, (rows, 1), np.float32)
        dx_view = rt.allocator.view(x.grad.storage, dx_off - x.grad.storage.ptr.offset, (rows, C), np.float32)
        check(f"fused_ce/row_loss[{row_base}:{row_base + rows}]", read_storage_f32(rt, rl_view, (rows, 1)), oracle["row_loss"][row_base:row_base + rows])
        check(f"fused_ce/dlogits[{row_base}:{row_base + rows}]", read_storage_f32(rt, logits, (rows, V)), oracle["dlogits"][row_base:row_base + rows])
        check(f"fused_ce/x.grad[{row_base}:{row_base + rows}]", read_storage_f32(rt, dx_view, (rows, C)), oracle["dx"].reshape(N, C)[row_base:row_base + rows])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch(
        "fused_linear_ce_reduce_kernel",
        [row_loss.ptr.buffer_id, loss.storage.ptr.buffer_id],
        [row_loss.ptr.offset, loss.storage.ptr.offset],
        [N], 1, 1, 1, 256,
    ))

    if defer_dw_bf16:
        capacity = logits.nbytes // 4
        tile_rows = min(chunk_rows, max(1, capacity - C))
        tile_v = max(1, min(V, capacity // (tile_rows + C)))
        dw_off = logits.ptr.offset + tile_rows * tile_v * 4
        for vocab_base in range(0, V, tile_v):
            cols = min(tile_v, V - vocab_base)
            w_off = w.storage.ptr.offset + vocab_base * C * 4
            for row_base in range(0, N, tile_rows):
                rows = min(tile_rows, N - row_base)
                x_off = x.storage.ptr.offset + row_base * C * 4
                count = rows * cols
                _dispatch(rt, lambda rows=rows, cols=cols, vocab_base=vocab_base, row_base=row_base, x_off=x_off, w_off=w_off, count=count: (
                    rt.backend.ctx.dispatch(
                        "cuda_sgemm_nt_kernel",
                        [x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
                        [x_off, w_off, logits.ptr.offset],
                        [rows, cols, C, 0], (cols + 63) // 64, (rows + 63) // 64, 1, 256,
                    ),
                    rt.backend.ctx.dispatch(
                        "ce_tile_logits_to_dlogits_kernel",
                        [logits.ptr.buffer_id, lse.ptr.buffer_id, labels.storage.ptr.buffer_id, 0],
                        [logits.ptr.offset, lse.ptr.offset, labels.storage.ptr.offset, 0],
                        [rows, cols, vocab_base, V, row_base, scale_bits, ignore_bits],
                        (count + 255) // 256, 1, 1, 256,
                    ),
                    rt.backend.ctx.dispatch(
                        "cuda_sgemm_tn_kernel",
                        [logits.ptr.buffer_id, x.storage.ptr.buffer_id, logits.ptr.buffer_id],
                        [logits.ptr.offset, x_off, dw_off],
                        [cols, C, rows, int(row_base != 0), 0],
                        (C + 63) // 64, (cols + 63) // 64, 1, 256,
                    ),
                ))
            dw_count = cols * C
            _dispatch(rt, lambda cols=cols, vocab_base=vocab_base, dw_count=dw_count: rt.backend.ctx.dispatch(
                "fp32_to_bf16_accum_kernel",
                [logits.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
                [dw_off, w.grad.storage.ptr.offset + vocab_base * C * 2],
                [dw_count, 0], (dw_count + 255) // 256, 1, 1, 256,
            ))

    check("fused_ce/loss", copy_to_host(rt, loss), oracle["loss"])
    check("fused_ce/lse", read_storage_f32(rt, lse, (N, 1)), oracle["lse"])
    check("fused_ce/x.grad", copy_to_host(rt, x.grad), oracle["dx"])
    check_parameter_gradient("fused_ce/w.grad", copy_to_host(rt, w.grad), oracle["dw"], grad_dtype)

    # 若默认小尺寸 N <= 512 但开启了 bf16，额外覆盖一次 N=520 > 512 的多分块 BF16 二阶段路径
    if dw_bf16 and N <= 512:
        rng = np.random.default_rng(SEED + 999)
        mc_x_np = (rng.standard_normal((2, 260, 64)).astype(np.float32) * 0.02).copy()
        mc_w_np = (rng.standard_normal((129, 64)).astype(np.float32) * 0.02).copy()
        mc_y_np = rng.integers(0, 129, (2, 260), dtype=np.int32)
        mc_ref = ce_oracle(mc_x_np, mc_w_np, mc_y_np)
        mc_rt = reset_runtime("cuda")
        mc_x = make_tensor(mc_x_np.shape, np.float32, "cuda", True, mc_x_np)
        mc_w = make_tensor(mc_w_np.shape, np.float32, "cuda", True, mc_w_np, grad_dtype=grad_dtype)
        mc_y = make_tensor(mc_y_np.shape, np.int32, "cuda", False, mc_y_np)
        mc_loss = FusedLinearCrossEntropyOp()(mc_x, mc_w, mc_y)
        mc_rt.backward(mc_loss, grad_scale=1.0, accumulate_parameter_grads=False)
        mc_rt.backend.synchronize()
        check("fused_ce/multichunk_bf16_v129/loss", copy_to_host(mc_rt, mc_loss), mc_ref["loss"])
        check("fused_ce/multichunk_bf16_v129/x.grad", copy_to_host(mc_rt, mc_x.grad), mc_ref["dx"])
        check_parameter_gradient("fused_ce/multichunk_bf16_v129/w.grad", copy_to_host(mc_rt, mc_w.grad), mc_ref["dw"], grad_dtype)

def probe_staged_ops_and_accumulation(cpu_trace, block, seed, grad_dtype=np.float32, check_accum=True):
    act_atol, act_rtol = tolerances_for_grad_dtype(grad_dtype, "activation")
    loss_atol, loss_rtol = tolerances_for_grad_dtype(grad_dtype, "loss")
    act_grad_atol, act_grad_rtol = tolerances_for_grad_dtype(grad_dtype, "activation_grad")

    print("\n=== Per-operator exact training path: Gather[0] & FusedLinearAdd ===")
    ids_np, tok_w_np = cpu_trace["forward_inputs"]["Gather[0]"]
    gather_upstream = cpu_trace["backward"]["Gather[0]"]["output_grads"][0]
    cpu_tok_dw = next(item["grad"] for item in cpu_trace["backward"]["Gather[0]"]["inputs"] if item["parameter"] == "tok_emb.weight")

    def build_gather(dev, gdtype):
        ids = make_tensor(ids_np.shape, np.int32, dev, False, ids_np)
        w = make_tensor(tok_w_np.shape, np.float32, dev, True, tok_w_np, grad_dtype=gdtype)
        return ids.gather(w), {"ids": ids, "tok_emb.weight": w}

    cuda_g = _run_op_graph("cuda", build_gather, gather_upstream, grad_dtype=grad_dtype)
    check("Gather[0]/forward/output[0]", cuda_g["out"], cpu_trace["forward"]["Gather[0]"][0], atol=act_atol, rtol=act_rtol)
    check_parameter_gradient("Gather[0]/backward/tok_emb.weight.grad", cuda_g["grads"]["tok_emb.weight"], cpu_tok_dw, grad_dtype)

    p = f"blocks[{block}]"
    fla_label = f"FusedLinearAdd[{block}]"
    x_np, act_np, w_np, b_np = cpu_trace["forward_inputs"][fla_label]
    fla_upstream = cpu_trace["backward"][fla_label]["output_grads"][0]
    cpu_fla_bwd = {item["index"]: item["grad"] for item in cpu_trace["backward"][fla_label]["inputs"]}

    def build_fla(dev, gdtype):
        x = make_tensor(x_np.shape, np.float32, dev, True, x_np)
        act = make_tensor(act_np.shape, np.float32, dev, True, act_np)
        w = make_tensor(w_np.shape, np.float32, dev, True, w_np, grad_dtype=gdtype)
        b = make_tensor(b_np.shape, np.float32, dev, True, b_np, grad_dtype=gdtype)
        out = FusedLinearAddOp()(x, act, w, b)
        return out, {"x": x, "act": act, f"{p}.attn.proj.weight": w, f"{p}.attn.proj.bias": b}

    cuda_fla = _run_op_graph("cuda", build_fla, fla_upstream, grad_dtype=grad_dtype)
    check(f"{fla_label}/forward/output[0]", cuda_fla["out"], cpu_trace["forward"][fla_label][0], atol=act_atol, rtol=act_rtol)
    check(f"{fla_label}/backward/input[0].grad", cuda_fla["grads"]["x"], cpu_fla_bwd[0], atol=act_grad_atol, rtol=act_grad_rtol)
    check(f"{fla_label}/backward/input[1].grad", cuda_fla["grads"]["act"], cpu_fla_bwd[1], atol=act_grad_atol, rtol=act_grad_rtol)
    check_parameter_gradient(f"{fla_label}/backward/{p}.attn.proj.weight.grad", cuda_fla["grads"][f"{p}.attn.proj.weight"], cpu_fla_bwd[2], grad_dtype)
    check_parameter_gradient(f"{fla_label}/backward/{p}.attn.proj.bias.grad", cuda_fla["grads"][f"{p}.attn.proj.bias"], cpu_fla_bwd[3], grad_dtype)

    fa_label = f"FusedRMSNormFlashAttention[{block}]"
    fa_x_np, wq_np, bq_np, wk_np, bk_np, wv_np, bv_np, rcos_np, rsin_np = cpu_trace["forward_inputs"][fa_label]
    fa_upstream = cpu_trace["backward"][fa_label]["output_grads"][0]
    fa_residual = cpu_trace["backward"][fa_label]["input_grads_before"][0]
    cpu_fa_bwd = {item["index"]: item["grad"] for item in cpu_trace["backward"][fa_label]["inputs"]}

    def build_fa(dev, gdtype):
        fx = make_tensor(fa_x_np.shape, np.float32, dev, True, fa_x_np)
        wq = make_tensor(wq_np.shape, np.float32, dev, True, wq_np, grad_dtype=gdtype)
        bq = make_tensor(bq_np.shape, np.float32, dev, True, bq_np, grad_dtype=gdtype)
        wk = make_tensor(wk_np.shape, np.float32, dev, True, wk_np, grad_dtype=gdtype)
        bk = make_tensor(bk_np.shape, np.float32, dev, True, bk_np, grad_dtype=gdtype)
        wv = make_tensor(wv_np.shape, np.float32, dev, True, wv_np, grad_dtype=gdtype)
        bv = make_tensor(bv_np.shape, np.float32, dev, True, bv_np, grad_dtype=gdtype)
        rcos = make_tensor(rcos_np.shape, np.float32, dev, False, rcos_np)
        rsin = make_tensor(rsin_np.shape, np.float32, dev, False, rsin_np)
        out = FusedRMSNormFlashAttentionOp(n_head=fa_x_np.shape[-1] // (rcos_np.shape[-1] * 2), eps=1e-6, is_causal=True)(
            fx, wq, bq, wk, bk, wv, bv, rcos, rsin
        )
        return out, {
            "fa_x": fx,
            f"{p}.attn.q_proj.weight": wq, f"{p}.attn.q_proj.bias": bq,
            f"{p}.attn.k_proj.weight": wk, f"{p}.attn.k_proj.bias": bk,
            f"{p}.attn.v_proj.weight": wv, f"{p}.attn.v_proj.bias": bv,
        }

    cuda_fa = _run_op_graph("cuda", build_fa, fa_upstream, grad_dtype=grad_dtype)
    check(f"{fa_label}/forward/output[0]", cuda_fa["out"], cpu_trace["forward"][fa_label][0], atol=act_atol, rtol=act_rtol)
    check(f"{fa_label}/backward/input[0].grad", cuda_fa["grads"]["fa_x"], cpu_fa_bwd[0] - fa_residual, atol=act_grad_atol, rtol=act_grad_rtol)
    for idx, suffix in ((1, "q_proj.weight"), (2, "q_proj.bias"), (3, "k_proj.weight"), (4, "k_proj.bias"), (5, "v_proj.weight"), (6, "v_proj.bias")):
        check_parameter_gradient(f"{fa_label}/backward/{p}.attn.{suffix}.grad", cuda_fa["grads"][f"{p}.attn.{suffix}"], cpu_fa_bwd[idx], grad_dtype)

    mlp_label = f"FusedRMSNormMLP[{block}]"
    mlp_x_np, fc1_w_np, fc2_w_np = cpu_trace["forward_inputs"][mlp_label]
    mlp_upstream = cpu_trace["backward"][mlp_label]["output_grads"][0]
    mlp_residual = cpu_trace["backward"][mlp_label]["input_grads_before"][0]
    cpu_mlp_bwd = {item["index"]: item["grad"] for item in cpu_trace["backward"][mlp_label]["inputs"]}

    def build_mlp(dev, gdtype):
        mx = make_tensor(mlp_x_np.shape, np.float32, dev, True, mlp_x_np)
        w1 = make_tensor(fc1_w_np.shape, np.float32, dev, True, fc1_w_np, grad_dtype=gdtype)
        w2 = make_tensor(fc2_w_np.shape, np.float32, dev, True, fc2_w_np, grad_dtype=gdtype)
        out = FusedRMSNormMLPOp(eps=1e-6)(mx, w1, w2)
        return out, {"mlp_x": mx, f"{p}.mlp.fc1.weight": w1, f"{p}.mlp.fc2.weight": w2}

    cuda_mlp = _run_op_graph("cuda", build_mlp, mlp_upstream, grad_dtype=grad_dtype)
    check(f"{mlp_label}/forward/output[0]", cuda_mlp["out"], cpu_trace["forward"][mlp_label][0], atol=act_atol, rtol=act_rtol)
    check(f"{mlp_label}/backward/input[0].grad", cuda_mlp["grads"]["mlp_x"], cpu_mlp_bwd[0] - mlp_residual, atol=act_grad_atol, rtol=act_grad_rtol)
    check_parameter_gradient(f"{mlp_label}/backward/{p}.mlp.fc1.weight.grad", cuda_mlp["grads"][f"{p}.mlp.fc1.weight"], cpu_mlp_bwd[1], grad_dtype)
    check_parameter_gradient(f"{mlp_label}/backward/{p}.mlp.fc2.weight.grad", cuda_mlp["grads"][f"{p}.mlp.fc2.weight"], cpu_mlp_bwd[2], grad_dtype)

    ce_label = "FusedLinearCrossEntropy[0]"
    ce_x_np, lm_w_np, ce_labels_np = cpu_trace["forward_inputs"][ce_label]
    ce_upstream = cpu_trace["backward"][ce_label]["output_grads"][0]
    cpu_ce_bwd = {item["index"]: item["grad"] for item in cpu_trace["backward"][ce_label]["inputs"]}

    def build_ce(dev, gdtype):
        cx = make_tensor(ce_x_np.shape, np.float32, dev, True, ce_x_np)
        cw = make_tensor(lm_w_np.shape, np.float32, dev, True, lm_w_np, grad_dtype=gdtype)
        cy = make_tensor(ce_labels_np.shape, np.int32, dev, False, ce_labels_np)
        out = FusedLinearCrossEntropyOp()(cx, cw, cy)
        return out, {"ce_x": cx, "lm_head.weight": cw, "ce_labels": cy}

    cuda_ce = _run_op_graph("cuda", build_ce, ce_upstream, grad_dtype=grad_dtype)
    check(f"{ce_label}/forward/output[0]", cuda_ce["out"], cpu_trace["forward"][ce_label][0], atol=loss_atol, rtol=loss_rtol)
    check(f"{ce_label}/backward/input[0].grad", cuda_ce["grads"]["ce_x"], cpu_ce_bwd[0], atol=act_grad_atol, rtol=act_grad_rtol)
    check_parameter_gradient(f"{ce_label}/backward/lm_head.weight.grad", cuda_ce["grads"]["lm_head.weight"], cpu_ce_bwd[1], grad_dtype)

    if check_accum:
        print("\n=== No-extra-buffer parameter gradient accumulation ===")
        for k_steps in (1, 2, 4):
            atol_g, rtol_g = tolerances_for_grad_dtype(grad_dtype, "parameter_grad")
            if str(grad_dtype).lower() in ("bf16", "bfloat16"):
                atol_g *= k_steps
                rtol_g *= k_steps
            cpu_g_acc = _run_op_graph("cpu", build_gather, gather_upstream, accum_steps=k_steps, seed=seed + 500 + k_steps)
            cuda_g_acc = _run_op_graph("cuda", build_gather, gather_upstream, grad_dtype=grad_dtype, accum_steps=k_steps, seed=seed + 500 + k_steps)
            check(f"grad_accum[{k_steps}]/tok_emb.weight.grad", cuda_g_acc["grads"]["tok_emb.weight"], cpu_g_acc["grads"]["tok_emb.weight"], atol=atol_g, rtol=rtol_g)

            cpu_fla_acc = _run_op_graph("cpu", build_fla, fla_upstream, accum_steps=k_steps, seed=seed + 600 + k_steps)
            cuda_fla_acc = _run_op_graph("cuda", build_fla, fla_upstream, grad_dtype=grad_dtype, accum_steps=k_steps, seed=seed + 600 + k_steps)
            for pname in (f"{p}.attn.proj.weight", f"{p}.attn.proj.bias"):
                check(f"grad_accum[{k_steps}]/{pname}.grad", cuda_fla_acc["grads"][pname], cpu_fla_acc["grads"][pname], atol=atol_g, rtol=rtol_g)

            cpu_mlp_acc = _run_op_graph("cpu", build_mlp, mlp_upstream, accum_steps=k_steps, seed=seed + 700 + k_steps)
            cuda_mlp_acc = _run_op_graph("cuda", build_mlp, mlp_upstream, grad_dtype=grad_dtype, accum_steps=k_steps, seed=seed + 700 + k_steps)
            for pname in (f"{p}.mlp.fc1.weight", f"{p}.mlp.fc2.weight"):
                check(f"grad_accum[{k_steps}]/{pname}.grad", cuda_mlp_acc["grads"][pname], cpu_mlp_acc["grads"][pname], atol=atol_g, rtol=rtol_g)

            cpu_fa_acc = _run_op_graph("cpu", build_fa, fa_upstream, accum_steps=k_steps, seed=seed + 800 + k_steps)
            cuda_fa_acc = _run_op_graph("cuda", build_fa, fa_upstream, grad_dtype=grad_dtype, accum_steps=k_steps, seed=seed + 800 + k_steps)
            for suffix in ("q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias", "v_proj.weight", "v_proj.bias"):
                pname = f"{p}.attn.{suffix}"
                check(f"grad_accum[{k_steps}]/{pname}.grad", cuda_fa_acc["grads"][pname], cpu_fa_acc["grads"][pname], atol=atol_g, rtol=rtol_g)

            cpu_ce_acc = _run_op_graph("cpu", build_ce, ce_upstream, accum_steps=k_steps, seed=seed + 900 + k_steps)
            cuda_ce_acc = _run_op_graph("cuda", build_ce, ce_upstream, grad_dtype=grad_dtype, accum_steps=k_steps, seed=seed + 900 + k_steps)
            check(
                f"grad_accum[{k_steps}]/lm_head.weight.grad",
                cuda_ce_acc["grads"]["lm_head.weight"],
                cpu_ce_acc["grads"]["lm_head.weight"],
                atol=atol_g, rtol=rtol_g,
            )

def main():
    global PRINT_PASSES
    args, cfg = parse_args()
    PRINT_PASSES = not args.compact
    grad_dtype = grad_dtype_value(args.grad_dtype)
    print(f"grad_dtype={args.grad_dtype} (CPU oracle fixed to fp32)")
    print(f"project_root={PROJECT_ROOT}")
    print(f"runtime={runtime.__file__}")
    print(
        "config: "
        f"B={cfg.B}, T={cfg.T}, V={cfg.V}, C={cfg.C}, H={cfg.H}, "
        f"D={cfg.D}, L={cfg.L}, atol={ATOL:g}, rtol={RTOL:g}"
    )

    arrays = build_parameter_arrays(cfg, args.seed)
    input_ids, labels = build_inputs(cfg, args.seed)

    cuda_exact = run_exact_training_graph(build_gpt2_graph("cuda", cfg, arrays, input_ids, labels, grad_dtype=grad_dtype))
    cpu_exact = run_exact_training_graph(build_gpt2_graph("cpu", cfg, arrays, input_ids, labels))
    compare_exact_graph(cpu_exact, cuda_exact, cuda_grad_dtype=grad_dtype)

    cpu_trace = run_staged_graph(build_gpt2_graph("cpu", cfg, arrays, input_ids, labels))
    cuda_trace = run_staged_graph(
        build_gpt2_graph("cuda", cfg, arrays, input_ids, labels, grad_dtype=grad_dtype),
        reference_inputs=cpu_trace["forward_inputs"],
    )
    compare_staged_graph(cpu_trace, cuda_trace, cuda_grad_dtype=grad_dtype)

    probe_staged_ops_and_accumulation(
        cpu_trace,
        args.kernel_block,
        args.seed,
        grad_dtype=grad_dtype,
        check_accum=not args.skip_gradient_accumulation,
    )

    if not args.skip_gradient_accumulation:
        probe_gpt2_grad_accumulation(cfg, arrays, args.seed, grad_dtype=grad_dtype)

    if not args.skip_internal_kernels:
        ids_np, tok_w_np = cpu_trace["forward_inputs"]["Gather[0]"]
        gather_out = cpu_trace["forward"]["Gather[0]"][0]
        gather_upstream = cpu_trace["backward"]["Gather[0]"]["output_grads"][0]
        gather_dw = next(item["grad"] for item in cpu_trace["backward"]["Gather[0]"]["inputs"] if item["parameter"] == "tok_emb.weight")
        probe_gather_internal(cfg, ids_np, tok_w_np, gather_out, gather_upstream, gather_dw, grad_dtype=grad_dtype)

        fla_label = f"FusedLinearAdd[{args.kernel_block}]"
        x_np, act_np, _, _ = cpu_trace["forward_inputs"][fla_label]
        fla_upstream = cpu_trace["backward"][fla_label]["output_grads"][0]
        probe_linear_add_internal(cfg, x_np, act_np, arrays, args.kernel_block, fla_upstream, grad_dtype=grad_dtype)

        mlp_label = f"FusedRMSNormMLP[{args.kernel_block}]"
        mlp_bwd = cpu_trace["backward"][mlp_label]
        probe_mlp_internal(
            cfg,
            x_np,
            act_np,
            arrays,
            args.kernel_block,
            mlp_bwd["output_grads"][0],
            mlp_bwd["input_grads_before"][0],
            grad_dtype=grad_dtype,
        )

        fa_label = f"FusedRMSNormFlashAttention[{args.kernel_block}]"
        fa_x_np = cpu_trace["forward_inputs"][fa_label][0]
        fa_upstream = cpu_trace["backward"][fa_label]["output_grads"][0]
        probe_fa_internal(cfg, fa_x_np, arrays, args.kernel_block, fa_upstream, grad_dtype=grad_dtype)

        ce_x_np, lm_w_np, ce_labels_np = cpu_trace["forward_inputs"]["FusedLinearCrossEntropy[0]"]
        probe_ce_internal(cfg, ce_x_np, lm_w_np, ce_labels_np, grad_dtype=grad_dtype)

        probe_add_and_mlp_recompute(
            cfg,
            fa_x_np,
            arrays,
            args.kernel_block,
            cpu_trace["backward"][f"Add[{args.kernel_block}]"]["output_grads"][0],
            grad_dtype=grad_dtype,
        )

    if not args.skip_optimizer:
        probe_grad_clip_and_adamw(cfg, arrays, args.seed, grad_dtype=grad_dtype)

    if FAILURES:
        print(f"\n{len(FAILURES)} correctness check(s) failed:")
        for name, max_abs, max_rel in FAILURES:
            print(f"  {name}: max_abs={max_abs:.8e}, max_rel={max_rel:.8e}")
        raise AssertionError("real GPT2 training-path CUDA correctness validation failed")
    print("\nAll full multi-layer GPT2 CUDA training-path checks passed.")


if __name__ == "__main__":
    main()
