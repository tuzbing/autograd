#!/usr/bin/env python3
"""Correctness checker for GatherOp and FusedLinearAddOp on CUDA against the FP32 CPU GPT2 oracle."""

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
from op import FusedLinearAddOp, FusedRMSNormMLPOp  # noqa: E402
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


def build_cpu_gpt2_graph(cfg, arrays, input_ids, labels):
    rt = reset_runtime("cpu")
    model = GPT2(cfg.V, cfg.C, cfg.T, cfg.H, cfg.B, cfg.L, cfg.dtype, "cpu")
    initialize_model(model, arrays)
    input_t = make_tensor(input_ids.shape, np.int32, "cpu", False, input_ids)
    labels_t = make_tensor(labels.shape, np.int32, "cpu", False, labels)
    loss = model(input_t, targets=labels_t)
    rt.loss = loss
    rt.build_topo(loss)
    verify_training_topology(rt.topo, cfg)
    rt.bind_memory()
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


def run_staged_cpu_graph(state):
    rt = state.rt
    forward = {}
    forward_inputs = {}
    backward = {}

    rt.backend.begin_pass()
    for node in rt.topo:
        label = state.labels_by_node[id(node)]
        forward_inputs[label] = [copy_to_host(rt, tensor) for tensor in node.inputs]
        node.forward(node.inputs, node.outputs)
        rt.backend.end_pass()
        rt.backend.synchronize()
        forward[label] = [copy_to_host(rt, out) for out in node.outputs]
        rt.backend.begin_pass()
    rt.backend.end_pass()

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
        backward[label] = {"output_grads": output_grads, 
                           "input_grads_before": input_grads_before,
                           "inputs": inputs}
        rt.backend.begin_pass()
    rt.backend.end_pass()
    return {"forward": forward, "forward_inputs": forward_inputs, "backward": backward}


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

        rt.accumulate_parameter_grads = step > 0
        rt.backend.accumulate_parameter_grads = step > 0
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
    proj_w_np = arrays[f"{p}.attn.proj.weight"]
    proj_b_np = arrays[f"{p}.attn.proj.bias"]
    expected_out = (
        x_np.reshape(-1, cfg.C)
        + act_np.reshape(-1, cfg.C) @ proj_w_np
        + proj_b_np
    ).reshape(x_np.shape).astype(np.float32)
    dO = np.asarray(upstream, dtype=np.float32)
    expected_dw = (act_np.reshape(-1, cfg.C).T @ dO.reshape(-1, cfg.C)).astype(np.float32)
    expected_dact = (dO.reshape(-1, cfg.C) @ proj_w_np.T).reshape(x_np.shape).astype(np.float32)
    expected_db = np.sum(dO.reshape(-1, cfg.C), axis=0).astype(np.float32)

    rt = reset_runtime("cuda")
    x = make_tensor(x_np.shape, np.float32, "cuda", True, x_np)
    act = make_tensor(act_np.shape, np.float32, "cuda", True, act_np)
    proj_w = make_tensor(proj_w_np.shape, np.float32, "cuda", True, proj_w_np, grad_dtype=grad_dtype)
    proj_b = make_tensor(proj_b_np.shape, np.float32, "cuda", True, proj_b_np, grad_dtype=grad_dtype)
    out = FusedLinearAddOp()(x, act, proj_w, proj_b)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()

    M = cfg.B * cfg.T
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch(
            "fused_linear_add_forward_kernel",
            [x.storage.ptr.buffer_id, act.storage.ptr.buffer_id, proj_w.storage.ptr.buffer_id, proj_b.storage.ptr.buffer_id, out.storage.ptr.buffer_id],
            [x.storage.ptr.offset, act.storage.ptr.offset, proj_w.storage.ptr.offset, proj_b.storage.ptr.offset, out.storage.ptr.offset],
            [M, cfg.C, cfg.C],
            (cfg.C + 63) // 64, (M + 63) // 64, 1, 256,
        ),
    )
    check("fused_linear_add_forward_kernel/output", copy_to_host(rt, out), expected_out)

    for tensor in (x, act, proj_w, proj_b, out):
        if tensor.grad is not None:
            rt.backend.fill(tensor.grad, 0.0)
    rt.backend.copy_from_host(out.grad, dO)

    dw_bf16 = int(np.dtype(proj_w.grad.storage.dtype) == np.dtype(np.uint16))
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch(
            "cuda_sgemm_tn_kernel",
            [act.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, proj_w.grad.storage.ptr.buffer_id],
            [act.storage.ptr.offset, out.grad.storage.ptr.offset, proj_w.grad.storage.ptr.offset],
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
        [N, H], (H + 255) // 256, min(N, 65535), 1, 256,
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

def probe_staged_ops_and_accumulation(cpu_trace, block, seed, grad_dtype=np.float32, check_accum=True):
    act_atol, act_rtol = tolerances_for_grad_dtype(grad_dtype, "activation")
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

    cpu_trace = run_staged_cpu_graph(
        build_cpu_gpt2_graph(cfg, arrays, input_ids, labels)
    )
    probe_staged_ops_and_accumulation(
        cpu_trace,
        args.kernel_block,
        args.seed,
        grad_dtype=grad_dtype,
        check_accum=not args.skip_gradient_accumulation,
    )

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

    if FAILURES:
        print(f"\n{len(FAILURES)} correctness check(s) failed:")
        for name, max_abs, max_rel in FAILURES:
            print(f"  {name}: max_abs={max_abs:.8e}, max_rel={max_rel:.8e}")
        raise AssertionError("real GPT2 training-path CUDA correctness validation failed")
    print("\nAll GatherOp and FusedLinearAddOp CUDA checks passed.")


if __name__ == "__main__":
    main()
