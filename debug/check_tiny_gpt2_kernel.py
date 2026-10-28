#!/usr/bin/env python3
"""Correctness checker for the exact GPT training graph used by train_autograd.py.

/Users/a58/Documents/code/nanochat/.venv/bin/python check_tiny_gpt2_kernel.py --T=256

"""

import argparse
import functools
import gc
import math
import os
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np


ATOL = 1e-6
RTOL = 1e-6
SEED = 1333
FAILURES = []
PRINT_PASSES = True

OPTIMIZER_STATE_MODE = "fp32"  # "fp32" / "8bit" / "mini"


def _locate_project_root():
    candidates = []
    env_root = os.environ.get("AUTOGRAD_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve().parent
    candidates.extend((Path.cwd(), Path.cwd().parent, here, here.parent))
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "debug" / "runtime.py").is_file() and (candidate / "train_autograd.py").is_file():
            return candidate
        if candidate.name == "debug" and (candidate / "runtime.py").is_file():
            return candidate.parent
    raise RuntimeError(
        "Cannot locate autograd project. Run from the repository/debug directory "
        "or set AUTOGRAD_ROOT=/absolute/path/to/autograd."
    )


PROJECT_ROOT = _locate_project_root()
DEBUG_ROOT = PROJECT_ROOT / "debug"
os.chdir(DEBUG_ROOT)
sys.path.insert(0, str(DEBUG_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

import runtime  # noqa: E402
from module import GPT2  # noqa: E402
from op import (  # noqa: E402
    FusedRMSNormFlashAttentionOp,
    FusedRMSNormMLPOp,
    FusedLinearAddOp,
    FusedLinearCrossEntropyOp,
    KVCacheAttentionOp,
    LMHeadOp,
    RolloutTopkOp,
    RolloutSampleOp,
)
from optimizer import AdamW # noqa: E402
from tensor import Tensor  # noqa: E402

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
        description="Verify the exact GPT2 training operator path on CPU and MPS"
    )
    # Small defaults keep the CPU oracle practical.  The dimension formula and
    # option names match train_autograd.py, so production shapes can be supplied
    # directly on the command line.
    parser.add_argument("--depth", "--L", dest="depth", type=int, default=2)
    parser.add_argument("--aspect-ratio", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-seq-len", "--T", dest="max_seq_len", type=int, default=32)
    parser.add_argument("--device-batch-size", "--B", dest="device_batch_size", type=int, default=2)
    parser.add_argument("--vocab-size", "--V", dest="vocab_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--kernel-block",
        type=int,
        default=0,
        help="block used for internal fused-kernel diagnostics (default: 0)",
    )
    parser.add_argument(
        "--skip-internal-kernels",
        action="store_true",
        help="run exact graph and per-operator checks only",
    )
    parser.add_argument(
        "--skip-optimizer",
        action="store_true",
        help="skip the AdamW training-step check",
    )
    parser.add_argument(
        "--skip-gradient-accumulation",
        action="store_true",
        help="skip the no-extra-buffer gradient accumulation checks",
    )
    parser.add_argument(
        "--gradient-accumulation-only",
        action="store_true",
        help="run only the no-extra-buffer gradient accumulation checks",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="print failures and summary events, but omit individual passing comparisons",
    )
    parser.add_argument(
        "--grad-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help=(
            "Logical dtype of model parameter gradients. "
            "CPU reference remains FP32."
        ),
    )
    parser.add_argument("--use-fp16-gemm", action="store_true", help="validate the fused FP16 GEMM path")
    args = parser.parse_args()

    if min(
        args.depth,
        args.aspect_ratio,
        args.head_dim,
        args.max_seq_len,
        args.device_batch_size,
        args.vocab_size,
    ) <= 0:
        parser.error("all model dimensions must be positive")

    if args.head_dim > 512:
        parser.error(
            "--head-dim must be <= 512: the current FA dispatch "
            "only registers 64/128/256/512 capacity kernels"
        )

    base_dim = args.depth * args.aspect_ratio
    model_dim = (
        (base_dim + args.head_dim - 1)
        // args.head_dim
        * args.head_dim
    )

    if model_dim % 4 != 0:
        parser.error(
            f"model dimension C={model_dim} is not divisible by 4; "
            "fused_ln_fa_qkv and fused_ln_fa_project_kv currently "
            "require C % 4 == 0"
        )

    heads = model_dim // args.head_dim
    cfg = Config(
        B=args.device_batch_size,
        T=args.max_seq_len,
        V=args.vocab_size,
        C=model_dim,
        H=heads,
        D=args.head_dim,
        L=args.depth,
    )

    if not 0 <= args.kernel_block < cfg.L:
        parser.error(f"--kernel-block must be in [0, {cfg.L})")

    return args, cfg


def reset_runtime(device):
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device)
    Runtime._default_runtime = rt
    return rt


def copy_to_host(rt, tensor):
    if rt.allocator.device != "mps":
        return np.asarray(rt.backend.copy_to_host(tensor)).copy()
        # return np.asarray(rt.backend.numpy_view(tensor)).copy()
    storage = tensor.storage
    storage_dtype = np.dtype(storage.dtype)
    shape = tuple(tensor.shape)
    count = prod(shape)
    nbytes = count * storage_dtype.itemsize

    if nbytes > storage.nbytes:
        raise RuntimeError(
            f"Host read exceeds storage: "
            f"requested={nbytes}, available={storage.nbytes}"
        )

    raw = np.asarray(
        rt.backend.ctx.numpy_view_uint8(
            storage.ptr.buffer_id,
            storage.ptr.offset,
            (nbytes,),
        )
    ).copy()

    values = raw.view(storage_dtype).reshape(shape)
    is_bf16 = (storage_dtype == np.dtype(np.uint16)
        and (
            str(tensor.dtype).lower() in ("bf16", "bfloat16")
            or str(getattr(tensor, "grad_dtype", "")).lower()
            in ("bf16", "bfloat16")
        )
    )

    if is_bf16:
        bits32 = values.astype(np.uint32)
        bits32 <<= np.uint32(16)
        return bits32.view(np.float32)

    return values


def read_storage_f32(rt, storage, shape=None):
    shape = tuple(shape if shape is not None else storage.shape)
    if rt.allocator.device == "cpu":
        return np.asarray(storage.data).reshape(shape).copy()
    return np.asarray(
        rt.backend.ctx.numpy_view_float32(
            storage.ptr.buffer_id, storage.ptr.offset, shape
        )
    ).copy()


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

def tolerances_for_tensor(tensor, *, value_kind):
    """
    CPU reference 始终保持 FP32。

    value_kind:
        "parameter"
        "parameter_grad"
        "activation"
        "activation_grad"
    """
    if value_kind == "parameter":
        # 参数本身仍然是 FP32。
        return 2e-7, 3e-5

    if value_kind == "parameter_grad":
        grad_dtype = getattr(tensor, "grad_dtype", np.float32)

        if str(grad_dtype).lower() in ("bf16", "bfloat16"):
            # CPU reference 仍是 FP32，只放宽比较误差。
            #
            # BF16 的相对精度约为 3.9e-3。
            # 额外保留少量 GEMM 累加顺序误差。
            return 2e-5, 4e-3

        return 1e-6, 1e-6

    if value_kind in ("activation", "activation_grad"):
        return 1e-6, 1e-6

    raise ValueError(f"unknown value_kind={value_kind!r}")

def tolerances_for_grad_dtype(
    grad_dtype,
    value_kind,
):
    """
    CPU reference 始终使用 FP32。

    grad_dtype 只决定比较时允许的误差。
    """
    if value_kind == "parameter":
        # 参数值始终是 FP32。
        return 2e-7, 3e-5

    if value_kind == "parameter_grad":
        if str(grad_dtype).lower() in (
            "bf16",
            "bfloat16",
        ):
            # CPU reference 仍然是 FP32。
            # MPS 参数梯度为 BF16 精度时放宽比较误差。
            return 2e-5, 4e-3

        return 1e-6, 1e-6

    if value_kind in (
        "activation",
        "activation_grad",
    ):
        return 1e-6, 1e-6

    raise ValueError(
        f"unknown value_kind={value_kind!r}"
    )

def check_parameter_gradient(
    name,
    got,
    reference,
    grad_dtype,
    fail_fast=False,
):
    atol, rtol = tolerances_for_grad_dtype(
        grad_dtype,
        "parameter_grad",
    )

    return check(
        name,
        got,
        reference,          # [保持] FP32 CPU reference
        fail_fast=fail_fast,
        atol=atol,
        rtol=rtol,
    )

def check_tensor_value(
    name,
    got,
    reference,
    *,
    tensor=None,
    value_kind="activation",
    fail_fast=False,
):
    atol, rtol = tolerances_for_tensor(
        tensor,
        value_kind=value_kind,
    )

    return check(
        name,
        got,
        reference,
        fail_fast=fail_fast,
        atol=atol,
        rtol=rtol,
    )

def _optimizer_tolerances(
    state_mode,
    value_kind,
    *,
    quantized=False,
    stochastic=False,
):
    """
    value_kind:
        "parameter" - 参数值
        "m"         - 一阶动量
        "v"         - 二阶动量
    """

    if value_kind not in ("parameter", "m", "v"):
        raise ValueError(
            f"invalid optimizer value kind: {value_kind!r}"
        )

    if state_mode not in ("fp32", "8bit", "mini"):
        raise ValueError(
            f"invalid optimizer state_mode: {state_mode!r}"
        )

    # [保持原有容差]
    # FP32 和 8bit 继续使用原有验证标准。
    if state_mode in ("fp32", "8bit"):
        if value_kind == "parameter":
            return 3e-5, 2e-7

        return 3e-5, 2e-10

    # mini 模式下，小参数仍然使用 FP32 m。
    # 只有 quantized=True 时才需要放宽容差。
    if not quantized:
        if value_kind == "parameter":
            return 3e-5, 2e-7

        return 3e-5, 2e-10

    # [新增] mini 量化参数的容差。
    #
    # m 被逐 block 量化后，下一步参数更新会放大量化误差。
    # 因此参数的容差应比 m 更宽。
    if value_kind == "parameter":
        if stochastic:
            return 5e-3, 1e-4
        return 3e-3, 5e-5

    if value_kind == "m":
        if stochastic:
            return 5e-3, 2e-5
        return 3e-3, 1e-5

    # 当前 mini 的 v 仍然是 FP32 分组统计值，
    # 不应因为 m 被量化而大幅放宽。
    return 3e-5, 2e-10

def _check_optimizer_value(
    name,
    got,
    ref,
    *,
    state_mode,
    value_kind,
    quantized=False,
    stochastic=False,
    fail_fast=True,
):
    rtol, atol = _optimizer_tolerances(
        state_mode,
        value_kind,
        quantized=quantized,
        stochastic=stochastic,
    )

    check(
        name,
        got,
        ref,
        fail_fast=fail_fast,
        atol=atol,
        rtol=rtol,
    )

def _check_grad_accum_optimizer_value(
    name,
    got,
    ref,
    *,
    state_mode,
    value_kind,
    quantized=False,
    stochastic=False,
):
    rtol, atol = _optimizer_tolerances(
        state_mode,
        value_kind,
        quantized=quantized,
        stochastic=stochastic,
    )

    check(
        name,
        got,
        ref,
        fail_fast=False,
        atol=max(atol, 1e-6),
        rtol=max(rtol, 1e-6),
    )


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
            "GPT2.forward topology differs from expected training graph:\n"
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

    print(
        "Training topology:",
        " -> ".join(topology_labels(topo)),
    )


def build_parameter_arrays(cfg, seed):
    rng = np.random.default_rng(seed)
    model = GPT2(cfg.V, cfg.C, cfg.T, cfg.H, cfg.B, cfg.L, cfg.dtype, "cpu")
    arrays = {}
    for name, tensor in iter_gpt2_named_tensors(model):
        arrays[name] = (rng.standard_normal(tensor.shape).astype(np.float32) * 0.02).copy()
    return arrays


def initialize_model(model, arrays):
    names = []
    for name, tensor in iter_gpt2_named_tensors(model):
        if name not in arrays:
            raise KeyError(f"missing parameter {name}")
        tensor.init_val = arrays[name].copy()
        names.append(name)
    if set(names) != set(arrays):
        raise AssertionError("parameter array set does not match GPT2")


def build_inputs(cfg, seed):
    rng = np.random.default_rng(seed + 1)
    input_ids = rng.integers(0, cfg.V, (cfg.B, cfg.T), dtype=np.int32)
    # pos_ids = np.broadcast_to(np.arange(cfg.T, dtype=np.int32), (cfg.B, cfg.T)).copy()
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


def build_graph(device, cfg, arrays, input_ids, labels, grad_dtype=np.float32, use_fp16_gemm=False):
    rt = reset_runtime(device)
    rt.use_fp16_gemm = bool(use_fp16_gemm)
    rt.backend.use_fp16_gemm = bool(use_fp16_gemm)
    model = GPT2(cfg.V, cfg.C, cfg.T, cfg.H, cfg.B, cfg.L, cfg.dtype, device)
    for parameter in model.parameters():
        parameter.grad_dtype = grad_dtype
    initialize_model(model, arrays)
    input_t = make_tensor(input_ids.shape, np.int32, device, False, input_ids)
    labels_t = make_tensor(labels.shape, np.int32, device, False, labels)
    # This is deliberately the model call used by train_autograd.py.  Do not
    # manually reconstruct the graph here.
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


def compare_exact_graph(
    cpu,
    mps,
    mps_grad_dtype=np.float32,
):
    print(
        "\n=== Exact Runtime.execute() training path ==="
    )

    if len(cpu) != len(mps):
        raise AssertionError(
            "CPU/MPS repeat count differs"
        )

    for repeat, (
        cpu_result,
        mps_result,
    ) in enumerate(zip(cpu, mps), 1):
        print(
            f"  [training execution {repeat}]"
        )

        check(
            f"repeat[{repeat}]/loss",
            mps_result["loss"],
            cpu_result["loss"],
        )

        for name in sorted(
            cpu_result["parameter_grads"]
        ):
            check_parameter_gradient(
                f"repeat[{repeat}]"
                f"/parameter.grad/{name}",
                mps_result["parameter_grads"][name],
                cpu_result["parameter_grads"][name],
                mps_grad_dtype,
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
                raise AssertionError(
                    f"{label}: input count differs"
                )

            for tensor, value in zip(
                node.inputs,
                expected_inputs,
            ):
                rt.backend.copy_from_host(
                    tensor,
                    np.asarray(
                        value,
                        dtype=tensor.dtype,
                    ),
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
                "parameter": state.param_name_by_id.get(
                    id(tensor)
                ),
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


def compare_staged_graph(
    cpu,
    mps,
    mps_grad_dtype=np.float32,
):
    print(
        "\n=== Per-operator exact training path ==="
    )

    if list(cpu["forward"]) != list(mps["forward"]):
        raise AssertionError(
            "CPU/MPS forward topology labels differ"
        )

    for label in cpu["forward"]:
        for index, (
            mps_value,
            cpu_value,
        ) in enumerate(
            zip(
                mps["forward"][label],
                cpu["forward"][label],
            )
        ):
            check(
                f"{label}/forward/output[{index}]",
                mps_value,
                cpu_value,
            )

    for label in reversed(
        list(cpu["forward"])
    ):
        cpu_bwd = cpu["backward"][label]
        mps_bwd = mps["backward"][label]

        for index, (
            mps_value,
            cpu_value,
        ) in enumerate(
            zip(
                mps_bwd["output_grads"],
                cpu_bwd["output_grads"],
            )
        ):
            if cpu_value is not None:
                check(
                    f"{label}"
                    f"/backward/output[{index}].grad",
                    mps_value,
                    cpu_value,
                )

        if len(cpu_bwd["inputs"]) != len(
            mps_bwd["inputs"]
        ):
            raise AssertionError(
                f"{label}: different "
                "differentiable input count"
            )

        for cpu_item, mps_item in zip(
            cpu_bwd["inputs"],
            mps_bwd["inputs"],
        ):
            if (
                cpu_item["index"],
                cpu_item["parameter"],
            ) != (
                mps_item["index"],
                mps_item["parameter"],
            ):
                raise AssertionError(
                    f"{label}: input mapping differs"
                )

            parameter_name = cpu_item["parameter"]

            if parameter_name is not None:
                check_parameter_gradient(
                    f"{label}"
                    f"/backward/"
                    f"{parameter_name}.grad",
                    mps_item["grad"],
                    cpu_item["grad"],
                    mps_grad_dtype,
                )
            else:
                check(
                    f"{label}"
                    f"/backward/input"
                    f"[{cpu_item['index']}].grad",
                    mps_item["grad"],
                    cpu_item["grad"],
                )


def layernorm_values(x, gamma, beta, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    inv_std = np.float32(1.0) / np.sqrt(np.var(x, axis=-1, keepdims=True) + np.float32(eps))
    x_hat = (x - mean) * inv_std
    x_norm = x_hat * gamma + beta
    return mean.astype(np.float32), inv_std.astype(np.float32), x_hat.astype(np.float32), x_norm.astype(np.float32)

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


def gelu(x):
    x = np.asarray(x, dtype=np.float32)
    inner = np.sqrt(np.float32(2.0 / np.pi)) * (x + np.float32(0.044715) * x * x * x)
    inner = np.clip(inner, -20.0, 20.0)
    return (np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)


def gelu_grad(x):
    x = np.asarray(x, dtype=np.float32)
    c = np.sqrt(np.float32(2.0 / np.pi))
    inner = np.clip(c * (x + np.float32(0.044715) * x * x * x), -20.0, 20.0)
    t = np.tanh(inner)
    return (
        np.float32(0.5) * (np.float32(1.0) + t)
        + np.float32(0.5) * x * (np.float32(1.0) - t * t)
        * c * (np.float32(1.0) + np.float32(3.0 * 0.044715) * x * x)
    ).astype(np.float32)

def apply_rope_reference(q, k, cos, sin):
    # q/k: (B, H, T, D)
    d2 = q.shape[-1] // 2

    q1 = q[..., :d2].copy()
    q2 = q[..., d2:].copy()
    k1 = k[..., :d2].copy()
    k2 = k[..., d2:].copy()

    c = cos[None, None, :, :]
    s = sin[None, None, :, :]

    q[..., :d2] = q1 * c + q2 * s
    q[..., d2:] = -q1 * s + q2 * c

    k[..., :d2] = k1 * c + k2 * s
    k[..., d2:] = -k1 * s + k2 * c

def apply_rope_backward_reference(dq, dk, cos, sin):
    d2 = dq.shape[-1] // 2

    dq1 = dq[..., :d2].copy()
    dq2 = dq[..., d2:].copy()
    dk1 = dk[..., :d2].copy()
    dk2 = dk[..., d2:].copy()

    c = cos[None, None, :, :]
    s = sin[None, None, :, :]

    dq[..., :d2] = dq1 * c - dq2 * s
    dq[..., d2:] = dq1 * s + dq2 * c

    dk[..., :d2] = dk1 * c - dk2 * s
    dk[..., d2:] = dk1 * s + dk2 * c

def attention_oracle(
    x,
    params,
    upstream,
    heads,
    eps=1e-6,
    causal=True,
    residual_grad=None,
    rope_theta=100000.0,
):
    x = np.asarray(x, dtype=np.float32)
    upstream = np.asarray(upstream, dtype=np.float32)

    B, T, C = x.shape
    H = heads
    D = C // H

    if D % 2 != 0:
        raise ValueError(
            f"RoPE requires an even head dimension, got D={D}"
        )

    positions = np.arange(
        T,
        dtype=np.float32,
    )[:, None]

    channels = np.arange(
        0,
        D,
        2,
        dtype=np.float32,
    )[None, :]

    inv_freq = np.asarray(
        rope_theta ** (-channels / D),
        dtype=np.float32,
    )

    freqs = positions * inv_freq

    rope_cos = np.cos(freqs).astype(np.float32)
    rope_sin = np.sin(freqs).astype(np.float32)

    mean_square, inv_rms, x_norm = rmsnorm_values(
        x,
        eps=eps,
    )

    x_flat = x_norm.reshape(B * T, C)

    q_flat = (
        x_flat @ params["wq"] + params["bq"]
    ).astype(np.float32)

    k_flat = (
        x_flat @ params["wk"] + params["bk"]
    ).astype(np.float32)

    v_flat = (
        x_flat @ params["wv"] + params["bv"]
    ).astype(np.float32)

    q = (
        q_flat
        .reshape(B, T, H, D)
        .transpose(0, 2, 1, 3)
        .copy()
    )
    k = (
        k_flat
        .reshape(B, T, H, D)
        .transpose(0, 2, 1, 3)
        .copy()
    )
    v = (
        v_flat
        .reshape(B, T, H, D)
        .transpose(0, 2, 1, 3)
        .copy()
    )

    q_raw = q.copy()
    k_raw = k.copy()
    apply_rope_reference(q, k, rope_cos, rope_sin)

    scale = np.float32(1.0 / math.sqrt(D))

    scores = np.matmul(
        q,
        k.transpose(0, 1, 3, 2),
    )
    scores *= scale

    if causal:
        mask = np.triu(
            np.ones((T, T), dtype=bool),
            1,
        )
        scores[..., mask] = -np.inf

    row_max = np.max(
        scores,
        axis=-1,
        keepdims=True,
    )

    exp_scores = np.exp(scores - row_max)
    row_sum = np.sum(
        exp_scores,
        axis=-1,
        keepdims=True,
    )
    probs = exp_scores / row_sum

    out_h = np.matmul(probs, v)
    out = (
        out_h
        .transpose(0, 2, 1, 3)
        .reshape(B, T, C)
        .astype(np.float32)
    )

    dO = (
        upstream
        .reshape(B, T, H, D)
        .transpose(0, 2, 1, 3)
    )

    work = np.matmul(
        dO,
        v.transpose(0, 1, 3, 2),
    )

    d_scores = (
        work
        - np.sum(
            probs * work,
            axis=-1,
            keepdims=True,
        )
    )
    d_scores *= probs
    d_scores *= scale

    dq_rot = np.matmul(d_scores, k).astype(np.float32)
    dk_rot = np.matmul(
        d_scores.transpose(0, 1, 3, 2),
        q,
    ).astype(np.float32)
    dv = np.matmul(
        probs.transpose(0, 1, 3, 2),
        dO,
    )

    dq = dq_rot.copy()
    dk = dk_rot.copy()
    apply_rope_backward_reference(dq, dk, rope_cos, rope_sin)

    dq2 = (
        dq
        .transpose(0, 2, 1, 3)
        .reshape(B * T, C)
    )
    dk2 = (
        dk
        .transpose(0, 2, 1, 3)
        .reshape(B * T, C)
    )
    dv2 = (
        dv
        .transpose(0, 2, 1, 3)
        .reshape(B * T, C)
    )

    dx_norm = (
        dq2 @ params["wq"].T
        + dk2 @ params["wk"].T
        + dv2 @ params["wv"].T
    ).reshape(B, T, C).astype(np.float32)

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

    if residual_grad is None:
        dx = dx_local
    else:
        dx = (
            dx_local
            + np.asarray(residual_grad, dtype=np.float32)
        ).astype(np.float32)

    return {
        "mean_square": mean_square,
        "inv_rms": inv_rms,
        "x_norm": x_norm,
        "q": q.astype(np.float32),
        "k": k.astype(np.float32),
        "v": v.astype(np.float32),
        "q_raw": q_raw.astype(np.float32),
        "k_raw": k_raw.astype(np.float32),
        "probs": probs.astype(np.float32),
        "out": out,
        "lse": (
            row_max[..., 0]
            + np.log(row_sum[..., 0])
        ).astype(np.float32),
        "d_val": np.sum(
            out_h * dO,
            axis=-1,
        ).astype(np.float32),
        "dq": dq.astype(np.float32),
        "dk": dk.astype(np.float32),
        "dv": dv.astype(np.float32),
        "dq_rot": dq_rot,
        "dk_rot": dk_rot,
        "dx_norm": dx_norm,
        "dx_local": dx_local,
        "dx": dx,
        "dwq": (
            x_flat.T @ dq2
        ).astype(np.float32),
        "dwk": (
            x_flat.T @ dk2
        ).astype(np.float32),
        "dwv": (
            x_flat.T @ dv2
        ).astype(np.float32),
        "dbq": np.sum(
            dq2,
            axis=0,
        ).astype(np.float32),
        "dbk": np.sum(
            dk2,
            axis=0,
        ).astype(np.float32),
        "dbv": np.sum(
            dv2,
            axis=0,
        ).astype(np.float32),
    }


def heads_to_btc(value):
    value = np.asarray(value)
    B, H, T, D = value.shape
    return value.transpose(0, 2, 1, 3).reshape(B, T, H*D).copy()

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

    XN = x_norm.reshape(N, C)

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

def block_parameter_dict(arrays, block, which):
    p = f"blocks[{block}]"

    if which == "fa":
        return {
            "wq": arrays[f"{p}.attn.q_proj.weight"],
            "bq": arrays[f"{p}.attn.q_proj.bias"],
            "wk": arrays[f"{p}.attn.k_proj.weight"],
            "bk": arrays[f"{p}.attn.k_proj.bias"],
            "wv": arrays[f"{p}.attn.v_proj.weight"],
            "bv": arrays[f"{p}.attn.v_proj.bias"],
        }

    if which == "mlp":
        return {
            "w1": arrays[f"{p}.mlp.fc1.weight"],
            "w2": arrays[f"{p}.mlp.fc2.weight"],
        }

    raise ValueError(which)


def _eps_bits(eps=1e-6):
    return struct.unpack("<i", struct.pack("<f", np.float32(eps)))[0]


def _dispatch(rt, callback):
    rt.backend.begin_pass()
    callback()
    rt.backend.end_pass()
    rt.backend.synchronize()


def _zero(rt, tensor):
    rt.backend.fill(tensor, 0.0)


def _read_at(rt, storage, byte_offset, shape):
    return np.asarray(
        rt.backend.ctx.numpy_view_float32(
            storage.ptr.buffer_id, int(byte_offset), tuple(shape)
        )
    ).copy()


def _scratch_from_partial(bound):
    if not isinstance(bound, functools.partial) or "scratch" not in (bound.keywords or {}):
        raise AssertionError("expected Runtime.bind_memory() to bind scratch via functools.partial")
    return bound.keywords["scratch"]

def build_rope_cache(cfg, theta=100000.0):
    positions = np.arange(cfg.T, dtype=np.float32)[:, None]
    channels = np.arange(
        0,
        cfg.D,
        2,
        dtype=np.float32,
    )[None, :]

    inv_freq = theta ** (-channels / cfg.D)
    freqs = positions * inv_freq

    return (
        np.cos(freqs).astype(np.float32),
        np.sin(freqs).astype(np.float32),
    )

def _make_fa_inputs(device, cfg, x_np, params):
    x = make_tensor(
        x_np.shape,
        np.float32,
        device,
        True,
        x_np,
    )

    names = (
        "wq",
        "bq",
        "wk",
        "bk",
        "wv",
        "bv",
    )

    tensors = [
        make_tensor(
            params[name].shape,
            np.float32,
            device,
            True,
            params[name],
        )
        for name in names
    ]

    rope_cos_np, rope_sin_np = build_rope_cache(cfg)

    rope_cos = make_tensor(
        rope_cos_np.shape,
        np.float32,
        device,
        False,
        rope_cos_np,
    )

    rope_sin = make_tensor(
        rope_sin_np.shape,
        np.float32,
        device,
        False,
        rope_sin_np,
    )

    out = FusedRMSNormFlashAttentionOp(
        n_head=cfg.H,
        eps=1e-6,
        is_causal=True,
    )(x, *tensors, rope_cos, rope_sin)

    tracked = {
        "x": x,
        **dict(zip(names, tensors)),
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "out": out,
    }

    return out, tracked


def probe_fa_internal(cfg, x_np, params, upstream):
    print(
        "\n=== Internal kernels: "
        "FusedRMSNormFlashAttention ==="
    )

    oracle = attention_oracle(
        x_np,
        params,
        upstream,
        cfg.H,
    )

    rt = reset_runtime("mps")
    out, t = _make_fa_inputs(
        "mps",
        cfg,
        x_np,
        params,
    )

    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()

    node = next(
        n
        for n in rt.topo
        if op_name(n) == "FusedRMSNormFlashAttention"
    )

    sf = _scratch_from_partial(node.forward)
    sb = _scratch_from_partial(node.backward)

    B, T, C = cfg.B, cfg.T, cfg.C
    H, D = cfg.H, cfg.D
    eps_bits = _eps_bits()

    def dispatch_qkv(scratch):
        input_names = (
            "x",
            "wq",
            "bq",
            "wk",
            "bk",
            "wv",
            "bv",
        )

        rt.backend.ctx.dispatch_with_args_3d(
            "fused_rms_fa_qkv",
            [
                t[name].storage.ptr.buffer_id
                for name in input_names
            ] + [
                scratch[name].ptr.buffer_id
                for name in ("q", "k", "v")
            ],
            [
                t[name].storage.ptr.offset
                for name in input_names
            ] + [
                scratch[name].ptr.offset
                for name in ("q", "k", "v")
            ],
            [B, T, C, H, eps_bits],
            ((C+127)//128)*256,
            # 256,
            (B * T + 15) // 16,
            1,
            256,
        )

    def dispatch_rope(scratch, mode):
        rt.backend.ctx.dispatch_with_args_3d(
            "rope_qk_inplace",
            [
                scratch["q"].ptr.buffer_id,
                scratch["k"].ptr.buffer_id,
                t["rope_cos"].storage.ptr.buffer_id,
                t["rope_sin"].storage.ptr.buffer_id,
            ],
            [
                scratch["q"].ptr.offset,
                scratch["k"].ptr.offset,
                t["rope_cos"].storage.ptr.offset,
                t["rope_sin"].storage.ptr.offset,
            ],
            [B, H, T, D, mode],
            D // 2,
            T,
            B * H,
            D // 2,
        )

    # Forward QKV
    _dispatch(rt, lambda: dispatch_qkv(sf))

    check(
        "fused_rms_fa_qkv/Q",
        read_storage_f32(rt, sf["q"]),
        oracle["q_raw"], atol=1e-5, rtol=1e-5,
    )
    check(
        "fused_rms_fa_qkv/K",
        read_storage_f32(rt, sf["k"]),
        oracle["k_raw"], atol=1e-5, rtol=1e-5,
    )
    check(
        "fused_rms_fa_qkv/V",
        read_storage_f32(rt, sf["v"]),
        oracle["v"], atol=1e-5, rtol=1e-5,
    )

    _dispatch(rt, lambda: dispatch_rope(sf, 0))

    fa_kernel, fa_grid_x, fa_grid_y, fa_threads = (
        rt.backend.fa_from_qkv_dispatch(D, T)
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            fa_kernel,
            [
                sf["q"].ptr.buffer_id,
                sf["k"].ptr.buffer_id,
                sf["v"].ptr.buffer_id,
                out.storage.ptr.buffer_id,
                sf["q"].ptr.buffer_id,
            ],
            [
                sf["q"].ptr.offset,
                sf["k"].ptr.offset,
                sf["v"].ptr.offset,
                out.storage.ptr.offset,
                sf["q"].ptr.offset,
            ],
            [B, H, T, C, 1, 0],
            fa_grid_x, fa_grid_y, B * H, fa_threads,
        ),
    )

    check(
        f"{fa_kernel}/output",
        copy_to_host(rt, out),
        oracle["out"],
    )

    zero_every = [
        t["x"],
        t["wq"],
        t["bq"],
        t["wk"],
        t["bk"],
        t["wv"],
        t["bv"],
        out,
    ]

    for tensor in zero_every:
        if tensor.grad is not None:
            _zero(rt, tensor.grad)

    rt.backend.copy_from_host(
        out.grad,
        np.asarray(upstream, dtype=np.float32),
    )

    # Backward QKV recomputation
    _dispatch(rt, lambda: dispatch_qkv(sb))

    check(
        "backward/recompute_qkv/Q",
        read_storage_f32(rt, sb["q"]),
        oracle["q_raw"], atol=1e-5, rtol=1e-5,
    )
    check(
        "backward/recompute_qkv/K",
        read_storage_f32(rt, sb["k"]),
        oracle["k_raw"], atol=1e-5, rtol=1e-5,
    )
    check(
        "backward/recompute_qkv/V",
        read_storage_f32(rt, sb["v"]),
        oracle["v"], atol=1e-5, rtol=1e-5,
    )

    _dispatch(rt, lambda: dispatch_rope(sb, 0))

    recompute_kernel, gx, gy, threads = (
        rt.backend.fa_from_qkv_dispatch(D, T)
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            recompute_kernel,
            [
                sb["q"].ptr.buffer_id,
                sb["k"].ptr.buffer_id,
                sb["v"].ptr.buffer_id,
                sb["dk"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
            ],
            [
                sb["q"].ptr.offset,
                sb["k"].ptr.offset,
                sb["v"].ptr.offset,
                sb["dk"].ptr.offset,
                sb["lse"].ptr.offset,
            ],
            [B, H, T, C, 1, 1],
            gx, gy, B * H, threads,
        ),
    )

    check(
        f"backward/{recompute_kernel}/output",
        read_storage_f32(
            rt,
            sb["dk"],
            (B, T, C),
        ),
        oracle["out"],
    )
    check(
        f"backward/{recompute_kernel}/lse",
        read_storage_f32(rt, sb["lse"]),
        oracle["lse"],
    )

    # FA backward preparation
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "fused_fa_bwd_prepare",
            [
                sb["dk"].ptr.buffer_id,
                out.grad.storage.ptr.buffer_id,
                sb["d_val"].ptr.buffer_id,
            ],
            [
                sb["dk"].ptr.offset,
                out.grad.storage.ptr.offset,
                sb["d_val"].ptr.offset,
            ],
            [B, H, T, C],
            128,
            (T + 3) // 4,
            B * H,
            128,
        ),
    )

    check(
        "fused_fa_bwd_prepare/d_val",
        read_storage_f32(rt, sb["d_val"]),
        oracle["d_val"],
    )

    dq_kernel, dq_gx, dq_gy, dq_threads = (
        rt.backend.fa_bwd_dq_dispatch(D, T)
    )

    dkv_kernel, dkv_gx, dkv_gy, dkv_threads = (
        rt.backend.fa_bwd_dkv_dispatch(D, T)
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            dq_kernel,
            [
                sb["q"].ptr.buffer_id,
                sb["k"].ptr.buffer_id,
                sb["v"].ptr.buffer_id,
                out.grad.storage.ptr.buffer_id,
                sb["dq"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
                sb["d_val"].ptr.buffer_id,
            ],
            [
                sb["q"].ptr.offset,
                sb["k"].ptr.offset,
                sb["v"].ptr.offset,
                out.grad.storage.ptr.offset,
                sb["dq"].ptr.offset,
                sb["lse"].ptr.offset,
                sb["d_val"].ptr.offset,
            ],
            [B, H, T, C, 1],
            dq_gx,
            dq_gy,
            B * H,
            dq_threads,
        ),
    )

    check(
        f"{dq_kernel}/dQ",
        read_storage_f32(
            rt,
            sb["dq"],
            (B, T, C),
        ),
        heads_to_btc(oracle["dq_rot"]),
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            dkv_kernel,
            [
                sb["q"].ptr.buffer_id,
                sb["k"].ptr.buffer_id,
                sb["v"].ptr.buffer_id,
                out.grad.storage.ptr.buffer_id,
                sb["dk"].ptr.buffer_id,
                sb["dv"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
                sb["d_val"].ptr.buffer_id,
            ],
            [
                sb["q"].ptr.offset,
                sb["k"].ptr.offset,
                sb["v"].ptr.offset,
                out.grad.storage.ptr.offset,
                sb["dk"].ptr.offset,
                sb["dv"].ptr.offset,
                sb["lse"].ptr.offset,
                sb["d_val"].ptr.offset,
            ],
            [B, H, T, C, 1],
            dkv_gx,
            dkv_gy,
            B * H,
            dkv_threads,
        ),
    )

    check(
        f"{dkv_kernel}/dK",
        read_storage_f32(
            rt,
            sb["dk"],
            (B, T, C),
        ),
        heads_to_btc(oracle["dk_rot"]),
    )
    check(
        f"{dkv_kernel}/dV",
        read_storage_f32(
            rt,
            sb["dv"],
            (B, T, C),
        ),
        heads_to_btc(oracle["dv"]),
    )

    # Recompute RMSNorm(X).
    # q is reused for x_norm; the first B*T elements of lse store inv_rms.
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "fused_rms_recompute",
            [
                t["x"].storage.ptr.buffer_id,
                sb["q"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
            ],
            [
                t["x"].storage.ptr.offset,
                sb["q"].ptr.offset,
                sb["lse"].ptr.offset,
            ],
            [B, T, C, eps_bits],
            128,
            (B * T + 3) // 4,
            1,
            128,
        ),
    )

    check(
        "fused_rms_recompute/x_norm",
        read_storage_f32(
            rt,
            sb["q"],
            (B, T, C),
        ),
        oracle["x_norm"],
    )
    check(
        "fused_rms_recompute/inv_rms",
        _read_at(
            rt,
            sb["lse"],
            sb["lse"].ptr.offset,
            (B, T, 1),
        ),
        oracle["inv_rms"],
    )

    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "rope_qk_inplace",
            [
                sb["dq"].ptr.buffer_id,
                sb["dk"].ptr.buffer_id,
                t["rope_cos"].storage.ptr.buffer_id,
                t["rope_sin"].storage.ptr.buffer_id,
            ],
            [
                sb["dq"].ptr.offset,
                sb["dk"].ptr.offset,
                t["rope_cos"].storage.ptr.offset,
                t["rope_sin"].storage.ptr.offset,
            ],
            [B, H, T, D, 1],
            D // 2,
            T,
            B * H,
            D // 2,
        ),
    )

    check(
        "rope_qk_inplace_backward/dQ",
        read_storage_f32(rt, sb["dq"], (B, T, C)),
        heads_to_btc(oracle["dq"]),
    )

    check(
        "rope_qk_inplace_backward/dK",
        read_storage_f32(rt, sb["dk"], (B, T, C)),
        heads_to_btc(oracle["dk"]),
    )

    # Projection weight gradients
    for weight, dproj, expected in (
        ("wq", "dq", "dwq"),
        ("wk", "dk", "dwk"),
        ("wv", "dv", "dwv"),
    ):
        _dispatch(
            rt,
            lambda weight=weight, dproj=dproj:
            rt.backend.ctx.dispatch_gemm_mps(
                [
                    sb["q"].ptr.buffer_id,
                    sb[dproj].ptr.buffer_id,
                    t[weight].grad.storage.ptr.buffer_id,
                ],
                [
                    sb["q"].ptr.offset,
                    sb[dproj].ptr.offset,
                    t[weight].grad.storage.ptr.offset,
                ],
                [1, C, C, B * T, 1, 0, 0, 0],
                1.0,
                0.0,
            ),
        )

        check(
            f"gemm/{weight}.grad",
            copy_to_host(rt, t[weight].grad),
            oracle[expected],
        )

    # Projection bias gradients
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "fused_qkv_bias_grad",
            [
                sb["dq"].ptr.buffer_id,
                sb["dk"].ptr.buffer_id,
                sb["dv"].ptr.buffer_id,
                t["bq"].grad.storage.ptr.buffer_id,
                t["bk"].grad.storage.ptr.buffer_id,
                t["bv"].grad.storage.ptr.buffer_id,
            ],
            [
                sb["dq"].ptr.offset,
                sb["dk"].ptr.offset,
                sb["dv"].ptr.offset,
                t["bq"].grad.storage.ptr.offset,
                t["bk"].grad.storage.ptr.offset,
                t["bv"].grad.storage.ptr.offset,
            ],
            [B * T, C, 0],
            128,
            (C + 3) // 4,
            1,
            128,
        ),
    )

    for bias in ("bq", "bk", "bv"):
        check(
            f"fused_qkv_bias_grad/{bias}.grad",
            copy_to_host(rt, t[bias].grad),
            oracle[f"d{bias}"],
        )

    # dRMSNormOutput
    for index, (dproj, weight, beta) in enumerate((
        ("dq", "wq", 0.0),
        ("dk", "wk", 1.0),
        ("dv", "wv", 1.0),
    )):
        _dispatch(
            rt,
            lambda dproj=dproj, weight=weight, beta=beta:
            rt.backend.ctx.dispatch_gemm_mps(
                [
                    sb[dproj].ptr.buffer_id,
                    t[weight].storage.ptr.buffer_id,
                    sb["k"].ptr.buffer_id,
                ],
                [
                    sb[dproj].ptr.offset,
                    t[weight].storage.ptr.offset,
                    sb["k"].ptr.offset,
                ],
                [1, B * T, C, C, 0, 1, 0, 0],
                1.0,
                beta,
            ),
        )

        if index == 2:
            check(
                "three projection GEMMs/dx_norm",
                read_storage_f32(
                    rt,
                    sb["k"],
                    (B, T, C),
                ),
                oracle["dx_norm"],
            )

    # RMSNorm backward
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "fused_rmsnorm_bwd_dx",
            [
                t["x"].storage.ptr.buffer_id,
                sb["k"].ptr.buffer_id,
                sb["lse"].ptr.buffer_id,
                t["x"].grad.storage.ptr.buffer_id,
            ],
            [
                t["x"].storage.ptr.offset,
                sb["k"].ptr.offset,
                sb["lse"].ptr.offset,
                t["x"].grad.storage.ptr.offset,
            ],
            [B * T, C],
            32,
            B * T,
            1,
            32,
        ),
    )

    check(
        "fused_rmsnorm_bwd_dx/x.grad",
        copy_to_host(rt, t["x"].grad),
        oracle["dx_local"],
    )

def make_kv_case(cfg, seed, mode, cache_host=None, position=0):
    rng = np.random.default_rng(seed)

    B = cfg.B
    T = cfg.T if mode == "prefill" else 1
    C = cfg.C
    H = cfg.H
    D = cfg.D
    S = cfg.T

    x_host = (
        rng.standard_normal((B, T, C))
        .astype(np.float32)
        * 0.02
    )

    wq_host = (
        rng.standard_normal((C, C))
        .astype(np.float32)
        * 0.02
    )
    wk_host = (
        rng.standard_normal((C, C))
        .astype(np.float32)
        * 0.02
    )
    wv_host = (
        rng.standard_normal((C, C))
        .astype(np.float32)
        * 0.02
    )

    bq_host = np.zeros((C,), dtype=np.float32)
    bk_host = np.zeros((C,), dtype=np.float32)
    bv_host = np.zeros((C,), dtype=np.float32)

    positions = np.arange(S, dtype=np.float32)[:, None]
    channels = np.arange(
        0,
        D,
        2,
        dtype=np.float32,
    )[None, :]
    inv_freq = 100000.0 ** (-channels / D)
    freqs = positions * inv_freq

    rope_cos_host = np.cos(freqs).astype(np.float32)
    rope_sin_host = np.sin(freqs).astype(np.float32)

    x = make_tensor(
        x_host.shape,
        np.float32,
        device=None,
        requires_grad=False,
        init_val=x_host,
    )

def make_kv_graph(
    device,
    cfg,
    x_host,
    wq_host,
    bq_host,
    wk_host,
    bk_host,
    wv_host,
    bv_host,
    rope_cos_host,
    rope_sin_host,
    k_cache_host,
    v_cache_host,
    position,
    mode,
):
    B, T, C = x_host.shape
    H = cfg.H
    D = cfg.D
    S = cfg.T

    def tensor(shape, dtype, value, name):
        return Tensor(
            shape=shape,
            dtype=dtype,
            device=device,
            requires_grad=False,
            init_val=np.asarray(value, dtype=dtype).copy(),
            name=name,
        )

    x = tensor(
        (B, T, C),
        np.float32,
        x_host,
        "kv_x",
    )
    wq = tensor((C, C), np.float32, wq_host, "wq")
    bq = tensor((C,), np.float32, bq_host, "bq")
    wk = tensor((C, C), np.float32, wk_host, "wk")
    bk = tensor((C,), np.float32, bk_host, "bk")
    wv = tensor((C, C), np.float32, wv_host, "wv")
    bv = tensor((C,), np.float32, bv_host, "bv")

    k_cache = tensor(
        (B, S, H, D),
        np.float32,
        k_cache_host,
        "k_cache",
    )

    v_cache = tensor(
        (B, S, H, D),
        np.float32,
        v_cache_host,
        "v_cache",
    )

    position_t = tensor(
        (1,),
        np.int32,
        np.asarray([position], dtype=np.int32),
        "position",
    )

    rope_cos = tensor(
        rope_cos_host.shape,
        np.float32,
        rope_cos_host,
        "rope_cos",
    )
    rope_sin = tensor(
        rope_sin_host.shape,
        np.float32,
        rope_sin_host,
        "rope_sin",
    )

    out = KVCacheAttentionOp(
        n_head=H,
        layer_idx=0,
        eps=1e-6,
        mode=mode,
    )(
        x,
        wq,
        bq,
        wk,
        bk,
        wv,
        bv,
        k_cache,
        v_cache,
        position_t,
        rope_cos,
        rope_sin,
    )

    runtime = Runtime(
        device=device,
        forward_only=True,
    )

    value = runtime.run_forward(
        out,
        return_value=True,
    )

    return {
        "runtime": runtime,
        "output": np.asarray(value).copy(),
        "k_cache": copy_to_host(runtime, k_cache),
        "v_cache": copy_to_host(runtime, v_cache),
    }

def probe_kv_cache_attention(cfg, seed):
    print("\n=== KVCacheAttentionOp CPU/MPS forward ===")

    rng = np.random.default_rng(seed)

    B = cfg.B
    T = min(cfg.T, 4)
    C = cfg.C
    H = cfg.H
    D = cfg.D
    S = cfg.T

    x_host = (
        rng.standard_normal((B, T, C))
        .astype(np.float32)
        * 0.02
    )

    wq_host = (
        rng.standard_normal((C, C))
        .astype(np.float32)
        * 0.02
    )
    wk_host = (
        rng.standard_normal((C, C))
        .astype(np.float32)
        * 0.02
    )
    wv_host = (
        rng.standard_normal((C, C))
        .astype(np.float32)
        * 0.02
    )

    bq_host = (
        rng.standard_normal((C,))
        .astype(np.float32)
        * 0.02
    )
    bk_host = (
        rng.standard_normal((C,))
        .astype(np.float32)
        * 0.02
    )
    bv_host = (
        rng.standard_normal((C,))
        .astype(np.float32)
        * 0.02
    )

    positions = np.arange(S, dtype=np.float32)[:, None]
    channels = np.arange(
        0,
        D,
        2,
        dtype=np.float32,
    )[None, :]
    inv_freq = 100000.0 ** (-channels / D)
    freqs = positions * inv_freq

    rope_cos_host = np.cos(freqs).astype(np.float32)
    rope_sin_host = np.sin(freqs).astype(np.float32)

    k_cache_host = np.zeros(
        (B, S, H, D),
        dtype=np.float32,
    )
    v_cache_host = np.zeros_like(k_cache_host)

    cpu = make_kv_graph(
        "cpu",
        cfg,
        x_host,
        wq_host,
        bq_host,
        wk_host,
        bk_host,
        wv_host,
        bv_host,
        rope_cos_host,
        rope_sin_host,
        k_cache_host,
        v_cache_host,
        0,
        "prefill",
    )

    mps = make_kv_graph(
        "mps",
        cfg,
        x_host,
        wq_host,
        bq_host,
        wk_host,
        bk_host,
        wv_host,
        bv_host,
        rope_cos_host,
        rope_sin_host,
        k_cache_host,
        v_cache_host,
        0,
        "prefill",
    )

    check(
        "KVCacheAttention/prefill/output",
        mps["output"],
        cpu["output"],
        atol=ATOL,
        rtol=RTOL,
    )

    check(
        "KVCacheAttention/prefill/k_cache",
        mps["k_cache"],
        cpu["k_cache"],
        atol=3e-6,
        rtol=2e-4,
    )

    check(
        "KVCacheAttention/prefill/v_cache",
        mps["v_cache"],
        cpu["v_cache"],
        atol=ATOL,
        rtol=RTOL,
    )

    decode_x = (
        rng.standard_normal((B, 1, C))
        .astype(np.float32)
        * 0.02
    )

    cpu_decode = make_kv_graph(
        "cpu",
        cfg,
        decode_x,
        wq_host,
        bq_host,
        wk_host,
        bk_host,
        wv_host,
        bv_host,
        rope_cos_host,
        rope_sin_host,
        cpu["k_cache"],
        cpu["v_cache"],
        T,
        "decode",
    )

    mps_decode = make_kv_graph(
        "mps",
        cfg,
        decode_x,
        wq_host,
        bq_host,
        wk_host,
        bk_host,
        wv_host,
        bv_host,
        rope_cos_host,
        rope_sin_host,
        mps["k_cache"],
        mps["v_cache"],
        T,
        "decode",
    )

    check(
        "KVCacheAttention/decode/output",
        mps_decode["output"],
        cpu_decode["output"],
        atol=ATOL,
        rtol=RTOL,
    )

def ce_oracle(x, w, labels):
    B, T, C = x.shape
    N, V = B * T, w.shape[0]

    X = x.reshape(N, C)
    y = labels.reshape(N)

    logits = X @ w.T

    row_max = np.max(
        logits,
        axis=1,
        keepdims=True,
    )
    exp_logits = np.exp(logits - row_max)
    sums = np.sum(
        exp_logits,
        axis=1,
        keepdims=True,
    )

    lse = row_max + np.log(sums)

    row_loss = (
        lse[:, 0]
        - logits[np.arange(N), y]
    ).astype(np.float32)

    probs = exp_logits / sums
    probs[np.arange(N), y] -= 1.0

    dlogits = probs / np.float32(N)

    return {
        "logits": logits.astype(np.float32),
        "target_logits": (
            logits[np.arange(N), y]
            .astype(np.float32)
            .reshape(N, 1)
        ),
        "row_loss": row_loss.reshape(N, 1),
        "loss": np.asarray(
            [np.mean(row_loss)],
            dtype=np.float32,
        ),
        "lse": lse.astype(np.float32),
        "dlogits": dlogits.astype(np.float32),
        "dx": (
            dlogits @ w
        ).reshape(B, T, C).astype(np.float32),
        "dw": (
            dlogits.T @ X
        ).astype(np.float32),
    }

def probe_ce_internal(cfg, x_np, w_np, labels_np):
    print(
        "\n=== Internal kernels: "
        "Optimized FusedLinearCrossEntropy ==="
    )

    oracle = ce_oracle(
        x_np,
        w_np,
        labels_np,
    )

    rt = reset_runtime("mps")

    x = make_tensor(
        x_np.shape,
        np.float32,
        "mps",
        True,
        x_np,
    )
    w = make_tensor(
        w_np.shape,
        np.float32,
        "mps",
        True,
        w_np,
    )
    labels = make_tensor(
        labels_np.shape,
        np.int32,
        "mps",
        False,
        labels_np,
    )

    loss = FusedLinearCrossEntropyOp()(
        x,
        w,
        labels,
    )

    rt.loss = loss
    rt.build_topo(loss)
    rt.bind_memory()

    node = rt.topo[-1]
    sf = _scratch_from_partial(node.forward)
    # 融合后 fwd/bwd 共用同一份 scratch，backward 不再绑 scratch
    assert not isinstance(node.backward, functools.partial), \
        "fused CE backward should not carry scratch anymore"
    N, C, V = cfg.B * cfg.T, cfg.C, cfg.V
    itemsize = np.dtype(x.dtype).itemsize
    logits, row_loss, lse = sf["logits"], sf["row_loss"], sf["lse"]
    chunk_rows = min(logits.shape[0], N)
    # 与算子内一致：grad_scale/N 作为 host 常量传入，不再读 grad_out[0]
    scale_bits = struct.unpack('<i', struct.pack('<f', 1.0 / N))[0]
    accumulate = bool(getattr(rt.backend, "accumulate_parameter_grads", False))
    _zero(rt, x.grad)
    _zero(rt, w.grad)
    for row_base in range(0, N, chunk_rows):
        rows = min(chunk_rows, N - row_base)
        x_offset = x.storage.ptr.offset + row_base * C * itemsize
        grad_x_offset = x.grad.storage.ptr.offset + row_base * C * itemsize
        # --- 阶段 1：只提交 logits，便于在被覆盖前校验 ---
        def submit_logits(rows=rows, x_offset=x_offset):
            rt.backend.ctx.dispatch_with_args_3d(
                "fused_linear_ce_chunk_logits",
                [x.storage.ptr.buffer_id, w.storage.ptr.buffer_id, logits.ptr.buffer_id],
                [x_offset, w.storage.ptr.offset, logits.ptr.offset],
                [rows, C, V], 256, (rows + 63) // 64, (V + 95) // 96, 256)
        _dispatch(rt, submit_logits)
        check(f"fused_ce/logits[{row_base}:{row_base + rows}]",
              read_storage_f32(rt, logits, (rows, V)),
              oracle["logits"][row_base:row_base + rows])
        # --- 阶段 2：row_loss + lse + 就地 dlogits + dX + dW ---
        def submit_grads(rows=rows, row_base=row_base,
                         x_offset=x_offset, grad_x_offset=grad_x_offset):
            rt.backend.ctx.dispatch_with_args_3d(
                "fused_linear_ce_logits_to_dlogits",
                [logits.ptr.buffer_id, labels.storage.ptr.buffer_id,
                 row_loss.ptr.buffer_id, lse.ptr.buffer_id],
                [logits.ptr.offset, labels.storage.ptr.offset,
                 row_loss.ptr.offset, lse.ptr.offset],
                [rows, V, row_base, scale_bits, 1], 256, rows, 1, 256)

            rt.backend.ctx.dispatch_gemm_mps(
                [logits.ptr.buffer_id, w.storage.ptr.buffer_id, x.grad.storage.ptr.buffer_id],
                [logits.ptr.offset, w.storage.ptr.offset, grad_x_offset],
                [1, rows, C, V, 0, 0, 0, 0], 1.0, 0.0)
            rt.backend.ctx.dispatch_gemm_mps(
                [logits.ptr.buffer_id, x.storage.ptr.buffer_id, w.grad.storage.ptr.buffer_id],
                [logits.ptr.offset, x_offset, w.grad.storage.ptr.offset],
                [1, V, C, rows, 1, 0, 0, 0], 1.0,
                1.0 if (accumulate or row_base != 0) else 0.0)
        _dispatch(rt, submit_grads)
        check(f"fused_ce/row_loss[{row_base}:{row_base + rows}]",
              _read_at(rt, row_loss, row_loss.ptr.offset + row_base * itemsize, (rows, 1)),
              oracle["row_loss"][row_base:row_base + rows])
        check(f"fused_ce/dlogits[{row_base}:{row_base + rows}]",
              read_storage_f32(rt, logits, (rows, V)),
              oracle["dlogits"][row_base:row_base + rows])
        check(f"fused_ce/x.grad[{row_base}:{row_base + rows}]",
              _read_at(rt, x.grad.storage, grad_x_offset, (rows, C)),
              oracle["dx"].reshape(N, C)[row_base:row_base + rows])
    # 全部 row_loss 就绪后再 reduce（保持原样）
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_with_args_3d(
        "fused_linear_ce_forward_reduce",
        [row_loss.ptr.buffer_id, loss.storage.ptr.buffer_id],
        [row_loss.ptr.offset, loss.storage.ptr.offset],
        [N], 256, 1, 1, 256))
    check("fused_ce/loss",   copy_to_host(rt, loss),   oracle["loss"])
    check("fused_ce/lse",    read_storage_f32(rt, lse, (N, 1)), oracle["lse"])
    check("fused_ce/x.grad", copy_to_host(rt, x.grad), oracle["dx"])
    check("fused_ce/w.grad", copy_to_host(rt, w.grad), oracle["dw"])

def run_lm_head_forward(device, x_np, weight_np):
    Runtime._default_runtime = None

    rt = Runtime(
        device=device,
        forward_only=True,
        owns_params=False,
    )
    Runtime._default_runtime = rt

    x = make_tensor(
        x_np.shape,
        np.float32,
        device,
        requires_grad=False,
        init_val=x_np,
    )

    weight = make_tensor(
        weight_np.shape,
        np.float32,
        device,
        requires_grad=False,
        init_val=weight_np,
    )

    out = LMHeadOp()(x, weight)

    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()
    rt.is_compiled = True

    value = rt.run_forward(
        out,
        return_value=True,
    )

    return np.asarray(value).copy()

def run_rollout_forward(
    device,
    logits_np,
    top_k_value,
    temperature_value,
    rng_state_np,
    finished_np,
    eos_token=99,
    bos_token=98,
):
    Runtime._default_runtime = None
    gc.collect()

    rt = Runtime(
        device=device,
        forward_only=True,
        owns_params=False,
    )

    Runtime._default_runtime = rt

    logits = make_tensor(
        logits_np.shape,
        np.float32,
        device,
        requires_grad=False,
        init_val=logits_np,
    )

    top_k = make_tensor(
        (1,),
        np.int32,
        device,
        requires_grad=False,
        init_val=np.asarray(
            [top_k_value],
            dtype=np.int32,
        ),
    )

    temperature = make_tensor(
        (1,),
        np.float32,
        device,
        requires_grad=False,
        init_val=np.asarray(
            [temperature_value],
            dtype=np.float32,
        ),
    )

    rng_state = make_tensor(
        rng_state_np.shape,
        np.uint32,
        device,
        requires_grad=False,
        init_val=rng_state_np,
    )

    finished = make_tensor(
        finished_np.shape,
        np.int32,
        device,
        requires_grad=False,
        init_val=finished_np,
    )

    logits_position = make_tensor((1,), np.int32, device, requires_grad=False, init_val=np.asarray([0], dtype=np.int32),)
    topk_values, topk_indices = RolloutTopkOp(max_top_k=64, mode="decode")(logits, top_k, logits_position)

    next_tokens = RolloutSampleOp(
        eos_token=eos_token,
        bos_token=bos_token,
    )(
        topk_values,
        topk_indices,
        rng_state,
        finished,
        temperature,
        top_k,
    )

    rt.loss = next_tokens
    rt.build_topo(next_tokens)
    rt.bind_memory()
    rt.is_compiled = True

    value = rt.run_forward(
        next_tokens,
        return_value=True,
    )

    return {
        "topk_values": copy_to_host(rt, topk_values),
        "topk_indices": copy_to_host(rt, topk_indices),
        "next_tokens": np.asarray(value).copy(),
        "rng_state": copy_to_host(rt, rng_state),
    }

def probe_lm_head_op(cfg, seed):
    print("\n=== LMHeadOp CPU/MPS forward ===")

    rng = np.random.default_rng(seed)

    B = cfg.B
    T = min(cfg.T, 8)
    C = cfg.C
    V = cfg.V

    x_np = (
        rng.standard_normal((B, T, C))
        .astype(np.float32)
        * np.float32(0.1)
    )

    weight_np = (
        rng.standard_normal((V, C))
        .astype(np.float32)
        * np.float32(0.1)
    )

    reference = np.matmul(
        x_np.reshape(-1, C),
        weight_np.T,
    ).reshape(B, T, V)

    cpu_out = run_lm_head_forward(
        "cpu",
        x_np,
        weight_np,
    )

    mps_out = run_lm_head_forward(
        "mps",
        x_np,
        weight_np,
    )

    check(
        "LMHeadOp CPU vs NumPy",
        cpu_out,
        reference,
        atol=1e-6,
        rtol=1e-6,
    )

    check(
        "LMHeadOp MPS vs NumPy",
        mps_out,
        reference,
        atol=1e-6,
        rtol=1e-6,
    )

    check(
        "LMHeadOp MPS vs CPU",
        mps_out,
        cpu_out,
        atol=1e-6,
        rtol=1e-6,
    )

def rollout_topk_reference(logits_np, top_k):
    B, _, V = logits_np.shape
    K = min(int(top_k), 64, V)

    logits_2d = logits_np.reshape(B, V)

    candidate_indices = np.argpartition(
        logits_2d,
        V - K,
        axis=-1,
    )[:, V - K:]

    candidate_values = np.take_along_axis(
        logits_2d,
        candidate_indices,
        axis=-1,
    )

    order = np.argsort(
        candidate_values,
        axis=-1,
        kind="stable",
    )[:, ::-1]

    values = np.full(
        (B, 64),
        -np.inf,
        dtype=np.float32,
    )
    indices = np.zeros(
        (B, 64),
        dtype=np.int32,
    )

    values[:, :K] = np.take_along_axis(
        candidate_values,
        order,
        axis=-1,
    )
    indices[:, :K] = np.take_along_axis(
        candidate_indices,
        order,
        axis=-1,
    )

    return values, indices

def run_rollout_forward_stepwise(
    device,
    logits_np,
    top_k_value,
    temperature_value,
    rng_state_np,
    finished_np,
    eos_token=99,
    bos_token=98,
):
    Runtime._default_runtime = None
    gc.collect()

    rt = Runtime(
        device=device,
        forward_only=True,
        owns_params=False,
    )
    Runtime._default_runtime = rt

    logits = make_tensor(
        logits_np.shape,
        np.float32,
        device,
        requires_grad=False,
        init_val=logits_np,
    )

    top_k = make_tensor(
        (1,),
        np.int32,
        device,
        requires_grad=False,
        init_val=np.asarray(
            [top_k_value],
            dtype=np.int32,
        ),
    )

    temperature = make_tensor(
        (1,),
        np.float32,
        device,
        requires_grad=False,
        init_val=np.asarray(
            [temperature_value],
            dtype=np.float32,
        ),
    )

    rng_state = make_tensor(
        rng_state_np.shape,
        np.uint32,
        device,
        requires_grad=False,
        init_val=rng_state_np,
    )

    finished = make_tensor(
        finished_np.shape,
        np.int32,
        device,
        requires_grad=False,
        init_val=finished_np,
    )

    logits_position = make_tensor(
        (1,),
        np.int32,
        device,
        requires_grad=False,
        init_val=np.asarray(
            [prompt_len - 1],
            dtype=np.int32,
        ),
    )

    topk_values, topk_indices = RolloutTopkOp(
        max_top_k=64,
        mode="prefill",
    )(
        logits,
        top_k,
        logits_position,
    )

    next_tokens = RolloutSampleOp(
        eos_token=eos_token,
        bos_token=bos_token,
    )(
        topk_values,
        topk_indices,
        rng_state,
        finished,
        temperature,
        top_k,
    )

    rt.loss = next_tokens
    rt.build_topo(next_tokens)
    rt.bind_memory()
    rt.is_compiled = True

    topk_node = None
    sample_node = None

    for node in rt.topo:
        if isinstance(node.op, RolloutTopkOp):
            topk_node = node
        elif isinstance(node.op, RolloutSampleOp):
            sample_node = node

    if topk_node is None or sample_node is None:
        raise AssertionError(
            "RolloutTopkOp or RolloutSampleOp is missing from topology"
        )

    backend = rt.backend

    # 执行 Top-k 节点，并立即同步、复制输出。
    backend.begin_pass()
    topk_node.forward(
        topk_node.inputs,
        topk_node.outputs,
    )
    backend.end_pass()

    if hasattr(backend, "synchronize"):
        backend.synchronize()

    topk_values_host = copy_to_host(
        rt,
        topk_values,
    )
    topk_indices_host = copy_to_host(
        rt,
        topk_indices,
    )

    # 继续执行 Sample 节点。
    backend.begin_pass()
    sample_node.forward(
        sample_node.inputs,
        sample_node.outputs,
    )
    backend.end_pass()

    if hasattr(backend, "synchronize"):
        backend.synchronize()

    next_tokens_host = copy_to_host(
        rt,
        next_tokens,
    )

    return {
        "topk_values": topk_values_host,
        "topk_indices": topk_indices_host,
        "next_tokens": next_tokens_host,
    }

def probe_rollout_ops(cfg, seed):
    print("\n=== RolloutTopkOp/RolloutSampleOp CPU/MPS forward ===")

    rng = np.random.default_rng(seed)

    B = max(2, min(cfg.B, 4))
    V = 97
    K = 50

    logits_np = (
        rng.standard_normal((B, 1, V))
        .astype(np.float32)
        * np.float32(0.1)
    )

    # 每一行使用非零状态，避免 xorshift 进入零状态。
    rng_state_np = np.zeros(
        (B, 4),
        dtype=np.uint32,
    )
    rng_state_np[:, 0] = np.asarray(
        [12345 + 7919 * i for i in range(B)],
        dtype=np.uint32,
    )

    finished_np = np.zeros(
        (B,),
        dtype=np.int32,
    )

    ref_values, ref_indices = rollout_topk_reference(
        logits_np,
        K,
    )

    cpu_result = run_rollout_forward_stepwise(
        "cpu",
        logits_np,
        top_k_value=K,
        temperature_value=0.0,
        rng_state_np=rng_state_np,
        finished_np=finished_np,
    )

    mps_result = run_rollout_forward_stepwise(
        "mps",
        logits_np,
        top_k_value=K,
        temperature_value=0.0,
        rng_state_np=rng_state_np,
        finished_np=finished_np,
    )

    check(
        "RolloutTopk CPU values",
        cpu_result["topk_values"][:, :K],
        ref_values[:, :K],
        atol=1e-6,
        rtol=1e-6,
    )

    check(
        "RolloutTopk CPU indices",
        cpu_result["topk_indices"],
        ref_indices,
        atol=0.0,
        rtol=0.0,
    )

    check(
        "RolloutTopk MPS values",
        mps_result["topk_values"][:, :K],
        ref_values[:, :K],
        atol=1e-6,
        rtol=1e-6,
    )

    check(
        "RolloutTopk MPS indices",
        mps_result["topk_indices"],
        ref_indices,
        atol=0.0,
        rtol=0.0,
    )

    check(
        "RolloutSample CPU greedy",
        cpu_result["next_tokens"].reshape(-1),
        ref_indices[:, 0],
        atol=0.0,
        rtol=0.0,
    )

    check(
        "RolloutSample MPS greedy",
        mps_result["next_tokens"].reshape(-1),
        ref_indices[:, 0],
        atol=0.0,
        rtol=0.0,
    )

    # finished 行必须直接输出 eos_token。
    finished_np = np.zeros((B,), dtype=np.int32)
    finished_np[1] = 1

    mps_finished = run_rollout_forward(
        "mps",
        logits_np,
        top_k_value=K,
        temperature_value=0.0,
        rng_state_np=rng_state_np,
        finished_np=finished_np,
        eos_token=99,
    )

    check(
        "RolloutSample MPS finished row",
        mps_finished["next_tokens"].reshape(-1)[1:2],
        np.asarray([99], dtype=np.int32),
        atol=0.0,
        rtol=0.0,
    )

    # temperature > 0 时，CPU 与 Metal 的随机数算法不同，
    # 只验证输出属于有效 top-k 候选。
    cpu_sample = run_rollout_forward(
        "cpu",
        logits_np,
        top_k_value=K,
        temperature_value=1.0,
        rng_state_np=rng_state_np,
        finished_np=np.zeros((B,), dtype=np.int32),
    )

    mps_sample = run_rollout_forward(
        "mps",
        logits_np,
        top_k_value=K,
        temperature_value=1.0,
        rng_state_np=rng_state_np,
        finished_np=np.zeros((B,), dtype=np.int32),
    )

    for name, result in (
        ("CPU", cpu_result),
        ("MPS", mps_result),
    ):
        sampled = result["next_tokens"].reshape(-1)
        valid_indices = result["topk_indices"][:, :K]

        for row in range(B):
            row_indices = valid_indices[row]
            if int(sampled[row]) not in row_indices:
                raise AssertionError(
                    f"RolloutSample {name} row {row} "
                    f"returned token {sampled[row]} outside top-k"
                )

def _make_block_prefix(
    device,
    cfg,
    x_np,
    arrays,
    block,
    include_mlp=False,
):
    p = f"blocks[{block}]"

    fa_params = block_parameter_dict(
        arrays,
        block,
        "fa",
    )

    attn, t = _make_fa_inputs(
        device,
        cfg,
        x_np,
        fa_params,
    )

    proj_w = make_tensor(
        arrays[f"{p}.attn.proj.weight"].shape,
        np.float32,
        device,
        True,
        arrays[f"{p}.attn.proj.weight"],
    )

    proj_b = make_tensor(
        arrays[f"{p}.attn.proj.bias"].shape,
        np.float32,
        device,
        True,
        arrays[f"{p}.attn.proj.bias"],
    )

    x_after = FusedLinearAddOp()(
        t["x"],
        attn,
        proj_w,
        proj_b,
    )

    t.update({
        "attn": attn,
        "proj_w": proj_w,
        "proj_b": proj_b,
        "x_after": x_after,
    })

    if not include_mlp:
        return x_after, t

    mp = block_parameter_dict(
        arrays,
        block,
        "mlp",
    )

    wgu = make_tensor(
        mp["w1"].shape,
        np.float32,
        device,
        True,
        mp["w1"],
    )

    wdown = make_tensor(
        mp["w2"].shape,
        np.float32,
        device,
        True,
        mp["w2"],
    )

    mlp = FusedRMSNormMLPOp(
        eps=1e-6,
    )(
        x_after,
        wgu,
        wdown,
    )

    t.update({
        "w1": wgu,
        "w2": wdown,
        "mlp": mlp,
    })

    return mlp, t


def _dispatch_fa_recompute_project(
    rt,
    cfg,
    act,
    scratch,
    out_storage,
):
    x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = act.op.inputs

    rt.backend.ctx.dispatch_with_args_3d(
        "fused_rms_fa_project_kv",
        [
            x.storage.ptr.buffer_id,
            wq.storage.ptr.buffer_id,
            bq.storage.ptr.buffer_id,
            wk.storage.ptr.buffer_id,
            bk.storage.ptr.buffer_id,
            wv.storage.ptr.buffer_id,
            bv.storage.ptr.buffer_id,
            out_storage.ptr.buffer_id,
            scratch["fa_k"].ptr.buffer_id,
            scratch["fa_v"].ptr.buffer_id,
        ],
        [
            x.storage.ptr.offset,
            wq.storage.ptr.offset,
            bq.storage.ptr.offset,
            wk.storage.ptr.offset,
            bk.storage.ptr.offset,
            wv.storage.ptr.offset,
            bv.storage.ptr.offset,
            out_storage.ptr.offset,
            scratch["fa_k"].ptr.offset,
            scratch["fa_v"].ptr.offset,
        ],
        [
            cfg.B,
            cfg.T,
            cfg.C,
            cfg.H,
            _eps_bits(),
        ],
        512, (cfg.B * cfg.T + 31) // 32, 1, 512,)

def probe_linear_add_internal(cfg, x_np, arrays, block, upstream):
    print("\n=== Internal kernels: FusedLinearAdd (real FA producer path) ===")
    p = f"blocks[{block}]"
    fa_params = block_parameter_dict(arrays, block, "fa")
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

    rt = reset_runtime("mps")
    out, t = _make_block_prefix("mps", cfg, x_np, arrays, block, include_mlp=False)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()
    fla = next(n for n in rt.topo if op_name(n) == "FusedLinearAdd")
    scratch = _scratch_from_partial(fla.backward)
    rt.forward(); rt.backend.synchronize()
    check("fused_linear_add_forward_kernel/output", copy_to_host(rt, out), expected_out)
    for tensor in t.values():
        if isinstance(tensor, Tensor) and tensor.grad is not None:
            _zero(rt, tensor.grad)
    rt.backend.copy_from_host(out.grad, dO)

    _dispatch(rt, lambda: _dispatch_fa_recompute_project(rt, cfg, t["attn"], scratch, t["attn"].grad.storage))
    # 这里只检查 fused_rms_fa_project_kv 的原始投影结果。
    check(
        "fused_rms_fa_project_kv/Q",
        read_storage_f32(
            rt,
            t["attn"].grad.storage,
            (cfg.B, cfg.T, cfg.C),
        ),
        heads_to_btc(fa_ref["q_raw"]),
    )

    check(
        "fused_rms_fa_project_kv/K",
        read_storage_f32(
            rt,
            scratch["fa_k"],
        ),
        fa_ref["k_raw"],
    )

    check(
        "fused_rms_fa_project_kv/V",
        read_storage_f32(
            rt,
            scratch["fa_v"],
        ),
        fa_ref["v"],
    )

    rope_cos = t["attn"].op.inputs[-2]
    rope_sin = t["attn"].op.inputs[-1]

    # project_kv 的 Q 是 BTHD，K 是 BHTD，因此使用 mode 2。
    _dispatch(
        rt,
        lambda: rt.backend.ctx.dispatch_with_args_3d(
            "rope_qk_inplace",
            [
                t["attn"].grad.storage.ptr.buffer_id,
                scratch["fa_k"].ptr.buffer_id,
                rope_cos.storage.ptr.buffer_id,
                rope_sin.storage.ptr.buffer_id,
            ],
            [
                t["attn"].grad.storage.ptr.offset,
                scratch["fa_k"].ptr.offset,
                rope_cos.storage.ptr.offset,
                rope_sin.storage.ptr.offset,
            ],
            [
                cfg.B,
                cfg.H,
                cfg.T,
                cfg.D,
                2,
            ],
            cfg.D // 2,
            cfg.T,
            cfg.B * cfg.H,
            cfg.D // 2,
        ),
    )

    kernel, gx, gy, threads = (
        rt.backend.fa_recompute_dispatch(
            cfg.D,
            cfg.T,
        )
    )

    kernel, gx, gy, threads = rt.backend.fa_recompute_dispatch(cfg.D, cfg.T)
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_with_args_3d(
        kernel,
        [t["attn"].grad.storage.ptr.buffer_id, scratch["fa_k"].ptr.buffer_id, scratch["fa_v"].ptr.buffer_id],
        [t["attn"].grad.storage.ptr.offset, scratch["fa_k"].ptr.offset, scratch["fa_v"].ptr.offset],
        [cfg.B, cfg.H, cfg.T, cfg.C, 1], gx, gy, cfg.B*cfg.H, threads,
    ))
    check(f"{kernel}/attention_output", copy_to_host(rt, t["attn"].grad), fa_ref["out"])

    _dispatch(rt, lambda: rt.backend.ctx.dispatch_gemm_mps(
        [t["attn"].grad.storage.ptr.buffer_id, out.grad.storage.ptr.buffer_id, t["proj_w"].grad.storage.ptr.buffer_id],
        [t["attn"].grad.storage.ptr.offset, out.grad.storage.ptr.offset, t["proj_w"].grad.storage.ptr.offset],
        [1, cfg.C, cfg.C, cfg.B*cfg.T, 1, 0, 0, 0], 1.0, 0.0,
    ))
    check("FusedLinearAdd GEMM/proj_w.grad", copy_to_host(rt, t["proj_w"].grad), expected_dw)
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_strided(
        "strided_add_backward_kernel",
        out.grad.storage.ptr.buffer_id, out.grad.storage.ptr.offset,
        t["x"].grad.storage.ptr.buffer_id, t["x"].grad.storage.ptr.offset,
        list(t["x"].shape), t["x"].strides, prod(t["x"].shape),
    ))
    check("strided_add_backward_kernel/residual.grad", copy_to_host(rt, t["x"].grad), dO)
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_gemm_mps(
        [out.grad.storage.ptr.buffer_id, t["proj_w"].storage.ptr.buffer_id, t["attn"].grad.storage.ptr.buffer_id],
        [out.grad.storage.ptr.offset, t["proj_w"].storage.ptr.offset, t["attn"].grad.storage.ptr.offset],
        [1, cfg.B*cfg.T, cfg.C, cfg.C, 0, 1, 0, 0], 1.0, 0.0,
    ))
    check("FusedLinearAdd GEMM/attention.grad", copy_to_host(rt, t["attn"].grad), expected_dact)
    _dispatch(rt, lambda: rt.backend.ctx.dispatch_with_args_3d(
        "bias_grad_reduce",
        [out.grad.storage.ptr.buffer_id, t["proj_b"].grad.storage.ptr.buffer_id],
        [out.grad.storage.ptr.offset, t["proj_b"].grad.storage.ptr.offset],
        # [cfg.B*cfg.T, cfg.C], cfg.C, 1, 1, min(cfg.C, 256),
        [cfg.B*cfg.T, cfg.C, 0], 256, (cfg.C+15)//16, 1, 256,
    ))
    check("bias_grad_reduce/proj_b.grad", copy_to_host(rt, t["proj_b"].grad), expected_db)


def _dispatch_context_recompute(rt, cfg, t, h, dh):
    del dh

    B, T, C, H = cfg.B, cfg.T, cfg.C, cfg.H
    N = B * T
    x_nbytes = N * C * 4

    q_offset = h.ptr.offset
    k_offset = q_offset + x_nbytes
    v_offset = q_offset + 2 * x_nbytes

    x, wq, bq, wk, bk, wv, bv, rope_cos, rope_sin = (
        t["attn"].op.inputs
    )

    rt.backend.ctx.dispatch_with_args_3d(
        "fused_rms_fa_project_kv",
        [
            x.storage.ptr.buffer_id,
            wq.storage.ptr.buffer_id,
            bq.storage.ptr.buffer_id,
            wk.storage.ptr.buffer_id,
            bk.storage.ptr.buffer_id,
            wv.storage.ptr.buffer_id,
            bv.storage.ptr.buffer_id,
            h.ptr.buffer_id,
            h.ptr.buffer_id,
            h.ptr.buffer_id,
        ],
        [
            x.storage.ptr.offset,
            wq.storage.ptr.offset,
            bq.storage.ptr.offset,
            wk.storage.ptr.offset,
            bk.storage.ptr.offset,
            wv.storage.ptr.offset,
            bv.storage.ptr.offset,
            q_offset,
            k_offset,
            v_offset,
        ],
        [
            B,
            T,
            C,
            H,
            _eps_bits(),
        ],
        512, (B * T + 31) // 32, 1, 512,)

    return q_offset, k_offset, v_offset


def _dispatch_context_attention(rt, cfg, h, q_off, k_off, v_off):
    kernel, gx, gy, threads = rt.backend.fa_recompute_dispatch(cfg.D, cfg.T)
    rt.backend.ctx.dispatch_with_args_3d(
        kernel, [h.ptr.buffer_id]*3, [q_off, k_off, v_off],
        [cfg.B, cfg.H, cfg.T, cfg.C, 1], gx, gy, cfg.B*cfg.H, threads,
    )
    return kernel


def _dispatch_context_linear_add(rt, cfg, t, h, dh, q_off):
    rt.backend.ctx.dispatch_with_args_3d(
        "fused_linear_add_forward_kernel",
        [t["x"].storage.ptr.buffer_id, h.ptr.buffer_id, t["proj_w"].storage.ptr.buffer_id,
         t["proj_b"].storage.ptr.buffer_id, dh.ptr.buffer_id],
        [t["x"].storage.ptr.offset, q_off, t["proj_w"].storage.ptr.offset,
         t["proj_b"].storage.ptr.offset, dh.ptr.offset],
        [cfg.B*cfg.T, cfg.C, cfg.C, 0], 256, (cfg.C+63)//64, (cfg.B*cfg.T+63)//64, 256,
    )


def probe_mlp_internal(
    cfg,
    x_np,
    arrays,
    block,
    upstream,
    residual_grad,
):
    print(
        "\n=== FusedRMSNormMLP SwiGLU validation ==="
    )

    p = f"blocks[{block}]"

    upstream = np.asarray(
        upstream,
        dtype=np.float32,
    ).copy()

    residual_grad = np.asarray(
        residual_grad,
        dtype=np.float32,
    ).copy()

    # 独立计算 FusedRMSNormFlashAttention 的参考输出。
    fa_params = block_parameter_dict(
        arrays,
        block,
        "fa",
    )

    fa_ref = attention_oracle(
        x_np,
        fa_params,
        np.zeros_like(x_np, dtype=np.float32),
        cfg.H,
    )

    # 独立计算 FusedLinearAdd 输出。
    x_after_ref = (
        np.asarray(x_np, dtype=np.float32).reshape(-1, cfg.C)
        + fa_ref["out"].reshape(-1, cfg.C)
        @ np.asarray(
            arrays[f"{p}.attn.proj.weight"],
            dtype=np.float32,
        )
        + np.asarray(
            arrays[f"{p}.attn.proj.bias"],
            dtype=np.float32,
        )
    ).reshape(
        np.asarray(x_np).shape
    ).astype(np.float32)

    mp = block_parameter_dict(
        arrays,
        block,
        "mlp",
    )

    # MLP oracle 的输入必须是 x_after_ref，而不是 block 输入 x_np。
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

        mlp, t = _make_block_prefix(
            device,
            cfg,
            x_np,
            arrays,
            block,
            include_mlp=True,
        )

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
            old_mps_bytes = 8 * N * C * itemsize

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

            if new_peak_bytes > old_mps_bytes:
                raise AssertionError(
                    "MPS SwiGLU scratch memory increased: "
                    f"new={new_peak_bytes}, old={old_mps_bytes}"
                )

        rt.forward()
        rt.backend.synchronize()

        # 先验证 MLP 收到的输入正确。
        check(
            f"{device}/FusedLinearAdd/x_after_vs_oracle",
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
        )
        check(
            f"{device}/FusedRMSNormMLP/backward/wgu.grad_vs_oracle",
            got_dwgu,
            oracle["dw1"],
            fail_fast=True,
        )
        check(
            f"{device}/FusedRMSNormMLP/backward/wdown.grad_vs_oracle",
            got_dwdown,
            oracle["dw2"],
            fail_fast=True,
        )

        results[device] = {
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
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/x.grad",
        results["mps"]["dx"],
        results["cpu"]["dx"],
        fail_fast=True,
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/wgu.grad",
        results["mps"]["dwgu"],
        results["cpu"]["dwgu"],
        fail_fast=True,
    )
    check(
        "FusedRMSNormMLP/CPU_vs_Metal/wdown.grad",
        results["mps"]["dwdown"],
        results["cpu"]["dwdown"],
        fail_fast=True,
    )

def _make_optimizer_leaf(
    rt,
    array,
    grad,
    grad_dtype=np.float32,
):
    tensor = Tensor(
        array.shape,
        np.float32,
        device=rt.allocator.device,
        requires_grad=True,
        grad_dtype=grad_dtype,
        init_val=np.asarray(
            array,
            dtype=np.float32,
        ).copy(),
    )

    tensor.storage = rt.allocator.alloc(
        tensor.shape,
        np.float32,
    )
    rt.backend.init(tensor)

    tensor.grad = Tensor(
        tensor.shape,
        np.float32,
        device=rt.allocator.device,
        requires_grad=False,
        grad_dtype=grad_dtype,
        init_val=np.asarray(
            grad,
            dtype=np.float32,
        ).copy(),
    )

    grad_storage_dtype = np.uint16 if rt.allocator.device == "mps" and str(grad_dtype).lower() in ("bf16", "bfloat16") else np.float32

    tensor.grad.storage = rt.allocator.alloc(
        tensor.shape,
        grad_storage_dtype,
    )
    rt.backend.copy_from_host(tensor.grad, np.ascontiguousarray(grad, dtype=np.float32))

    return tensor


def _adamw_numpy_step(p, g, m, v, step, lr, betas, eps, wd):
    b1, b2 = betas
    lr_t = np.float32(lr * math.sqrt(1.0 - b2**step) / (1.0 - b1**step))
    p = p.copy(); m = m.copy(); v = v.copy()
    if wd:
        p -= np.float32(lr * wd) * p
    m = np.float32(b1) * m + np.float32(1.0-b1) * g
    v = np.float32(b2) * v + np.float32(1.0-b2) * g * g
    p -= lr_t * m / (np.sqrt(v) + np.float32(eps))
    return p.astype(np.float32), m.astype(np.float32), v.astype(np.float32)

def _optimizer_quantized_parts(optimizer, index):
    if optimizer.state_mode == "fp32":
        return ()

    # 8bit/mini 的 dispatch 与 params 一一对应；
    # fp32 的 dispatch 会合并 arena，不能按此方式索引。
    item = optimizer.dispatches[index]
    if item[0] == "fp32":
        return ()

    if optimizer.state_mode == "8bit" and item[0] == "q8":
        return (
            ("m", item[3], item[5], optimizer.qmap_m),
            ("v", item[4], item[6], optimizer.qmap_v),
        )

    if optimizer.state_mode == "mini" and item[0] == "mini8bit":
        # item[13]表示是否真正使用uint8量化
        # 小参数虽然是mini8bit dispatch, 但quantized=False, 其m state是FP32, 不能进入量化验证
        if not bool(item[13]):
            return ()
        # mini模式没有全局qmap, item[4]是每块的[scale, zero] metadata
        return (("m", item[3], item[4], None, int(item[9]), bool(item[12])),)

    raise AssertionError(
        f"mode/dispatch mismatch: {optimizer.state_mode}, {item[0]}"
    )

def _read_optimizer_u8(rt, storage):
    return np.asarray(
        rt.backend.ctx.numpy_view_uint8(
            storage.ptr.buffer_id, storage.ptr.offset, storage.shape
        )
    ).copy().reshape(-1)

def read_adamw_state(rt, optimizer, index):
    """调用前 synchronize；量化状态须在至少一次 step 后读取。"""
    p = optimizer.params[index]

    if optimizer.state_mode == "mini":
        item = optimizer.dispatches[index]

        if item[0] == "mini8bit":
            m_storage = item[3]
            meta_storage = item[4]
            v_storage = item[5]
            quantized = bool(item[13])

            if quantized:
                codes = _read_optimizer_u8(rt, m_storage)
                meta = read_storage_f32(rt, meta_storage).reshape(-1, 2)

                blocks = (np.arange(p.numel, dtype=np.int64) // optimizer.BLOCK_SIZE)
                decoded = (codes.astype(np.float32) - meta[blocks, 1]) * meta[blocks, 0]

                if bool(item[12]): # transpose_m
                    cols = int(item[9])
                    rows = p.numel // cols
                    j = np.arange(p.numel, dtype=np.int64)
                    physical_index = (j % rows) * cols + j // rows
                    logical = np.empty_like(decoded)
                    logical[physical_index] = decoded
                    m = logical
                else:
                    m = decoded
            else:
                m = read_storage_f32(rt, m_storage).reshape(-1)

            v = read_storage_f32(rt, v_storage).reshape(-1)
            return m.reshape(p.shape), v

    parts = _optimizer_quantized_parts(optimizer, index)

    if not parts:
        return (
            read_storage_f32(rt, optimizer.m[index], p.shape),
            read_storage_f32(rt, optimizer.v[index], p.shape),
        )

    decoded = {}
    blocks = np.arange(p.numel) // optimizer.BLOCK_SIZE

    for name, codes_storage, scale_storage, table_storage in parts:
        codes = _read_optimizer_u8(rt, codes_storage)
        scale = read_storage_f32(rt, scale_storage).reshape(-1)
        table = read_storage_f32(rt, table_storage, (256,))
        decoded[name] = (
            table[codes] * scale[blocks]
        ).reshape(p.shape)

    v = (
        decoded["v"]
        if "v" in decoded
        else read_storage_f32(rt, optimizer.v[index])
    )
    return decoded["m"], v

def _adamw_reference_step(p, g, m, v, step, optimizer, group_ids):
    def ema_f32(old, target, beta):
        old = old.astype(np.float32)
        target = target.astype(np.float32)
        beta = np.float32(beta)
        return (old + (target - old) * (np.float32(1.0) - beta)).astype(np.float32)

    b1, b2 = optimizer.betas

    if group_ids is None:
        target_v = g.astype(np.float32) * g.astype(np.float32)
        v_indices = np.arange(g.size)
    else:
        count = np.bincount(group_ids)
        target_v = (np.bincount(group_ids, weights=g.astype(np.float64) ** 2) / count).astype(np.float32)
        v_indices = group_ids

    new_m = ema_f32(m, g, b1)
    new_v = ema_f32(v, target_v, b2)

    lr_bc1 = np.float32(optimizer.lr / (1.0 - b1**step))
    inv_sqrt_bc2 = np.float32(1.0 / math.sqrt(1.0 - b2**step))
    denom = np.sqrt(np.maximum(new_v[v_indices], np.float32(0.0))) * inv_sqrt_bc2 + np.float32(optimizer.eps)
    direction = -new_m / denom

    # p 已由调用方应用 decay。
    # 参数更新使用本步未量化的 m/v，随后才保存量化状态。
    new_p = (p.astype(np.float32) + direction.astype(np.float32) * lr_bc1).astype(np.float32)

    return new_p, new_m, new_v

def _check_optimizer_quantization(rt, label, part, expected):
    if len(part) == 6:
        _, codes_storage, scale_storage, table_storage, item_cols, transpose_m = part
    else:
        _, codes_storage, scale_storage, table_storage = part
        item_cols = None
        transpose_m = False 
    codes = _read_optimizer_u8(rt, codes_storage)

    # mini8bit: 每个256元素块使用独立的scale/zero,不存在256项全局查找表
    if table_storage is None:
        meta = read_storage_f32(rt, scale_storage).reshape(-1, 2)
        expected = np.asarray(expected, dtype=np.float32).reshape(-1)
        if transpose_m:
            cols = int(item_cols)
            rows = expected.size // cols 
            j = np.arange(expected.size, dtype=np.int64)
            physical_index = ((j % rows) * cols) + j // rows
            expected_quant = expected[physical_index]
        else:
            expected_quant = expected
        n_blocks = (expected_quant.size + 255) // 256
        assert meta.shape == (n_blocks, 2), (f"{label}/meta shape dismatch: got {meta.shape}, expected {(n_blocks, 2)}")
        blocks = np.arange(expected.size, dtype=np.int64) // 256
        block_min = np.minimum.reduceat(expected_quant, np.arange(0, expected_quant.size, 256))
        block_max = np.maximum.reduceat(expected_quant, np.arange(0, expected_quant.size, 256))
        expected_scale = np.maximum(block_max - block_min, np.float32(1.0e-5)) / np.float32(255.0)
        expected_zero = np.clip(-np.rint(block_min/expected_scale), 0.0, 255.0).astype(np.float32)
        np.testing.assert_allclose(meta[:, 0], expected_scale, rtol=3e-5, atol=1e-12, err_msg=label+"/scale")
        raw_zero = (-block_min/expected_scale).astype(np.float64)
        zero_slack = 8.0 * float(np.finfo(np.float32).eps) * np.maximum(1.0, np.abs(raw_zero))
        zero_lower = np.clip(np.rint(raw_zero - zero_slack), 0.0, 255.0)
        zero_upper = np.clip(np.rint(raw_zero + zero_slack), 0.0, 255.0)
        got_zero = meta[:, 1].astype(np.float64)
        valid_zero = np.isfinite(got_zero) & (got_zero == np.rint(got_zero)) & (got_zero >= zero_lower) & (got_zero <= zero_upper)
        if not np.all(valid_zero):
            bad = np.flatnonzero(~valid_zero)
            index = int(bad[0])
            raise AssertionError(
                f"{label}/zero: block={index}, "
                f"got={got_zero[index]}, "
                f"reference_unrounded={raw_zero[index]}, "
                f"allowed=[{zero_lower[index]}, "
                f"{zero_upper[index]}]"
            )
        decoded = (codes.astype(np.float32) - meta[blocks, 1]) * meta[blocks, 0]

        # mini kernel的量化误差最多约为半个量化步长,这里使用一个完整的step容差，兼容stochastic rounding
        error = np.abs(decoded - expected_quant)
        allowed = meta[blocks, 0] + 3e-6
        assert np.all(error <= allowed), f"{label}/decoded: max_error={error.max()}, max_allowed={allowed.max()}"
        return

    # 8bit动态查找表路径
    table = read_storage_f32(rt, table_storage, (256, ))
    assert np.all(np.isfinite(table))
    assert np.all(np.diff(table) >= 0)

    expected = np.asarray(expected, dtype=np.float32).reshape(-1)
    expected_scale = np.maximum(np.maximum.reduceat(np.abs(expected), np.arange(0, expected.size, 256)), np.float32(1e-30))
    actual_scale = read_storage_f32(rt, scale_storage).reshape(-1)
    np.testing.assert_allclose(actual_scale, expected_scale, rtol=3e-5, atol=1e-30, err_msg=label+"/scale")

    normalized = np.clip(expected / expected_scale[np.arange(expected.size) // 256], table[0], table[-1])
    upper = np.clip(np.searchsorted(table, normalized, side="left"), 0, 255)
    lower = np.maximum(upper - 1, 0)

    best = np.minimum(np.abs(table[upper] - normalized), np.abs(table[lower] - normalized))
    actual = np.abs(table[codes] - normalized)

    # 允许舍入误差使量化中点附近的值选择相邻编码。
    assert np.all(actual <= best + 3e-6), label + "/nearest-code"

def probe_adamw(state_mode=OPTIMIZER_STATE_MODE, grad_dtype=np.float32):
    if state_mode not in ("fp32", "8bit", "mini"):
        raise ValueError(state_mode)

    print(
        f"\n=== AdamW state_mode={state_mode}: Metal vs NumPy ==="
    )

    cfg = Config(B=1, T=8, V=128, C=128, H=2, D=64, L=1)
    arrays = build_parameter_arrays(cfg, SEED + 701)
    state = build_graph(
        "mps", cfg, arrays, *build_inputs(cfg, SEED + 702), grad_dtype=grad_dtype,
    )
    rt = state.rt
    rng = np.random.default_rng(SEED + 703)

    # 保留全部模型参数，兼容 FP32 arena 合并。
    cases = list(state.param_by_name.items())
    cases.append((
        "extra_vector",
        _make_optimizer_leaf(
            rt,
            rng.standard_normal((4101,)).astype(np.float32) * 0.02,
            np.zeros((4101,), dtype=np.float32),
            grad_dtype=grad_dtype,
        ),
    ))

    if state_mode != "mini":
        shape = (17, 13)
        p0 = (
            rng.standard_normal(shape) * 0.02
        ).astype(np.float32)
        cases.append((
            "small_matrix",
            _make_optimizer_leaf(
                rt,
                p0,
                np.zeros_like(p0),
            ),
        ))

    optimizer = AdamW(
        [p for _, p in cases],
        lr=0.002,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.2,
        state_mode=state_mode,
    )
    assert [id(p) for p in optimizer.params] == [
        id(p) for _, p in cases
    ]

    # 独立从模型定义取得 head 信息，不从优化器 dispatch 反推。
    heads = {}
    column_ids = set()
    stochastic_ids = set()
    for node in state.rt.topo:
        op_name = node.op.op_name
        if op_name == "Gather":
            stochastic_ids.add(id(node.inputs[1]))
        elif op_name == "FusedLinearCrossEntropy":
            stochastic_ids.add(id(node.inputs[1]))
        elif op_name == "FusedRMSNormMLP":
            column_ids.add(id(node.inputs[1]))
            column_ids.add(id(node.inputs[2]))
        elif op_name == "FusedLinearAdd":
            column_ids.add(id(node.inputs[2]))

    for block in state.model.blocks:
        for p in (
            block.attn.q_proj.weight,
            block.attn.k_proj.weight,
            block.attn.v_proj.weight,
        ):
            heads[id(p)] = block.attn.head_dim

    group_ids_by_param = []
    for i, (_, p) in enumerate(cases):
        large = p.numel >= 4096
        ids = None

        if state_mode != "fp32":
            if state_mode == "8bit":
                wanted = "q8" if large else "fp32"
            elif state_mode == "mini":
                wanted = "mini8bit"
            assert optimizer.dispatches[i][0] == wanted

        if state_mode == "mini" and large:
            cols = p.shape[-1]
            head_cols = 0
            if id(p) in heads:
                # Q/K权重: 按attention head分组
                head_cols = int(heads[id(p)])
                block_elems = int(p.shape[0]) * head_cols
                ids = (np.arange(p.numel) % cols) // head_cols
            elif id(p) in column_ids:
                # MLP/Linear权重: 按列分组
                head_cols = 1
                block_elems = int(p.shape[0])
                ids = (np.arange(p.numel) % cols)
            elif id(p) in stochastic_ids:
                # embedding/stochastic参数: 按连续列块分组
                head_cols = 0
                block_elems = cols
                ids = np.arange(p.numel) // block_elems
            else:
                # 普通向量或未分组参数
                # block_elems = int(p.numel)
                head_cols = 0
                block_elems = 1
                ids = np.arange(p.numel, dtype=np.int64)

            assert optimizer.dispatches[i][8:11] == (block_elems, cols, head_cols)
            assert prod(optimizer.v[i].shape) == int(ids.max()) + 1

        group_ids_by_param.append(ids)

    rt.backend.synchronize()

    # 首步零梯度：验证初始化和零 scale。
    # 中间两步非零：验证量化状态解码及更新。
    # 最后零梯度：验证历史动量延续。
    for step, wd in enumerate((0.2, 0.0, 0.07, 0.13), 1):
        optimizer.lr = 0.002 / step
        optimizer.weight_decay = wd
        expected = []

        for i, (_, p) in enumerate(cases):
            ids = group_ids_by_param[i]

            if step == 1:
                old_m = np.zeros(p.numel, np.float32)
                old_v = np.zeros(
                    p.numel if ids is None else int(ids.max()) + 1,
                    np.float32,
                )
            else:
                old_m, old_v = read_adamw_state(rt, optimizer, i)
                old_m = old_m.reshape(-1)
                old_v = old_v.reshape(-1)

            g = (
                rng.standard_normal(p.shape) * 0.03
            ).astype(np.float32)

            if p.ndim > 1:
                # 各列幅度不同，便于识别错误的 head 分组。
                g *= np.linspace(
                    0.2, 2.0, p.shape[-1], dtype=np.float32
                )
            if step in (1, 4):
                g.fill(0)

            if np.dtype(p.grad.storage.dtype) == np.dtype(np.uint16):
                bits = np.ascontiguousarray(g).view(np.uint32)
                rounded_bits = (
                    bits
                    + np.uint32(0x7FFF)
                    + ((bits >> np.uint32(16)) & np.uint32(1))
                ) & np.uint32(0xFFFF0000)
                g = rounded_bits.view(np.float32)

            rt.backend.copy_from_host(
                p.grad, np.ascontiguousarray(g)
            )

            old_p = copy_to_host(rt, p).reshape(-1)
            decay = np.float32(
                1.0 - optimizer.lr * (wd if p.ndim > 1 else 0.0)
            )
            expected.append(_adamw_reference_step(
                old_p * decay,
                g.reshape(-1),
                old_m,
                old_v,
                step,
                optimizer,
                ids,
            ))

        optimizer.step()
        rt.backend.synchronize()

        for i, (name, p) in enumerate(cases):
            ref_p, ref_m, ref_v = expected[i]
            label = f"AdamW/{state_mode}/{name}/step{step}"

            if optimizer.state_mode == "fp32":
                quantized = False
                stochastic = False
            elif optimizer.state_mode == "mini":
                item = optimizer.dispatches[i]
                quantized = item[0] == "mini8bit" and bool(item[13])
                stochastic = item[0] == "mini8bit" and bool(item[11])
            else:
                item = optimizer.dispatches[i]
                quantized = optimizer.state_mode == "8bit" and item[0] == "q8"
                stochastic = False
            _check_optimizer_value(label + "/p", 
                copy_to_host(rt, p).reshape(-1),
                ref_p,
                state_mode=optimizer.state_mode,
                value_kind="parameter",
                quantized=quantized,
                stochastic=stochastic,
                fail_fast=True)

            got_m, got_v = read_adamw_state(rt, optimizer, i)
            parts = {
                part[0]: part
                for part in _optimizer_quantized_parts(optimizer, i)
            }

            for moment, got, ref in (
                ("m", got_m, ref_m),
                ("v", got_v, ref_v),
            ):
                if moment in parts:
                    _check_optimizer_quantization(
                        rt, label + "/" + moment, parts[moment], ref
                    )
                else:
                    _check_optimizer_value(
                        label + "/" + moment,
                        got.reshape(-1),
                        ref,
                        state_mode=optimizer.state_mode,
                        value_kind=moment,
                        quantized=quantized,
                        stochastic=stochastic,
                        fail_fast=True,
                    )

    print(
        f"  PASS AdamW/{state_mode}: "
        "parameters, moments and quantization"
    )

def _copy_training_batch(state, input_ids, labels):
    state.rt.backend.copy_from_host(state.input_ids, input_ids)
    state.rt.backend.copy_from_host(state.labels, labels)


def _parameter_snapshot(state):
    return {
        name: copy_to_host(state.rt, tensor)
        for name, tensor in state.param_by_name.items()
    }


def _gradient_snapshot(state):
    return {
        name: copy_to_host(state.rt, tensor.grad)
        for name, tensor in state.param_by_name.items()
    }


def _optimizer_snapshot(state, optimizer):
    moments = {}

    for index, tensor in enumerate(optimizer.params):
        name = state.param_name_by_id[id(tensor)]
        m, v = read_adamw_state(state.rt, optimizer, index)
        moments[name] = {"m": m, "v": v}

    return moments

def _gradient_storage_layout(state):
    layout = {}
    for name, tensor in state.param_by_name.items():
        storage = tensor.grad.storage
        layout[name] = (
            storage.ptr.buffer_id,
            storage.ptr.offset,
            storage.nbytes,
        )
    return layout


def _assert_step_updated(label, before, after):
    max_change = max(
        float(np.max(np.abs(after[name] - before[name]), initial=0.0))
        for name in before
    )
    ok = max_change > 0.0
    print(f"  {'PASS' if ok else 'FAIL'} {label}: max_parameter_change={max_change:.8e}")
    if not ok:
        FAILURES.append((label, 0.0, 0.0))


def _make_accumulation_batches(cfg, accumulation_steps, training_steps, seed):
    large_cfg = Config(
        B=cfg.B * accumulation_steps,
        T=cfg.T,
        V=cfg.V,
        C=cfg.C,
        H=cfg.H,
        D=cfg.D,
        L=cfg.L,
        dtype=cfg.dtype,
    )
    batches = []
    for step in range(training_steps):
        large_batch = build_inputs(large_cfg, seed + 1000 + step * 17)
        micro_batches = []
        for micro_step in range(accumulation_steps):
            start = micro_step * cfg.B
            end = start + cfg.B
            micro_batches.append(tuple(array[start:end].copy() for array in large_batch))
        batches.append((large_batch, micro_batches))
    return large_cfg, batches


def _run_training_trajectory(cfg, arrays, batches, accumulation_steps, enabled, 
                             optimizer_gradients=None, grad_dtype=np.float32):
    initial = batches[0][1][0]
    state = build_graph("mps", cfg, arrays, *initial, grad_dtype=grad_dtype)
    optimizer = AdamW(
        state.model.parameters(),
        lr=0.002,
        weight_decay=0.234903,
        betas=(0.9, 0.999),
        eps=1e-8,
        state_mode=OPTIMIZER_STATE_MODE,
    )
    optimizer_metadata = {}
    for index, t in enumerate(optimizer.params):
        name = state.param_name_by_id[id(t)]
        quantized = False
        stochastic = False
        if optimizer.state_mode == "mini" and isinstance(optimizer.dispatches, tuple):
            item = optimizer.dispatches[index]
            if item[0] == "mini8bit":
                stochastic = bool(item[11])
                quantized = bool(item[13])
        # elif optimizer.state_mode == "8bit" and isinstance(optimizer.dispatches, tuple):
        #     item = optimizer.dispatches[index]
        #     quantized = item[0] == "q8"
        optimizer_metadata[name] = {"quantized": quantized, "stochastic": stochastic}

    initial_layout = _gradient_storage_layout(state)
    history = []
    for step, (large_batch, micro_batches) in enumerate(batches, 1):
        before = _parameter_snapshot(state)
        if enabled:
            for accumulation_step, micro_batch in enumerate(micro_batches):
                _copy_training_batch(state, *micro_batch)
                state.loss.backward(
                    grad_scale=1.0 / accumulation_steps,
                    # Step 0 overwrites the existing parameter-gradient
                    # storage. Later micro-steps add into the same storage,
                    # so no zero_grad call or second accumulation buffer is
                    # needed.
                    accumulate_parameter_grads=accumulation_step > 0,
                )
        else:
            micro_gradients = []

            for micro_batch in micro_batches:
                _copy_training_batch(state, *micro_batch)

                # 与真实累加路径一致：先对每个 micro-batch 缩放。
                state.loss.backward(
                    grad_scale=1.0 / accumulation_steps,
                )
                state.rt.backend.synchronize()

                micro_gradients.append(
                    _gradient_snapshot(state)
                )

            # 每个 micro-batch 已经缩放过，因此这里求和而不是求平均。
            gradients = {
                name: np.sum(
                    np.stack(
                        [
                            item[name]
                            for item in micro_gradients
                        ],
                        axis=0,
                    ),
                    axis=0,
                    dtype=np.float32,
                ).astype(np.float32)
                for name in state.param_by_name
            }

            for name, tensor in state.param_by_name.items():
                state.rt.backend.copy_from_host(
                    tensor.grad,
                    gradients[name],
                )
        state.rt.backend.synchronize()
        if enabled:
            gradients = _gradient_snapshot(state)
        if optimizer_gradients is not None:
            reference_gradients = optimizer_gradients[step - 1]

            for name, tensor in state.param_by_name.items():
                state.rt.backend.copy_from_host(
                    tensor.grad,
                    reference_gradients[name],
                )
        optimizer.step()
        state.rt.backend.synchronize()
        parameters = _parameter_snapshot(state)
        _assert_step_updated(
            f"grad_accum[{accumulation_steps}]/{'enabled' if enabled else 'baseline'}/step[{step}]/updated",
            before,
            parameters,
        )
        history.append({
            "gradients": gradients,
            "parameters": parameters,
            "optimizer": _optimizer_snapshot(state, optimizer),
            "optimizer_metadata": optimizer_metadata,
        })
    if _gradient_storage_layout(state) != initial_layout:
        raise AssertionError("gradient accumulation replaced or reallocated parameter gradient storage")
    return history

def _compare_training_trajectories(reference, accumulated, accumulation_steps, state_mode, grad_dtype=np.float32):
    gradient_atol, gradient_rtol = ATOL, RTOL
    if str(grad_dtype).lower() in ("bf16", "bfloat16"):
        gradient_atol, gradient_rtol = tolerances_for_grad_dtype(grad_dtype, "parameter_grad")
        gradient_atol *= accumulation_steps
        gradient_rtol *= accumulation_steps

    bitwise_equal = 0
    tensor_count = 0
    max_abs = {"gradient": 0.0, "parameter": 0.0, "optimizer.m": 0.0, "optimizer.v": 0.0}
    for step, (ref_step, acc_step) in enumerate(zip(reference, accumulated), 1):
        metadata = acc_step.get("optimizer_metadata", {})
        for name in sorted(ref_step["gradients"]):
            meta = metadata.get(name, {"quantized": False, "stochastic": False,})
            quantized = bool(meta["quantized"])
            stochastic = bool(meta["stochastic"])
            gradient_label = f"grad_accum[{accumulation_steps}]/step[{step}]/gradient/{name}"
            max_abs["gradient"] = max(
                max_abs["gradient"],
                float(np.max(np.abs(
                    acc_step["gradients"][name] - ref_step["gradients"][name]
                ), initial=0.0)),
            )
            tensor_count += 1
            bitwise_equal += int(np.array_equal(
                acc_step["gradients"][name], ref_step["gradients"][name]
            ))
            # Separate MPS graph executions can differ in the last few bits
            # for parallel reductions. K=1 still takes the exact overwrite
            # path (accumulation_step == 0); numerical equality is therefore
            # the meaningful cross-run assertion here.
            check(gradient_label, acc_step["gradients"][name], ref_step["gradients"][name],
                atol=gradient_atol, rtol=gradient_rtol)
            parameter_label = f"grad_accum[{accumulation_steps}]/step[{step}]/parameter/{name}"
            _check_grad_accum_optimizer_value(parameter_label,
                                              acc_step["parameters"][name],
                                              ref_step["parameters"][name],
                                              state_mode=state_mode,
                                              value_kind="parameter",
                                              quantized=quantized,
                                              stochastic=stochastic)
            max_abs["parameter"] = max(
                max_abs["parameter"],
                float(np.max(np.abs(
                    acc_step["parameters"][name] - ref_step["parameters"][name]
                ), initial=0.0)),
            )
            for moment in ("m", "v"):
                key = f"optimizer.{moment}"
                _check_grad_accum_optimizer_value(
                    f"grad_accum[{accumulation_steps}]/step[{step}]/{key}/{name}",
                    acc_step["optimizer"][name][moment],
                    ref_step["optimizer"][name][moment],
                    state_mode=state_mode,
                    value_kind=moment,
                    quantized=quantized,
                    stochastic=stochastic,
                )
                max_abs[key] = max(
                    max_abs[key],
                    float(np.max(np.abs(
                        acc_step["optimizer"][name][moment]
                        - ref_step["optimizer"][name][moment]
                    ), initial=0.0)),
                )
    print(
        f"  INFO grad_accum[{accumulation_steps}]/max_abs: "
        + ", ".join(f"{key}={value:.8e}" for key, value in max_abs.items())
    )
    if accumulation_steps == 1:
        print(
            "  INFO grad_accum[1]/bitwise-identical-gradient-tensors: "
            f"{bitwise_equal}/{tensor_count}; K=1 equivalence is asserted "
            "with the configured numerical tolerances because independent "
            "Metal reductions are not guaranteed bitwise deterministic"
        )

def probe_gradient_accumulation(cfg, arrays, seed, grad_dtype=np.float32):
    print("\n=== No-extra-buffer parameter gradient accumulation ===")
    training_steps = 2
    for accumulation_steps in (1, 2, 4):
        print(f"  [accumulation_steps={accumulation_steps}]")
        _, batches = _make_accumulation_batches(
            cfg, accumulation_steps, training_steps, seed
        )
        reference = _run_training_trajectory(
            cfg, arrays, batches, accumulation_steps, enabled=False, grad_dtype=grad_dtype,
        )
        accumulated = _run_training_trajectory(
            cfg, arrays, batches, accumulation_steps, enabled=True,
            optimizer_gradients=[item["gradients"] for item in reference],
            grad_dtype=grad_dtype,
        )
        _compare_training_trajectories(reference, accumulated, accumulation_steps, OPTIMIZER_STATE_MODE, grad_dtype=grad_dtype)

def main():
    global PRINT_PASSES
    args, cfg = parse_args()
    PRINT_PASSES = not args.compact
    grad_dtype = grad_dtype_value(args.grad_dtype)
    print(f"grad_dtype={args.grad_dtype}")
    print(f"project_root={PROJECT_ROOT}")
    print(f"runtime={runtime.__file__}")
    print(
        "config: "
        f"B={cfg.B}, T={cfg.T}, V={cfg.V}, C={cfg.C}, H={cfg.H}, "
        f"D={cfg.D}, L={cfg.L}, atol={ATOL:g}, rtol={RTOL:g}"
    )
    arrays = build_parameter_arrays(cfg, args.seed)
    input_ids, labels = build_inputs(cfg, args.seed)

    if args.gradient_accumulation_only:
        probe_gradient_accumulation(cfg, arrays, args.seed, grad_dtype=grad_dtype)
        if FAILURES:
            print(f"\n{len(FAILURES)} gradient accumulation correctness check(s) failed:")
            for name, max_abs, max_rel in FAILURES:
                print(f"  {name}: max_abs={max_abs:.8e}, max_rel={max_rel:.8e}")
            raise AssertionError(
                f"{len(FAILURES)} gradient accumulation correctness check(s) failed"
            )
        print("\nAll gradient accumulation correctness checks passed.")
        return

    # The current extension must initialize its Metal contexts before the first
    # CPUBackend instance is constructed.  Device caches both afterwards.
    mps_exact = run_exact_training_graph(
        build_graph("mps", cfg, arrays, input_ids, labels, grad_dtype=grad_dtype, use_fp16_gemm=args.use_fp16_gemm)
    )
    cpu_exact = run_exact_training_graph(
        build_graph("cpu", cfg, arrays, input_ids, labels, grad_dtype=np.float32)
    )
    compare_exact_graph(cpu_exact, mps_exact, mps_grad_dtype=grad_dtype)

    cpu_trace = run_staged_graph(
        build_graph("cpu", cfg, arrays, input_ids, labels, grad_dtype=np.float32)
    )
    mps_trace = run_staged_graph(
        build_graph("mps", cfg, arrays, input_ids, labels, grad_dtype=grad_dtype, use_fp16_gemm=args.use_fp16_gemm),
        reference_inputs=cpu_trace["forward_inputs"],
    )
    compare_staged_graph(cpu_trace, mps_trace, mps_grad_dtype=grad_dtype)

    probe_kv_cache_attention(cfg, args.seed + 1000)

    probe_lm_head_op(cfg, args.seed + 2000,)

    # probe_rollout_ops(cfg, args.seed + 3000)

    if not args.skip_internal_kernels:
        block = args.kernel_block
        if block == 0:
            x_in = cpu_trace["forward"][f"Add[{block}]"][0]
        else:
            x_in = cpu_trace["forward"][f"Add[{block-1}]"][0]
        fa_upstream = cpu_trace["backward"][f"FusedRMSNormFlashAttention[{block}]"]["output_grads"][0]
        fla_upstream = cpu_trace["backward"][f"FusedLinearAdd[{block}]"]["output_grads"][0]
        mlp_bwd = cpu_trace["backward"][f"FusedRMSNormMLP[{block}]"]
        mlp_upstream = mlp_bwd["output_grads"][0]
        mlp_residual = mlp_bwd["input_grads_before"][0]
        probe_fa_internal(cfg, x_in, block_parameter_dict(arrays, block, "fa"), fa_upstream)
        probe_linear_add_internal(cfg, x_in, arrays, block, fla_upstream)
        probe_mlp_internal(cfg, x_in, arrays, block, mlp_upstream, mlp_residual)
        final_x = cpu_trace["forward"][f"Add[{cfg.L-1}]"][0]
        probe_ce_internal(cfg, final_x, arrays["lm_head.weight"], labels)

    if not args.skip_gradient_accumulation:
        probe_gradient_accumulation(cfg, arrays, args.seed, grad_dtype=grad_dtype)

    if not args.skip_optimizer:
        probe_adamw(grad_dtype=grad_dtype)
    if FAILURES:
        print(f"\n{len(FAILURES)} correctness check(s) failed:")
        for name, max_abs, max_rel in FAILURES:
            print(f"  {name}: max_abs={max_abs:.8e}, max_rel={max_rel:.8e}")
        raise AssertionError("real GPT2 training-path correctness validation failed")
    print("\nAll real GPT2 training-path correctness checks passed.")

if __name__ == "__main__":
    main()
