#!/usr/bin/env python3
"""Correctness checker for CUDA kernels against the immutable FP32 CPU oracle.

Usage (in NVIDIA GPU environment after running ./test.sh):
    python3 debug/check_cuda_kernel.py
    python3 debug/check_cuda_kernel.py --grad-dtype bf16
    python3 debug/check_cuda_kernel.py --B 4 --T 128 --depth 4
"""

import argparse
import functools
import gc
import os
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


def _locate_project_root():
    candidates = []
    env_root = os.environ.get("AUTOGRAD_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve().parent
    candidates.extend((Path.cwd(), Path.cwd().parent, here, here.parent))
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "runtime.py").is_file() and (candidate / "backend" / "cuda.py").is_file():
            return candidate
    raise RuntimeError(
        "Cannot locate autograd project root. Run from the repository/debug directory "
        "or set AUTOGRAD_ROOT=/absolute/path/to/autograd."
    )


PROJECT_ROOT = _locate_project_root()
DEBUG_DIR = (PROJECT_ROOT / "debug").resolve()
os.chdir(PROJECT_ROOT)
# 移除 debug 目录路径，确保始终加载根目录拆分后的 runtime.py、backend/ 与 cuda_ext*.so
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != DEBUG_DIR]
sys.path.insert(0, str(PROJECT_ROOT))

import runtime  # noqa: E402
from module import Embedding, Module  # noqa: E402
from op import AddOp, GatherOp  # noqa: E402
from tensor import Tensor  # noqa: E402

Runtime = runtime.Runtime


def make_tensor(shape, dtype, device, requires_grad, init_val, grad_dtype=None, name=""):
    return Tensor(
        shape=shape,
        dtype=dtype,
        grad_dtype=grad_dtype,
        device=device,
        requires_grad=requires_grad,
        init_val=np.asarray(init_val, dtype=dtype).copy(),
        name=name,
    )


def grad_dtype_value(value):
    if value == "fp32":
        return np.float32
    if value == "bf16":
        return "bf16"
    raise ValueError(f"unsupported grad_dtype={value!r}")


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
        description="Verify CUDA operator kernels against the FP32 CPU reference"
    )
    parser.add_argument("--depth", "--L", dest="depth", type=int, default=2)
    parser.add_argument("--aspect-ratio", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-seq-len", "--T", dest="max_seq_len", type=int, default=32)
    parser.add_argument("--device-batch-size", "--B", dest="device_batch_size", type=int, default=2)
    parser.add_argument("--vocab-size", "--V", dest="vocab_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-internal-kernels",
        action="store_true",
        help="run exact graph and per-operator staged checks only",
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
        help="Logical dtype of CUDA parameter gradients. CPU reference always remains FP32.",
    )
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

    base_dim = args.depth * args.aspect_ratio
    model_dim = (base_dim + args.head_dim - 1) // args.head_dim * args.head_dim
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
    return args, cfg


def reset_runtime(device):
    Runtime._default_runtime = None
    gc.collect()
    rt = Runtime(device=device, use_fp16_gemm=False)
    Runtime._default_runtime = rt
    return rt


def copy_to_host(rt, tensor):
    return np.asarray(rt.backend.copy_to_host(tensor)).copy()


def read_storage_f32(rt, storage, shape=None):
    shape = tuple(shape if shape is not None else storage.shape)
    if rt.allocator.device == "cpu":
        return np.asarray(storage.data).reshape(shape).copy()
    st_dtype = np.dtype(storage.dtype)
    raw = rt.backend.ctx.copy_bytes_to_host(
        storage.ptr.buffer_id,
        storage.ptr.offset,
        prod(shape) * st_dtype.itemsize,
    )
    arr = np.frombuffer(raw, dtype=st_dtype).reshape(shape)
    if st_dtype == np.dtype(np.uint16):
        return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()
    return arr.astype(np.float32).copy()


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
    """CPU reference 始终保持 FP32；仅根据待验证 GPU 参数梯度类型决定允许容差。"""
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


# ---------------------------------------------------------------------------
# 当前已实现的 CUDA 算子组合模型（GatherOp + AddOp 偏置广播/同形状残差）
# 后续补齐其余 4 个融合算子后，可直接替换为 module.GPT2 与原版期望拓扑
# ---------------------------------------------------------------------------
class ImplementedCudaBlock(Module):
    def __init__(self, V, C, dtype, device):
        self.bias = Tensor(shape=(C,), init_val="zeros", dtype=dtype, requires_grad=True, device=device, name="bias")
        self.res_emb = Embedding(V, C, dtype, device)

    def forward(self, x, aux_ids):
        h = x + self.bias
        r = self.res_emb(aux_ids)
        return h + r


class ImplementedCudaModel(Module):
    def __init__(self, V, C, n_layer, dtype, device):
        self.tok_emb = Embedding(V, C, dtype, device)
        self.blocks = [ImplementedCudaBlock(V, C, dtype, device) for _ in range(n_layer)]

    def forward(self, input_ids, aux_ids):
        x = self.tok_emb(input_ids)
        for blk in self.blocks:
            x = blk(x, aux_ids)
        return x


def iter_named_tensors(model):
    yield "tok_emb.weight", model.tok_emb.weight
    for index, block in enumerate(model.blocks):
        p = f"blocks[{index}]"
        yield f"{p}.bias", block.bias
        yield f"{p}.res_emb.weight", block.res_emb.weight


def verify_training_topology(topo, cfg):
    expected = Counter({
        "Gather": 1 + cfg.L,
        "Add": 2 * cfg.L,
    })
    actual = Counter(op_name(node) for node in topo)
    if actual != expected:
        raise AssertionError(
            "Topology differs from expected graph:\n"
            f"  expected={dict(expected)}\n"
            f"  actual={dict(actual)}"
        )
    print("Training topology:", " -> ".join(topology_labels(topo)))


def build_parameter_arrays(cfg, seed):
    rng = np.random.default_rng(seed)
    model = ImplementedCudaModel(cfg.V, cfg.C, cfg.L, cfg.dtype, "cpu")
    arrays = {}
    for name, tensor in iter_named_tensors(model):
        arrays[name] = (rng.standard_normal(tensor.shape).astype(np.float32) * 0.02).copy()
    return arrays


def initialize_model(model, arrays):
    names = []
    for name, tensor in iter_named_tensors(model):
        if name not in arrays:
            raise KeyError(f"missing parameter {name}")
        tensor.init_val = arrays[name].copy()
        names.append(name)
    if set(names) != set(arrays):
        raise AssertionError("parameter array set does not match model")


def build_inputs(cfg, seed):
    rng = np.random.default_rng(seed + 1)
    input_ids = rng.integers(0, cfg.V, (cfg.B, cfg.T), dtype=np.int32)
    aux_ids = rng.integers(0, cfg.V, (cfg.B, cfg.T), dtype=np.int32)
    # 注入重复 token ID，覆盖 Gather 反向的 atomicAdd 并发冲突与 BF16 warp 去重累加逻辑
    hot_vocab = min(cfg.V, 8)
    half_t = max(1, cfg.T // 2)
    input_ids[:, :half_t] = rng.integers(0, hot_vocab, (cfg.B, half_t), dtype=np.int32)
    aux_ids[:, :half_t] = rng.integers(0, hot_vocab, (cfg.B, half_t), dtype=np.int32)
    return input_ids, aux_ids


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
    """为 CPU 参考路径上的每个张量、梯度和 scratch 分配独立私有内存，彻底解耦 GPU 内存复用策略。"""
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
    aux_ids: object
    param_by_name: dict
    param_name_by_id: dict
    labels_by_node: dict


def build_graph(device, cfg, arrays, input_ids, aux_ids, grad_dtype=np.float32):
    # CPU 参考路径强制固定为 FP32
    effective_grad_dtype = np.float32 if device == "cpu" else grad_dtype
    rt = reset_runtime(device)
    model = ImplementedCudaModel(cfg.V, cfg.C, cfg.L, cfg.dtype, device)
    for parameter in model.parameters():
        parameter.grad_dtype = effective_grad_dtype
    initialize_model(model, arrays)
    input_t = make_tensor(input_ids.shape, np.int32, device, False, input_ids, name="input_ids")
    aux_t = make_tensor(aux_ids.shape, np.int32, device, False, aux_ids, name="aux_ids")
    out = model(input_t, aux_t)
    rt.loss = out
    rt.build_topo(out)
    verify_training_topology(rt.topo, cfg)
    rt.bind_memory()
    if device == "cpu":
        isolate_cpu_reference_memory(rt)
    rt.is_compiled = True
    param_by_name = dict(iter_named_tensors(model))
    return GraphState(
        rt=rt,
        model=model,
        loss=out,
        input_ids=input_t,
        aux_ids=aux_t,
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
    """与真实训练循环一致地连续调用 Runtime.execute()，验证前反向与非累加模式梯度覆写。"""
    results = []
    for _ in range(repeats):
        state.rt.execute()
        state.rt.backend.synchronize()
        results.append({
            "output": copy_to_host(state.rt, state.loss),
            "parameter_grads": {
                name: copy_to_host(state.rt, tensor.grad)
                for name, tensor in state.param_by_name.items()
            },
        })
    return results


def compare_exact_graph(cpu, cuda, cuda_grad_dtype=np.float32):
    print("\n=== Exact Runtime.execute() training path (CUDA vs CPU FP32) ===")
    if len(cpu) != len(cuda):
        raise AssertionError("CPU/CUDA repeat count differs")

    out_atol, out_rtol = tolerances_for_grad_dtype(cuda_grad_dtype, "activation")
    for repeat, (cpu_result, cuda_result) in enumerate(zip(cpu, cuda), 1):
        print(f"  [training execution {repeat}]")
        check(
            f"repeat[{repeat}]/output",
            cuda_result["output"],
            cpu_result["output"],
            atol=out_atol,
            rtol=out_rtol,
        )
        for name in sorted(cpu_result["parameter_grads"]):
            check_parameter_gradient(
                f"repeat[{repeat}]/parameter.grad/{name}",
                cuda_result["parameter_grads"][name],
                cpu_result["parameter_grads"][name],
                cuda_grad_dtype,
            )


def run_staged_graph(state, upstream_grad, reference_inputs=None):
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
                rt.backend.copy_from_host(tensor, np.asarray(value, dtype=tensor.dtype))

        forward_inputs[label] = [copy_to_host(rt, tensor) for tensor in node.inputs]
        node.forward(node.inputs, node.outputs)
        rt.backend.end_pass()
        rt.backend.synchronize()

        forward[label] = [copy_to_host(rt, out) for out in node.outputs]
        rt.backend.begin_pass()
    rt.backend.end_pass()

    if rt.allocator.device == "cpu":
        zero_every_grad(state)

    # 注入非均匀随机上游梯度，确保任意行列/通道错位都不会因常数梯度相消而被掩盖
    rt.backend.copy_from_host(state.loss.grad, np.asarray(upstream_grad, dtype=np.float32))
    rt.backend.begin_pass()

    for node in reversed(rt.topo):
        label = state.labels_by_node[id(node)]
        output_grads = [
            None if out.grad is None else copy_to_host(rt, out.grad)
            for out in node.outputs
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
            "inputs": inputs,
        }
        rt.backend.begin_pass()
    rt.backend.end_pass()

    return {
        "forward": forward,
        "forward_inputs": forward_inputs,
        "backward": backward,
    }


def compare_staged_graph(cpu, cuda, cuda_grad_dtype=np.float32):
    print("\n=== Per-operator staged path (CUDA vs CPU FP32) ===")
    if list(cpu["forward"]) != list(cuda["forward"]):
        raise AssertionError("CPU/CUDA forward topology labels differ")

    act_atol, act_rtol = tolerances_for_grad_dtype(cuda_grad_dtype, "activation")
    act_grad_atol, act_grad_rtol = tolerances_for_grad_dtype(cuda_grad_dtype, "activation_grad")

    for label in cpu["forward"]:
        for index, (cuda_value, cpu_value) in enumerate(zip(cuda["forward"][label], cpu["forward"][label])):
            check(f"{label}/forward/output[{index}]", cuda_value, cpu_value, atol=act_atol, rtol=act_rtol)

    for label in reversed(list(cpu["forward"])):
        cpu_bwd = cpu["backward"][label]
        cuda_bwd = cuda["backward"][label]

        for index, (cuda_value, cpu_value) in enumerate(zip(cuda_bwd["output_grads"], cpu_bwd["output_grads"])):
            if cpu_value is not None:
                check(
                    f"{label}/backward/output[{index}].grad",
                    cuda_value,
                    cpu_value,
                    atol=act_grad_atol,
                    rtol=act_grad_rtol,
                )

        if len(cpu_bwd["inputs"]) != len(cuda_bwd["inputs"]):
            raise AssertionError(f"{label}: different differentiable input count")

        for cpu_item, cuda_item in zip(cpu_bwd["inputs"], cuda_bwd["inputs"]):
            if (cpu_item["index"], cpu_item["parameter"]) != (cuda_item["index"], cuda_item["parameter"]):
                raise AssertionError(f"{label}: input mapping differs")
            parameter_name = cpu_item["parameter"]
            if parameter_name is not None:
                check_parameter_gradient(
                    f"{label}/backward/{parameter_name}.grad",
                    cuda_item["grad"],
                    cpu_item["grad"],
                    cuda_grad_dtype,
                )
            else:
                check(
                    f"{label}/backward/input[{cpu_item['index']}].grad",
                    cuda_item["grad"],
                    cpu_item["grad"],
                    atol=act_grad_atol,
                    rtol=act_grad_rtol,
                )


def _run_single_op_graph(device, build_fn, upstream_np):
    rt = reset_runtime(device)
    out, named_inputs = build_fn(device)
    rt.loss = out
    rt.build_topo(out)
    rt.bind_memory()
    if device == "cpu":
        isolate_cpu_reference_memory(rt)
    rt.is_compiled = True

    rt.forward()
    rt.backend.synchronize()
    fwd_out = copy_to_host(rt, out)

    if device == "cpu":
        for t in named_inputs.values():
            if t.grad is not None:
                rt.backend.fill(t.grad, 0.0)

    rt.backend.copy_from_host(out.grad, np.asarray(upstream_np, dtype=np.float32))
    rt.backend.begin_pass()
    for node in reversed(rt.topo):
        node.backward(node.inputs, node.outputs)
    rt.backend.end_pass()
    rt.backend.synchronize()

    grads = {
        name: copy_to_host(rt, t.grad)
        for name, t in named_inputs.items()
        if t.requires_grad and t.grad is not None
    }
    return fwd_out, grads


def probe_storage_and_fill():
    print("\n=== Internal kernels: CudaContext copy & fill_scalar_kernel2 ===")
    rt_cpu = reset_runtime("cpu")
    rt_cuda = reset_runtime("cuda")

    # FP32 非零/零填充与读写
    s_cpu = rt_cpu.allocator.alloc((259,), np.float32)
    s_cuda = rt_cuda.allocator.alloc((259,), np.float32)
    rt_cpu.backend.fill_storage(s_cpu, -3.25)
    rt_cuda.backend.fill_storage(s_cuda, -3.25)
    rt_cuda.backend.synchronize()
    check("fill_scalar_kernel2/fp32_nonzero", read_storage_f32(rt_cuda, s_cuda), read_storage_f32(rt_cpu, s_cpu))

    rt_cpu.backend.fill_storage(s_cpu, 0.0)
    rt_cuda.backend.fill_storage(s_cuda, 0.0)
    rt_cuda.backend.synchronize()
    check("memset_zero/fp32_zero", read_storage_f32(rt_cuda, s_cuda), read_storage_f32(rt_cpu, s_cpu))

    # BF16 (uint16) 非零填充
    bf16_cuda = rt_cuda.allocator.alloc((259,), np.uint16)
    rt_cuda.backend.fill_storage(bf16_cuda, 1.375)
    rt_cuda.backend.synchronize()
    check(
        "fill_scalar_kernel2/bf16_nonzero",
        read_storage_f32(rt_cuda, bf16_cuda),
        np.full((259,), 1.375, dtype=np.float32),
        atol=1e-2,
        rtol=4e-3,
    )


def probe_gather_variants(seed, grad_dtype):
    print("\n=== Internal kernels: GatherOp (vec4 / scalar fallback / duplicate tokens) ===")
    rng = np.random.default_rng(seed + 101)
    # 同时覆盖 C % 4 == 0 (vec4=1) 与 C % 4 != 0 (vec4=0 及 BF16 尾列 valid < 4)
    for case_name, (B, T, V, C) in (
        ("vec4_aligned_C128", (3, 24, 128, 128)),
        ("scalar_tail_C66", (3, 19, 97, 66)),
    ):
        idx_np = rng.integers(0, V, (B, T), dtype=np.int32)
        idx_np[:, : max(1, T // 2)] = rng.integers(0, min(V, 6), (B, max(1, T // 2)), dtype=np.int32)
        w_np = (rng.standard_normal((V, C)) * 0.05).astype(np.float32)
        upstream_np = (rng.standard_normal((B, T, C)) * 0.1).astype(np.float32)

        def build_fn(dev, idx_np=idx_np, w_np=w_np):
            g_dt = np.float32 if dev == "cpu" else grad_dtype
            idx = make_tensor(idx_np.shape, np.int32, dev, False, idx_np)
            w = make_tensor(w_np.shape, np.float32, dev, True, w_np, grad_dtype=g_dt)
            out = GatherOp()(idx, w)
            return out, {"weight": w}

        cpu_out, cpu_grads = _run_single_op_graph("cpu", build_fn, upstream_np)
        cuda_out, cuda_grads = _run_single_op_graph("cuda", build_fn, upstream_np)
        check(f"Gather/{case_name}/forward", cuda_out, cpu_out)
        check_parameter_gradient(
            f"Gather/{case_name}/weight.grad",
            cuda_grads["weight"],
            cpu_grads["weight"],
            grad_dtype,
        )


def probe_add_variants(seed, grad_dtype):
    print("\n=== Internal kernels: AddOp (residual / pos / bias & vec4 / scalar tail) ===")
    rng = np.random.default_rng(seed + 202)

    # 1) 同形状残差相加：覆盖 vec4 与非 4 倍数标量路径，以及单侧 requires_grad=False (nullptr) 分支
    for case_name, shape, req_x, req_y in (
        ("residual_vec4_both_grad", (2, 16, 64), True, True),
        ("residual_scalar_455_both_grad", (1, 7, 65), True, True),
        ("residual_vec4_x_only", (2, 8, 64), True, False),
        ("residual_scalar_y_only", (1, 7, 65), False, True),
    ):
        x_np = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        y_np = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        upstream_np = (rng.standard_normal(shape) * 0.1).astype(np.float32)

        def build_res(dev, x_np=x_np, y_np=y_np, req_x=req_x, req_y=req_y):
            # 激活张量梯度恒为 FP32
            x = make_tensor(x_np.shape, np.float32, dev, req_x, x_np, grad_dtype=np.float32)
            y = make_tensor(y_np.shape, np.float32, dev, req_y, y_np, grad_dtype=np.float32)
            out = AddOp()(x, y)
            return out, {"x": x, "y": y}

        cpu_out, cpu_grads = _run_single_op_graph("cpu", build_res, upstream_np)
        cuda_out, cuda_grads = _run_single_op_graph("cuda", build_res, upstream_np)
        check(f"Add/{case_name}/forward", cuda_out, cpu_out)
        for k in cpu_grads:
            check(f"Add/{case_name}/{k}.grad", cuda_grads[k], cpu_grads[k])

    # 2) 位置编码广播相加：[B, T, C] + [T, C] 与 [B, H, T, T] + [1, 1, T, T]
    for case_name, x_shape, pos_shape in (
        ("pos_2d_C66", (3, 11, 66), (11, 66)),
        ("pos_4d_mask", (2, 4, 13, 13), (1, 1, 13, 13)),
    ):
        x_np = (rng.standard_normal(x_shape) * 0.1).astype(np.float32)
        pos_np = (rng.standard_normal(pos_shape) * 0.1).astype(np.float32)
        upstream_np = (rng.standard_normal(x_shape) * 0.1).astype(np.float32)

        def build_pos(dev, x_np=x_np, pos_np=pos_np):
            x = make_tensor(x_np.shape, np.float32, dev, True, x_np, grad_dtype=np.float32)
            pos = make_tensor(pos_np.shape, np.float32, dev, True, pos_np, grad_dtype=np.float32)
            out = AddOp()(x, pos)
            return out, {"x": x, "pos": pos}

        cpu_out, cpu_grads = _run_single_op_graph("cpu", build_pos, upstream_np)
        cuda_out, cuda_grads = _run_single_op_graph("cuda", build_pos, upstream_np)
        check(f"Add/{case_name}/forward", cuda_out, cpu_out)
        check(f"Add/{case_name}/x.grad", cuda_grads["x"], cpu_grads["x"])
        check(f"Add/{case_name}/pos.grad", cuda_grads["pos"], cpu_grads["pos"], atol=2e-6, rtol=2e-6)

    # 3) 偏置广播相加：覆盖 aligned4=True (C=128) 与 aligned4=False 尾块 (C=65)，同时测 FP32 / BF16 梯度
    for case_name, x_shape, b_shape in (
        ("bias_1d_aligned_C128", (3, 17, 128), (128,)),
        ("bias_1d_tail_C65", (3, 17, 65), (65,)),
        ("bias_3d_tail_C65", (2, 11, 65), (1, 1, 65)),
    ):
        x_np = (rng.standard_normal(x_shape) * 0.1).astype(np.float32)
        b_np = (rng.standard_normal(b_shape) * 0.1).astype(np.float32)
        upstream_np = (rng.standard_normal(x_shape) * 0.1).astype(np.float32)

        def build_bias(dev, x_np=x_np, b_np=b_np):
            g_dt = np.float32 if dev == "cpu" else grad_dtype
            x = make_tensor(x_np.shape, np.float32, dev, True, x_np, grad_dtype=np.float32)
            b = make_tensor(b_np.shape, np.float32, dev, True, b_np, grad_dtype=g_dt)
            out = AddOp()(x, b)
            return out, {"x": x, "bias": b}

        cpu_out, cpu_grads = _run_single_op_graph("cpu", build_bias, upstream_np)
        cuda_out, cuda_grads = _run_single_op_graph("cuda", build_bias, upstream_np)
        check(f"Add/{case_name}/forward", cuda_out, cpu_out)
        check(f"Add/{case_name}/x.grad", cuda_grads["x"], cpu_grads["x"])
        check_parameter_gradient(
            f"Add/{case_name}/bias.grad",
            cuda_grads["bias"],
            cpu_grads["bias"],
            grad_dtype,
        )


def _copy_training_batch(state, input_ids, aux_ids):
    state.rt.backend.copy_from_host(state.input_ids, input_ids)
    state.rt.backend.copy_from_host(state.aux_ids, aux_ids)


def _gradient_snapshot(state):
    return {
        name: copy_to_host(state.rt, tensor.grad)
        for name, tensor in state.param_by_name.items()
    }


def _gradient_storage_layout(state):
    return {
        name: (
            tensor.grad.storage.ptr.buffer_id,
            tensor.grad.storage.ptr.offset,
            tensor.grad.storage.nbytes,
        )
        for name, tensor in state.param_by_name.items()
    }


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
    return batches


def probe_gradient_accumulation(cfg, arrays, seed, grad_dtype=np.float32):
    print("\n=== No-extra-buffer parameter gradient accumulation (CUDA vs CPU FP32) ===")
    training_steps = 2
    for accumulation_steps in (1, 2, 4):
        print(f"  [accumulation_steps={accumulation_steps}]")
        batches = _make_accumulation_batches(cfg, accumulation_steps, training_steps, seed)

        initial = batches[0][1][0]
        # 参考真值始终由独立隔离的 CPU FP32 图逐微批计算并求和
        cpu_state = build_graph("cpu", cfg, arrays, *initial, grad_dtype=np.float32)
        cuda_state = build_graph("cuda", cfg, arrays, *initial, grad_dtype=grad_dtype)
        initial_layout = _gradient_storage_layout(cuda_state)

        grad_atol, grad_rtol = tolerances_for_grad_dtype(grad_dtype, "parameter_grad")
        if str(grad_dtype).lower() in ("bf16", "bfloat16"):
            grad_atol *= accumulation_steps
            grad_rtol *= accumulation_steps

        for step, (_, micro_batches) in enumerate(batches, 1):
            cpu_micro_grads = []
            for micro_batch in micro_batches:
                _copy_training_batch(cpu_state, *micro_batch)
                cpu_state.rt.backward(
                    cpu_state.loss,
                    grad_scale=1.0 / accumulation_steps,
                    accumulate_parameter_grads=False,
                )
                cpu_micro_grads.append(_gradient_snapshot(cpu_state))

            cpu_ref_grads = {
                name: np.sum(
                    np.stack([m[name] for m in cpu_micro_grads], axis=0),
                    axis=0,
                    dtype=np.float32,
                ).astype(np.float32)
                for name in cpu_state.param_by_name
            }

            for acc_step, micro_batch in enumerate(micro_batches):
                _copy_training_batch(cuda_state, *micro_batch)
                cuda_state.rt.backward(
                    cuda_state.loss,
                    grad_scale=1.0 / accumulation_steps,
                    accumulate_parameter_grads=(acc_step > 0),
                )
            cuda_state.rt.backend.synchronize()
            cuda_grads = _gradient_snapshot(cuda_state)

            for name in sorted(cpu_ref_grads):
                check(
                    f"grad_accum[{accumulation_steps}]/step[{step}]/gradient/{name}",
                    cuda_grads[name],
                    cpu_ref_grads[name],
                    atol=grad_atol,
                    rtol=grad_rtol,
                )

        if _gradient_storage_layout(cuda_state) != initial_layout:
            raise AssertionError("CUDA gradient accumulation reallocated parameter gradient storage")


def main():
    global PRINT_PASSES
    args, cfg = parse_args()
    PRINT_PASSES = not args.compact
    grad_dtype = grad_dtype_value(args.grad_dtype)

    print(f"grad_dtype={args.grad_dtype} (CPU oracle fixed to fp32)")
    print(f"project_root={PROJECT_ROOT}")
    print(f"runtime={runtime.__file__}")
    print(
        f"config: B={cfg.B}, T={cfg.T}, V={cfg.V}, C={cfg.C}, H={cfg.H}, "
        f"D={cfg.D}, L={cfg.L}, atol={ATOL:g}, rtol={RTOL:g}"
    )

    arrays = build_parameter_arrays(cfg, args.seed)
    input_ids, aux_ids = build_inputs(cfg, args.seed)

    if args.gradient_accumulation_only:
        probe_gradient_accumulation(cfg, arrays, args.seed, grad_dtype=grad_dtype)
        if FAILURES:
            print(f"\n{len(FAILURES)} gradient accumulation correctness check(s) failed:")
            for name, max_abs, max_rel in FAILURES:
                print(f"  {name}: max_abs={max_abs:.8e}, max_rel={max_rel:.8e}")
            raise AssertionError(f"{len(FAILURES)} gradient accumulation check(s) failed")
        print("\nAll CUDA gradient accumulation correctness checks passed.")
        return

    cuda_exact = run_exact_training_graph(
        build_graph("cuda", cfg, arrays, input_ids, aux_ids, grad_dtype=grad_dtype)
    )
    cpu_exact = run_exact_training_graph(
        build_graph("cpu", cfg, arrays, input_ids, aux_ids, grad_dtype=np.float32)
    )
    compare_exact_graph(cpu_exact, cuda_exact, cuda_grad_dtype=grad_dtype)

    rng = np.random.default_rng(args.seed + 99)
    upstream_grad = (rng.standard_normal((cfg.B, cfg.T, cfg.C)) * 0.1).astype(np.float32)
    cpu_trace = run_staged_graph(
        build_graph("cpu", cfg, arrays, input_ids, aux_ids, grad_dtype=np.float32),
        upstream_grad=upstream_grad,
    )
    cuda_trace = run_staged_graph(
        build_graph("cuda", cfg, arrays, input_ids, aux_ids, grad_dtype=grad_dtype),
        upstream_grad=upstream_grad,
        reference_inputs=cpu_trace["forward_inputs"],
    )
    compare_staged_graph(cpu_trace, cuda_trace, cuda_grad_dtype=grad_dtype)

    if not args.skip_internal_kernels:
        probe_storage_and_fill()
        probe_gather_variants(args.seed, grad_dtype)
        probe_add_variants(args.seed, grad_dtype)

    if not args.skip_gradient_accumulation:
        probe_gradient_accumulation(cfg, arrays, args.seed, grad_dtype=grad_dtype)

    if FAILURES:
        print(f"\n{len(FAILURES)} correctness check(s) failed:")
        for name, max_abs, max_rel in FAILURES:
            print(f"  {name}: max_abs={max_abs:.8e}, max_rel={max_rel:.8e}")
        raise AssertionError("CUDA kernel correctness validation against CPU failed")

    print("\nAll implemented CUDA kernel correctness checks against CPU passed.")


if __name__ == "__main__":
    main()