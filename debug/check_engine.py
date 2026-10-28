"""
PYTHONDONTWRITEBYTECODE=1 \
/Users/a58/Documents/code/nanochat/.venv/bin/python check_engine.py \
  --batch-size 2 \
  --n-layer 2 \
  --max-new-tokens 8 \
  --max-seq-len 64
"""

import argparse
import gc
import functools
import sys
import signal
import warnings
from pathlib import Path

import numpy as np
from collections import deque


PROJECT_ROOT = Path("/Users/a58/Documents/code/autograd")
sys.path.insert(0, str(PROJECT_ROOT))

from tensor import Tensor
from module import GPT2
from runtime import Runtime, ExecutionPlan
from engine import Engine


EOS_TOKEN = 1
BOS_TOKEN = 0
PYTHON_START = 120
PYTHON_END = 121
OUTPUT_START = 122
OUTPUT_END = 123
USER_START = 124
USER_END = 125
ASSISTANT_START = 126

K_MAX = 64

def named_tensors(root):
    result = {}
    visited = set()

    def visit(value, name):
        if isinstance(value, Tensor):
            if id(value) not in visited:
                visited.add(id(value))
                result[name] = value
            return

        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{name}.{index}")
            return

        if hasattr(value, "__dict__"):
            if id(value) in visited:
                return
            visited.add(id(value))

            for key, item in value.__dict__.items():
                child_name = (
                    f"{name}.{key}"
                    if name
                    else key
                )
                visit(item, child_name)

    visit(root, "")
    return result

class SyntheticTokenizer:
    def get_bos_token_id(self):
        return BOS_TOKEN

    def encode_special(self, text):
        mapping = {
            "<|assistant_end|>": EOS_TOKEN,
            "<|python_start|>": PYTHON_START,
            "<|python_end|>": PYTHON_END,
            "<|output_start|>": OUTPUT_START,
            "<|output_end|>": OUTPUT_END,
            "<|user_start|>": USER_START,
            "<|user_end|>": USER_END,
            "<|assistant_start|>": ASSISTANT_START,
        }

        if text not in mapping:
            raise KeyError(text)

        return mapping[text]

    def encode(self, text):
        text = str(text).strip()

        if not text:
            return []

        values = []

        for item in text.split():
            try:
                values.append(int(item))
            except ValueError:
                return []

        return values

    def decode(self, tokens):
        return " ".join(
            str(int(token))
            for token in tokens
        )

    def render_for_completion(self, conversation):
        # This is the same token structure produced by nanochat's
        # render_for_completion: render the conversation up to an empty
        # assistant turn, retaining all prompt/control tokens.
        if not isinstance(conversation, dict):
            raise TypeError("conversation must be a dict")
        messages = conversation.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(
                "conversation must contain user and assistant messages"
            )
        if messages[-1].get("role") != "assistant":
            raise ValueError(
                "last conversation message must be assistant"
            )
        prompt_messages = messages[:-1]
        if not prompt_messages or prompt_messages[-1].get("role") != "user":
            raise ValueError("completion prompt must end with user message")
        ids = [BOS_TOKEN]
        for message in prompt_messages:
            if message.get("role") != "user":
                raise ValueError("synthetic completion supports user prompt only")
            ids.append(USER_START)
            ids.extend(self.encode(message.get("content", "")))
            ids.append(USER_END)
        ids.append(ASSISTANT_START)
        return ids

def fail(message):
    raise AssertionError(message)

def check_equal(name, actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    if actual.shape != expected.shape:
        fail(
            f"{name}: shape mismatch: "
            f"actual={actual.shape}, expected={expected.shape}"
        )

    if not np.array_equal(actual, expected):
        indices = np.argwhere(actual != expected)
        index = tuple(indices[0]) if len(indices) else ()
        fail(
            f"{name}: mismatch at {index}: "
            f"actual={actual[index]}, expected={expected[index]}"
        )


def check_close(name, actual, expected, atol=1e-6, rtol=1e-6):
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    if actual.shape != expected.shape:
        fail(
            f"{name}: shape mismatch: "
            f"actual={actual.shape}, expected={expected.shape}"
        )

    try:
        np.testing.assert_allclose(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as exc:
        raise AssertionError(f"{name}: {exc}") from exc


def xorshift32(value):
    value = int(value) & 0xFFFFFFFF
    value ^= (value << 13) & 0xFFFFFFFF
    value ^= value >> 17
    value ^= (value << 5) & 0xFFFFFFFF
    return np.uint32(value & 0xFFFFFFFF)


def numpy_topk(logits, top_k, capacity=K_MAX):
    """
    NumPy reference for the intended top-k semantics.

    Values:
      descending value;
      ascending token index for ties.
    """
    logits = np.asarray(logits, dtype=np.float32)

    if logits.ndim != 2:
        fail(f"expected logits [B,V], got {logits.shape}")

    batch, vocab = logits.shape
    top_k = int(top_k)

    if top_k <= 0:
        fail(
            "This checker uses top_k=50. "
            "Full-vocabulary top_k<=0 requires a separate capacity."
        )

    count = min(top_k, vocab)

    if count > capacity:
        fail(
            f"top_k={count} exceeds RolloutTopk capacity={capacity}"
        )

    values = np.full(
        (batch, capacity),
        -np.inf,
        dtype=np.float32,
    )
    indices = np.zeros(
        (batch, capacity),
        dtype=np.int32,
    )

    token_ids = np.arange(vocab, dtype=np.int32)

    for row in range(batch):
        order = np.lexsort(
            (token_ids, -logits[row])
        )[:count]

        values[row, :count] = logits[row, order]
        indices[row, :count] = order

    return values, indices, count


def numpy_sample_greedy(values, indices, top_k):
    """
    CPU correctness mode.

    temperature=0 is used so that CPU and Metal token results are
    deterministic without relying on different RNG implementations.
    """
    count = min(int(top_k), values.shape[-1])
    index = int(np.argmax(values[:count]))
    return int(indices[index])


def make_reference_state(batch, max_tokens, seed):
    rng = np.zeros(
        (batch, 4),
        dtype=np.uint32,
    )
    rng[:, 0] = np.uint32(seed & 0xFFFFFFFF)

    return {
        "rng_state": rng,
        "finished": np.zeros(
            (batch,),
            dtype=np.int32,
        ),
        "next_tokens": np.zeros(
            (batch, 1),
            dtype=np.int32,
        ),
        "generated_ids": np.zeros(
            (batch, max_tokens),
            dtype=np.int32,
        ),
        "lengths": np.zeros(
            (batch,),
            dtype=np.int32,
        ),
        "position": np.zeros(
            (1,),
            dtype=np.int32,
        ),
        "step": np.zeros(
            (1,),
            dtype=np.int32,
        ),
    }

def calculator_reference(expr, max_time=3):
    expr = str(expr).replace(",", "")

    def timeout_handler(signum, frame):
        raise TimeoutError(
            f"calculator expression timed out: {expr}"
        )

    previous_handler = signal.signal(
        signal.SIGALRM,
        timeout_handler,
    )
    signal.alarm(int(max_time))

    try:
        if not expr:
            return None

        if all(
            ch in "0123456789*+-/.() "
            for ch in expr
        ):
            if "**" in expr:
                return None

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    SyntaxWarning,
                )
                return eval(
                    expr,
                    {"__builtins__": {}},
                    {},
                )

        allowed_chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789'\"()._ "
        )

        if not all(ch in allowed_chars for ch in expr):
            return None

        dangerous = (
            "__",
            "import",
            "exec",
            "eval",
            "compile",
            "open",
            "file",
            "input",
            "raw_input",
            "globals",
            "locals",
            "vars",
            "dir",
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
        )

        expression_lower = expr.lower()

        if any(
            item in expression_lower
            for item in dangerous
        ):
            return None

        if ".count(" not in expr:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                SyntaxWarning,
            )
            return eval(
                expr,
                {"__builtins__": {}},
                {},
            )
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(
            signal.SIGALRM,
            previous_handler,
        )

def numpy_rollout_reference(
    cpu_logits_trace,
    prompt_tokens,
    num_samples,
    max_tokens,
    top_k,
    eos_token,
    bos_token,
    seed,
):
    state = make_reference_state(
        num_samples, max_tokens, seed
    )

    prompt_len = len(prompt_tokens)
    if prompt_len == 0:
        raise ValueError("prompt_tokens must not be empty")

    state["position"][0] = prompt_len - 1

    results = [
        list(prompt_tokens) for _ in range(num_samples)
    ]
    masks = [
        [0] * prompt_len for _ in range(num_samples)
    ]
    trace = []

    for iteration, raw_logits in enumerate(cpu_logits_trace):
        mode = "prefill" if iteration == 0 else "decode"

        logits = select_sampling_logits(
            raw_logits, mode, prompt_len
        )
        expected_rows = 1 if mode == "prefill" else num_samples

        if logits.shape[0] != expected_rows:
            fail(
                f"{mode}: invalid CPU logits shape "
                f"{logits.shape}"
            )

        values, indices, count = numpy_topk_reference(
            logits, top_k
        )
        current_step = int(state["step"][0])

        if current_step >= max_tokens:
            fail("CPU reference generated_ids capacity exhausted")

        previous_next_tokens = state["next_tokens"].copy()

        for row in range(num_samples):
            if state["finished"][row]:
                state["next_tokens"][row, 0] = eos_token
                continue

            source_row = 0 if mode == "prefill" else row
            token = numpy_sample_greedy(
                values[source_row],
                indices[source_row],
                count,
            )
            state["next_tokens"][row, 0] = token

            if token == eos_token or token == bos_token:
                state["finished"][row] = 1
            else:
                state["generated_ids"][row, current_step] = token
                state["lengths"][row] += 1
                results[row].append(token)
                masks[row].append(1)

        # prefill 生成首 token 后，第一次 decode 应写入 prompt_len。
        state["position"][0] = prompt_len + iteration
        state["step"][0] = iteration + 1
        done = int(np.all(state["finished"] != 0))

        entry = {
            "mode": mode,
            "logits": np.array(logits, copy=True),
            "topk_values": values.copy(),
            "topk_indices": indices.copy(),
            "previous_next_tokens": previous_next_tokens,
            "finished_all": np.asarray(
                [done], dtype=np.int32
            ),
        }
        entry.update({
            key: value.copy()
            for key, value in state.items()
        })
        trace.append(entry)

        if done:
            break

    return {
        "results": results,
        "masks": masks,
        "trace": trace,
    }

def reward_reference(sequence, prompt_len):
    """
    Constructed reward.

    The reward is derived from the generated result, not injected into the
    generation path. This avoids dependence on GSM8K or nanochat tasks.
    """
    response = sequence[prompt_len:]

    if not response:
        return np.float32(0.0)

    non_special = [
        token
        for token in response
        if token not in (EOS_TOKEN, BOS_TOKEN)
    ]

    if not non_special:
        return np.float32(0.0)

    checksum = sum(
        (index + 1) * int(token)
        for index, token in enumerate(non_special)
    )

    return np.float32(checksum % 17) / np.float32(17.0)


def build_rl_batch(
    sequences,
    masks,
    rewards,
    assistant_end,
):
    return build_nanochat_batch_reference(
        sequences=sequences,
        masks=masks,
        rewards=rewards,
        assistant_end=assistant_end,
    )

def read_tensor(backend, tensor):
    backend.synchronize()

    if hasattr(backend, "numpy_view"):
        return np.array(
            backend.numpy_view(tensor),
            copy=True,
        )

    return np.array(
        backend.copy_to_host(tensor),
        copy=True,
    )

def iter_model_tensors(model):
    yield model.tok_emb.weight

    for block in model.blocks:
        yield block.attn.q_proj.weight
        yield block.attn.q_proj.bias
        yield block.attn.k_proj.weight
        yield block.attn.k_proj.bias
        yield block.attn.v_proj.weight
        yield block.attn.v_proj.bias
        yield block.attn.proj.weight
        yield block.attn.proj.bias
        yield block.mlp.fc1.weight
        yield block.mlp.fc2.weight

    yield model.lm_head.weight


def snapshot_parameters(model, backend):
    snapshot = []

    for tensor in iter_model_tensors(model):
        snapshot.append(
            (
                tensor,
                np.array(
                    backend.copy_to_host(tensor),
                    copy=True,
                ),
            )
        )

    return snapshot

def copy_parameters(
    source_model,
    target_model,
    source_backend,
    target_backend,
):
    source_tensors = list(iter_model_tensors(source_model))
    target_tensors = list(iter_model_tensors(target_model))

    if len(source_tensors) != len(target_tensors):
        fail(
            "CPU/Metal parameter count mismatch"
        )

    for source, target in zip(
        source_tensors,
        target_tensors,
    ):
        value = np.array(
            source_backend.copy_to_host(source),
            copy=True,
        )

        target_backend.copy_from_host(
            target,
            value,
        )

def build_cpu_logits_plan(
    model,
    input_ids,
    kv_cache,
    position,
    mode,
):
    x = model.tok_emb(input_ids)

    for block in model.blocks:
        x = block(
            x,
            model.rope_cos,
            model.rope_sin,
            kv_cache=kv_cache,
            position=position,
            mode=mode,
        )

    logits = model.lm_head(x)

    runtime = Runtime(
        device="cpu",
        forward_only=True,
        owns_params=False,
    )
    runtime.loss = logits
    runtime.build_topo(logits)
    runtime.plan_memory()
    runtime.bind_memory()
    runtime.is_compiled = True

    return ExecutionPlan(
        runtime=runtime,
        input_ids=input_ids,
        target_ids=None,
        output=logits,
        seq_len=input_ids.shape[1],
        mode=mode,
    )

def run_cpu_prefill_logits(
    cpu_model,
    cpu_runtime,
    cpu_kv_cache,
    prompt_tokens,
    bucket_len,
):
    input_ids = Tensor(
        shape=(1, bucket_len),
        dtype=np.int32,
        device="cpu",
        requires_grad=False,
        init_val=None,
        name="cpu_prefill_input",
    )

    plan = build_cpu_logits_plan(
        model=cpu_model,
        input_ids=input_ids,
        kv_cache=cpu_kv_cache,
        position=cpu_kv_cache.position,
        mode="prefill",
    )

    padded = np.full(
        (1, bucket_len),
        BOS_TOKEN,
        dtype=np.int32,
    )
    padded[0, :len(prompt_tokens)] = prompt_tokens

    result = plan.forward(
        feeds={
            plan.input_ids: padded,
            cpu_kv_cache.position: np.asarray(
                [0],
                dtype=np.int32,
            ),
        },
        return_value=True,
    )

    return np.array(result, copy=True)

def select_sampling_logits(raw_logits, mode, prompt_len):
    raw_logits = np.asarray(raw_logits)

    if mode == "prefill":
        if raw_logits.ndim != 3:
            raise AssertionError(
                f"prefill logits must be [1,T,V], "
                f"got {raw_logits.shape}"
            )

        if raw_logits.shape[0] != 1:
            raise AssertionError(
                f"prefill batch must be 1, "
                f"got {raw_logits.shape}"
            )

        if raw_logits.shape[1] != 1:
        	raise AssertionError(f"prefill logits must have T=1, got {raw_logits.shape}")
        return raw_logits[:, 0, :]

    if mode == "decode":
        if raw_logits.ndim == 3:
            if raw_logits.shape[1] != 1:
                raise AssertionError(
                    f"decode logits must have T=1, "
                    f"got {raw_logits.shape}"
                )
            return raw_logits[:, 0, :]

        if raw_logits.ndim == 2:
            return raw_logits

        raise AssertionError(
            f"invalid decode logits shape={raw_logits.shape}"
        )

    raise AssertionError(f"unknown mode={mode}")

def numpy_topk_reference(logits, top_k):
    logits = np.asarray(logits, dtype=np.float32)

    if logits.ndim != 2:
        raise AssertionError(
            f"expected [B,V], got {logits.shape}"
        )

    batch, vocab = logits.shape
    k = min(int(top_k), vocab)

    values = np.full(
        (batch, K_MAX),
        -np.inf,
        dtype=np.float32,
    )
    indices = np.zeros(
        (batch, K_MAX),
        dtype=np.int32,
    )

    token_ids = np.arange(vocab, dtype=np.int32)

    for row in range(batch):
        order = np.lexsort(
            (token_ids, -logits[row])
        )[:k]

        values[row, :k] = logits[row, order]
        indices[row, :k] = order

    return values, indices, k

def numpy_rl_objective(
    logits,
    targets,
    advantages,
    num_passes,
    examples_per_rank,
):
    return nanochat_objective_reference(
        logits=logits,
        targets=targets,
        advantages=advantages,
        num_passes=num_passes,
        examples_per_rank=examples_per_rank,
    )

def validate_nanochat_rl_batch(
    sequences,
    masks,
    rewards,
    assistant_end,
):
    """
    验证 generate_batch_new() 输出经过 RL shift 后，
    与 nanochat chat_rl.py 的 inputs/targets/mask 语义一致。
    """
    if len(sequences) == 0:
        raise AssertionError(
            "RL batch must not be empty"
        )

    if len(sequences) != len(masks):
        raise AssertionError(
            "sequence/mask batch size mismatch"
        )

    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    if rewards.shape != (len(sequences),):
        raise AssertionError(
            "reward shape mismatch"
        )

    batch = build_rl_batch(
        sequences=sequences,
        masks=masks,
        rewards=rewards,
        assistant_end=assistant_end,
    )

    ids = batch["ids"]
    mask_ids = batch["mask_ids"]
    inputs = batch["inputs"]
    targets = batch["targets"]

    if ids.ndim != 2:
        raise AssertionError(
            f"ids must be rank 2, got {ids.shape}"
        )

    if mask_ids.shape != ids.shape:
        raise AssertionError(
            "mask_ids and ids shape mismatch"
        )

    if inputs.shape[0] != ids.shape[0]:
        raise AssertionError(
            "inputs batch mismatch"
        )

    if targets.shape != inputs.shape:
        raise AssertionError(
            "targets and inputs shape mismatch"
        )

    shifted_masks = mask_ids[:, 1:]
    shifted_targets = targets

    # mask=0 的位置必须被 ignore。
    if np.any(
        shifted_targets[shifted_masks == 0] != -1
    ):
        raise AssertionError(
            "masked target contains a trainable token"
        )

    # mask=1 的位置必须是合法 token。
    if np.any(
        shifted_targets[shifted_masks != 0] < 0
    ):
        raise AssertionError(
            "unmasked target contains ignore_index"
        )

    # inputs/targets 必须保持自回归一位偏移。
    if not np.array_equal(
        inputs,
        ids[:, :-1],
    ):
        raise AssertionError(
            "inputs are not ids[:, :-1]"
        )

    raw_targets = ids[:, 1:].copy()
    raw_targets[shifted_masks == 0] = -1

    if not np.array_equal(
        targets,
        raw_targets,
    ):
        raise AssertionError(
            "targets do not follow nanochat mask semantics"
        )

    expected_advantages = (
        rewards - rewards.mean()
    )

    if not np.allclose(
        batch["advantages"],
        expected_advantages,
        atol=0.0,
        rtol=0.0,
    ):
        raise AssertionError(
            "advantages are not reward - mean(reward)"
        )

    return batch

def read_parameter_gradients(
    backend,
    parameters,
):
    backend.synchronize()

    return [
        np.array(
            backend.copy_to_host(parameter.grad),
            copy=True,
        )
        for parameter in parameters
    ]


def zero_parameter_gradients(
    backend,
    parameters,
):
    for parameter in parameters:
        if parameter.grad is not None:
            backend.fill(
                parameter.grad,
                0.0,
            )


def validate_metal_gradient_accumulation(
    train_plan,
    backend,
    parameters,
    microbatches,
    batch_size,
    sequence_length,
    expected_grad_scales,
):
    """
    验证：

      accumulated_gradient
      ==
      sum(independent_microbatch_gradient)

    microbatches:
        [(inputs, targets), ...]

    expected_grad_scales:
        每个 microbatch 对应的 grad_scale。
    """
    if len(microbatches) == 0:
        raise AssertionError(
            "microbatches must not be empty"
        )

    if len(microbatches) != len(
        expected_grad_scales
    ):
        raise AssertionError(
            "microbatches/grad_scales mismatch"
        )

    if sequence_length <= 0:
        raise AssertionError(
            "sequence_length must be positive"
        )

    # 先分别计算每个 microbatch 的 Metal 梯度。
    independent = []

    for (inputs, targets), grad_scale in zip(
        microbatches,
        expected_grad_scales,
    ):
        if inputs.shape != (
            batch_size,
            sequence_length,
        ):
            raise AssertionError(
                f"invalid input shape: {inputs.shape}"
            )

        if targets.shape != inputs.shape:
            raise AssertionError(
                "input/target shape mismatch"
            )

        zero_parameter_gradients(
            backend,
            parameters,
        )

        train_plan.set_inputs(inputs)
        train_plan.set_targets(targets)

        train_plan.backward(
            grad_scale=float(grad_scale),
            accumulate_parameter_grads=False,
        )

        independent.append(
            read_parameter_gradients(
                backend,
                parameters,
            )
        )

    expected = []

    for parameter_index in range(
        len(parameters)
    ):
        value = np.zeros_like(
            independent[0][parameter_index],
            dtype=np.float32,
        )

        for contribution in independent:
            value += contribution[
                parameter_index
            ]

        expected.append(value)

    # 再按生产路径执行一次跨 microbatch 累积。
    zero_parameter_gradients(
        backend,
        parameters,
    )

    for index, (
        inputs,
        targets,
    ) in enumerate(microbatches):
        train_plan.set_inputs(inputs)
        train_plan.set_targets(targets)

        train_plan.backward(
            grad_scale=float(
                expected_grad_scales[index]
            ),
            accumulate_parameter_grads=(
                index != 0
            ),
        )

    actual = read_parameter_gradients(
        backend,
        parameters,
    )

    for index, (
        actual_gradient,
        expected_gradient,
    ) in enumerate(zip(actual, expected)):
        np.testing.assert_allclose(
            actual_gradient,
            expected_gradient,
            atol=1e-6,
            rtol=1e-6,
            err_msg=(
                "Metal accumulated gradient mismatch "
                f"for parameter {index}"
            ),
        )

    return {
        "independent": independent,
        "expected": expected,
        "actual": actual,
    }

def build_nanochat_microbatch_reference(
    targets,
    advantages,
    num_passes,
    examples_per_rank,
    batch_size,
):
    targets = np.asarray(
        targets,
        dtype=np.int32,
    )

    advantages = np.asarray(
        advantages,
        dtype=np.float32,
    )

    if targets.ndim != 2:
        raise ValueError(
            "targets must have shape [B,T]"
        )

    if targets.shape[0] != batch_size:
        raise ValueError(
            "targets batch size mismatch"
        )

    if advantages.shape != (batch_size,):
        raise ValueError(
            "advantages shape mismatch"
        )

    valid_tokens = int(
        np.count_nonzero(targets >= 0)
    )

    denominator = float(
        max(valid_tokens, 1)
        * int(num_passes)
        * int(examples_per_rank)
    )

    sequence_length = targets.shape[1]

    references = []

    for advantage in sorted(
        set(
            float(value)
            for value in advantages
        )
    ):
        row_mask = (
            np.abs(
                advantages - advantage
            )
            <= 1e-12
        )

        grouped_targets = targets.copy()

        grouped_targets[
            ~row_mask,
            :,
        ] = -1

        grad_scale = (
            float(batch_size)
            * float(sequence_length)
            * float(advantage)
            / denominator
        )

        references.append(
            {
                "advantage": float(advantage),
                "targets": grouped_targets,
                "valid_tokens": valid_tokens,
                "grouped_valid_tokens": int(
                    np.count_nonzero(
                        grouped_targets >= 0
                    )
                ),
                "denominator": denominator,
                "grad_scale": float(grad_scale),
            }
        )

    return references

def capture_logits_wrapper(
    engine,
    prompt_len,
    captured_logits,
    captured_inputs,
    captured_topk,
):
    prefill_plan = engine._choose(prompt_len)
    if prefill_plan is None:
        fail(f"no prefill plan for prompt_len={prompt_len}")

    pending = []

    for mode, plan in (
        ("prefill", prefill_plan),
        ("decode", engine.decode_plan),
    ):
        nodes = [
            node
            for node in plan.runtime.topo
            if node.op.__class__.__name__ == "RolloutTopkOp"
        ]
        if len(nodes) != 1:
            fail(
                f"{mode}: expected one RolloutTopkOp, "
                f"got {len(nodes)}"
            )

        node = nodes[0]
        original_forward = node.forward

        def wrapped_forward(
            inputs,
            outputs,
            _mode=mode,
            _plan=plan,
            _node=node,
            _original=original_forward,
        ):
            backend = _plan.runtime.backend

            # 仅验证路径：先完成前面的 GPU 操作，
            # 再读取 TopK 实际使用的 logits 和位置。
            backend.end_pass()
            backend.synchronize()

            record = {
                "mode": _mode,
                "plan": _plan,
                "node": _node,
                "logits": read_tensor(backend, inputs[0]),
                "position_before": read_tensor(
                    backend, inputs[2]
                ),
                "input_ids_before": read_tensor(
                    backend, _plan.input_ids
                ),
            }

            backend.begin_pass()
            _original(inputs, outputs)
            backend.end_pass()
            backend.synchronize()

            record["topk_values"] = read_tensor(
                backend, outputs[0]
            )
            record["topk_indices"] = read_tensor(
                backend, outputs[1]
            )
            pending.append(record)

            # 恢复 pass，Runtime 随后正常执行 Sample 并结束 pass。
            backend.begin_pass()

        node.forward = wrapped_forward

    return pending

def capture_engine_state(
    engine,
    trace,
    mode,
    finished_all,
):
    backend = engine.decode_plan.runtime.backend
    state = engine.rollout_state

    trace.append(
        {
            "mode": mode,
            "next_tokens": read_tensor(
                backend,
                state.next_tokens,
            ),
            "generated_ids": read_tensor(
                backend,
                state.generated_ids,
            ),
            "lengths": read_tensor(
                backend,
                state.lengths,
            ),
            "finished": read_tensor(
                backend,
                state.finished,
            ),
            "rng_state": read_tensor(
                backend,
                state.rng_state,
            ),
            "position": read_tensor(
                backend,
                state.position,
            ),
            "step": read_tensor(
                backend,
                state.step,
            ),
            "finished_all": np.asarray(
                [int(bool(finished_all))],
                dtype=np.int32,
            ),
        }
    )

def compare_metal_topk_against_numpy(
    captured_logits,
    captured_topk,
    prompt_len,
    top_k,
):
    if not captured_logits:
        fail("no captured Metal logits")

    if not captured_topk:
        fail("no captured Metal topk output")

    logits_item = captured_logits[0]
    topk_item = captured_topk[0]

    if logits_item["mode"] != "prefill":
        fail(
            f"first logits mode must be prefill, "
            f"got {logits_item['mode']}"
        )

    raw_logits = np.asarray(
        logits_item["logits"],
        dtype=np.float32,
    )

    if raw_logits.ndim == 3:
        if raw_logits.shape[0] != 1:
            fail(
                f"prefill logits batch must be 1, "
                f"got {raw_logits.shape}"
            )
        cpu_selected = raw_logits[:, 0, :]
    elif raw_logits.ndim == 2:
        cpu_selected = raw_logits[:1]
    else:
        fail(
            f"invalid prefill logits shape {raw_logits.shape}"
        )

    expected_values, expected_indices, count = (
        numpy_topk_reference(
            cpu_selected,
            top_k,
        )
    )

    actual_values = np.asarray(
        topk_item["values"],
        dtype=np.float32,
    )
    actual_indices = np.asarray(
        topk_item["indices"],
        dtype=np.int32,
    )

    print(
        "[PREFILL-TOPK-DIAGNOSTIC]",
        f"raw_logits_shape={raw_logits.shape}",
        f"selected_shape={cpu_selected.shape}",
        f"position={logits_item['position'].reshape(-1).tolist()}",
        f"mode_id={logits_item['mode_id']}",
        f"expected_indices={expected_indices[0, :10].tolist()}",
        f"actual_indices={actual_indices[0, :10].tolist()}",
        flush=True,
    )

    check_close(
        "METAL_PREFILL_TOPK_VALUES",
        actual_values[:, :count],
        expected_values[:, :count],
    )

    check_equal(
        "METAL_PREFILL_TOPK_INDICES",
        actual_indices[:, :count],
        expected_indices[:, :count],
    )

def wrap_engine_rollout_methods(
    engine,
    state_trace,
    captured_logits,
    captured_inputs,
    captured_topk,
    pending,
):
    original_prefill = engine._prefill_rollout
    original_decode = engine._decode_rollout

    def materialize_step(finished_all, mode):
        index = len(state_trace)

        if index >= len(pending):
            fail(f"missing capture for step={index}")

        record = pending[index]
        if record["mode"] != mode:
            fail(f"capture mode mismatch at step={index}")

        backend = record["plan"].runtime.backend
        position_before = record["position_before"].copy()

        captured_logits.append({
            "mode": mode,
            "logits": record["logits"].copy(),
            "position": position_before,
            "mode_id": int(record["node"].op.mode_id),
        })

        captured_topk.append({
            "mode": mode,
            "values": record["topk_values"].copy(),
            "indices": record["topk_indices"].copy(),
            "position": position_before.copy(),
        })

        if mode == "decode":
            captured_inputs.append({
                "mode": mode,
                "input_ids": record["input_ids_before"].copy(),
                "position": position_before.copy(),
            })

        # 这里读取 Sample 执行后的状态。
        state = engine.rollout_state
        entry = {"mode": mode}

        for key in (
            "next_tokens",
            "generated_ids",
            "lengths",
            "finished",
            "rng_state",
            "position",
            "step",
        ):
            entry[key] = read_tensor(
                backend, getattr(state, key)
            )

        entry["finished_all"] = read_tensor(
            backend, record["plan"].output
        )

        check_equal(
            f"FINISHED_RETURN_MISMATCH step={index}",
            np.asarray(
                [int(bool(finished_all))], dtype=np.int32
            ),
            entry["finished_all"],
        )

        state_trace.append(entry)

    def wrapped_prefill(prompt_ids):
        result = original_prefill(prompt_ids)
        materialize_step(result, "prefill")
        return result

    def wrapped_decode():
        result = original_decode()
        materialize_step(result, "decode")
        return result

    engine._prefill_rollout = wrapped_prefill
    engine._decode_rollout = wrapped_decode

def compare_engine_trace(
    metal_state_trace,
    metal_logits_trace,
    metal_topk_trace,
    metal_input_trace,
    reference,
    prompt_len,
    top_k,
):
    ref_trace = reference["trace"]

    for name, actual in (
        ("STATE", metal_state_trace),
        ("LOGITS", metal_logits_trace),
        ("TOPK", metal_topk_trace),
    ):
        if len(actual) != len(ref_trace):
            fail(
                f"{name}_TRACE_LENGTH_MISMATCH: "
                f"metal={len(actual)}, cpu={len(ref_trace)}"
            )

    for step, expected in enumerate(ref_trace):
        mode = expected["mode"]
        logits_item = metal_logits_trace[step]
        topk_item = metal_topk_trace[step]
        state = metal_state_trace[step]

        if any(
            item["mode"] != mode
            for item in (logits_item, topk_item, state)
        ):
            fail(f"MODE_MISMATCH step={step}")

        check_close(
            f"MODEL_LOGITS_MISMATCH step={step}",
            select_sampling_logits(
                logits_item["logits"], mode, prompt_len
            ),
            expected["logits"],
            atol=1e-5,
            rtol=1e-5,
        )

        count = min(
            int(top_k),
            expected["topk_indices"].shape[-1],
        )

        check_close(
            f"TOPK_VALUES_MISMATCH step={step}",
            topk_item["values"][:, :count],
            expected["topk_values"][:, :count],
            atol=1e-5,
            rtol=1e-5,
        )
        check_equal(
            f"TOPK_INDICES_MISMATCH step={step}",
            topk_item["indices"][:, :count],
            expected["topk_indices"][:, :count],
        )

        expected_before = np.asarray(
            [prompt_len - 1 + step], dtype=np.int32
        )
        check_equal(
            f"INPUT_POSITION_MISMATCH step={step}",
            logits_item["position"],
            expected_before,
        )

        for key in (
            "next_tokens",
            "lengths",
            "finished",
            "rng_state",
            "position",
            "step",
            "finished_all",
        ):
            check_equal(
                f"{key.upper()}_MISMATCH step={step}",
                state[key],
                expected[key],
            )

        actual_ids = state["generated_ids"]
        expected_ids = expected["generated_ids"]
        capacity = expected_ids.shape[1]

        if (
            actual_ids.ndim != 2
            or actual_ids.shape[0] != expected_ids.shape[0]
            or actual_ids.shape[1] < capacity
        ):
            fail(
                f"GENERATED_IDS_SHAPE_MISMATCH step={step}: "
                f"{actual_ids.shape}"
            )

        check_equal(
            f"GENERATED_IDS_MISMATCH step={step}",
            actual_ids[:, :capacity],
            expected_ids,
        )

        # 缓冲区可以更大，但生成上限之外不能被写入。
        check_equal(
            f"GENERATED_IDS_TAIL_MISMATCH step={step}",
            actual_ids[:, capacity:],
            np.zeros_like(actual_ids[:, capacity:]),
        )

    decode_inputs = [
        item for item in metal_input_trace
        if item["mode"] == "decode"
    ]
    decode_refs = [
        item for item in ref_trace
        if item["mode"] == "decode"
    ]

    if len(decode_inputs) != len(decode_refs):
        fail("DECODE_INPUT_TRACE_LENGTH_MISMATCH")

    for index, (actual, expected) in enumerate(
        zip(decode_inputs, decode_refs)
    ):
        check_equal(
            f"DECODE_INPUT_MISMATCH step={index}",
            actual["input_ids"],
            expected["previous_next_tokens"],
        )
        check_equal(
            f"DECODE_POSITION_MISMATCH step={index}",
            actual["position"],
            np.asarray(
                [prompt_len + index], dtype=np.int32
            ),
        )

def compare_cpu_metal_logits(
    cpu_logits_trace,
    metal_logits_trace,
    prompt_len,
):
    if len(cpu_logits_trace) != len(metal_logits_trace):
        fail(
            "MODEL_LOGITS_STEP_COUNT_MISMATCH: "
            f"cpu={len(cpu_logits_trace)} "
            f"metal={len(metal_logits_trace)}"
        )

    for step, (cpu_raw, metal_item) in enumerate(
        zip(cpu_logits_trace, metal_logits_trace)
    ):
        mode = "prefill" if step == 0 else "decode"

        cpu_selected = select_sampling_logits(
            cpu_raw,
            mode=mode,
            prompt_len=prompt_len,
        )

        metal_selected = select_sampling_logits(
            metal_item["logits"],
            mode=mode,
            prompt_len=prompt_len,
        )

        check_close(
            f"MODEL_LOGITS_MISMATCH step={step}",
            metal_selected,
            cpu_selected,
            atol=1e-5,
            rtol=1e-5,
        )

        if mode == "prefill":
            expected_position = prompt_len - 1
            actual_position = int(
                np.asarray(
                    metal_item["position"]
                ).reshape(-1)[0]
            )

            check_equal(
                "PREFILL_POSITION_MISMATCH",
                np.asarray(
                    [actual_position],
                    dtype=np.int32,
                ),
                np.asarray(
                    [expected_position],
                    dtype=np.int32,
                ),
            )

def run_cpu_generation_logits(
    cpu_model,
    cpu_runtime,
    prompt_tokens,
    num_samples,
    max_seq_len,
    max_tokens,
    top_k,
    temperature,
    seed,
):
    """
    使用当前项目的 CPU 模型生成 logits，
    使用独立 NumPy TopK/贪心采样驱动后续 decode。

    返回：
        prefill: [1, T, V]
        decode:  [B, 1, V]

    当前仅支持 temperature=0.0；此时 seed 不参与采样。
    """
    from engine import KVCache

    if temperature != 0.0:
        raise ValueError(
            "run_cpu_generation_logits currently requires "
            "temperature=0.0 for deterministic CPU/Metal comparison"
        )

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if max_tokens <= 0:
        return []

    prompt_tokens = np.asarray(
        prompt_tokens, dtype=np.int32
    ).reshape(-1)
    prompt_len = int(prompt_tokens.shape[0])

    if prompt_len <= 0:
        raise ValueError("prompt_tokens must not be empty")

    if prompt_len + max_tokens > max_seq_len:
        raise ValueError(
            "prompt_len + max_tokens exceeds max_seq_len: "
            f"{prompt_len} + {max_tokens} > {max_seq_len}"
        )

    cpu_backend = cpu_runtime.backend
    first_block = cpu_model.blocks[0]
    num_heads = first_block.attn.n_head
    head_dim = first_block.attn.head_dim
    num_layers = len(cpu_model.blocks)

    prefill_bucket = 128
    while (
        prefill_bucket < prompt_len
        and prefill_bucket < max_seq_len
    ):
        prefill_bucket *= 2
    prefill_bucket = min(prefill_bucket, max_seq_len)

    prefill_cache = KVCache(
        batch_size=1,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=np.float32,
    )

    prefill_input = Tensor(
        shape=(1, prefill_bucket),
        dtype=np.int32,
        device="cpu",
        requires_grad=False,
        init_val=None,
        name="check_cpu_prefill_input",
    )

    prefill_plan = build_cpu_logits_plan(
        model=cpu_model,
        input_ids=prefill_input,
        kv_cache=prefill_cache,
        position=prefill_cache.position,
        mode="prefill",
    )

    padded_prompt = np.full(
        (1, prefill_bucket),
        BOS_TOKEN,
        dtype=np.int32,
    )
    padded_prompt[0, :prompt_len] = prompt_tokens

    prefill_result = prefill_plan.forward(
        feeds={
            prefill_plan.input_ids: padded_prompt,
            prefill_cache.position: np.asarray(
                [0], dtype=np.int32
            ),
        },
        return_value=True,
    )

    prefill_logits = np.array(
        prefill_result,
        dtype=np.float32,
        copy=True,
    )

    if prefill_logits.ndim != 3:
        raise AssertionError(
            "CPU prefill logits must be [1,1,V], "
            f"got {prefill_logits.shape}"
        )
    if prefill_logits.shape[0] != 1:
        raise AssertionError(
            "CPU prefill batch must be 1, "
            f"got {prefill_logits.shape}"
        )
    if prefill_logits.shape[1] != 1:
        raise AssertionError(
            "CPU prefill logits do not contain the final prompt "
            f"position: shape={prefill_logits.shape}, "
            f"prompt_len={prompt_len}"
        )

    logits_trace = [prefill_logits]

    prefill_selected = select_sampling_logits(
        prefill_logits,
        mode="prefill",
        prompt_len=prompt_len,
    )
    prefill_values, prefill_indices, count = (
        numpy_topk_reference(prefill_selected, top_k)
    )
    first_token = numpy_sample_greedy(
        prefill_values[0],
        prefill_indices[0],
        count,
    )

    finished = np.zeros((num_samples,), dtype=np.int32)
    next_tokens = np.full(
        (num_samples, 1),
        first_token,
        dtype=np.int32,
    )
    lengths = np.zeros((num_samples,), dtype=np.int32)

    if first_token == EOS_TOKEN or first_token == BOS_TOKEN:
        finished[:] = 1
    else:
        lengths[:] = 1

    if np.all(finished != 0):
        return logits_trace

    decode_cache = KVCache(
        batch_size=num_samples,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=np.float32,
    )

    decode_input = Tensor(
        shape=(num_samples, 1),
        dtype=np.int32,
        device="cpu",
        requires_grad=False,
        init_val=None,
        name="check_cpu_decode_input",
    )

    # 先构建并绑定 plan，确保 decode KV cache 已分配 storage。
    decode_plan = build_cpu_logits_plan(
        model=cpu_model,
        input_ids=decode_input,
        kv_cache=decode_cache,
        position=decode_cache.position,
        mode="decode",
    )

    # 当前 CPUBackend.copy_to_host() 返回 storage.data 本身，
    # 因此 np.copyto() 会直接更新目标 KV cache。
    for source_k, target_k in zip(
        prefill_cache.k_cache,
        decode_cache.k_cache,
    ):
        source = cpu_backend.copy_to_host(source_k)
        target = cpu_backend.copy_to_host(target_k)
        np.copyto(
            target[:num_samples, :prompt_len],
            source[0:1, :prompt_len],
        )

    for source_v, target_v in zip(
        prefill_cache.v_cache,
        decode_cache.v_cache,
    ):
        source = cpu_backend.copy_to_host(source_v)
        target = cpu_backend.copy_to_host(target_v)
        np.copyto(
            target[:num_samples, :prompt_len],
            source[0:1, :prompt_len],
        )

    for generation_step in range(1, max_tokens):
        if np.all(finished != 0):
            break

        # 第一次 decode 消费 prefill 生成的 token，
        # 将它写入 prompt 后的第一个 KV 位置。
        decode_position = prompt_len + generation_step - 1

        decode_result = decode_plan.forward(
            feeds={
                decode_plan.input_ids: next_tokens,
                decode_cache.position: np.asarray(
                    [decode_position], dtype=np.int32
                ),
            },
            return_value=True,
        )

        decode_logits = np.array(
            decode_result,
            dtype=np.float32,
            copy=True,
        )

        if decode_logits.ndim != 3:
            raise AssertionError(
                "CPU decode logits must be [B,1,V], "
                f"got {decode_logits.shape}"
            )
        if decode_logits.shape[0] != num_samples:
            raise AssertionError(
                "CPU decode batch mismatch: "
                f"got {decode_logits.shape[0]}, "
                f"expected {num_samples}"
            )
        if decode_logits.shape[1] != 1:
            raise AssertionError(
                "CPU decode time dimension must be 1, "
                f"got {decode_logits.shape}"
            )

        logits_trace.append(decode_logits)

        decode_selected = select_sampling_logits(
            decode_logits,
            mode="decode",
            prompt_len=prompt_len,
        )
        decode_values, decode_indices, count = (
            numpy_topk_reference(decode_selected, top_k)
        )

        new_next_tokens = np.full(
            (num_samples, 1),
            EOS_TOKEN,
            dtype=np.int32,
        )

        for row in range(num_samples):
            if finished[row] != 0:
                continue

            token = numpy_sample_greedy(
                decode_values[row],
                decode_indices[row],
                count,
            )
            new_next_tokens[row, 0] = token

            if token == EOS_TOKEN or token == BOS_TOKEN:
                finished[row] = 1
            else:
                lengths[row] += 1

        next_tokens = new_next_tokens

    return logits_trace

def make_nanochat_microbatch_cases(
    batch,
    batch_size,
    examples_per_rank=1,
):
    inputs = np.asarray(
        batch["inputs"],
        dtype=np.int32,
    )

    targets = np.asarray(
        batch["targets"],
        dtype=np.int32,
    )

    advantages = np.asarray(
        batch["advantages"],
        dtype=np.float32,
    )

    if inputs.ndim != 2:
        raise ValueError(
            "inputs must have shape [B,T]"
        )

    if targets.shape != inputs.shape:
        raise ValueError(
            "inputs/targets shape mismatch"
        )

    if advantages.shape != (inputs.shape[0],):
        raise ValueError(
            "advantages shape mismatch"
        )

    total_samples, sequence_length = (
        inputs.shape
    )

    if total_samples % batch_size != 0:
        raise ValueError(
            "sample count is not divisible by batch_size"
        )

    num_passes = total_samples // batch_size

    microbatches = []
    grad_scales = []
    references = []

    for pass_idx in range(num_passes):
        begin = pass_idx * batch_size
        end = begin + batch_size

        inputs_pass = inputs[
            begin:end
        ].copy()

        targets_pass = targets[
            begin:end
        ].copy()

        advantages_pass = advantages[
            begin:end
        ].copy()

        pass_references = (
            build_nanochat_microbatch_reference(
                targets=targets_pass,
                advantages=advantages_pass,
                num_passes=num_passes,
                examples_per_rank=examples_per_rank,
                batch_size=batch_size,
            )
        )

        references.append(
            pass_references
        )

        for reference in pass_references:
            microbatches.append(
                (
                    inputs_pass.copy(),
                    reference["targets"].copy(),
                )
            )

            grad_scales.append(
                float(
                    reference["grad_scale"]
                )
            )

    return {
        "inputs": inputs,
        "targets": targets,
        "advantages": advantages,
        "num_passes": num_passes,
        "sequence_length": sequence_length,
        "valid_tokens": int(
            np.count_nonzero(targets >= 0)
        ),
        "all_zero_advantage": bool(
            np.all(
                np.abs(advantages) <= 1e-12
            )
        ),
        "microbatches": microbatches,
        "grad_scales": grad_scales,
        "references": references,
    }

def _duplicate_rollout_batch(
    sequences,
    masks,
    rewards,
    assistant_end,
    repeats,
):
    if repeats <= 0:
        raise ValueError(
            "repeats must be positive"
        )

    expanded_sequences = []
    expanded_masks = []
    expanded_rewards = []

    for repeat_index in range(repeats):
        expanded_sequences.extend(
            [
                list(sequence)
                for sequence in sequences
            ]
        )

        expanded_masks.extend(
            [
                list(mask)
                for mask in masks
            ]
        )

        expanded_rewards.extend(
            np.asarray(
                rewards,
                dtype=np.float32,
            )
            + np.float32(repeat_index)
        )

    return build_nanochat_batch_reference(
        sequences=expanded_sequences,
        masks=expanded_masks,
        rewards=np.asarray(
            expanded_rewards,
            dtype=np.float32,
        ),
        assistant_end=assistant_end,
    )

def compare_exact_sampled_tokens(
    metal_state_trace,
    cpu_trace,
):
    if len(metal_state_trace) != len(cpu_trace):
        raise AssertionError(
            "SAMPLE_TRACE_LENGTH_MISMATCH: "
            f"metal={len(metal_state_trace)} "
            f"cpu={len(cpu_trace)}"
        )

    for step, (metal_state, cpu_step) in enumerate(
        zip(metal_state_trace, cpu_trace)
    ):
        actual = np.asarray(
            metal_state["next_tokens"],
            dtype=np.int32,
        ).reshape(-1)

        expected = np.asarray(
            cpu_step["sampled_tokens"],
            dtype=np.int32,
        ).reshape(-1)

        check_equal(
            f"SAMPLED_TOKENS_MISMATCH step={step}",
            actual,
            expected,
        )

def run_random_sampling_semantics_case(
    model,
    runtime,
    arena,
    cpu_reference_model,
    prompt_tokens,
    batch_size,
    max_seq_len,
    max_new_tokens,
    top_k,
    seed,
):
    tokenizer = SyntheticTokenizer()

    metal_engine = Engine(
        model=model,
        tokenizer=tokenizer,
        device="mps",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        parameter_runtime=runtime,
        arena=arena,
        max_top_k=K_MAX,
    )

    metal_logits_trace = []
    metal_input_trace = []
    metal_topk_trace = []
    metal_state_trace = []

    pending_capture = capture_logits_wrapper(
        engine=metal_engine,
        prompt_len=len(prompt_tokens),
        captured_logits=metal_logits_trace,
        captured_inputs=metal_input_trace,
        captured_topk=metal_topk_trace,
    )

    wrap_engine_rollout_methods(
        engine=metal_engine,
        state_trace=metal_state_trace,
        captured_logits=metal_logits_trace,
        captured_inputs=metal_input_trace,
        captured_topk=metal_topk_trace,
        pending=pending_capture,
    )

    metal_engine.generate_batch_new(
        tokens=prompt_tokens,
        num_samples=batch_size,
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_k=top_k,
        seed=seed,
    )

    cpu_rollout = generate_cpu_nanochat_reference(
        cpu_model=cpu_reference_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        num_samples=batch_size,
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_k=top_k,
        seed=seed,
    )

    if not metal_logits_trace:
        raise AssertionError(
            "random sampling produced no Metal logits"
        )

    if not metal_topk_trace:
        raise AssertionError(
            "random sampling produced no Metal TopK"
        )

    cpu_prefill_logits = np.asarray(
        cpu_rollout["trace"][0]["logits"],
        dtype=np.float32,
    )

    metal_prefill_logits = select_sampling_logits(
        metal_logits_trace[0]["logits"],
        mode="prefill",
        prompt_len=len(prompt_tokens),
    )

    metal_prefill_logits = _align_rows(
        metal_prefill_logits,
        cpu_prefill_logits.shape[0],
    )

    check_close(
        "RANDOM_PREFILL_LOGITS_MISMATCH",
        metal_prefill_logits,
        cpu_prefill_logits,
        atol=1e-5,
        rtol=1e-6,
    )

    expected_values, expected_indices, count = (
        numpy_topk_reference(
            cpu_prefill_logits,
            top_k,
        )
    )

    actual_values = _align_rows(
        metal_topk_trace[0]["values"],
        batch_size,
    )

    actual_indices = _align_rows(
        metal_topk_trace[0]["indices"],
        batch_size,
    )

    check_close(
        "RANDOM_PREFILL_TOPK_VALUES_MISMATCH",
        actual_values[:, :count],
        expected_values[:, :count],
        atol=1e-5,
        rtol=1e-5,
    )

    check_equal(
        "RANDOM_PREFILL_TOPK_INDICES_MISMATCH",
        actual_indices[:, :count],
        expected_indices[:, :count],
    )

    # PyTorch nanochat RNG 与 Metal RNG 实现不同，
    # 因此随机 token 不能逐 token 要求相等。
    # 必须验证 Metal 输出属于当前 TopK 分布支持集合。
    for step, state in enumerate(metal_state_trace):
        if step >= len(metal_topk_trace):
            break

        next_tokens = np.asarray(
            state["next_tokens"],
            dtype=np.int32,
        ).reshape(-1)

        topk_indices = _align_rows(
            metal_topk_trace[step]["indices"],
            next_tokens.shape[0],
        )

        current_count = min(
            int(top_k),
            topk_indices.shape[1],
        )

        for row, token in enumerate(next_tokens):
            token = int(token)

            if token in (
                EOS_TOKEN,
                BOS_TOKEN,
            ):
                continue

            if token not in topk_indices[
                row,
                :current_count,
            ]:
                raise AssertionError(
                    "RANDOM_SAMPLE_OUTSIDE_TOPK: "
                    f"step={step}, row={row}, "
                    f"token={token}"
                )

    compare_metal_cpu_policy_tokens(
	    metal_state_trace=metal_state_trace,
	    cpu_trace=cpu_rollout["trace"],
	    eos_token=EOS_TOKEN,
	    bos_token=BOS_TOKEN,
	)

    print(
        "[RANDOM-SAMPLING-SEMANTICS-PASS]",
        flush=True,
    )

class NanochatCPUModel:
    """
    独立 CPU 参考模型。

    这里只复用当前项目的 GPT2 结构和 NumPy CPU kernel，
    不复用 Metal runtime、Metal activation、Metal KV cache，
    也不使用 FusedLinearCrossEntropy 作为参考 loss。
    """

    def __init__(self, model):
        self.model = model
        self.max_seq_len = int(model.rope_cos.shape[0])
        self.parameters = {
            name: tensor
            for name, tensor in named_tensors(model).items()
            if tensor.requires_grad
        }
        self._plans = {}

    def _build_plan(self, batch_size, seq_len):
        key = (int(batch_size), int(seq_len))
        if key in self._plans:
            return self._plans[key]

        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("invalid CPU reference shape")

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds "
                f"CPU model capacity {self.max_seq_len}"
            )

        runtime = Runtime(
            device="cpu",
            forward_only=False,
            owns_params=False,
        )

        input_ids = Tensor(
            shape=(batch_size, seq_len),
            dtype=np.int32,
            device="cpu",
            requires_grad=False,
            init_val=None,
            name=f"cpu_reference_input_{batch_size}_{seq_len}",
        )

        # 当前项目 CPU RoPE kernel 要求 RoPE 表与实际 T 一致。
        rope_cos = Tensor(
            shape=(seq_len, self.model.rope_cos.shape[1]),
            dtype=np.float32,
            device="cpu",
            requires_grad=False,
            init_val=self.model.rope_cos.storage.data[:seq_len].copy(),
            name=f"cpu_reference_rope_cos_{seq_len}",
        )

        rope_sin = Tensor(
            shape=(seq_len, self.model.rope_sin.shape[1]),
            dtype=np.float32,
            device="cpu",
            requires_grad=False,
            init_val=self.model.rope_sin.storage.data[:seq_len].copy(),
            name=f"cpu_reference_rope_sin_{seq_len}",
        )

        hidden = self.model.tok_emb(input_ids)

        for block in self.model.blocks:
            hidden = block(
                hidden,
                rope_cos,
                rope_sin,
                mode="train",
            )

        runtime.loss = hidden
        runtime.build_topo(hidden)

        # CPU 参考不使用 activation aliasing。
        tensors = {}

        for node in runtime.topo:
            for tensor in node.inputs + node.outputs:
                tensors[id(tensor)] = tensor

                if tensor.storage is None:
                    tensor.storage = runtime.allocator.alloc(
                        tensor.shape,
                        tensor.dtype,
                    )
                    runtime.backend.init(tensor)

                if (
                    tensor.grad is not None
                    and tensor.grad.storage is None
                ):
                    tensor.grad.storage = runtime.allocator.zeros(
                        tensor.grad.shape,
                        tensor.grad.dtype,
                    )

            for phase, layout in (
                ("forward", node.scratch_fwd_layout),
                ("backward", node.scratch_bwd_layout),
            ):
                kernel = getattr(node, phase)
                if kernel is None:
                    raise RuntimeError(
                        f"missing CPU {phase} kernel for {node.op}"
                    )

                if layout is not None:
                    scratch_storage = runtime.allocator.empty(
                        (layout.nbytes,),
                        np.uint8,
                    )
                    scratch_views = runtime.scratch_bind(
                        scratch_storage,
                        layout,
                    )
                    setattr(
                        node,
                        phase,
                        functools.partial(
                            kernel,
                            scratch=scratch_views,
                        ),
                    )

        runtime.is_compiled = True

        result = {
            "runtime": runtime,
            "input_ids": input_ids,
            "hidden": hidden,
            "tensors": list(tensors.values()),
        }
        self._plans[key] = result
        return result

    def forward(self, input_ids):
        input_ids = np.asarray(input_ids, dtype=np.int32)

        if input_ids.ndim != 2:
            raise ValueError(
                f"CPU input must have shape [B,T], got {input_ids.shape}"
            )

        vocab_size = int(self.model.tok_emb.weight.shape[0])
        if np.any(input_ids < 0) or np.any(input_ids >= vocab_size):
            raise ValueError("CPU input token is outside vocabulary")

        plan = self._build_plan(
            batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
        )

        runtime = plan["runtime"]
        input_tensor = plan["input_ids"]
        hidden = plan["hidden"]

        np.copyto(
            input_tensor.storage.data,
            input_ids,
        )

        runtime.forward()

        hidden_value = np.array(
            hidden.storage.data,
            dtype=np.float32,
            copy=True,
        )

        logits = np.matmul(
            hidden_value,
            self.model.lm_head.weight.storage.data.T,
        ).astype(np.float32)

        return logits, {
            "runtime": runtime,
            "hidden": hidden,
            "tensors": plan["tensors"],
            "hidden_value": hidden_value,
        }

    def backward(self, context, dlogits):
        runtime = context["runtime"]
        hidden = context["hidden"]
        tensors = context["tensors"]
        hidden_value = context["hidden_value"]
        lm_head = self.model.lm_head.weight

        dlogits = np.asarray(dlogits, dtype=np.float32)

        expected_shape = (
            hidden_value.shape[:-1]
            + (lm_head.shape[0],)
        )
        if dlogits.shape != expected_shape:
            raise ValueError(
                f"dlogits shape {dlogits.shape} != "
                f"{expected_shape}"
            )

        # 只清理 activation gradient；
        # parameter gradient 在多个 microbatch 之间累积。
        for tensor in tensors:
            if tensor.op is not None and tensor.grad is not None:
                tensor.grad.storage.data.fill(0.0)

        np.copyto(
            hidden.grad.storage.data,
            np.matmul(
                dlogits,
                lm_head.storage.data,
            ),
        )

        lm_head.grad.storage.data[...] += np.matmul(
            dlogits.reshape(-1, lm_head.shape[0]).T,
            hidden_value.reshape(-1, hidden_value.shape[-1]),
        ).astype(np.float32)

        for node in reversed(runtime.topo):
            node.backward(
                node.inputs,
                node.outputs,
            )

    def zero_grad(self):
        for parameter in self.parameters.values():
            if parameter.grad is not None:
                parameter.grad.storage.data.fill(0.0)

    def gradients(self):
        return {
            name: np.array(
                parameter.grad.storage.data,
                dtype=np.float32,
                copy=True,
            )
            for name, parameter in self.parameters.items()
        }


def clone_model_to_cpu_reference(
    metal_model,
    metal_backend,
):
    """
    从 Metal 模型复制参数到独立 CPU GPT2。
    CPU 模型只作为参考，不共享任何 Metal activation 或 runtime。
    """
    source_tensors = named_tensors(metal_model)

    first_block = metal_model.blocks[0]

    cpu_model = GPT2(
        V=int(metal_model.tok_emb.weight.shape[0]),
        C=int(metal_model.tok_emb.weight.shape[1]),
        max_T=int(metal_model.rope_cos.shape[0]),
        n_head=int(first_block.attn.n_head),
        batch_size=1,
        n_layer=len(metal_model.blocks),
        dtype=np.float32,
        device="cpu",
        rollout_max_topk=K_MAX,
    )

    cpu_runtime = Runtime(
        device="cpu",
        forward_only=False,
        owns_params=True,
    )

    target_tensors = named_tensors(cpu_model)

    if set(source_tensors) != set(target_tensors):
        missing = sorted(set(source_tensors) - set(target_tensors))
        extra = sorted(set(target_tensors) - set(source_tensors))
        raise AssertionError(
            f"CPU/Metal model tensor mismatch: "
            f"missing={missing}, extra={extra}"
        )

    for name, target in target_tensors.items():
        source = source_tensors[name]
        value = np.array(
            metal_backend.copy_to_host(source),
            dtype=np.float32,
            copy=True,
        )

        if tuple(value.shape) != tuple(target.shape):
            raise AssertionError(
                f"{name}: source shape {value.shape} != "
                f"target shape {target.shape}"
            )

        target.storage = cpu_runtime.allocator.alloc(
            target.shape,
            target.dtype,
        )
        np.copyto(
            target.storage.data,
            value,
        )
        target.init_val = None

        if target.requires_grad:
            target.grad = Tensor(
                shape=target.shape,
                dtype=target.dtype,
                device="cpu",
                requires_grad=False,
                init_val=None,
            )
            target.grad.storage = cpu_runtime.allocator.zeros(
                target.shape,
                target.dtype,
            )

    return NanochatCPUModel(cpu_model)

class NanochatPhilox4x32:
    """Torch MPS generator-compatible Philox stream, implemented in NumPy.

    nanochat creates one ``torch.Generator(device=device)`` per rollout and
    seeds it once before the prefill.  MPS uses Philox4x32-10 internally.  The
    checker keeps the same 64-bit seed and advances the stream in row-major
    order; no torch or nanochat code is imported here.
    """

    _M0 = np.uint32(0xD2511F53)
    _M1 = np.uint32(0xCD9E8D57)
    _W0 = np.uint32(0x9E3779B9)
    _W1 = np.uint32(0xBB67AE85)
    _MASK = np.uint64(0xFFFFFFFF)

    def __init__(self, seed):
        seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.seed = seed
        self.counter = 0
        self.buffer = np.empty((0,), dtype=np.uint32)

    @staticmethod
    def _mulhilo(a, b):
        product = np.uint64(int(a) & 0xFFFFFFFF) * np.uint64(int(b) & 0xFFFFFFFF)
        return np.uint32((product >> np.uint64(32)) & np.uint64(0xFFFFFFFF)), np.uint32(product & np.uint64(0xFFFFFFFF))

    def _block(self, counter):
        c0 = np.uint32(counter & 0xFFFFFFFF)
        c1 = np.uint32((counter >> 32) & 0xFFFFFFFF)
        c2 = np.uint32(0)
        c3 = np.uint32(0)
        k0 = np.uint32(self.seed & 0xFFFFFFFF)
        k1 = np.uint32((self.seed >> 32) & 0xFFFFFFFF)
        for _ in range(10):
            hi0, lo0 = self._mulhilo(self._M0, c0)
            hi1, lo1 = self._mulhilo(self._M1, c2)
            c0, c1, c2, c3 = (
                np.uint32(hi1 ^ c1 ^ k0),
                np.uint32(lo1),
                np.uint32(hi0 ^ c3 ^ k1),
                np.uint32(lo0),
            )
            k0 = np.uint32((int(k0) + int(self._W0)) & 0xFFFFFFFF)
            k1 = np.uint32((int(k1) + int(self._W1)) & 0xFFFFFFFF)
        return np.asarray((c0, c1, c2, c3), dtype=np.uint32)

    def uint32(self, count):
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        out = np.empty((count,), dtype=np.uint32)
        written = 0
        if self.buffer.size:
            take = min(count, self.buffer.size)
            out[:take] = self.buffer[:take]
            self.buffer = self.buffer[take:]
            written = take
        while written < count:
            block = self._block(self.counter)
            self.counter += 1
            take = min(count - written, 4)
            out[written:written + take] = block[:take]
            written += take
            if take < 4:
                self.buffer = block[take:].copy()
        return out

    def uniform(self, count):
        # This is ATen's Philox conversion used by the generator.  Masking the
        # sign bit is intentional and differs from NumPy's PCG64 conversion.
        raw = self.uint32(count)
        return ((raw & np.uint32(0x7FFFFFFF)).astype(np.float64) * 4.6566127342e-10).astype(np.float32)


def numpy_sample_nanochat(
    logits,
    rng,
    temperature,
    top_k,
):
    logits = np.asarray(logits, dtype=np.float32)

    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [B,V], got {logits.shape}"
        )

    if not np.all(np.isfinite(logits)):
        raise ValueError("logits contain non-finite values")

    temperature = float(temperature)
    if temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative")

    batch_size, vocab_size = logits.shape

    if temperature == 0.0:
        return np.argmax(
            logits,
            axis=-1,
        ).astype(np.int32)

    if int(top_k) > 0:
        count = min(int(top_k), vocab_size)

        # 与 torch.topk 的语义一致：只在候选集合中采样。
        order = np.argsort(
            -logits,
            axis=-1,
            kind="stable",
        )[:, :count]
    else:
        count = vocab_size
        order = np.broadcast_to(
            np.arange(vocab_size, dtype=np.int32),
            (batch_size, vocab_size),
        ).copy()

    selected = np.take_along_axis(
        logits,
        order,
        axis=-1,
    )

    selected = selected / np.float32(temperature)
    selected = selected - np.max(
        selected,
        axis=-1,
        keepdims=True,
    )

    probabilities = np.exp(selected).astype(np.float32)
    probabilities /= np.sum(
        probabilities,
        axis=-1,
        keepdims=True,
    )

    uniforms = rng.uniform(batch_size)
    cumulative = np.cumsum(
        probabilities,
        axis=-1,
    )
    cumulative[:, -1] = 1.0

    choices = np.sum(
        cumulative <= uniforms[:, None],
        axis=-1,
    ).astype(np.int32)

    return order[
        np.arange(batch_size),
        choices,
    ].astype(np.int32)

def generate_cpu_nanochat_reference(
    cpu_model,
    tokenizer,
    prompt_tokens,
    num_samples,
    max_tokens,
    temperature,
    top_k,
    seed,
):
    prompt_tokens = [
        int(token)
        for token in prompt_tokens
    ]

    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")

    if int(num_samples) <= 0:
        raise ValueError("num_samples must be positive")

    if int(max_tokens) < 0:
        raise ValueError("max_tokens must be non-negative")

    if (
        len(prompt_tokens) + int(max_tokens)
        > int(cpu_model.max_seq_len)
    ):
        raise ValueError(
            "prompt plus generated tokens exceeds CPU model capacity"
        )

    assistant_end = int(
        tokenizer.encode_special("<|assistant_end|>")
    )
    bos_token = int(
        tokenizer.get_bos_token_id()
    )
    python_start = int(
        tokenizer.encode_special("<|python_start|>")
    )
    python_end = int(
        tokenizer.encode_special("<|python_end|>")
    )
    output_start = int(
        tokenizer.encode_special("<|output_start|>")
    )
    output_end = int(
        tokenizer.encode_special("<|output_end|>")
    )

    rng = NanochatPhilox4x32(
        int(seed) & 0xFFFFFFFFFFFFFFFF
    )

    contexts = [
        list(prompt_tokens)
        for _ in range(int(num_samples))
    ]

    sequences = [
        list(prompt_tokens)
        for _ in range(int(num_samples))
    ]

    masks = [
        [0] * len(prompt_tokens)
        for _ in range(int(num_samples))
    ]

    completed = np.zeros(
        (int(num_samples),),
        dtype=np.bool_,
    )

    forced_tokens = [
        deque()
        for _ in range(int(num_samples))
    ]

    in_python_block = [
        False
        for _ in range(int(num_samples))
    ]

    python_expression_tokens = [
        []
        for _ in range(int(num_samples))
    ]

    trace = []

    # nanochat always performs the one-row prefill before entering its
    # generation loop, including the max_tokens=0 case.
    raw_logits, _ = cpu_model.forward(
        np.asarray([prompt_tokens], dtype=np.int32)
    )
    sampled_logits = np.repeat(
        np.asarray(raw_logits[:, -1, :], dtype=np.float32),
        int(num_samples),
        axis=0,
    )

    for generation_step in range(int(max_tokens)):
        if bool(np.all(completed)):
            break

        sampled_tokens = numpy_sample_nanochat(
            logits=sampled_logits,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
        )

        token_column = sampled_tokens.copy()

        token_masks = np.ones(
            (int(num_samples),),
            dtype=np.int32,
        )

        for row in range(int(num_samples)):
            was_completed = bool(completed[row])

            is_forced = bool(forced_tokens[row])

            if is_forced:
                token_column[row] = (
                    forced_tokens[row].popleft()
                )
                token_masks[row] = 0

            token = int(token_column[row])

            # nanochat Engine.generate 始终推进内部 row state，
            # 即使该 row 已经完成。
            contexts[row].append(token)

            if token == assistant_end or token == bos_token:
                completed[row] = True

            if token == python_start:
                in_python_block[row] = True
                python_expression_tokens[row] = []

            elif (
                token == python_end
                and in_python_block[row]
            ):
                in_python_block[row] = False

                if python_expression_tokens[row]:
                    expression = tokenizer.decode(
                        python_expression_tokens[row]
                    )

                    result = calculator_reference(
                        expression
                    )

                    if result is not None:
                        result_tokens = tokenizer.encode(
                            str(result)
                        )

                        forced_tokens[row].append(
                            output_start
                        )
                        forced_tokens[row].extend(
                            int(value)
                            for value in result_tokens
                        )
                        forced_tokens[row].append(
                            output_end
                        )

                python_expression_tokens[row] = []

            elif in_python_block[row]:
                python_expression_tokens[row].append(token)

            # 对应 nanochat Engine.generate_batch：
            # prompt 已经在 sequences 中；
            # EOS/BOS 不加入结果；
            # 已完成 row 后续 token 不加入结果；
            # forced token 的 mask 为 0。
            if (
                not was_completed
                and token != assistant_end
                and token != bos_token
            ):
                sequences[row].append(token)
                masks[row].append(
                    int(token_masks[row])
                )

        trace.append(
            {
                "step": generation_step,
                "logits": sampled_logits.copy(),
                "sampled_tokens": sampled_tokens.copy(),
                "tokens": token_column.copy(),
                "token_masks": token_masks.copy(),
                "completed": completed.copy(),
            }
        )

        # nanochat updates the KV cache immediately after yielding each token,
        # even when that token completes the last row.  The CPU reference has
        # no KV cache, so it recomputes the same next-token logits from the full
        # contexts; this preserves the call order and row-state semantics.
        raw_logits, _ = cpu_model.forward(
            np.asarray(contexts, dtype=np.int32)
        )
        sampled_logits = np.asarray(
            raw_logits[:, -1, :],
            dtype=np.float32,
        )

    return {
        "sequences": sequences,
        "masks": masks,
        "trace": trace,
    }

def build_nanochat_batch_reference(
    sequences,
    masks,
    rewards,
    assistant_end,
):
    if not sequences:
        raise ValueError(
            "sequences must not be empty"
        )

    if len(sequences) != len(masks):
        raise ValueError(
            "sequence/mask count mismatch"
        )

    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    if rewards.shape != (len(sequences),):
        raise ValueError(
            "reward shape mismatch"
        )

    if not np.all(np.isfinite(rewards)):
        raise ValueError(
            "rewards contain non-finite values"
        )

    max_length = max(
        len(sequence)
        for sequence in sequences
    )

    ids = np.full(
        (len(sequences), max_length),
        int(assistant_end),
        dtype=np.int32,
    )

    mask_ids = np.zeros_like(ids)

    for row, (sequence, mask) in enumerate(
        zip(sequences, masks)
    ):
        if len(sequence) != len(mask):
            raise ValueError(
                "sequence and mask length mismatch"
            )

        sequence = np.asarray(
            sequence,
            dtype=np.int32,
        )
        mask = np.asarray(
            mask,
            dtype=np.int32,
        )

        if np.any((mask != 0) & (mask != 1)):
            raise ValueError(
                "mask values must be 0 or 1"
            )

        ids[row, :len(sequence)] = sequence
        mask_ids[row, :len(mask)] = mask

    # 与 chat_rl.py 完全一致。
    inputs = ids[:, :-1].copy()
    targets = ids[:, 1:].copy()

    targets[
        mask_ids[:, 1:] == 0
    ] = -1

    # torch.float32 mean 的 NumPy 对应实现。
    mean_reward = np.mean(
        rewards,
        dtype=np.float32,
    )

    advantages = (
        rewards - mean_reward
    ).astype(np.float32)

    return {
        "ids": ids,
        "mask_ids": mask_ids,
        "inputs": inputs,
        "targets": targets,
        "rewards": rewards,
        "advantages": advantages,
    }

def build_cpu_nanochat_get_batch_reference(
    cpu_model,
    tokenizer,
    conversation,
    prompt_tokens,
    num_samples,
    device_batch_size,
    max_tokens,
    temperature,
    top_k,
    step,
    example_idx,
    reward_fn,
    assistant_end,
):
    rollout = generate_cpu_nanochat_get_batch_rollouts(
        cpu_model=cpu_model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        num_samples=num_samples,
        device_batch_size=device_batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        step=step,
        example_idx=example_idx,
    )

    prefix_length = len(prompt_tokens)
    rewards = []

    for sequence in rollout["sequences"]:
        generated_tokens = sequence[prefix_length:]
        generated_text = tokenizer.decode(
            generated_tokens
        )

        # 与 chat_rl.py 中：
        # train_task.reward(conversation, generated_text)
        # 保持同样的调用形状。
        reward = reward_fn(
            conversation,
            generated_text,
        )

        rewards.append(float(reward))

    batch = build_nanochat_batch_reference(
        sequences=rollout["sequences"],
        masks=rollout["masks"],
        rewards=np.asarray(
            rewards,
            dtype=np.float32,
        ),
        assistant_end=assistant_end,
    )

    batch["conversation"] = conversation
    batch["rollout_sequences"] = rollout["sequences"]
    batch["rollout_masks"] = rollout["masks"]
    batch["rollout_traces"] = rollout["traces"]
    batch["num_sampling_steps"] = rollout["num_sampling_steps"]

    return batch
def generate_metal_nanochat_get_batch_rollouts(
    engine,
    prompt_tokens,
    num_samples,
    device_batch_size,
    max_tokens,
    temperature,
    top_k,
    step,
    example_idx,
):
    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive"
        )

    if device_batch_size <= 0:
        raise ValueError(
            "device_batch_size must be positive"
        )

    num_sampling_steps = (
        num_samples // device_batch_size
    )

    sequences = []
    masks = []

    for sampling_step in range(
        num_sampling_steps
    ):
        sampling_seed = hash(
            (
                int(step),
                int(example_idx),
                int(sampling_step),
            )
        ) & 0x7FFFFFFF

        generated_sequences, generated_masks = (
            engine.generate_batch_new(
                tokens=prompt_tokens,
                num_samples=device_batch_size,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=sampling_seed,
            )
        )

        sequences.extend(
            generated_sequences
        )

        masks.extend(
            generated_masks
        )

    return {
        "sequences": sequences,
        "masks": masks,
        "num_sampling_steps": num_sampling_steps,
    }

def nanochat_objective_reference(
    logits,
    targets,
    advantages,
    num_passes,
    examples_per_rank,
):
    logits = np.asarray(
        logits,
        dtype=np.float32,
    )

    targets = np.asarray(
        targets,
        dtype=np.int32,
    )

    advantages = np.asarray(
        advantages,
        dtype=np.float32,
    )

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [B,T,V]"
        )

    batch_size, sequence_length, vocab_size = (
        logits.shape
    )

    if targets.shape != (
        batch_size,
        sequence_length,
    ):
        raise ValueError(
            "targets shape mismatch"
        )

    if advantages.shape != (batch_size,):
        raise ValueError(
            "advantages shape mismatch"
        )

    num_passes = int(num_passes)
    examples_per_rank = int(examples_per_rank)

    if num_passes <= 0:
        raise ValueError("num_passes must be positive")

    if examples_per_rank <= 0:
        raise ValueError(
            "examples_per_rank must be positive"
        )

    if np.any(targets < -1):
        raise ValueError(
            "targets contain invalid negative token"
        )

    if np.any(targets >= vocab_size):
        raise ValueError(
            "targets contain out-of-range token"
        )

    valid = targets >= 0
    valid_tokens = int(np.count_nonzero(valid))

    denominator = float(
        max(valid_tokens, 1)
        * num_passes
        * examples_per_rank
    )

    # 对应 torch.nn.functional.cross_entropy(..., reduction='none')
    row_max = np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    shifted = (
        logits - row_max
    ).astype(np.float32)

    exp_logits = np.exp(
        shifted
    ).astype(np.float32)

    sum_exp = np.sum(
        exp_logits,
        axis=-1,
        keepdims=True,
        dtype=np.float32,
    )

    log_sum_exp = np.log(
        sum_exp
    ).astype(np.float32)

    probabilities = (
        exp_logits / sum_exp
    ).astype(np.float32)

    safe_targets = np.where(
        valid,
        targets,
        0,
    )

    target_shifted = np.take_along_axis(
        shifted,
        safe_targets[..., None],
        axis=-1,
    )[..., 0]

    token_nll = np.where(
        valid,
        log_sum_exp[..., 0] - target_shifted,
        np.float32(0.0),
    ).astype(np.float32)

    # chat_rl.py:
    # logp = -nll
    # pg_obj = (logp * advantage).sum()
    # loss = -pg_obj / denominator
    loss = np.sum(
        token_nll
        * advantages[:, None],
        dtype=np.float32,
    ) / np.float32(denominator)

    dlogits = probabilities.copy()

    flat_dlogits = dlogits.reshape(
        -1,
        vocab_size,
    )
    flat_targets = targets.reshape(-1)
    flat_valid = valid.reshape(-1)

    valid_rows = np.flatnonzero(flat_valid)

    flat_dlogits[
        valid_rows,
        flat_targets[valid_rows],
    ] -= np.float32(1.0)

    dlogits[~valid] = np.float32(0.0)

    dlogits *= (
        advantages[:, None, None]
        / np.float32(denominator)
    )

    return {
        "loss": float(loss),
        "token_nll": token_nll,
        "dlogits": dlogits.astype(np.float32),
        "valid_tokens": valid_tokens,
        "denominator": denominator,
    }


def evaluate_cpu_nanochat_rl_batch(
    cpu_model,
    batch,
    batch_size,
    examples_per_rank,
    reset_grad,
):
    inputs = np.asarray(
        batch["inputs"],
        dtype=np.int32,
    )
    targets = np.asarray(
        batch["targets"],
        dtype=np.int32,
    )
    advantages = np.asarray(
        batch["advantages"],
        dtype=np.float32,
    )

    if inputs.ndim != 2:
        raise ValueError(
            "inputs must have shape [B,T]"
        )

    if targets.shape != inputs.shape:
        raise ValueError(
            "targets shape mismatch"
        )

    if advantages.shape != (inputs.shape[0],):
        raise ValueError(
            "advantages shape mismatch"
        )

    total_samples = int(inputs.shape[0])
    batch_size = int(batch_size)

    if total_samples <= 0:
        raise ValueError(
            "RL batch must not be empty"
        )

    if total_samples % batch_size != 0:
        raise ValueError(
            "sample count must be divisible by batch_size"
        )

    num_passes = total_samples // batch_size

    if reset_grad:
        cpu_model.zero_grad()

    total_loss = 0.0
    microbatches = []

    for pass_idx in range(num_passes):
        begin = pass_idx * batch_size
        end = begin + batch_size

        inputs_pass = inputs[begin:end]
        targets_pass = targets[begin:end]
        advantages_pass = advantages[begin:end]

        logits, context = cpu_model.forward(
            inputs_pass
        )

        objective = nanochat_objective_reference(
            logits=logits,
            targets=targets_pass,
            advantages=advantages_pass,
            num_passes=num_passes,
            examples_per_rank=examples_per_rank,
        )

        cpu_model.backward(
            context=context,
            dlogits=objective["dlogits"],
        )

        total_loss += objective["loss"]

        microbatches.append(
            {
                "pass_idx": pass_idx,
                "inputs": inputs_pass.copy(),
                "targets": targets_pass.copy(),
                "advantages": advantages_pass.copy(),
                "logits": np.asarray(
                    logits,
                    dtype=np.float32,
                ).copy(),
                "loss": objective["loss"],
                "dlogits": objective["dlogits"].copy(),
                "valid_tokens": objective["valid_tokens"],
                "denominator": objective["denominator"],
            }
        )

    return {
        "loss": float(total_loss),
        "gradients": cpu_model.gradients(),
        "microbatches": microbatches,
        "num_passes": num_passes,
    }


def evaluate_cpu_nanochat_rl_step(
    cpu_model,
    batches,
    batch_size,
    examples_per_rank,
):
    batches = list(batches)
    examples_per_rank = int(examples_per_rank)

    if len(batches) != examples_per_rank:
        raise ValueError(
            "batch count must equal examples_per_rank"
        )

    # 对应 chat_rl.py 在一个 optimization step 开始时
    # 已经处于清零后的参数梯度状态。
    cpu_model.zero_grad()

    total_loss = 0.0
    all_microbatches = []

    for batch in batches:
        result = evaluate_cpu_nanochat_rl_batch(
            cpu_model=cpu_model,
            batch=batch,
            batch_size=batch_size,
            examples_per_rank=examples_per_rank,
            reset_grad=False,
        )

        total_loss += result["loss"]
        all_microbatches.extend(
            result["microbatches"]
        )

    return {
        "loss": float(total_loss),
        "gradients": cpu_model.gradients(),
        "microbatches": all_microbatches,
        "examples_per_rank": examples_per_rank,
    }


def make_checker_reward(tokenizer):
    def checker_reward(
        conversation,
        generated_text,
    ):
        del conversation

        generated_tokens = tokenizer.encode(
            generated_text
        )

        value = np.float32(0.0)

        for index, token in enumerate(generated_tokens):
            token = int(token)

            if token in (
                EOS_TOKEN,
                BOS_TOKEN,
            ):
                continue

            value += np.float32(
                (index + 1) * (token + 1)
            )

        return float(value)

    return checker_reward

def assert_cpu_zero_result(
    result,
    label,
):
    if abs(float(result["loss"])) > 1e-6:
        raise AssertionError(
            f"{label}: expected zero loss, "
            f"got {result['loss']}"
        )

    for name, gradient in result["gradients"].items():
        if not np.allclose(
            gradient,
            0.0,
            atol=1e-6,
            rtol=0.0,
        ):
            raise AssertionError(
                f"{label}: expected zero gradient for {name}"
            )


def run_cpu_nanochat_rl_reference(
    cpu_model,
    tokenizer,
    examples,
    num_samples,
    device_batch_size,
    max_tokens,
    temperature,
    top_k,
    optimization_step,
    examples_per_rank,
    reward_fn,
    assistant_end,
):
    examples = list(examples)

    num_samples = int(num_samples)
    device_batch_size = int(device_batch_size)
    examples_per_rank = int(examples_per_rank)

    if len(examples) != examples_per_rank:
        raise ValueError(
            "examples count must equal examples_per_rank"
        )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive"
        )

    if device_batch_size <= 0:
        raise ValueError(
            "device_batch_size must be positive"
        )

    if num_samples < device_batch_size:
        raise ValueError(
            "num_samples must be at least device_batch_size"
        )

    cpu_batches = []

    for local_example_idx, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(
                "example must be a dict"
            )

        conversation = example["conversation"]
        # chat_rl.py seeds each rollout with the dataset's global
        # ``example_idx``.  Do not silently replace it with the position in
        # this locally constructed per-rank list.
        example_idx = int(
            example.get("example_index", local_example_idx)
        )

        prompt_tokens = tokenizer.render_for_completion(
            conversation
        )
        supplied_prompt = example.get("prompt_tokens")
        if supplied_prompt is not None and list(supplied_prompt) != list(prompt_tokens):
            raise AssertionError(
                "example.prompt_tokens differs from tokenizer.render_for_completion; "
                "the CPU reference must use the chat_rl prompt path"
            )

        batch = build_cpu_nanochat_get_batch_reference(
            cpu_model=cpu_model,
            tokenizer=tokenizer,
            conversation=conversation,
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            device_batch_size=device_batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            step=optimization_step,
            example_idx=example_idx,
            reward_fn=reward_fn,
            assistant_end=assistant_end,
        )

        expected_samples = (
            num_samples
            // device_batch_size
            * device_batch_size
        )

        if len(batch["rollout_sequences"]) != expected_samples:
            raise AssertionError(
                "CPU rollout sample count mismatch"
            )

        if len(batch["rollout_masks"]) != expected_samples:
            raise AssertionError(
                "CPU rollout mask count mismatch"
            )

        cpu_batches.append(batch)

    actual_step = evaluate_cpu_nanochat_rl_step(
        cpu_model=cpu_model,
        batches=cpu_batches,
        batch_size=device_batch_size,
        examples_per_rank=examples_per_rank,
    )

    if not np.isfinite(actual_step["loss"]):
        raise AssertionError(
            "CPU RL loss is non-finite"
        )

    for name, gradient in actual_step["gradients"].items():
        if not np.all(np.isfinite(gradient)):
            raise AssertionError(
                f"CPU gradient is non-finite: {name}"
            )

    base_batch = cpu_batches[0]
    stress_count = 2 * device_batch_size

    stress_sequences = [
        list(
            base_batch["rollout_sequences"][
                index
                % len(base_batch["rollout_sequences"])
            ]
        )
        for index in range(stress_count)
    ]

    stress_masks = [
        list(
            base_batch["rollout_masks"][
                index
                % len(base_batch["rollout_masks"])
            ]
        )
        for index in range(stress_count)
    ]

    # 对任意 stress_count=2*batch_size：
    # rewards=[0,2,1,1,...] 的均值恒为 1，
    # 因而必然包含 zero 和 non-zero advantage。
    mixed_rewards = np.ones(
        (stress_count,),
        dtype=np.float32,
    )
    mixed_rewards[0] = np.float32(0.0)
    mixed_rewards[1] = np.float32(2.0)

    mixed_batch = build_nanochat_batch_reference(
        sequences=stress_sequences,
        masks=stress_masks,
        rewards=mixed_rewards,
        assistant_end=assistant_end,
    )

    mixed_advantages = mixed_batch["advantages"]

    if not np.any(
        np.abs(mixed_advantages) > 1e-12
    ):
        raise AssertionError(
            "mixed case has no non-zero advantage"
        )

    if not np.any(
        np.abs(mixed_advantages) <= 1e-12
    ):
        raise AssertionError(
            "mixed case has no zero advantage"
        )

    mixed_result = evaluate_cpu_nanochat_rl_step(
        cpu_model=cpu_model,
        batches=[mixed_batch],
        batch_size=device_batch_size,
        examples_per_rank=1,
    )

    if not np.isfinite(mixed_result["loss"]):
        raise AssertionError(
            "mixed CPU loss is non-finite"
        )

    zero_batch = build_nanochat_batch_reference(
        sequences=stress_sequences,
        masks=stress_masks,
        rewards=np.ones(
            (stress_count,),
            dtype=np.float32,
        ),
        assistant_end=assistant_end,
    )

    zero_result = evaluate_cpu_nanochat_rl_step(
        cpu_model=cpu_model,
        batches=[zero_batch],
        batch_size=device_batch_size,
        examples_per_rank=1,
    )

    assert_cpu_zero_result(
        zero_result,
        "CPU all-zero advantage",
    )

    masked_batch = build_nanochat_batch_reference(
        sequences=stress_sequences,
        masks=[
            [0] * len(sequence)
            for sequence in stress_sequences
        ],
        rewards=mixed_rewards,
        assistant_end=assistant_end,
    )

    masked_result = evaluate_cpu_nanochat_rl_step(
        cpu_model=cpu_model,
        batches=[masked_batch],
        batch_size=device_batch_size,
        examples_per_rank=1,
    )

    assert_cpu_zero_result(
        masked_result,
        "CPU all-targets-masked",
    )

    return {
        "batches": cpu_batches,
        "actual_step": actual_step,
        "mixed_batch": mixed_batch,
        "mixed_result": mixed_result,
        "zero_batch": zero_batch,
        "zero_result": zero_result,
        "masked_batch": masked_batch,
        "masked_result": masked_result,
    }

def compare_cpu_and_metal_rl(
    cpu_result,
    metal_result,
    label,
    atol=1e-6,
    rtol=1e-6,
):
    if not np.isfinite(cpu_result["loss"]):
        raise AssertionError(
            f"{label}: CPU loss is non-finite"
        )

    if not np.isfinite(metal_result["loss"]):
        raise AssertionError(
            f"{label}: Metal loss is non-finite"
        )

    np.testing.assert_allclose(
        metal_result["loss"],
        cpu_result["loss"],
        atol=atol,
        rtol=rtol,
        err_msg=f"{label}: loss mismatch",
    )

    cpu_gradients = cpu_result["gradients"]
    metal_gradients = metal_result["gradients"]

    if set(cpu_gradients) != set(metal_gradients):
        raise AssertionError(
            f"{label}: gradient parameter set mismatch"
        )

    for name in sorted(cpu_gradients):
        cpu_gradient = np.asarray(
            cpu_gradients[name],
            dtype=np.float32,
        )

        metal_gradient = np.asarray(
            metal_gradients[name],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(cpu_gradient)):
            raise AssertionError(
                f"{label}: CPU gradient {name} is non-finite"
            )

        if not np.all(np.isfinite(metal_gradient)):
            raise AssertionError(
                f"{label}: Metal gradient {name} is non-finite"
            )

        np.testing.assert_allclose(
            metal_gradient,
            cpu_gradient,
            atol=atol,
            rtol=rtol,
            err_msg=(
                f"{label}: gradient mismatch: {name}"
            ),
        )

def validate_all_examples_metal_cpu_rollouts(
    engine,
    cpu_model,
    tokenizer,
    examples,
    num_samples,
    device_batch_size,
    max_tokens,
    temperature,
    top_k,
    optimization_step,
    reward_fn,
    assistant_end,
):
    """Run and compare the complete rollout/batch path for every example."""
    cpu_batches = []
    metal_batches = []

    for local_index, example in enumerate(examples):
        conversation = example["conversation"]
        example_index = int(example.get("example_index", local_index))
        prompt = tokenizer.render_for_completion(conversation)

        cpu_batch = build_cpu_nanochat_get_batch_reference(
            cpu_model=cpu_model,
            tokenizer=tokenizer,
            conversation=conversation,
            prompt_tokens=prompt,
            num_samples=num_samples,
            device_batch_size=device_batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            step=optimization_step,
            example_idx=example_index,
            reward_fn=reward_fn,
            assistant_end=assistant_end,
        )

        metal_rollout = generate_metal_nanochat_get_batch_rollouts(
            engine=engine,
            prompt_tokens=prompt,
            num_samples=num_samples,
            device_batch_size=device_batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            step=optimization_step,
            example_idx=example_index,
        )

        metal_rewards = np.asarray(
            [
                reward_fn(
                    conversation,
                    tokenizer.decode(sequence[len(prompt):]),
                )
                for sequence in metal_rollout["sequences"]
            ],
            dtype=np.float32,
        )

        metal_batch = build_nanochat_batch_reference(
            sequences=metal_rollout["sequences"],
            masks=metal_rollout["masks"],
            rewards=metal_rewards,
            assistant_end=assistant_end,
        )
        metal_batch["rollout_sequences"] = metal_rollout["sequences"]
        metal_batch["rollout_masks"] = metal_rollout["masks"]
        metal_batch["rewards"] = metal_rewards

        cpu_batches.append(cpu_batch)
        metal_batches.append(metal_batch)

    if len(cpu_batches) != len(examples):
        raise AssertionError("CPU example rollout count mismatch")
    if len(metal_batches) != len(examples):
        raise AssertionError("Metal example rollout count mismatch")

    for index, (cpu_batch, metal_batch) in enumerate(
        zip(cpu_batches, metal_batches)
    ):
        for field in (
            "rollout_sequences",
            "rollout_masks",
            "ids",
            "mask_ids",
            "inputs",
            "targets",
        ):
            check_equal(
                f"ALL_EXAMPLES_{index}_{field}",
                np.asarray(metal_batch[field], dtype=object),
                np.asarray(cpu_batch[field], dtype=object),
            )
        check_close(
            f"ALL_EXAMPLES_{index}_advantages",
            metal_batch["advantages"],
            cpu_batch["advantages"],
            atol=0.0,
            rtol=0.0,
        )

    print(
        "[ALL-EXAMPLES-METAL-CPU-RL-BATCH-PASS]",
        f"examples={len(examples)}",
        f"samples_per_example={num_samples}",
        flush=True,
    )
    return cpu_batches, metal_batches

def validate_extended_rl_coverage(
    *,
    cpu_reference_result,
    metal_training_result,
    cpu_tool_trace,
    metal_tool_trace,
    configuration_results,
    strict_logits_checked,
):
    """Reject incomplete coverage instead of silently reporting PASS."""
    if metal_training_result is None:
        raise AssertionError(
            "METAL_RL_TRAINING_TRACE_MISSING: "
            "Metal loss, dlogits, and gradients were not captured"
        )

    compare_cpu_and_metal_rl(
        cpu_result=cpu_reference_result["actual_step"],
        metal_result=metal_training_result,
        label="EXTENDED_METAL_CPU_RL_TRAINING",
        atol=3e-5,
        rtol=3e-5,
    )

    if not cpu_tool_trace or not metal_tool_trace:
        raise AssertionError(
            "METAL_TOOL_CALL_TRACE_MISSING: "
            "CPU and Metal forced-token traces are required"
        )

    if len(cpu_tool_trace) != len(metal_tool_trace):
        raise AssertionError(
            "METAL_TOOL_CALL_TRACE_LENGTH_MISMATCH"
        )

    saw_forced = False
    for step, (cpu_step, metal_step) in enumerate(
        zip(cpu_tool_trace, metal_tool_trace)
    ):
        check_equal(
            f"TOOL_TOKENS step={step}",
            np.asarray(metal_step["tokens"], dtype=np.int32),
            np.asarray(cpu_step["tokens"], dtype=np.int32),
        )
        check_equal(
            f"TOOL_MASKS step={step}",
            np.asarray(metal_step["token_masks"], dtype=np.int32),
            np.asarray(cpu_step["token_masks"], dtype=np.int32),
        )
        check_equal(
            f"TOOL_COMPLETED step={step}",
            np.asarray(metal_step["completed"], dtype=np.bool_),
            np.asarray(cpu_step["completed"], dtype=np.bool_),
        )
        saw_forced = saw_forced or bool(
            np.any(np.asarray(cpu_step["token_masks"]) == 0)
        )

    if not saw_forced:
        raise AssertionError(
            "METAL_TOOL_CALL_TRACE_HAS_NO_FORCED_TOKEN"
        )

    if not strict_logits_checked:
        raise AssertionError(
            "STRICT_LOGITS_VALIDATION_MISSING"
        )

    required_cases = {
        (0.0, 2, 1),
        (0.0, 2, 8),
        (1.0, 2, 8),
        (1.0, 4, 8),
        (1.0, 2, 0),
    }
    actual_cases = {
        (
            float(item["temperature"]),
            int(item["batch_size"]),
            int(item["max_new_tokens"]),
        )
        for item in configuration_results
    }
    missing = required_cases - actual_cases
    if missing:
        raise AssertionError(
            f"RL_CONFIGURATION_CASES_MISSING: {sorted(missing)}"
        )

    print(
        "[EXTENDED-METAL-CPU-RL-COVERAGE-PASS]",
        flush=True,
    )

def validate_metal_trace_artifacts(
    *,
    metal_training_result,
    cpu_tool_trace,
    metal_tool_trace,
    configuration_results,
    strict_logits_checked,
):
    """Explicit artifact gate used by the integrated checker.

    Every argument is required to come from an actual Metal execution.  The
    validator intentionally rejects ``None`` and synthetic placeholders.
    """
    if metal_training_result is None:
        raise AssertionError(
            "METAL_RL_TRAINING_TRACE_MISSING"
        )
    if cpu_tool_trace is None or metal_tool_trace is None:
        raise AssertionError(
            "METAL_TOOL_CALL_TRACE_MISSING"
        )
    if configuration_results is None:
        raise AssertionError(
            "RL_CONFIGURATION_RESULTS_MISSING"
        )
    if strict_logits_checked is not True:
        raise AssertionError(
            "STRICT_LOGITS_VALIDATION_MISSING"
        )

    return True


def run_required_rl_configuration_matrix(run_case):
    """Execute every required temperature/shape case through the same checker."""
    required = (
        {"temperature": 0.0, "batch_size": 2, "max_new_tokens": 1},
        {"temperature": 0.0, "batch_size": 2, "max_new_tokens": 8},
        {"temperature": 1.0, "batch_size": 2, "max_new_tokens": 8},
        {"temperature": 1.0, "batch_size": 4, "max_new_tokens": 8},
        {"temperature": 1.0, "batch_size": 2, "max_new_tokens": 0},
    )
    results = []
    for case in required:
        result = run_case(**case)
        if result is None:
            raise AssertionError(
                f"RL_CONFIGURATION_CASE_NOT_CAPTURED: {case}"
            )
        results.append(
            {
                **case,
                "result": result,
            }
        )
    return results

def validate_metal_native_rl_batch(cpu_batch, metal_batch):
    """Compare batches produced by the actual CPU and Metal RL batch builders."""
    required = ("ids", "mask_ids", "inputs", "targets", "advantages", "rewards")
    for field in required:
        if field not in cpu_batch or field not in metal_batch:
            raise AssertionError(f"NATIVE_RL_BATCH_FIELD_MISSING: {field}")
        actual = np.asarray(metal_batch[field])
        expected = np.asarray(cpu_batch[field])
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise AssertionError(f"NATIVE_RL_BATCH_MISMATCH: {field}")


def validate_metal_rl_training_update(cpu_result, metal_result, cpu_params_before, metal_params_before, cpu_params_after, metal_params_after):
    """Compare loss, dlogits, gradients, and parameter updates."""
    compare_cpu_and_metal_rl(
        cpu_result=cpu_result,
        metal_result=metal_result,
        label="METAL_CPU_RL_TRAINING",
        atol=3e-5,
        rtol=3e-5,
    )
    for name in sorted(cpu_params_before):
        for label, cpu_value, metal_value in (
            ("PARAM_BEFORE", cpu_params_before[name], metal_params_before[name]),
            ("PARAM_AFTER", cpu_params_after[name], metal_params_after[name]),
        ):
            np.testing.assert_allclose(
                np.asarray(metal_value, dtype=np.float32),
                np.asarray(cpu_value, dtype=np.float32),
                atol=3e-5,
                rtol=3e-5,
                err_msg=f"{label}_MISMATCH: {name}",
            )
        cpu_delta = np.asarray(cpu_params_after[name]) - np.asarray(cpu_params_before[name])
        metal_delta = np.asarray(metal_params_after[name]) - np.asarray(metal_params_before[name])
        np.testing.assert_allclose(
            metal_delta,
            cpu_delta,
            atol=3e-5,
            rtol=3e-5,
            err_msg=f"PARAM_UPDATE_MISMATCH: {name}",
        )


def validate_complete_tool_call_trace(cpu_trace, metal_trace, cpu_sequences, metal_sequences):
    """Validate tool boundaries, forced queues, masks, completion and sequences."""
    if not cpu_trace or not metal_trace:
        raise AssertionError("TOOL_TRACE_MISSING")
    if len(cpu_trace) != len(metal_trace):
        raise AssertionError("TOOL_TRACE_LENGTH_MISMATCH")
    if cpu_sequences is None or metal_sequences is None:
        raise AssertionError("TOOL_SEQUENCE_TRACE_MISSING")
    forced_seen = False
    for step, (cpu, metal) in enumerate(zip(cpu_trace, metal_trace)):
        for field in ("tokens", "token_masks", "completed"):
            if field not in cpu or field not in metal:
                raise AssertionError(f"TOOL_TRACE_FIELD_MISSING: {field}")
            check_equal(
                f"TOOL_{field}_STEP_{step}",
                np.asarray(metal[field]),
                np.asarray(cpu[field]),
            )
        forced_seen = forced_seen or bool(np.any(np.asarray(cpu["token_masks"]) == 0))
    if not forced_seen:
        raise AssertionError("TOOL_FORCED_TOKEN_NOT_EXERCISED")
    flattened_cpu = [int(token) for seq in cpu_sequences for token in seq]
    flattened_metal = [int(token) for seq in metal_sequences for token in seq]
    for required_token, label in (
        (int(PYTHON_START), "PYTHON_START"),
        (int(PYTHON_END), "PYTHON_END"),
        (int(OUTPUT_START), "OUTPUT_START"),
        (int(OUTPUT_END), "OUTPUT_END"),
    ):
        if required_token not in flattened_cpu:
            raise AssertionError(f"TOOL_CPU_TOKEN_MISSING: {label}")
        if required_token not in flattened_metal:
            raise AssertionError(f"TOOL_METAL_TOKEN_MISSING: {label}")
    check_equal("TOOL_FINAL_SEQUENCES", np.asarray(metal_sequences, dtype=object), np.asarray(cpu_sequences, dtype=object))


def validate_configuration_matrix_results(results):
    required = {
        (0.0, 2, 1), (0.0, 2, 8), (1.0, 2, 8),
        (1.0, 4, 8), (1.0, 2, 0),
    }
    actual = {
        (float(x["temperature"]), int(x["batch_size"]), int(x["max_new_tokens"]))
        for x in results
    }
    missing = required - actual
    if missing:
        raise AssertionError(f"CONFIGURATION_MATRIX_MISSING: {sorted(missing)}")
    for item in results:
        if item.get("metal_pass") is not True or item.get("cpu_pass") is not True:
            raise AssertionError(f"CONFIGURATION_CASE_NOT_VERIFIED: {item}")


def validate_strict_logits_and_topk(cpu_steps, metal_steps, select_logits, topk_reference, prompt_len, top_k):
    if len(cpu_steps) != len(metal_steps):
        raise AssertionError("STRICT_TRACE_LENGTH_MISMATCH")
    for step, (cpu, metal) in enumerate(zip(cpu_steps, metal_steps)):
        mode = "prefill" if step == 0 else "decode"
        expected = np.asarray(cpu["logits"], dtype=np.float32)
        actual = np.asarray(select_logits(metal["logits"], mode, prompt_len), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6, err_msg=f"STRICT_LOGITS_STEP_{step}")
        expected_values, expected_indices, count = topk_reference(expected, top_k)
        np.testing.assert_allclose(metal["values"][:, :count], expected_values[:, :count], atol=2e-6, rtol=2e-6, err_msg=f"STRICT_TOPK_VALUES_STEP_{step}")
        check_equal(f"STRICT_TOPK_INDICES_STEP_{step}", metal["indices"][:, :count], expected_indices[:, :count])

def _assert_zero_rl_result(result, label):
    if abs(float(result["loss"])) > 1e-6:
        raise AssertionError(
            f"{label}: expected zero loss, "
            f"got {result['loss']}"
        )

    for name, gradient in result["gradients"].items():
        if not np.allclose(
            gradient,
            0.0,
            atol=1e-6,
            rtol=0.0,
        ):
            raise AssertionError(
                f"{label}: expected zero gradient: {name}"
            )

def _align_rows(value, rows):
    value = np.asarray(value)

    if value.shape[0] == rows:
        return value

    if value.shape[0] == 1:
        return np.repeat(
            value,
            rows,
            axis=0,
        )

    raise AssertionError(
        f"cannot align rows: "
        f"value={value.shape}, rows={rows}"
    )

def generate_cpu_nanochat_get_batch_rollouts(
    cpu_model,
    tokenizer,
    prompt_tokens,
    num_samples,
    device_batch_size,
    max_tokens,
    temperature,
    top_k,
    step,
    example_idx,
):
    num_samples = int(num_samples)
    device_batch_size = int(device_batch_size)

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be positive"
        )

    if device_batch_size <= 0:
        raise ValueError(
            "device_batch_size must be positive"
        )

    num_sampling_steps = (
        num_samples // device_batch_size
    )

    if num_sampling_steps <= 0:
        raise ValueError(
            "num_samples must be at least device_batch_size"
        )

    generated_sequences = []
    generated_masks = []
    rollout_traces = []

    for sampling_step in range(num_sampling_steps):
        sampling_seed = hash(
            (
                int(step),
                int(example_idx),
                int(sampling_step),
            )
        ) & 0x7FFFFFFF

        rollout = generate_cpu_nanochat_reference(
            cpu_model=cpu_model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            num_samples=device_batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=sampling_seed,
        )

        generated_sequences.extend(
            rollout["sequences"]
        )

        generated_masks.extend(
            rollout["masks"]
        )

        rollout_traces.append(
            rollout["trace"]
        )

    expected_samples = (
        num_sampling_steps
        * device_batch_size
    )

    if len(generated_sequences) != expected_samples:
        raise AssertionError(
            "CPU get_batch rollout count mismatch"
        )

    if len(generated_masks) != expected_samples:
        raise AssertionError(
            "CPU get_batch mask count mismatch"
        )

    return {
        "sequences": generated_sequences,
        "masks": generated_masks,
        "traces": rollout_traces,
        "num_sampling_steps": num_sampling_steps,
    }


def run_cpu_forced_token_semantics_case():
    """Verify the NumPy reference state machine against nanochat's tool path.

    This test is CPU-only and does not use torch or nanochat imports. It forces
    the sequence: <python_start>, 2, <python_end>, then the calculator-generated
    <output_start>, 2, <output_end>. Tool tokens must carry mask=0.
    """
    tokenizer = SyntheticTokenizer()
    prompt = [7]
    vocab = 128
    expected_sampled = [
        PYTHON_START,
        2,
        PYTHON_END,
    ]

    class ScriptedCPUModel:
        max_seq_len = 32

        def forward(self, input_ids):
            input_ids = np.asarray(input_ids, dtype=np.int32)
            batch, time = input_ids.shape
            logits = np.zeros((batch, time, vocab), dtype=np.float32)
            step = time - len(prompt)
            token = expected_sampled[min(max(step, 0), len(expected_sampled) - 1)]
            logits[:, -1, token] = 100.0
            return logits, None

    result = generate_cpu_nanochat_reference(
        cpu_model=ScriptedCPUModel(),
        tokenizer=tokenizer,
        prompt_tokens=prompt,
        num_samples=1,
        max_tokens=6,
        temperature=0.0,
        top_k=50,
        seed=123,
    )

    sequence = result["sequences"][0]
    mask = result["masks"][0]
    expected_sequence = [
        7,
        PYTHON_START,
        2,
        PYTHON_END,
        OUTPUT_START,
        2,
        OUTPUT_END,
    ]
    expected_mask = [0, 1, 1, 1, 0, 0, 0]

    if sequence != expected_sequence:
        raise AssertionError(
            "CPU forced-token sequence mismatch: "
            f"actual={sequence}, expected={expected_sequence}"
        )
    if mask != expected_mask:
        raise AssertionError(
            "CPU forced-token mask mismatch: "
            f"actual={mask}, expected={expected_mask}"
        )

    trace = result["trace"]
    if len(trace) != 6:
        raise AssertionError(
            f"CPU forced-token step count mismatch: {len(trace)}"
        )
    if trace[3]["token_masks"][0] != 0:
        raise AssertionError("output_start must be forced with mask=0")
    if trace[4]["token_masks"][0] != 0:
        raise AssertionError("calculator result must be forced with mask=0")
    if trace[5]["token_masks"][0] != 0:
        raise AssertionError("output_end must be forced with mask=0")

    print("[CPU-FORCED-TOKEN-PASS]", flush=True)

def run_cpu_nanochat_policy_equivalence_case():
    """Exercise the complete CPU policy state machine independently.

    The scripted model emits python_start, ``2``, python_end, and then EOS.
    The CPU implementation is compared with a second transcription of the
    exact nanochat Engine.generate/generate_batch control flow.  This catches
    prompt construction, queue priority, mask values, terminal-token handling,
    and the requirement that every row (including completed rows) advances the
    decode model until the batch terminates.
    """
    tokenizer = SyntheticTokenizer()
    prompt = tokenizer.render_for_completion(
        {
            "messages": [
                {"role": "user", "content": "7"},
                {"role": "assistant", "content": ""},
            ]
        }
    )
    expected_prompt = [BOS_TOKEN, USER_START, 7, USER_END, ASSISTANT_START]
    if prompt != expected_prompt:
        raise AssertionError(
            f"nanochat prompt mismatch: actual={prompt}, expected={expected_prompt}"
        )

    schedule = [PYTHON_START, 2, PYTHON_END, EOS_TOKEN]
    vocab = 128

    class ScriptedModel:
        max_seq_len = 32

        def __init__(self):
            self.calls = 0

        def forward(self, input_ids):
            input_ids = np.asarray(input_ids, dtype=np.int32)
            batch, time = input_ids.shape
            logits = np.full((batch, time, vocab), -100.0, dtype=np.float32)
            token = schedule[min(self.calls, len(schedule) - 1)]
            logits[:, -1, token] = 100.0
            self.calls += 1
            return logits, None

    def oracle(seed):
        rng = NanochatPhilox4x32(seed)
        queues = [deque() for _ in range(2)]
        in_python = [False, False]
        expressions = [[], []]
        completed = [False, False]
        current = [list(prompt) for _ in range(2)]
        sequences = [list(prompt) for _ in range(2)]
        masks = [[0] * len(prompt) for _ in range(2)]
        trace = []
        sampled_logits = np.full((2, vocab), -100.0, dtype=np.float32)
        sampled_logits[:, schedule[0]] = 100.0
        for step in range(8):
            if all(completed):
                break
            sampled = numpy_sample_nanochat(
                sampled_logits, rng, 0.0, 50
            )
            column = []
            token_masks = []
            for row in range(2):
                was_completed = completed[row]
                forced = bool(queues[row])
                token_masks.append(0 if forced else 1)
                token = int(queues[row].popleft() if forced else sampled[row])
                column.append(token)
                current[row].append(token)
                if token in (EOS_TOKEN, BOS_TOKEN):
                    completed[row] = True
                if token == PYTHON_START:
                    in_python[row] = True
                    expressions[row] = []
                elif token == PYTHON_END and in_python[row]:
                    in_python[row] = False
                    if expressions[row]:
                        result = calculator_reference(
                            tokenizer.decode(expressions[row])
                        )
                        if result is not None:
                            result_tokens = tokenizer.encode(str(result))
                            queues[row].append(OUTPUT_START)
                            queues[row].extend(result_tokens)
                            queues[row].append(OUTPUT_END)
                    expressions[row] = []
                elif in_python[row]:
                    expressions[row].append(token)
                if (
                    not was_completed
                    and token not in (EOS_TOKEN, BOS_TOKEN)
                ):
                    sequences[row].append(token)
                    masks[row].append(token_masks[row])
            trace.append((tuple(column), tuple(token_masks), tuple(completed)))
            next_logits = np.full((2, vocab), -100.0, dtype=np.float32)
            # The first three sampled tokens are scripted.  The next loop
            # still samples every row before queue selection, exactly as
            # nanochat does; a forced token only replaces the sampled value.
            next_token = schedule[min(step + 1, len(schedule) - 1)]
            next_logits[:, next_token] = 100.0
            sampled_logits = next_logits
        return sequences, masks, trace

    model = ScriptedModel()
    actual = generate_cpu_nanochat_reference(
        cpu_model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt,
        num_samples=2,
        max_tokens=8,
        temperature=0.0,
        top_k=50,
        seed=123,
    )

    expected_sequences, expected_masks, expected_trace = oracle(123)
    if actual["sequences"] != expected_sequences:
        raise AssertionError(
            f"CPU policy sequence mismatch: actual={actual['sequences']}, expected={expected_sequences}"
        )
    if actual["masks"] != expected_masks:
        raise AssertionError(
            f"CPU policy mask mismatch: actual={actual['masks']}, expected={expected_masks}"
        )
    actual_trace = [
        (
            tuple(item["tokens"].tolist()),
            tuple(item["token_masks"].tolist()),
            tuple(item["completed"].tolist()),
        )
        for item in actual["trace"]
    ]
    if actual_trace != expected_trace:
        raise AssertionError(
            f"CPU policy trace mismatch: actual={actual_trace}, expected={expected_trace}"
        )
    print("[CPU-NANOCHAT-POLICY-PASS]", flush=True)


def run_cpu_philox_stream_case():
    """Check the seed/state contract used by temperature>0 rollouts."""
    first = NanochatPhilox4x32(123456789)
    second = NanochatPhilox4x32(123456789)
    other = NanochatPhilox4x32(123456790)
    a = first.uniform(17)
    b = second.uniform(17)
    c = other.uniform(17)
    if not np.array_equal(a, b):
        raise AssertionError("Philox stream is not reproducible for one seed")
    if np.array_equal(a, c):
        raise AssertionError("different Philox seeds produced identical stream")

    # The sampler consumes exactly one uniform per row per multinomial call;
    # temperature=0 returns before touching the generator.
    before = NanochatPhilox4x32(7)
    before_state = before.counter, before.buffer.size
    logits = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    numpy_sample_nanochat(logits, before, 0.0, 0)
    if (before.counter, before.buffer.size) != before_state:
        raise AssertionError("temperature=0 consumed RNG state")
    numpy_sample_nanochat(logits, before, 1.0, 2)
    if (before.counter, before.buffer.size) == before_state:
        raise AssertionError("temperature=1 did not consume RNG state")
    print("[CPU-PHILOX-STREAM-PASS]", flush=True)

def run_cpu_chat_rl_contract_case():
    """Check the CPU reference against the literal chat_rl contracts.

    This test deliberately computes the expected padding, target masking,
    mean-centered advantages, per-pass denominator, and sampling-pass seeds
    independently instead of reusing the reference helpers under test.
    """
    tokenizer = SyntheticTokenizer()
    assistant_end = EOS_TOKEN
    sequences = [
        [BOS_TOKEN, USER_START, 7, USER_END, ASSISTANT_START, 10, 11],
        [BOS_TOKEN, USER_START, 8, USER_END, ASSISTANT_START, 12],
    ]
    masks = [
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
    ]
    rewards = np.asarray([1.0, 3.0], dtype=np.float32)
    batch = build_nanochat_batch_reference(
        sequences=sequences,
        masks=masks,
        rewards=rewards,
        assistant_end=assistant_end,
    )

    max_length = 7
    expected_ids = np.full((2, max_length), assistant_end, dtype=np.int32)
    expected_masks = np.zeros_like(expected_ids)
    expected_ids[0, :7] = np.asarray(sequences[0], dtype=np.int32)
    expected_ids[1, :6] = np.asarray(sequences[1], dtype=np.int32)
    expected_masks[0, :7] = np.asarray(masks[0], dtype=np.int32)
    expected_masks[1, :6] = np.asarray(masks[1], dtype=np.int32)
    expected_inputs = expected_ids[:, :-1]
    expected_targets = expected_ids[:, 1:].copy()
    expected_targets[expected_masks[:, 1:] == 0] = -1
    expected_advantages = rewards - np.float32(2.0)
    check_equal("CHAT_RL_IDS", batch["ids"], expected_ids)
    check_equal("CHAT_RL_MASK_IDS", batch["mask_ids"], expected_masks)
    check_equal("CHAT_RL_INPUTS", batch["inputs"], expected_inputs)
    check_equal("CHAT_RL_TARGETS", batch["targets"], expected_targets)
    check_close("CHAT_RL_ADVANTAGES", batch["advantages"], expected_advantages)

    logits = np.asarray(
        [
            [[0.1, 1.2, -0.4, 0.7, 0.2],
             [0.6, -0.3, 1.1, 0.2, -0.1],
             [0.2, 0.4, -0.7, 0.9, 0.1]],
            [[-0.2, 0.3, 0.8, -0.1, 0.4],
             [0.9, -0.2, 0.1, 0.5, -0.6],
             [0.4, 0.2, -0.3, 0.7, 1.0]],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([[1, 2, -1], [3, -1, 4]], dtype=np.int32)
    advantages = np.asarray([0.5, -0.25], dtype=np.float32)
    objective = nanochat_objective_reference(
        logits=logits,
        targets=targets,
        advantages=advantages,
        num_passes=2,
        examples_per_rank=3,
    )
    valid = targets >= 0
    expected_valid = int(np.count_nonzero(valid))
    expected_denominator = float(expected_valid * 2 * 3)
    expected_loss_sum = np.float32(0.0)
    expected_dlogits = np.zeros_like(logits, dtype=np.float32)
    for row in range(logits.shape[0]):
        shifted = logits[row] - np.max(logits[row], axis=-1, keepdims=True)
        probs = np.exp(shifted).astype(np.float32)
        probs /= np.sum(probs, axis=-1, keepdims=True, dtype=np.float32)
        for time in range(logits.shape[1]):
            token = int(targets[row, time])
            if token < 0:
                continue
            nll = np.float32(-np.log(probs[time, token]))
            expected_loss_sum = np.float32(
                expected_loss_sum + nll * advantages[row]
            )
            expected_dlogits[row, time] = probs[time]
            expected_dlogits[row, time, token] -= np.float32(1.0)
            expected_dlogits[row, time] *= np.float32(
                advantages[row] / expected_denominator
            )
    check_equal(
        "CHAT_RL_VALID_TOKENS",
        np.asarray([objective["valid_tokens"]], dtype=np.int32),
        np.asarray([expected_valid], dtype=np.int32),
    )
    check_close(
        "CHAT_RL_DENOMINATOR",
        np.asarray([objective["denominator"]], dtype=np.float32),
        np.asarray([expected_denominator], dtype=np.float32),
    )
    check_close(
        "CHAT_RL_LOSS",
        np.asarray([objective["loss"]], dtype=np.float32),
        np.asarray([expected_loss_sum / expected_denominator], dtype=np.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    check_close(
        "CHAT_RL_DLOGITS",
        objective["dlogits"],
        expected_dlogits,
        atol=1e-6,
        rtol=1e-6,
    )

    class ConstantModel:
        max_seq_len = 64

        def forward(self, input_ids):
            input_ids = np.asarray(input_ids, dtype=np.int32)
            batch_size, time = input_ids.shape
            logits = np.full((batch_size, time, 128), -100.0, dtype=np.float32)
            logits[..., 10] = 100.0
            return logits, None

    prompt = tokenizer.render_for_completion(
        {"messages": [
            {"role": "user", "content": "7"},
            {"role": "assistant", "content": ""},
        ]}
    )
    step = 9
    example_index = 17
    rollout = generate_cpu_nanochat_get_batch_rollouts(
        cpu_model=ConstantModel(),
        tokenizer=tokenizer,
        prompt_tokens=prompt,
        num_samples=4,
        device_batch_size=2,
        max_tokens=2,
        temperature=0.0,
        top_k=50,
        step=step,
        example_idx=example_index,
    )
    expected_sequences = []
    expected_masks = []
    for sampling_step in range(2):
        seed = hash((step, example_index, sampling_step)) & 0x7FFFFFFF
        one = generate_cpu_nanochat_reference(
            cpu_model=ConstantModel(),
            tokenizer=tokenizer,
            prompt_tokens=prompt,
            num_samples=2,
            max_tokens=2,
            temperature=0.0,
            top_k=50,
            seed=seed,
        )
        expected_sequences.extend(one["sequences"])
        expected_masks.extend(one["masks"])
    if rollout["sequences"] != expected_sequences:
        raise AssertionError("chat_rl sampling-pass seed contract mismatch")
    if rollout["masks"] != expected_masks:
        raise AssertionError("chat_rl sampling-pass mask contract mismatch")
    if rollout["num_sampling_steps"] != 2:
        raise AssertionError("chat_rl num_sampling_steps contract mismatch")
    print("[CPU-CHAT-RL-CONTRACT-PASS]", flush=True)

def validate_sequence_terminal_tokens(
    cpu_sequences,
    metal_sequences,
    prompt_tokens,
    assistant_end,
    bos_token,
):
    """
    Validate terminal-token placement without treating prompt BOS as a
    generated terminal token.

    The prompt prefix is allowed to contain BOS (and any other tokenizer
    control token). Terminal checks apply only to the generated suffix. A
    generated assistant_end or BOS token must be the final token in that
    suffix if it is present.
    """
    prompt_len = len(prompt_tokens)
    terminal_tokens = {
        int(assistant_end),
        int(bos_token),
    }

    if len(cpu_sequences) != len(metal_sequences):
        raise AssertionError(
            "TERMINAL_SEQUENCE_COUNT_MISMATCH: "
            f"cpu={len(cpu_sequences)} "
            f"metal={len(metal_sequences)}"
        )

    for row, (cpu_sequence, metal_sequence) in enumerate(
        zip(cpu_sequences, metal_sequences)
    ):
        for name, sequence in (
            ("CPU", cpu_sequence),
            ("METAL", metal_sequence),
        ):
            if len(sequence) < prompt_len:
                raise AssertionError(
                    f"{name}_SEQUENCE_SHORTER_THAN_PROMPT: "
                    f"row={row} length={len(sequence)} "
                    f"prompt_len={prompt_len}"
                )

            generated_suffix = sequence[prompt_len:]

            for index, token in enumerate(generated_suffix):
                if int(token) in terminal_tokens and index + 1 < len(
                    generated_suffix
                ):
                    raise AssertionError(
                        "TOKENS_AFTER_TERMINAL_MISMATCH: "
                        f"row={row} generated_index={index} "
                        f"token={int(token)}"
                    )


def validate_complete_metal_cpu_rl_policy(
    metal_state_trace,
    metal_logits_trace,
    metal_topk_trace,
    metal_input_trace,
    metal_sequences,
    metal_masks,
    cpu_rollout,
    prompt_tokens,
    num_samples,
    top_k,
    assistant_end,
    tokenizer,
):
    """
    完整验证 Metal rollout 与 CPU RL reference。

    验证内容：

    1. sampled token；
    2. logits；
    3. TopK values/indices；
    4. decode input_ids；
    5. decode position；
    6. sequence；
    7. response mask；
    8. RL ids/mask_ids/inputs/targets/advantages；
    9. 工具 token 的 mask 语义。

    注意：
    decode step n 使用 CPU trace[n] 的 policy 后 tokens，
    而不是下一步 trace[n + 1] 的原始 sampled_tokens。
    """

    cpu_trace = cpu_rollout["trace"]
    cpu_sequences = [
        list(map(int, value))
        for value in cpu_rollout["sequences"]
    ]
    cpu_masks = [
        list(map(int, value))
        for value in cpu_rollout["masks"]
    ]

    metal_sequences = [
        list(map(int, value))
        for value in metal_sequences
    ]
    metal_masks = [
        list(map(int, value))
        for value in metal_masks
    ]

    if len(metal_state_trace) != len(cpu_trace):
        raise AssertionError(
            "STATE_TRACE_LENGTH_MISMATCH: "
            f"metal={len(metal_state_trace)} "
            f"cpu={len(cpu_trace)}"
        )

    if len(metal_logits_trace) != len(cpu_trace):
        raise AssertionError(
            "LOGITS_TRACE_LENGTH_MISMATCH: "
            f"metal={len(metal_logits_trace)} "
            f"cpu={len(cpu_trace)}"
        )

    if len(metal_topk_trace) != len(cpu_trace):
        raise AssertionError(
            "TOPK_TRACE_LENGTH_MISMATCH: "
            f"metal={len(metal_topk_trace)} "
            f"cpu={len(cpu_trace)}"
        )

    if len(metal_sequences) != len(cpu_sequences):
        raise AssertionError(
            "SEQUENCE_COUNT_MISMATCH: "
            f"metal={len(metal_sequences)} "
            f"cpu={len(cpu_sequences)}"
        )

    if len(metal_masks) != len(cpu_masks):
        raise AssertionError(
            "MASK_COUNT_MISMATCH: "
            f"metal={len(metal_masks)} "
            f"cpu={len(cpu_masks)}"
        )

    compare_metal_cpu_policy_tokens(
        metal_state_trace=metal_state_trace,
        cpu_trace=cpu_trace,
        eos_token=int(assistant_end),
        bos_token=int(tokenizer.get_bos_token_id()),
    )

    for step_index, cpu_step in enumerate(cpu_trace):
        mode = (
            "prefill"
            if step_index == 0
            else "decode"
        )

        metal_logits = select_sampling_logits(
            metal_logits_trace[step_index]["logits"],
            mode=mode,
            prompt_len=len(prompt_tokens),
        )

        cpu_logits = np.asarray(
            cpu_step["logits"],
            dtype=np.float32,
        )

        metal_logits = _align_rows(
            metal_logits,
            cpu_logits.shape[0],
        )

        check_close(
            f"METAL_CPU_LOGITS step={step_index}",
            metal_logits,
            cpu_logits,
            atol=1e-5,
            rtol=1e-5,
        )

        expected_values, expected_indices, count = (
            numpy_topk_reference(
                cpu_logits,
                top_k,
            )
        )

        actual_values = _align_rows(
            metal_topk_trace[step_index]["values"],
            cpu_logits.shape[0],
        )

        actual_indices = _align_rows(
            metal_topk_trace[step_index]["indices"],
            cpu_logits.shape[0],
        )

        check_close(
            f"METAL_CPU_TOPK_VALUES step={step_index}",
            actual_values[:, :count],
            expected_values[:, :count],
            atol=1e-5,
            rtol=1e-5,
        )

        check_equal(
            f"METAL_CPU_TOPK_INDICES step={step_index}",
            actual_indices[:, :count],
            expected_indices[:, :count],
        )

        actual_position = np.asarray(
            metal_logits_trace[step_index]["position"],
            dtype=np.int32,
        ).reshape(-1)

        expected_position = np.asarray(
            [len(prompt_tokens) - 1 + step_index],
            dtype=np.int32,
        )

        check_equal(
            f"METAL_CPU_LOGITS_POSITION step={step_index}",
            actual_position,
            expected_position,
        )

    decode_inputs = [
        item
        for item in metal_input_trace
        if item["mode"] == "decode"
    ]

    expected_decode_count = max(
        len(cpu_trace) - 1,
        0,
    )

    if len(decode_inputs) != expected_decode_count:
        raise AssertionError(
            "DECODE_INPUT_TRACE_LENGTH_MISMATCH: "
            f"metal={len(decode_inputs)} "
            f"expected={expected_decode_count}"
        )

    for decode_index, actual in enumerate(decode_inputs):
        cpu_source_step = decode_index

        expected_tokens = np.asarray(
            cpu_trace[cpu_source_step].get(
                "tokens",
                cpu_trace[cpu_source_step]["sampled_tokens"],
            ),
            dtype=np.int32,
        ).reshape(-1, 1)

        expected_tokens = expected_tokens[
            :num_samples,
            :1,
        ]

        actual_tokens = np.asarray(
            actual["input_ids"],
            dtype=np.int32,
        )[
            :num_samples,
            :1,
        ]

        check_equal(
            f"DECODE_INPUT_TOKENS step={decode_index}",
            actual_tokens,
            expected_tokens,
        )

        actual_position = np.asarray(
            actual["position"],
            dtype=np.int32,
        ).reshape(-1)

        expected_position = np.asarray(
            [len(prompt_tokens) + decode_index],
            dtype=np.int32,
        )

        check_equal(
            f"DECODE_INPUT_POSITION step={decode_index}",
            actual_position,
            expected_position,
        )

    check_equal(
        "METAL_CPU_SEQUENCES",
        np.asarray(
            metal_sequences,
            dtype=object,
        ),
        np.asarray(
            cpu_sequences,
            dtype=object,
        ),
    )

    check_equal(
        "METAL_CPU_MASKS",
        np.asarray(
            metal_masks,
            dtype=object,
        ),
        np.asarray(
            cpu_masks,
            dtype=object,
        ),
    )

    for row in range(len(cpu_sequences)):
        metal_sequence = metal_sequences[row]
        cpu_sequence = cpu_sequences[row]
        metal_mask = metal_masks[row]
        cpu_mask = cpu_masks[row]

        if len(metal_sequence) != len(metal_mask):
            raise AssertionError(
                "METAL_SEQUENCE_MASK_LENGTH_MISMATCH: "
                f"row={row}"
            )

        if len(cpu_sequence) != len(cpu_mask):
            raise AssertionError(
                "CPU_SEQUENCE_MASK_LENGTH_MISMATCH: "
                f"row={row}"
            )

        metal_mask_array = np.asarray(
            metal_mask,
            dtype=np.int32,
        )

        cpu_mask_array = np.asarray(
            cpu_mask,
            dtype=np.int32,
        )

        if np.any(
            (metal_mask_array != 0)
            & (metal_mask_array != 1)
        ):
            raise AssertionError(
                f"METAL_MASK_VALUE_INVALID row={row}"
            )

        if np.any(
            (cpu_mask_array != 0)
            & (cpu_mask_array != 1)
        ):
            raise AssertionError(
                f"CPU_MASK_VALUE_INVALID row={row}"
            )

    validate_sequence_terminal_tokens(
        cpu_sequences=cpu_sequences,
        metal_sequences=metal_sequences,
        prompt_tokens=prompt_tokens,
        assistant_end=assistant_end,
        bos_token=tokenizer.get_bos_token_id(),
    )

    rewards = np.arange(
        len(cpu_sequences),
        dtype=np.float32,
    )

    cpu_batch = build_nanochat_batch_reference(
        sequences=cpu_sequences,
        masks=cpu_masks,
        rewards=rewards,
        assistant_end=assistant_end,
    )

    metal_batch = build_nanochat_batch_reference(
        sequences=metal_sequences,
        masks=metal_masks,
        rewards=rewards,
        assistant_end=assistant_end,
    )

    check_equal(
        "METAL_CPU_RL_IDS",
        metal_batch["ids"],
        cpu_batch["ids"],
    )

    check_equal(
        "METAL_CPU_RL_MASK_IDS",
        metal_batch["mask_ids"],
        cpu_batch["mask_ids"],
    )

    check_equal(
        "METAL_CPU_RL_INPUTS",
        metal_batch["inputs"],
        cpu_batch["inputs"],
    )

    check_equal(
        "METAL_CPU_RL_TARGETS",
        metal_batch["targets"],
        cpu_batch["targets"],
    )

    check_close(
        "METAL_CPU_RL_ADVANTAGES",
        metal_batch["advantages"],
        cpu_batch["advantages"],
        atol=0.0,
        rtol=0.0,
    )

    print(
        "[METAL-CPU-RL-EQUIVALENCE-PASS]",
        f"steps={len(cpu_trace)}",
        f"rows={len(cpu_sequences)}",
        flush=True,
    )

def run_complete_metal_cpu_rl_validation(
    metal_state_trace,
    metal_logits_trace,
    metal_topk_trace,
    metal_input_trace,
    metal_sequences,
    metal_masks,
    cpu_rollout,
    prompt_tokens,
    num_samples,
    top_k,
    assistant_end,
    tokenizer,
):
    validate_complete_metal_cpu_rl_policy(
        metal_state_trace=metal_state_trace,
        metal_logits_trace=metal_logits_trace,
        metal_topk_trace=metal_topk_trace,
        metal_input_trace=metal_input_trace,
        metal_sequences=metal_sequences,
        metal_masks=metal_masks,
        cpu_rollout=cpu_rollout,
        prompt_tokens=prompt_tokens,
        num_samples=num_samples,
        top_k=top_k,
        assistant_end=assistant_end,
        tokenizer=tokenizer,
    )

def compare_metal_cpu_policy_tokens(metal_state_trace, cpu_trace, eos_token, bos_token):
    """Strict trace comparison against unchanged CPU policy outputs, including done rows.

    This checks internal trace equality separately from external sequence/mask
    equality. A different EOS convention is a reported trace difference, never
    a reason to overwrite the CPU reference token.
    """
    audit_equal('policy step count', len(metal_state_trace), len(cpu_trace))
    for step, (metal, cpu) in enumerate(zip(metal_state_trace, cpu_trace)):
        audit_equal(f'policy tokens {step}', np.asarray(metal['next_tokens']).reshape(-1),
                    np.asarray(cpu['tokens']).reshape(-1))
        finished = np.asarray(metal['finished']).reshape(-1)
        if np.any((finished != 0) & (finished != 1)):
            raise AssertionError(f'finished must be binary: step={step}')
        audit_equal(f'policy completed {step}', finished, np.asarray(cpu['completed']).reshape(-1))

def validate_metal_cpu_rl_equivalence(
    metal_state_trace,
    metal_logits_trace,
    metal_topk_trace,
    metal_input_trace,
    metal_sequences,
    metal_masks,
    cpu_rollout,
    prompt_tokens,
    num_samples,
    max_tokens,
    top_k,
    temperature,
    seed,
    assistant_end,
):
    """
    完整验证 Metal rollout 与 CPU RL reference 的一致性。

    必须传入：

    metal_state_trace:
        capture_engine_state() 产生的 Metal state trace。

    metal_logits_trace:
        capture_logits_wrapper() 产生的 Metal logits trace。

    metal_topk_trace:
        capture_logits_wrapper() 产生的 Metal TopK trace。

    metal_input_trace:
        decode 输入和 position trace。

    metal_sequences / metal_masks:
        Engine.generate_batch_new() 返回的 RL rollout 结果。

    cpu_rollout:
        generate_cpu_nanochat_reference() 的完整返回值。

    temperature=0:
        额外执行完整 state-machine 对比。

    temperature>0:
        使用 Philox 后，要求每个 sampled token 逐 token 相同。
    """

    cpu_trace = cpu_rollout["trace"]
    cpu_sequences = cpu_rollout["sequences"]
    cpu_masks = cpu_rollout["masks"]

    if len(metal_logits_trace) != len(cpu_trace):
        raise AssertionError(
            "LOGITS_TRACE_LENGTH_MISMATCH: "
            f"metal={len(metal_logits_trace)} "
            f"cpu={len(cpu_trace)}"
        )

    if len(metal_topk_trace) != len(cpu_trace):
        raise AssertionError(
            "TOPK_TRACE_LENGTH_MISMATCH: "
            f"metal={len(metal_topk_trace)} "
            f"cpu={len(cpu_trace)}"
        )

    if len(metal_state_trace) != len(cpu_trace):
        raise AssertionError(
            "STATE_TRACE_LENGTH_MISMATCH: "
            f"metal={len(metal_state_trace)} "
            f"cpu={len(cpu_trace)}"
        )

    if len(metal_sequences) != len(cpu_sequences):
        raise AssertionError(
            "SEQUENCE_BATCH_SIZE_MISMATCH: "
            f"metal={len(metal_sequences)} "
            f"cpu={len(cpu_sequences)}"
        )

    if len(metal_masks) != len(cpu_masks):
        raise AssertionError(
            "MASK_BATCH_SIZE_MISMATCH: "
            f"metal={len(metal_masks)} "
            f"cpu={len(cpu_masks)}"
        )

    compare_metal_cpu_policy_tokens(
        metal_state_trace=metal_state_trace,
        cpu_trace=cpu_trace,
        eos_token=int(assistant_end),
        bos_token=BOS_TOKEN,
    )

    for step_index, cpu_step in enumerate(cpu_trace):
        mode = (
            "prefill"
            if step_index == 0
            else "decode"
        )

        metal_logits = select_sampling_logits(
            metal_logits_trace[step_index]["logits"],
            mode=mode,
            prompt_len=len(prompt_tokens),
        )

        cpu_logits = np.asarray(
            cpu_step["logits"],
            dtype=np.float32,
        )

        metal_logits = _align_rows(
            metal_logits,
            cpu_logits.shape[0],
        )

        check_close(
            f"METAL_CPU_LOGITS step={step_index}",
            metal_logits,
            cpu_logits,
            atol=1e-5,
            rtol=1e-5,
        )

        expected_values, expected_indices, count = (
            numpy_topk_reference(
                cpu_logits,
                top_k,
            )
        )

        actual_values = _align_rows(
            metal_topk_trace[step_index]["values"],
            cpu_logits.shape[0],
        )

        actual_indices = _align_rows(
            metal_topk_trace[step_index]["indices"],
            cpu_logits.shape[0],
        )

        check_close(
            f"METAL_CPU_TOPK_VALUES step={step_index}",
            actual_values[:, :count],
            expected_values[:, :count],
            atol=1e-5,
            rtol=1e-5,
        )

        check_equal(
            f"METAL_CPU_TOPK_INDICES step={step_index}",
            actual_indices[:, :count],
            expected_indices[:, :count],
        )

        expected_position = np.asarray(
            [len(prompt_tokens) - 1 + step_index],
            dtype=np.int32,
        )

        actual_position = np.asarray(
            metal_logits_trace[step_index]["position"],
            dtype=np.int32,
        ).reshape(-1)

        check_equal(
            f"METAL_CPU_LOGITS_POSITION step={step_index}",
            actual_position,
            expected_position,
        )

    decode_inputs = [
        item
        for item in metal_input_trace
        if item["mode"] == "decode"
    ]

    expected_decode_steps = cpu_trace[:-1]

    if len(decode_inputs) != len(expected_decode_steps):
        raise AssertionError(
            "DECODE_INPUT_TRACE_LENGTH_MISMATCH: "
            f"metal={len(decode_inputs)} "
            f"cpu={len(expected_decode_steps)}"
        )

    for decode_index, (actual, cpu_step) in enumerate(
        zip(decode_inputs, expected_decode_steps)
    ):
        previous_policy_tokens = np.asarray(
            cpu_step.get(
                "tokens",
                cpu_step["sampled_tokens"],
            ),
            dtype=np.int32,
        ).reshape(-1, 1)

        previous_policy_tokens = previous_policy_tokens[
            :num_samples,
            :1,
        ]

        actual_input = np.asarray(
            actual["input_ids"],
            dtype=np.int32,
        )

        actual_input = actual_input[
            :num_samples,
            :1,
        ]

        check_equal(
            f"DECODE_INPUT_TOKENS step={decode_index}",
            actual_input,
            previous_policy_tokens,
        )

        expected_position = np.asarray(
            [len(prompt_tokens) + decode_index],
            dtype=np.int32,
        )

        actual_position = np.asarray(
            actual["position"],
            dtype=np.int32,
        ).reshape(-1)

        check_equal(
            f"DECODE_INPUT_POSITION step={decode_index}",
            actual_position,
            expected_position,
        )

    check_equal(
        "METAL_CPU_SEQUENCES",
        np.asarray(
            metal_sequences,
            dtype=object,
        ),
        np.asarray(
            cpu_sequences,
            dtype=object,
        ),
    )

    check_equal(
        "METAL_CPU_MASKS",
        np.asarray(
            metal_masks,
            dtype=object,
        ),
        np.asarray(
            cpu_masks,
            dtype=object,
        ),
    )

    for row, (sequence, mask) in enumerate(
        zip(metal_sequences, metal_masks)
    ):
        if len(sequence) != len(mask):
            raise AssertionError(
                f"METAL_SEQUENCE_MASK_LENGTH_MISMATCH "
                f"row={row}: "
                f"sequence={len(sequence)} "
                f"mask={len(mask)}"
            )

        mask_array = np.asarray(
            mask,
            dtype=np.int32,
        )

        if np.any(
            (mask_array != 0)
            & (mask_array != 1)
        ):
            raise AssertionError(
                f"METAL_MASK_VALUE_MISMATCH row={row}: "
                f"{mask_array.tolist()}"
            )

    rewards = np.arange(
        len(cpu_sequences),
        dtype=np.float32,
    )

    cpu_batch = build_nanochat_batch_reference(
        sequences=cpu_sequences,
        masks=cpu_masks,
        rewards=rewards,
        assistant_end=assistant_end,
    )

    metal_batch = build_nanochat_batch_reference(
        sequences=metal_sequences,
        masks=metal_masks,
        rewards=rewards,
        assistant_end=assistant_end,
    )

    check_equal(
        "METAL_CPU_RL_IDS",
        metal_batch["ids"],
        cpu_batch["ids"],
    )

    check_equal(
        "METAL_CPU_RL_MASK_IDS",
        metal_batch["mask_ids"],
        cpu_batch["mask_ids"],
    )

    check_equal(
        "METAL_CPU_RL_INPUTS",
        metal_batch["inputs"],
        cpu_batch["inputs"],
    )

    check_equal(
        "METAL_CPU_RL_TARGETS",
        metal_batch["targets"],
        cpu_batch["targets"],
    )

    check_close(
        "METAL_CPU_RL_ADVANTAGES",
        metal_batch["advantages"],
        cpu_batch["advantages"],
        atol=0.0,
        rtol=0.0,
    )

    if temperature == 0.0:
        cpu_logits_trace = [
            item["logits"]
            for item in cpu_trace
        ]

        cpu_state_reference = numpy_rollout_reference(
            cpu_logits_trace=cpu_logits_trace,
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            max_tokens=max_tokens,
            top_k=top_k,
            eos_token=EOS_TOKEN,
            bos_token=BOS_TOKEN,
            seed=seed,
        )

        compare_engine_trace(
            metal_state_trace=metal_state_trace,
            metal_logits_trace=metal_logits_trace,
            metal_topk_trace=metal_topk_trace,
            metal_input_trace=metal_input_trace,
            reference=cpu_state_reference,
            prompt_len=len(prompt_tokens),
            top_k=top_k,
        )

def audit_real_ce_multichunk(model,runtime,cpu):
    """Use an actual training shape with B*T>512; do not alter the allocator/kernel."""
    width=min(57,int(cpu.max_seq_len))
    if width<1:raise RuntimeError('No valid sequence width for CE coverage')
    batch_size=512//width+1
    # Avoid an exactly divisible final block under the production 512-row cap.
    while (batch_size*width)%512==0:
        batch_size+=1
    sequences=[[BOS_TOKEN]+[7+(row+i)%20 for i in range(width)] for row in range(2*batch_size)]
    masks=[[0]+[int(i%4!=0) for i in range(width)] for _ in sequences]
    rewards=np.asarray([row%2 for row in range(len(sequences))],np.float32)
    fixture=build_nanochat_batch_reference(sequences,masks,rewards,EOS_TOKEN)
    fixture['rollout_sequences']=sequences
    fixture['rollout_masks']=masks
    sink={}
    checks=AuditChecks('real GPU CE multiple chunks and short tail')
    from runtime import build_train_plan
    original_arena = runtime.unified_arena
    sizing_plans = [build_train_plan(model, batch_size, length, 'mps', owns_params=False) for length in (4, 8, 16, 32, 64)]
    required_bytes = max(plan.runtime.arena_bytes for plan in sizing_plans)
    test_arena = runtime.allocator.empty((required_bytes,), dtype=np.uint8)
    try:
    	runtime.unified_arena = test_arena
    	checks.check("chunked production training", audit_train_step, model, runtime, cpu, [fixture], batch_size, observation_sink=sink, require_multichunk=True)
    finally:
    	runtime.backend.synchronize()
    	runtime.unified_arena = original_arena
    if 'ce_chunks' not in sink:
        raise RuntimeError('Chunked training did not complete observation; ' + repr(checks.failures))
    coverage=sink['ce_chunks']
    if not coverage or not all(len(group)>1 for group in coverage):
        raise AssertionError('GPU did not execute multiple CE chunks in every group')
    if not any(group[-1][1]<group[0][1] for group in coverage):
        raise AssertionError('GPU did not execute a shorter final CE chunk')
    import json
    checks.finish()


def audit_real_nonfinite_optimizer(model,runtime,kind):
    """Real GPU gradient norm + original production skip block; no fake norm."""
    import math
    import time
    from optimizer import AdamW
    backend=runtime.backend
    params={n:t for n,t in named_tensors(model).items() if t.requires_grad}
    before={n:np.array(backend.copy_to_host(t),copy=True) for n,t in params.items()}
    grads={n:np.array(backend.copy_to_host(t.grad),copy=True) for n,t in params.items()}
    saved=Runtime._default_runtime
    checks=AuditChecks(f'real GPU {kind} norm and production skip')
    Runtime._default_runtime=runtime
    try:
        optimizer=AdamW(list(params.values()),lr=.001,state_mode='fp32')
        clip=optimizer.build_grad_clip()
        for tensor in params.values():backend.fill(tensor.grad,0.)
        first=next(iter(params.values()))
        injected=np.zeros(first.shape,np.float32)
        injected.flat[0]={'nan':np.nan,'inf':np.inf,'negative_inf':-np.inf}[kind]
        backend.copy_from_host(first.grad,injected)
        observed=[]
        def actual_clip(max_norm):
            result=clip(max_norm)
            observed.append(result)
            return result
        code=audit_optimizer_block_code()
        exec(code,dict(time=time,math=math,optimizer=optimizer,clip_grad_norm=actual_clip,
             step=1,num_steps=4,initial_lr=.001,print0=lambda *a,**k:None))
        backend.synchronize()
        checks.check('clip call count',audit_equal,'clip calls',len(observed),1)
        if observed:
            norm=observed[0][0]
            predicate=np.isnan(norm) if kind=='nan' else np.isposinf(norm)
            checks.check('real GPU nonfinite norm',audit_equal,'nonfinite norm',bool(predicate),True)
        checks.check('optimizer update must be skipped',audit_equal,'optimizer.t',optimizer.t,0)
        checks.check('skipped LR assignment',audit_equal,'optimizer.lr',optimizer.lr,.001)
        for index,(name,tensor) in enumerate(params.items()):
            checks.check(f'unchanged parameter {name}',audit_equal,name,backend.copy_to_host(tensor),before[name])
            for label,storage in (('m',optimizer.m[index]),('v',optimizer.v[index])):
                actual=backend.ctx.copy_to_host(storage.ptr.buffer_id,storage.ptr.offset,storage.shape)
                checks.check(f'unchanged {label} {name}',audit_equal,label,actual,np.zeros(storage.shape,np.float32))
    finally:
        for name,tensor in params.items():
            backend.copy_from_host(tensor,before[name])
            backend.copy_from_host(tensor.grad,grads[name])
        backend.synchronize()
        Runtime._default_runtime=saved
    checks.finish()

def audit_gpu_gradient(label, actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)

    if (
        a.shape != b.shape
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        raise AssertionError(f"{label}: shape/nonfinite mismatch")

    # 无反向贡献的整块梯度仍必须精确为零，不能容忍残留梯度。
    if not np.any(b):
        audit_equal(label, a, b)
        return

    delta = a - b
    maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
    rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    reference_rms = float(np.sqrt(np.mean(b * b))) if b.size else 0.0

    # 同时限制局部最大误差与整块误差，不能仅检查其中一个。
    limit = 1e-6 + 3e-5 * reference_rms
    if maximum > 1e-3 or rms > limit:
        raise AssertionError(
            f"{label}: max_abs={maximum:.9g}, limit=0.001; "
            f"rms={rms:.9g}, limit={limit:.9g}"
        )


def audit_model_topk_indices(actual, expected, cpu_logits):
    a = np.asarray(actual)
    b = np.asarray(expected)
    logits = np.asarray(cpu_logits)

    if (
        a.shape != b.shape
        or a.ndim != 2
        or a.shape[0] != logits.shape[0]
    ):
        raise AssertionError("model TopK index shape mismatch")

    if not np.issubdtype(a.dtype, np.integer):
        raise AssertionError("model TopK indices must be integers")

    if np.any(a < 0) or np.any(a >= logits.shape[1]):
        raise AssertionError("model TopK index outside vocabulary")

    if any(len(np.unique(row)) != len(row) for row in a):
        raise AssertionError("duplicate model TopK index")

    # 仅用于不同 CPU/GPU 前向结果之间的比较。
    # 在同一 GPU logits 上，TopK 索引仍另行严格比较。
    audit_close(
        "model TopK rank values",
        np.take_along_axis(logits, a, axis=1),
        np.take_along_axis(logits, b, axis=1),
        1e-5,
        1e-5,
    )


def audit_real_branch_cases(model,runtime,cpu,case):
    case('real GPU CE multiple chunks / short final chunk / accumulated gradients',
         lambda:audit_real_ce_multichunk(model,runtime,cpu))
    for kind in ('nan','inf','negative_inf'):
        case(f'real GPU nonfinite clip / production skip kind={kind}',
             lambda kind=kind:audit_real_nonfinite_optimizer(model,runtime,kind))


def audit_clone_cpu_state(cpu):
    """Copy a reference instance through its existing clone routine, never Metal."""
    class CPUReader:
        def copy_to_host(self,tensor):
            return np.array(tensor.storage.data,copy=True)
    return clone_model_to_cpu_reference(cpu.model,CPUReader())


def audit_optimizer_block_code():
    """Compile the original production LR/clip/skip/update block without CLI I/O."""
    import ast
    path=PROJECT_ROOT/'rl_autograd.py'
    tree=ast.parse(path.read_text(),filename=str(path))
    schedules=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_lr_multiplier']
    loops=[n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step']
    if len(schedules)!=1 or len(loops)!=1:
        raise RuntimeError('Production optimizer structure changed')
    def assigns(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    starts=[i for i,n in enumerate(loops[0].body) if assigns(n,'optimizer_start')]
    ends=[i for i,n in enumerate(loops[0].body) if assigns(n,'optimizer_dt') and starts and i>starts[0]]
    if len(starts)!=1 or len(ends)!=1:raise RuntimeError('Cannot isolate optimizer block')
    block=ast.For(target=ast.Name(id='_audit_once',ctx=ast.Store()),
                  iter=ast.Tuple(elts=[ast.Constant(value=0)],ctx=ast.Load()),
                  body=loops[0].body[starts[0]:ends[0]+1],orelse=[])
    return compile(ast.fix_missing_locations(ast.Module(body=schedules+[block],type_ignores=[])),str(path),'exec')


def audit_closed_loop(model,runtime,arena,examples,batch_size,capacity,max_tokens,
                      temperature,top_k,num_samples,start_step):
    """Two true rollout/train/update rounds with independently evolving weights.

    CPU reference routines remain unchanged. CPU optimizer state is a separate
    test oracle and never receives GPU gradients/weights after initialization.
    Real production get_batch, training loop and optimizer-control block run.
    All original GPU weights/gradients and default Runtime are restored finally.
    """
    import math
    import time
    import json
    from optimizer import AdamW
    checks=AuditChecks('two-round independent CPU/Metal RL closed loop')
    backend=runtime.backend
    params={n:t for n,t in named_tensors(model).items() if t.requires_grad}
    before={n:np.array(backend.copy_to_host(t),copy=True) for n,t in params.items()}
    saved_grads={n:np.array(backend.copy_to_host(t.grad),copy=True) for n,t in params.items()}
    saved_runtime=Runtime._default_runtime
    cpu=clone_model_to_cpu_reference(model,backend)
    expected_weights={n:np.array(t.storage.data,copy=True) for n,t in cpu.parameters.items()}
    moments={n:np.zeros_like(w) for n,w in expected_weights.items()}
    variances={n:np.zeros_like(w) for n,w in expected_weights.items()}
    tokenizer=SyntheticTokenizer()
    reward=make_checker_reward(tokenizer)
    lr,b1,b2,eps,wd=1e-3,.9,.999,1e-8,.01
    code=audit_optimizer_block_code()
    journal=[]
    oracle_updates=0
    Runtime._default_runtime=runtime
    try:
        optimizer=AdamW(list(params.values()),lr=lr,betas=(b1,b2),eps=eps,weight_decay=wd,state_mode='fp32')
        clip=optimizer.build_grad_clip()
        zero_parameter_gradients(backend,list(params.values()))
        for round_index in range(2):
            production_step=start_step+round_index
            record=dict(round=round_index,production_step=production_step,phases=[])
            journal.append(record)
            batches_sink={}
            checks.check(f'round {round_index} production rollout and batch',
                audit_native_batches,model,runtime,arena,cpu,tokenizer,examples,num_samples,
                batch_size,max_tokens,temperature,top_k,production_step,capacity,
                observation_sink=batches_sink)
            if 'batches' not in batches_sink:
                raise RuntimeError(f'round {round_index}: native batches unavailable; closed loop incomplete')
            record['phases'].append('production_rollout_and_batch')
            references=[]
            for index,example in enumerate(examples):
                conversation=example['conversation']
                references.append(build_cpu_nanochat_get_batch_reference(
                    cpu,tokenizer,conversation,tokenizer.render_for_completion(conversation),
                    num_samples,batch_size,max_tokens,temperature,top_k,production_step,index,reward,EOS_TOKEN))
            reference_hash=audit_digest(references)
            training_sink={}
            checks.check(f'round {round_index} production backward',audit_train_step,
                model,runtime,cpu,batches_sink['batches'],batch_size,reference_batches=references,
                observation_sink=training_sink,clear_initial_gradients=False)
            if not all(key in training_sink for key in ('expected','actual')):
                raise RuntimeError(f'round {round_index}: training observations unavailable; closed loop incomplete')
            record['phases'].append('production_backward')
            expected=training_sink['expected']
            expected_hash=audit_digest(expected)
            gradients=expected['gradients']
            norm=float(np.sqrt(sum(float(np.sum(np.asarray(g,np.float64)**2)) for g in gradients.values())))
            coefficient=min(1.,1./(norm+1e-6))
            effective_lr=lr*max(0.,1.-production_step/max(1,start_step+2))
            if math.isfinite(norm):
                oracle_updates+=1
                for name,weight in expected_weights.items():
                    grad=np.asarray(gradients[name]*coefficient,np.float32)
                    moments[name]=np.asarray(b1*moments[name]+(1-b1)*grad,np.float32)
                    variances[name]=np.asarray(b2*variances[name]+(1-b2)*grad*grad,np.float32)
                    decay=wd if weight.ndim>1 else 0.
                    expected_weights[name]=np.asarray(weight*(1-effective_lr*decay)-
                        (effective_lr*np.sqrt(1-b2**oracle_updates)/(1-b1**oracle_updates))*
                        moments[name]/(np.sqrt(variances[name])+eps),np.float32)
            # Advance only this independent CPU test instance using its own oracle.
            # No existing reference output or pre-existing CPU instance is changed.
            for name,tensor in cpu.parameters.items():
                np.copyto(tensor.storage.data,expected_weights[name])
            clipping=[]
            def observed_clip(max_norm):
                value=clip(max_norm)
                clipping.append(value)
                return value
            env=dict(time=time,math=math,optimizer=optimizer,clip_grad_norm=observed_clip,
                     step=production_step,num_steps=start_step+2,initial_lr=lr,
                     print0=lambda *a,**k:None)
            exec(code,env)
            backend.synchronize()
            record['phases'].append('production_optimizer_update')
            checks.check(f'round {round_index} clip invocation',audit_equal,'clip count',len(clipping),1)
            if clipping:
                checks.check(f'round {round_index} gradient norm',audit_close,'norm',clipping[0][0],norm)
                checks.check(f'round {round_index} clip coefficient',audit_close,'clip',clipping[0][1],coefficient)
            checks.check(f'round {round_index} optimizer update count',audit_equal,'optimizer.t',optimizer.t,oracle_updates)
            for index,(name,tensor) in enumerate(params.items()):
                checks.check(f'round {round_index} weight {name}',audit_close,'updated weight',
                             backend.copy_to_host(tensor),expected_weights[name])
                for label,storage,reference in (('m',optimizer.m[index],moments[name]),('v',optimizer.v[index],variances[name])):
                    actual=backend.ctx.copy_to_host(storage.ptr.buffer_id,storage.ptr.offset,storage.shape)
                    checks.check(f'round {round_index} {label} {name}',audit_close,label,actual,reference)
            checks.check(f'round {round_index} reference batches immutable',audit_equal,'hash',audit_digest(references),reference_hash)
            checks.check(f'round {round_index} reference outputs immutable',audit_equal,'hash',audit_digest(expected),expected_hash)
            record['comparisons_completed']=len(checks.attempted)
    finally:
        for name,tensor in params.items():
            backend.copy_from_host(tensor,before[name])
            backend.copy_from_host(tensor.grad,saved_grads[name])
        backend.synchronize()
        Runtime._default_runtime=saved_runtime
    checks.finish()


def audit_production_optimizer_control(step, num_steps, norm, coefficient, expected_lr, skip):
    """Execute original production control flow with recorded optimizer inputs.

    GPU clipping/AdamW arithmetic is covered separately by the real GPU tests.
    This test observes scheduling, ordering, scale forwarding and skip branches;
    it must not be reported as a GPU numerical test.
    """
    import ast
    import math
    import time
    source = PROJECT_ROOT / 'rl_autograd.py'
    tree = ast.parse(source.read_text(),filename=str(source))
    schedule = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_lr_multiplier']
    loops = [n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step']
    if len(schedule)!=1 or len(loops)!=1:
        raise RuntimeError('Production optimizer control structure changed; audit needs review')
    def assigns(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    starts=[i for i,n in enumerate(loops[0].body) if assigns(n,'optimizer_start')]
    ends=[i for i,n in enumerate(loops[0].body) if assigns(n,'optimizer_dt') and starts and i>starts[0]]
    if len(starts)!=1 or len(ends)!=1:
        raise RuntimeError('Cannot uniquely locate production optimizer block')
    statements=loops[0].body[starts[0]:ends[0]+1]
    # The outer one-iteration loop preserves the original continue semantics.
    wrapper=ast.For(target=ast.Name(id='_audit_once',ctx=ast.Store()),
                    iter=ast.Tuple(elts=[ast.Constant(value=0)],ctx=ast.Load()),body=statements,orelse=[])
    module=ast.fix_missing_locations(ast.Module(body=schedule+[wrapper],type_ignores=[]))
    events=[]
    class OptimizerObserver:
        def __init__(self): self._lr=0.0125
        @property
        def lr(self): return self._lr
        @lr.setter
        def lr(self,value):
            self._lr=value
            events.append(('lr',float(value)))
        def step(self,grad_scale): events.append(('step',float(self.lr),float(grad_scale)))
    optimizer=OptimizerObserver()
    def clip(max_norm):
        events.append(('clip',float(max_norm)))
        return norm,coefficient
    env=dict(time=time,math=math,optimizer=optimizer,clip_grad_norm=clip,
             step=step,num_steps=num_steps,initial_lr=.01,print0=lambda *a,**k:None)
    exec(compile(module,str(source),'exec'),env)
    expected=[('clip',1.)] if skip else [('clip',1.),('lr',expected_lr),('step',expected_lr,coefficient)]
    audit_equal('production optimizer call order / LR / forwarded clip scale',events,expected)
    if skip:
        audit_equal('skipped optimizer lr remains unchanged',optimizer.lr,.0125)


def audit_control_and_boundary_cases(model,runtime,arena,cpu,tokenizer,examples,capacity,seed,top_k,case):
    # Independent expected values; production get_lr_multiplier is executed as-is.
    for step,total,lr in ((0,4,.01),(1,4,.0075),(3,4,.0025),(4,4,0.),(7,4,0.),(0,0,.01)):
        case(f'production optimizer control step={step} total={total}',
             lambda step=step,total=total,lr=lr:audit_production_optimizer_control(step,total,2.,.25,lr,False))
    for name,norm in (('NaN',float('nan')),('positive infinity',float('inf')),('negative infinity',-float('inf'))):
        case(f'production optimizer skips norm={name}',
             lambda norm=norm:audit_production_optimizer_control(1,4,norm,float('nan'),None,True))
    prompt=tokenizer.render_for_completion(examples[0]['conversation'])
    # Single-row and non-power-of-two K are independent of CLI batch_size>=2.
    for k,temp in ((1,1.),(2,1.),(3,.7),(K_MAX-1,1.),(0,0.)):
        case(f'rollout single row K={k} T={temp}',lambda k=k,temp=temp:
             audit_rollout_case(model,runtime,arena,cpu,tokenizer,prompt,1,capacity,3,temp,k,seed))
    # Exercise prefill buckets around powers of two within the model capacity.
    lengths=sorted({x for p in (4,8,16,32) for x in (p-1,p,p+1) if 0<x<capacity})
    for length in lengths:
        variant=[BOS_TOKEN]+[7]*(length-1)
        case(f'prefill bucket boundary length={length}',lambda variant=variant:
             audit_rollout_case(model,runtime,arena,cpu,tokenizer,variant,2,capacity,1,0.,1,seed))



class AuditChecks:
    """Collect comparison failures; execution errors still propagate as ERROR."""
    def __init__(self, label):
        self.label = label
        self.failures = []
        self.attempted = []

    def check(self, label, function, *args, **kwargs):
        self.attempted.append(label)
        try:
            return function(*args, **kwargs)
        except AssertionError as exc:
            self.failures.append(f'{label}: {exc}')

    def wrap(self, function):
        def checked(*args, **kwargs):
            label = str(args[0]) if args and isinstance(args[0],str) else function.__name__
            return self.check(label,function,*args,**kwargs)
        return checked

    def finish(self):
        if self.failures:
            raise AssertionError(f'{self.label}: {len(self.failures)} failed comparisons; '
                                 f'{len(self.attempted)} comparisons executed\n' + '\n'.join(self.failures))


def audit_training_outputs_all(
    expected, actual, checks, atol=3e-5, rtol=3e-5
):
    checks.check(
        "total loss",
        audit_close,
        "total loss",
        actual["loss"],
        expected["loss"],
        atol,
        rtol,
    )

    cpu_grads = expected["gradients"]
    metal_grads = actual["gradients"]

    checks.check(
        "gradient names",
        audit_equal,
        "gradient names",
        sorted(metal_grads),
        sorted(cpu_grads),
    )

    for name in sorted(cpu_grads):
        if name not in metal_grads:
            checks.failures.append(f"missing Metal gradient: {name}")
            continue

        checks.check(
            f"gradient {name}",
            audit_gpu_gradient,
            f"gradient {name}",
            metal_grads[name],
            cpu_grads[name],
        )

    for name in sorted(set(metal_grads) - set(cpu_grads)):
        checks.failures.append(f"unexpected Metal gradient: {name}")


def audit_independent_optimizer_cases(model, runtime, case):
    """Same-input GPU optimizer tests independent of rollout/training success.

    These are gradient INPUT fixtures, not substituted GPU outputs. The actual
    GPU moments, clipping and updated weights are read by audit_optimizer_update.
    End-to-end training-to-optimizer validation remains registered separately.
    """
    params = {n:t for n,t in named_tensors(model).items() if t.requires_grad}
    for kind in ('zero','unclipped','clipped','sign_reversal'):
        def execute(kind=kind):
            pairs=[]
            for step in (0,1):
                grads={}
                for i,(name,tensor) in enumerate(params.items()):
                    size=int(np.prod(tensor.shape))
                    base=((np.arange(size,dtype=np.float32)%7)-3).reshape(tensor.shape)
                    scale=0. if kind=='zero' else (1e-7 if kind=='unclipped' else .125)
                    if kind=='sign_reversal' and step: scale *= -2
                    grads[name]=np.asarray(base*scale,np.float32)
                reference=dict(loss=0.,gradients=grads)
                gpu_inputs=dict(loss=0.,gradients={n:g.copy() for n,g in grads.items()})
                pairs.append((reference,gpu_inputs))
            before=audit_digest(pairs)
            audit_optimizer_update(model,runtime,*pairs[0],gradient_pairs=pairs)
            audit_equal('optimizer input fixtures immutable',audit_digest(pairs),before)
        case(f'independent GPU optimizer input={kind} / two steps / all parameters',execute)



def audit_stage_rows(value, mode, batch, label):
    value = np.asarray(value)
    rows = 1 if mode == 'prefill' else batch
    if mode not in ('prefill', 'decode') or value.ndim != 2 or value.shape[0] != rows:
        raise AssertionError(f'{label}: {mode} expected ({rows}, width), got {value.shape}')
    return np.repeat(value, batch, axis=0) if mode == 'prefill' else value


def audit_rollout_shapes(record, state, mode, batch, capacity, prompt):
    audit_equal('record mode', record['mode'], mode)
    audit_equal('state mode', state['mode'], mode)
    for name, shape in (('next_tokens',(batch,1)), ('lengths',(batch,)),
                        ('finished',(batch,)), ('generated_ids',(batch,capacity)),
                        ('rng_state',(batch,4)), ('position',(1,)), ('step',(1,)),
                        ('finished_all',(1,))):
        audit_equal(f'{name} shape', np.asarray(state[name]).shape, shape)
    ids = np.asarray(record['input_ids'])
    if mode == 'decode':
        audit_equal('decode input shape', ids.shape, (batch,1))
    else:
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < len(prompt):
            raise AssertionError(f'prefill input shape: {ids.shape}')
        expected = np.full(ids.shape, BOS_TOKEN, dtype=np.int32)
        expected[0,:len(prompt)] = prompt
        audit_equal('prefill prompt and bucket padding', ids, expected)
    audit_equal('input position shape', np.asarray(record['position']).shape, (1,))
    values = audit_stage_rows(record['values'], mode, batch, 'TopK values')
    indices = audit_stage_rows(record['indices'], mode, batch, 'TopK indices')
    audit_equal('TopK value/index shape', values.shape, indices.shape)
    if values.shape[1] != K_MAX:
        raise AssertionError(f'TopK capacity: {values.shape[1]} != {K_MAX}')


def audit_assemble_dlogits(captures, shape):
    if len(shape) != 3 or any(int(x) <= 0 for x in shape):
        raise AssertionError(f'invalid dlogits shape: {shape}')
    rows, vocab = int(shape[0])*int(shape[1]), int(shape[2])
    result = np.zeros((rows,vocab), dtype=np.float32)
    seen = np.zeros(rows, dtype=bool)
    for base, value in captures:
        value = np.asarray(value)
        if not isinstance(base, (int,np.integer)) or isinstance(base, (bool,np.bool_)):
            raise AssertionError('CE chunk base must be an integer')
        if value.ndim != 2 or value.shape[1] != vocab or len(value) == 0:
            raise AssertionError(f'invalid CE chunk shape: {value.shape}')
        end = int(base) + len(value)
        if base < 0 or end > rows:
            raise AssertionError(f'CE chunk outside [0,{rows}): [{base},{end})')
        if np.any(seen[base:end]):
            raise AssertionError('Duplicated CE chunk')
        if not np.all(np.isfinite(value)):
            raise AssertionError('Nonfinite CE dlogits')
        result[base:end] = value
        seen[base:end] = True
    if not np.all(seen):
        raise AssertionError('GPU dlogits chunks not fully captured')
    return result.reshape(shape)


def audit_masked_dlogits(dlogits, targets):
    dlogits, targets = np.asarray(dlogits), np.asarray(targets)
    if dlogits.ndim != 3 or dlogits.shape[:2] != targets.shape:
        raise AssertionError('Masked dlogits shape differs from targets')
    masked = dlogits[targets < 0]
    if np.any(masked != 0):
        raise AssertionError('Ignored targets have nonzero or nonfinite dlogits')


def audit_case(reports, name, function):
    if any(row['case'] == name for row in reports):
        raise AssertionError(f'Duplicate audit case: {name}')
    from contextlib import redirect_stdout
    try:
        with redirect_stdout(None):
            value = function()
        # value = function()
    except Exception as exc:
        status = 'FAIL' if isinstance(exc, AssertionError) else 'ERROR'
        reports.append(dict(case=name, status=status, error=f'{type(exc).__name__}: {exc}'))
        return None
    reports.append(dict(case=name, status='PASS'))
    return value


def audit_training_fixtures(batch_size, kind):
    """New test inputs passed through the unchanged CPU batch builder."""
    batches = []
    for example in range(2):
        sequences, masks = [], []
        for row in range(2*batch_size):
            width = 5 + example*3 + row % 3
            sequences.append([BOS_TOKEN] + [7+(row+i)%20 for i in range(width-1)])
            mask = [0] + [1]*(width-1)
            if kind == 'mask_holes':
                mask = [0] + [int(i%3 != 0) for i in range(1,width)]
            elif kind == 'empty_first_microbatch' and row < batch_size:
                mask = [0]*width
            elif kind == 'empty_last_microbatch' and row >= batch_size:
                mask = [0]*width
            elif kind == 'uneven_valid_counts':
                mask = [0]*width
                mask[-(1+row%3):] = [1]*(1+row%3)
            elif kind not in ('ragged_widths','empty_first_microbatch','empty_last_microbatch',
                               'uneven_valid_counts','mask_holes'):
                raise ValueError(kind)
            masks.append(mask)
        rewards = np.arange(2*batch_size, dtype=np.float32)
        fixture = build_nanochat_batch_reference(sequences,masks,rewards,EOS_TOKEN)
        fixture['rollout_sequences'] = sequences
        fixture['rollout_masks'] = masks
        batches.append(fixture)
    return batches


def audit_extra_training_cases(model, runtime, cpu, batch_size, case):
    kinds = (
        'mask_holes',
        'ragged_widths',
        'empty_first_microbatch',
        'empty_last_microbatch',
        'uneven_valid_counts',
    )

    for kind in kinds:
        def execute(kind=kind):
            batches = audit_training_fixtures(batch_size, kind)

            expected, actual = audit_train_step(
                model,
                runtime,
                cpu,
                batches,
                batch_size,
                poison_previous_gradients=True,
            )

            audit_optimizer_update(
                model,
                runtime,
                expected,
                actual,
            )

            return expected, actual

        case(
            f'production training boundary {kind}',
            execute,
        )


# This block is inserted before the checker's main guard, not imported separately.
def audit_equal(label, actual, expected):
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f'{label}: length {len(actual)} != {len(expected)}')
        for i, (a, b) in enumerate(zip(actual, expected)):
            audit_equal(f'{label}[{i}]', a, b)
        return
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise AssertionError(f'{label}: values/shapes differ: {a.shape}, {b.shape}')


def audit_close(label, actual, expected, atol=3e-5, rtol=3e-5):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise AssertionError(f'{label}: shape or nonfinite value')
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=label)


def audit_production_code():
    """Compile selected production AST only; never import CLI/training side effects."""
    import ast
    path = PROJECT_ROOT / 'rl_autograd.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    batch = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_batch']
    helper = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "choose_train_bucket"]
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'example_step']
    if len(batch) != 1 or len(loops) != 1:
        raise AssertionError('Production AST changed: expected get_batch and one example_step loop')
    if len(helper) != 1:
    	raise AssertionError("Production AST changed: expected exactly one choose_train_bucket()")
    return (compile(ast.Module(body=batch, type_ignores=[]), str(path), 'exec'),
            compile(ast.Module(body=[helper[0], loops[0]], type_ignores=[]), str(path), 'exec'))


def audit_new_engine(model, runtime, arena, tokenizer, batch, capacity):
    return Engine(model=model, tokenizer=tokenizer, device='mps', batch_size=batch,
                  max_seq_len=capacity, parameter_runtime=runtime, arena=arena,
                  max_top_k=K_MAX)


def audit_native_batches(model, runtime, arena, cpu, tokenizer, examples,
                         num_samples, batch_size, max_tokens, temperature, top_k,
                         step, capacity, observation_sink=None):
    """Execute production get_batch, not the CPU batch builder on both sides."""
    checks = AuditChecks('production native batches / all examples')
    audit_equal = checks.wrap(globals()['audit_equal'])
    import itertools
    import time
    from types import SimpleNamespace
    batch_code, _ = audit_production_code()
    reward = make_checker_reward(tokenizer)
    engine = audit_new_engine(model, runtime, arena, tokenizer, batch_size, capacity)
    recordings = []
    original = engine.generate_batch_new

    reward_calls = []
    trace_calls = []

    def record(*args, **kwargs):
        # Run the actual production call exactly once with its own Engine/seed.
        # Restore the bound method temporarily so the audit cannot recurse.
        import inspect
        call = inspect.signature(original).bind(*args, **kwargs)
        call.apply_defaults()
        options = call.arguments
        observed = {}
        engine.generate_batch_new = original
        try:
            checks.check(f'example {index} sampling pass {len(recordings)} trace',
                audit_rollout_case, model,runtime,arena,cpu,tokenizer,
                list(options['tokens']),int(options['num_samples']),capacity,
                int(options['max_tokens']),float(options['temperature']),int(options['top_k']),
                int(options['seed']),reuse_engine=engine,observation_sink=observed)
        finally:
            engine.generate_batch_new = record
        if 'sequences' not in observed or 'masks' not in observed:
            raise RuntimeError('Production rollout did not return observable outputs; batch verification incomplete')
        sequences, masks = observed['sequences'], observed['masks']
        trace_calls.append((index,len(recordings),int(options['seed']),
                            observed['captured_steps'],observed['captured_logits']))
        recordings.append((sequences, masks, options['seed']))
        return sequences, masks

    engine.generate_batch_new = record

    try:
        class Task:
            def __len__(self):
                return len(examples)
            def __getitem__(self, index):
                return examples[index]['conversation']
            def reward(self, conversation, text):
                reward_calls.append((audit_digest(conversation),text))
                return reward(conversation, text)

        env = dict(np=np, itertools=itertools, time=time, tokenizer=tokenizer,
                   train_task=Task(), engine=engine, ddp_rank=0, ddp_world_size=1,
                   step=step, rollout_seq_len=capacity - 1,
                   args=SimpleNamespace(num_samples=num_samples, device_batch_size=batch_size,
                                        max_new_tokens=max_tokens, temperature=temperature, top_k=top_k),
                   print0=lambda *a, **k: None)
        exec(batch_code, env)
        stream = env['get_batch']()
        result = []
        for index, example in enumerate(examples):
            recordings.clear()
            reward_calls.clear()
            sequence, inputs, targets, rewards, advantages = next(stream)
            prompt = tokenizer.render_for_completion(example['conversation'])
            ref = build_cpu_nanochat_get_batch_reference(
                cpu_model=cpu, tokenizer=tokenizer, conversation=example['conversation'],
                prompt_tokens=prompt, num_samples=num_samples, device_batch_size=batch_size,
                max_tokens=max_tokens, temperature=temperature, top_k=top_k,
                step=step, example_idx=index, reward_fn=reward, assistant_end=EOS_TOKEN)
            masks = [m for _, group, _ in recordings for m in group]
            expected_reward_calls = [(audit_digest(example['conversation']),
                tokenizer.decode(seq[len(prompt):])) for seq in ref['rollout_sequences']]
            audit_equal(f'example {index} reward call arguments',reward_calls,expected_reward_calls)
            audit_equal(f'example {index} actual sampling pass count',len(recordings),num_samples//batch_size)
            audit_equal(f'example {index} sequences', sequence, ref['rollout_sequences'])
            audit_equal(f'example {index} masks', masks, ref['rollout_masks'])
            audit_equal(f'example {index} rewards', rewards, ref['rewards'])
            audit_equal(f'example {index} advantages', advantages, ref['advantages'])
            audit_equal(f'example {index} seeds', [r[2] for r in recordings],
                        [hash((step, index, p)) & 0x7fffffff for p in range(num_samples // batch_size)])
            # Production pads to a fixed length; CPU reference uses the longest row.
            # Compare the meaningful prefix and require every extra target to be masked.
            width = ref['inputs'].shape[1]
            if inputs.shape != (len(sequence), capacity - 1) or targets.shape != inputs.shape:
                raise AssertionError('Production fixed batch shape mismatch')
            audit_equal('native input prefix', inputs[:, :width], ref['inputs'])
            audit_equal('native target prefix', targets[:, :width], ref['targets'])
            audit_equal('native padded targets', targets[:, width:], np.full_like(targets[:, width:], -1))
            ids = np.full((len(sequence), capacity), EOS_TOKEN, dtype=np.int32)
            mask_ids = np.zeros_like(ids)
            for row, (seq, mask) in enumerate(zip(sequence, masks)):
                if len(seq) > capacity:
                    raise AssertionError('Silent production truncation would lose response tokens')
                ids[row, :len(seq)] = seq
                mask_ids[row, :len(mask)] = mask
            expected_targets = ids[:, 1:].copy()
            expected_targets[mask_ids[:, 1:] == 0] = -1
            audit_equal('native full inputs', inputs, ids[:, :-1])
            audit_equal('native full targets', targets, expected_targets)
            result.append(dict(inputs=inputs, targets=targets, rewards=rewards,
                               advantages=advantages, rollout_sequences=sequence,
                               rollout_masks=masks, ids=ids, mask_ids=mask_ids))
        audit_equal('all production sampling passes observed',
                    [(e,p) for e,p,_,_,_ in trace_calls],
                    [(e,p) for e in range(len(examples)) for p in range(num_samples//batch_size)])
        if observation_sink is not None:
            observation_sink['batches'] = result
        checks.finish()
        return result

    finally:
        engine.generate_batch_new = original

def audit_multistep_rollout_case(
    model, runtime, arena, cpu, tokenizer, prompt, batch,
    capacity, max_tokens, temperature, top_k, seed,
    schedule=None, logits_factory=None, reuse_engine=None,
    compare_bitwise=False, observation_sink=None,
):
    """Compare committed logical row/steps to the unchanged independent CPU oracle.

    Speculative suffixes cannot be compared to a CPU prefix they did not use.
    Every live CPU row/step MUST instead have exactly one committed GPU record.
    Finished-row dummy work is observed but is not part of the external policy.
    """
    engine = reuse_engine or Engine(
        model, tokenizer, 'mps', batch, capacity, runtime, arena,
        max_top_k=K_MAX)
    checks = AuditChecks('multi-step committed policy')
    audit_equal = checks.wrap(globals()['audit_equal'])
    audit_close = checks.wrap(globals()['audit_close'])
    decode_steps = int(engine.decode_steps)
    if decode_steps < 1:
        raise AssertionError('decode_steps must be positive')
    vocab = int(model.tok_emb.weight.shape[0])
    injection = schedule is not None or logits_factory is not None

    def fixture(step, rows):
        if logits_factory is not None:
            value = np.asarray(logits_factory(step, rows, vocab), np.float32)
        else:
            ids = schedule[min(step, len(schedule) - 1)]
            value = np.full((rows, vocab), -100., np.float32)
            for row in range(rows):
                value[row, int(ids[min(row, len(ids) - 1)])] = 100.
        if value.shape != (rows, vocab) or not np.all(np.isfinite(value)):
            raise AssertionError('Invalid multi-step fixture')
        return value

    class FixtureModel:
        max_seq_len = capacity
        calls = 0
        def forward(self, ids):
            out = np.zeros((*ids.shape, vocab), np.float32)
            out[:, -1, :] = fixture(self.calls, len(ids))
            self.calls += 1
            return out, None

    # Oracle runs independently before GPU execution; never repair it from GPU state.
    expected = generate_cpu_nanochat_reference(
        FixtureModel() if injection else cpu, tokenizer, prompt, batch,
        max_tokens, temperature, top_k, seed)
    fingerprint = audit_digest(expected)
    pending, committed, boundaries, undo = [], {}, [], []
    submissions = [0]
    observed_substeps = [0]

    try:
        for mode, plan in (('prefill', engine._choose(len(prompt))),
                           ('decode', engine.decode_plan)):
            top_nodes = [n for n in plan.runtime.topo if type(n.op).__name__ == 'RolloutTopkOp']
            sample_nodes = [n for n in plan.runtime.topo if type(n.op).__name__ == 'RolloutSampleOp']
            wanted = 1 if mode == 'prefill' else decode_steps
            audit_equal(mode + ' TopK node count', len(top_nodes), wanted)
            audit_equal(mode + ' sample node count', len(sample_nodes), wanted)
            for top, sample in zip(top_nodes, sample_nodes):
                old_top, old_sample = top.forward, sample.forward
                undo.extend(((top, 'forward', old_top), (sample, 'forward', old_sample)))
                slot = {}

                def observe_top(ins, outs, old=old_top, mode=mode, plan=plan, slot=slot):
                    backend = plan.runtime.backend
                    backend.end_pass()
                    backend.synchronize()
                    state = engine.rollout_state
                    logical = (np.zeros(batch, np.int32) if mode == 'prefill' else
                               read_tensor(backend, state.decode_step).reshape(-1).copy())
                    if injection:
                        # Every substep and every row gets its own logical fixture.
                        if mode == 'prefill':
                            supplied = fixture(0, 1)
                        else:
                            supplied = np.stack([fixture(int(logical[r]), batch)[r]
                                                 for r in range(batch)])
                        value = supplied if len(ins[0].shape) == 2 else np.broadcast_to(
                            supplied[:, None, :], ins[0].shape).copy()
                        backend.copy_from_host(ins[0], value)
                    logits = audit_stage_rows(select_sampling_logits(
                        read_tensor(backend, ins[0]), mode, len(prompt)), mode, batch, 'logits')
                    record = dict(mode=mode, logical=logical, logits=logits.copy())
                    if mode == 'decode':
                        record['position'] = read_tensor(backend, state.decode_position).reshape(-1).copy()
                    backend.begin_pass()
                    old(ins, outs)
                    backend.end_pass()
                    backend.synchronize()
                    record['values'] = audit_stage_rows(read_tensor(backend, outs[0]), mode, batch, 'TopK values')
                    record['indices'] = audit_stage_rows(read_tensor(backend, outs[1]), mode, batch, 'TopK indices')
                    pending.append(record)
                    slot['record'] = record
                    observed_substeps[0] += 1
                    backend.begin_pass()

                def observe_sample(ins, outs, old=old_sample, mode=mode, plan=plan, slot=slot):
                    old(ins, outs)
                    backend = plan.runtime.backend
                    backend.end_pass()
                    backend.synchronize()
                    state = engine.rollout_state
                    raw = state.next_tokens if mode == 'prefill' else state.decode_candidates
                    slot['record']['raw'] = read_tensor(backend, raw).reshape(-1).copy()
                    backend.begin_pass()

                top.forward, sample.forward = observe_top, observe_sample

        old_policy = engine._apply_rollout_policy
        undo.append((engine, '_apply_rollout_policy', old_policy))

        def observe_commit(plan):
            backend = plan.runtime.backend
            backend.synchronize()
            state = engine.rollout_state
            before = read_tensor(backend, state.metadata).copy()
            was_done = read_tensor(backend, state.finished).reshape(-1).copy()
            count = 1 if plan is not engine.decode_plan else decode_steps
            audit_equal('all substeps captured before commit', len(pending), count)
            done = old_policy(plan)
            after = read_tensor(backend, state.metadata).copy()
            finished = read_tensor(backend, state.finished).reshape(-1).copy()
            tape = read_tensor(backend, state.tokens)
            masks = read_tensor(backend, state.mask)
            for row in range(batch):
                start, end = int(before[row, 0]), int(after[row, 0])
                if was_done[row]:
                    audit_equal('completed row g is frozen', end, start)
                    audit_equal('completed row remains finished', int(finished[row]), 1)
                    continue
                if not start <= end <= min(start + count, max_tokens):
                    raise AssertionError(f'invalid commit interval row={row}: {start}:{end}')
                for step in range(start, end):
                    record = pending[step - start]
                    audit_equal('record logical step', int(record['logical'][row]), step)
                    if record['mode'] == 'decode':
                        audit_equal('row KV position', int(record['position'][row]), len(prompt) + step - 1)
                    key = (row, step)
                    if key in committed:
                        raise AssertionError(f'duplicate committed record {key}')
                    committed[key] = dict(
                        logits=record['logits'][row].copy(), raw=int(record['raw'][row]),
                        values=record['values'][row].copy(), indices=record['indices'][row].copy(),
                        token=int(tape[row, step]), mask=int(masks[row, step]))
                if end:
                    if end > len(expected['trace']):
                        raise AssertionError(f'GPU committed past CPU trace: row={row}, end={end}')
                    audit_equal('committed finished state', int(finished[row]),
                                int(expected['trace'][end - 1]['completed'][row]))
                if not end <= int(after[row, 1]) <= max_tokens:
                    raise AssertionError('forced_end outside generation budget')
            audit_equal('policy done return', bool(done), bool(np.all(finished)))
            audit_equal('policy output tensor', int(read_tensor(backend, plan.output).reshape(-1)[0]), int(done))
            capture_engine_state(engine, boundaries, 'prefill' if plan is not engine.decode_plan else 'decode', done)
            submissions[0] += int(plan is engine.decode_plan)
            pending.clear()
            return done

        engine._apply_rollout_policy = observe_commit
        sequences, masks = engine.generate_batch_new(
            prompt, batch, max_tokens, temperature, top_k, seed)
        if observation_sink is not None:
            observation_sink.update(sequences=sequences, masks=masks, states=boundaries,
                                    captured_steps=len(boundaries), captured_logits=observed_substeps[0])
        audit_equal('multi-step sequences', sequences, expected['sequences'])
        audit_equal('multi-step masks', masks, expected['masks'])
        keys = set()
        prior_done = np.zeros(batch, bool)
        maximum_error = 0.
        bit_mismatches = 0
        for step, ref in enumerate(expected['trace']):
            for row in range(batch):
                if prior_done[row]:
                    continue
                key = (row, step)
                keys.add(key)
                if key not in committed:
                    raise AssertionError(f'missing committed GPU row/step {key}')
                actual = committed[key]
                audit_close(f'logits row={row} step={step}', actual['logits'], ref['logits'][row], 1e-5, 1e-5)
                maximum_error = max(maximum_error, float(np.max(np.abs(actual['logits'] - ref['logits'][row]))))
                bits = np.count_nonzero(actual['logits'].view(np.uint32) != ref['logits'][row].view(np.uint32))
                bit_mismatches += int(bits)
                rng = NanochatPhilox4x32(int(seed) & 0xffffffffffffffff)
                rng.uint32(step * batch)
                same_logits = np.repeat(actual['logits'][None, :], batch, axis=0)
                isolated = numpy_sample_nanochat(same_logits, rng, temperature, top_k)
                audit_equal(f'isolated raw sample {key}', actual['raw'], int(isolated[row]))
                audit_equal(f'raw sample {key}', actual['raw'], int(ref['sampled_tokens'][row]))
                audit_equal(f'policy token {key}', actual['token'], int(ref['tokens'][row]))
                audit_equal(f'policy mask {key}', actual['mask'], int(ref['token_masks'][row]))
                if 0 < top_k <= K_MAX:
                    values, indices, k = numpy_topk_reference(ref['logits'][row:row + 1], top_k)
                    iv, ii, ic = numpy_topk_reference(actual['logits'][None, :], top_k)
                    audit_close(f'isolated TopK values {key}', actual['values'][:ic], iv[0, :ic], 2e-6, 2e-6)
                    audit_equal(f'isolated TopK indices {key}', actual['indices'][:ic], ii[0, :ic])
                    audit_close(f'TopK values {key}', actual['values'][:k], values[0, :k], 1e-5, 1e-5)
                    checks.check(f'TopK indices {key}', audit_model_topk_indices,
                                 actual['indices'][None, :k], indices[:, :k], ref['logits'][row:row+1])
            prior_done = ref['completed'].astype(bool)
        audit_equal('complete committed key coverage', sorted(committed), sorted(keys))
        audit_equal('CPU reference immutability', audit_digest(expected), fingerprint)
        final_done = read_tensor(engine.decode_plan.runtime.backend, engine.rollout_state.finished).reshape(-1)
        audit_equal('final completed flags', final_done, prior_done.astype(np.uint8))
        stats = dict(sequences=sequences, masks=masks, captured_steps=len(boundaries),
                     captured_logits=observed_substeps[0], decode_submissions=submissions[0],
                     committed_row_steps=len(keys), discarded_row_steps=observed_substeps[0] * batch - len(keys),
                     max_logit_error=maximum_error, bitwise_mismatches=bit_mismatches)
        stats['states'] = boundaries  # Actual chunk boundaries, not fabricated single-step states.
        if observation_sink is not None:
            observation_sink.update(stats)
        if compare_bitwise:
            print('[FP32-BITWISE-DIAGNOSTIC]',
                  f'mismatched_elements={bit_mismatches}; pass/fail uses bounded numerical comparison', flush=True)
        checks.finish()
        return stats
    finally:
        for obj, name, original in reversed(undo):
            setattr(obj, name, original)


def audit_rollout_case(
    model, runtime, arena, cpu, tokenizer, prompt, batch,
    capacity, max_tokens, temperature, top_k, seed,
    schedule=None, logits_factory=None, reuse_engine=None,
    compare_bitwise=False, observation_sink=None,
):
    import copy

    engine = reuse_engine or audit_new_engine(
        model, runtime, arena, tokenizer, batch, capacity
    )
    if engine.decode_steps > 1:
        return audit_multistep_rollout_case(
            model, runtime, arena, cpu, tokenizer, prompt, batch, capacity,
            max_tokens, temperature, top_k, seed, schedule=schedule,
            logits_factory=logits_factory, reuse_engine=engine,
            compare_bitwise=compare_bitwise, observation_sink=observation_sink)

    vocab = int(model.tok_emb.weight.shape[0])

    failures, records, states, undo = [], [], [], []
    raw_samples, stored_masks, stored_done = [], [], []

    injection = schedule is not None or logits_factory is not None
    counter = [0]

    def verify(label, operation):
        try:
            operation()
        except (
            AssertionError, ValueError, KeyError, IndexError
        ) as exc:
            failures.append(f"{label}: {exc}")

    def fixture(step, rows):
        if logits_factory is not None:
            value = np.asarray(
                logits_factory(step, rows, vocab), np.float32
            )
            if (
                value.shape != (rows, vocab)
                or not np.all(np.isfinite(value))
            ):
                raise AssertionError("Invalid scripted-logit fixture")
            return value

        tokens = schedule[min(step, len(schedule) - 1)]
        value = np.full((rows, vocab), -100., np.float32)
        for row in range(rows):
            value[
                row, int(tokens[min(row, len(tokens) - 1)])
            ] = 100.
        return value

    class FixtureModel:
        max_seq_len = capacity
        calls = 0

        def forward(self, ids):
            out = np.zeros((*ids.shape, vocab), np.float32)
            out[:, -1, :] = fixture(self.calls, len(ids))
            self.calls += 1
            return out, None

    reference_model = FixtureModel() if injection else cpu

    try:
        plans = (
            ("prefill", engine._choose(len(prompt))),
            ("decode", engine.decode_plan),
        )

        for mode, plan in plans:
            node = next(
                n for n in plan.runtime.topo
                if type(n.op).__name__ == "RolloutTopkOp"
            )
            old = node.forward
            undo.append((node, "forward", old))

            def observe_topk(
                ins, outs, old=old, plan=plan, mode=mode
            ):
                backend = plan.runtime.backend
                backend.end_pass()
                backend.synchronize()

                if injection:
                    supplied = fixture(
                        counter[0], int(ins[0].shape[0])
                    )
                    value = (
                        supplied
                        if len(ins[0].shape) == 2
                        else np.broadcast_to(
                            supplied[:, None, :], ins[0].shape
                        ).copy()
                    )
                    backend.copy_from_host(ins[0], value)

                record = dict(
                    mode=mode,
                    logits=read_tensor(backend, ins[0]),
                    position=read_tensor(backend, ins[2]),
                    input_ids=read_tensor(backend, plan.input_ids),
                )

                backend.begin_pass()
                old(ins, outs)
                backend.end_pass()
                backend.synchronize()

                record["values"] = read_tensor(backend, outs[0])
                record["indices"] = read_tensor(backend, outs[1])
                records.append(record)
                counter[0] += 1
                backend.begin_pass()

            node.forward = observe_topk

        old_policy = engine._apply_rollout_policy
        undo.append((
            engine, "_apply_rollout_policy", old_policy
        ))

        def observe_policy(plan):
            backend = plan.runtime.backend
            backend.synchronize()

            # 读取工具队列覆盖之前的真实 GPU 抽样。
            raw_samples.append(
                read_tensor(
                    backend, engine.rollout_state.next_tokens
                ).copy()
            )

            done = old_policy(plan)

            # 读取真实持久化张量，不能用 Python 返回值伪装。
            stored_masks.append(
                read_tensor(
                    backend, engine.rollout_state.resp_mask
                ).copy()
            )
            stored_done.append(
                read_tensor(backend, plan.output).copy()
            )
            return done

        engine._apply_rollout_policy = observe_policy

        for method, mode in (
            ("_prefill_rollout", "prefill"),
            ("_decode_rollout", "decode"),
        ):
            old = getattr(engine, method)
            undo.append((engine, method, old))

            def observe_step(
                *args, old=old, mode=mode, **kwargs
            ):
                done = old(*args, **kwargs)
                capture_engine_state(
                    engine, states, mode, done
                )
                return done

            setattr(engine, method, observe_step)

        sequences, masks = engine.generate_batch_new(
            prompt,
            num_samples=batch,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )

        if observation_sink is not None:
            observation_sink.update(
                sequences=copy.deepcopy(sequences),
                masks=copy.deepcopy(masks),
                captured_steps=len(states),
                captured_logits=len(records),
                states=copy.deepcopy(states),
            )

        expected = generate_cpu_nanochat_reference(
            cpu_model=reference_model,
            tokenizer=tokenizer,
            prompt_tokens=prompt,
            num_samples=batch,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        fingerprint = audit_digest(expected)

        verify(
            "external sequences",
            lambda: audit_equal(
                "sequence", sequences, expected["sequences"]
            ),
        )
        verify(
            "external masks",
            lambda: audit_equal(
                "mask", masks, expected["masks"]
            ),
        )

        for name, items in (
            ("states", states),
            ("logits", records),
            ("raw samples", raw_samples),
            ("stored masks", stored_masks),
            ("output", stored_done),
        ):
            verify(
                name + " count",
                lambda items=items: audit_equal(
                    "count", len(items), len(expected["trace"])
                ),
            )

        verify(
            "internal policy trace",
            lambda: compare_metal_cpu_policy_tokens(
                states, expected["trace"], EOS_TOKEN, BOS_TOKEN
            ),
        )

        generated = [[] for _ in range(batch)]
        response_masks = [[] for _ in range(batch)]
        previous_done = np.zeros(batch, dtype=bool)
        sampler_rng = NanochatPhilox4x32(seed)
        numeric_errors, bitwise_mismatches = [], []

        for step, (record, state, ref) in enumerate(
            zip(records, states, expected["trace"])
        ):
            mode = "prefill" if step == 0 else "decode"

            verify(
                f"shapes {step}",
                lambda: audit_rollout_shapes(
                    record, state, mode, batch, capacity, prompt
                ),
            )

            actual_logits = audit_stage_rows(
                select_sampling_logits(
                    record["logits"], mode, len(prompt)
                ),
                mode,
                batch,
                "sampling logits",
            )
            numeric_errors.append(float(np.max(
                np.abs(actual_logits - ref["logits"])
            )))

            verify(
                f"logits {step}",
                lambda: audit_close(
                    "logits",
                    actual_logits,
                    ref["logits"],
                    1e-5,
                    1e-5,
                ),
            )

            if compare_bitwise:
                actual_bits = np.ascontiguousarray(
                    actual_logits, dtype=np.float32
                ).view(np.uint32)
                reference_bits = np.ascontiguousarray(
                    ref["logits"], dtype=np.float32
                ).view(np.uint32)
                bitwise_mismatches.append(int(
                    np.count_nonzero(actual_bits != reference_bits)
                ))

            # 原 CPU sampler 接收相同 GPU logits，隔离采样器错误。
            # 不修改独立 CPU 模型产生的 ref。
            sampled = numpy_sample_nanochat(
                actual_logits, sampler_rng, temperature, top_k
            )
            verify(
                f"raw sampler all rows {step}",
                lambda: audit_equal(
                    "raw sample",
                    np.asarray(raw_samples[step]).reshape(-1),
                    sampled,
                ),
            )

            # 全词表/大 K 的覆盖由上面的实际抽样检查证明，
            # 不要求固定 64 宽快路径数组容纳整个词表。
            wanted = vocab if top_k <= 0 else min(top_k, vocab)
            if 0 < top_k and wanted <= K_MAX:
                values, indices, count = numpy_topk_reference(
                    ref["logits"], top_k
                )
                iv, ii, ic = numpy_topk_reference(
                    actual_logits, top_k
                )
                av = audit_stage_rows(
                    record["values"], mode, batch, "TopK values"
                )
                ai = audit_stage_rows(
                    record["indices"], mode, batch, "TopK indices"
                )

                verify(
                    f"isolated TopK values {step}",
                    lambda: audit_close(
                        "TopK on GPU logits",
                        av[:, :ic], iv[:, :ic], 2e-6, 2e-6
                    ),
                )
                verify(
                    f"isolated TopK indices {step}",
                    lambda: audit_equal(
                        "TopK on GPU logits",
                        ai[:, :ic], ii[:, :ic]
                    ),
                )
                verify(
                    f"topk values {step}",
                    lambda: audit_close(
                        "topk values",
                        av[:, :count], values[:, :count],
                        1e-5, 1e-5,
                    ),
                )
                verify(
                    f"topk indices {step}",
                    lambda: audit_model_topk_indices(
                        ai[:, :count],
                        indices[:, :count],
                        ref["logits"],
                    ),
                )

            for row, token in enumerate(ref["tokens"]):
                if (
                    not previous_done[row]
                    and int(token) not in (EOS_TOKEN, BOS_TOKEN)
                ):
                    generated[row].append(int(token))
                    response_masks[row].append(
                        int(ref["token_masks"][row])
                    )

            verify(
                f"lengths {step}",
                lambda: audit_equal(
                    "lengths",
                    np.asarray(state["lengths"]).reshape(-1),
                    np.array([len(x) for x in generated]),
                ),
            )

            for row, response in enumerate(generated):
                verify(
                    f"generated ids {step}/{row}",
                    lambda row=row, response=response: audit_equal(
                        "response",
                        list(np.asarray(state["generated_ids"])[
                            row, :len(response)
                        ]),
                        response,
                    ),
                )
                verify(
                    f"persistent mask {step}/{row}",
                    lambda row=row: audit_equal(
                        "response mask",
                        np.asarray(stored_masks[step])[
                            row, :len(response_masks[row])
                        ],
                        np.asarray(response_masks[row], np.int32),
                    ),
                )

            done = np.array(
                [int(np.all(ref["completed"]))], np.int32
            )
            verify(
                f"all done {step}",
                lambda: audit_equal(
                    "all done",
                    np.asarray(state["finished_all"]).reshape(-1),
                    done,
                ),
            )
            verify(
                f"actual output {step}",
                lambda: audit_equal(
                    "output",
                    np.asarray(stored_done[step]).reshape(-1),
                    done,
                ),
            )
            verify(
                f"position {step}",
                lambda: audit_equal(
                    "position",
                    np.asarray(state["position"]).reshape(-1),
                    np.array([len(prompt) + step]),
                ),
            )
            verify(
                f"step {step}",
                lambda: audit_equal(
                    "step",
                    np.asarray(state["step"]).reshape(-1),
                    np.array([step + 1]),
                ),
            )

            if step:
                verify(
                    f"decode token {step}",
                    lambda: audit_equal(
                        "input",
                        record["input_ids"],
                        np.asarray(
                            expected["trace"][step - 1]["tokens"]
                        ).reshape(-1, 1),
                    ),
                )
            else:
                verify(
                    "prefill input",
                    lambda: audit_equal(
                        "prompt",
                        list(record["input_ids"][0, :len(prompt)]),
                        list(prompt),
                    ),
                )

            verify(
                f"input position {step}",
                lambda: audit_equal(
                    "input position",
                    np.asarray(record["position"]).reshape(-1),
                    np.array([len(prompt) - 1 + step]),
                ),
            )
            previous_done = np.asarray(ref["completed"]).copy()

        if max_tokens == 0:
            verify(
                "zero-token trace",
                lambda: audit_equal(
                    "sampling count", len(records), 0
                ),
            )

        verify(
            "CPU reference immutable",
            lambda: audit_equal(
                "fingerprint", audit_digest(expected), fingerprint
            ),
        )

        if compare_bitwise:
            print(
                "[FP32-BITWISE-DIAGNOSTIC]",
                f"mismatched_elements={sum(bitwise_mismatches)}; "
                "pass/fail uses bounded numerical comparison",
                flush=True,
            )

        if failures:
            raise AssertionError("\n".join(failures))

        return dict(
            sequences=copy.deepcopy(sequences),
            masks=copy.deepcopy(masks),
            states=copy.deepcopy(states),
            max_logit_error=max(numeric_errors, default=0.),
            bitwise_mismatched_elements=sum(bitwise_mismatches),
        )

    finally:
        for obj, attribute, old in reversed(undo):
            setattr(obj, attribute, old)

def build_audit_train_plans(model, runtime, batches, batch_size):
    """
    建立验证用固定长度 train_plan。

    固定桶选择规则：
        required_length <= bucket_length

    不把 batch 的原始宽度直接当作桶。
    例如当前 batch 宽度为 6，固定桶为 4/8/16，
    则实际使用 T=8。
    """
    from runtime import build_train_plan

    if not batches:
        raise AssertionError(
            "cannot build train plans from empty batches"
        )

    shared_arena = runtime.unified_arena
    if shared_arena is None:
        raise AssertionError(
            "runtime has no shared training arena"
        )

    max_input_width = max(
        int(batch["inputs"].shape[1])
        for batch in batches
    )

    # 小规模 checker 使用缩放后的固定桶，
    # 使 max-seq-len=64 的测试也能覆盖“向上取最近桶”。
    #
    # 真实生产长度较大时使用生产桶。
    if max_input_width < 128:
        fixed_bucket_lengths = (
            4,
            8,
            16,
            32,
            64,
        )
    else:
        fixed_bucket_lengths = (
            128,
            192,
            256,
            320,
            384,
            448,
            512,
        )

    bucket_lengths = [
        length
        for length in fixed_bucket_lengths
        if length >= max_input_width
    ]

    if not bucket_lengths:
        # 只有输入宽度超过最大固定桶时，
        # 才使用最大输入宽度作为最终容量桶。
        bucket_lengths = [max_input_width]

    # 只保留不超过最终桶的候选长度。
    max_bucket = bucket_lengths[0]
    bucket_lengths = [
        length
        for length in fixed_bucket_lengths
        if length <= max_bucket
    ]

    if max_bucket not in bucket_lengths:
        bucket_lengths.append(max_bucket)

    bucket_lengths = sorted(set(bucket_lengths))

    plans = []

    for length in bucket_lengths:
        plan = build_train_plan(
            model=model,
            batch_size=batch_size,
            seq_len=length,
            device="mps",
            owns_params=False,
        )

        if plan.runtime.arena_bytes > shared_arena.nbytes:
            raise AssertionError(
                f"train bucket T={length} requires "
                f"{plan.runtime.arena_bytes} bytes, "
                f"shared arena has {shared_arena.nbytes} bytes"
            )

        # 所有固定桶共享同一个原始训练 arena。
        plan.bind_to(shared_arena)
        plans.append(plan)

    return tuple(plans)

def pad_audit_batches_for_train_buckets(
    batches,
    execution_width,
):
    """
    只生成 Metal 生产训练使用的补齐副本。

    CPU reference 使用的 batches 完全不修改。
    """
    padded_batches = []

    for batch in batches:
        inputs = np.asarray(
            batch["inputs"],
            dtype=np.int32,
        )
        targets = np.asarray(
            batch["targets"],
            dtype=np.int32,
        )

        if inputs.ndim != 2 or targets.ndim != 2:
            raise AssertionError(
                "audit train inputs/targets must be rank-2"
            )

        if inputs.shape != targets.shape:
            raise AssertionError(
                "audit train inputs/targets shape mismatch"
            )

        batch_size, width = inputs.shape

        if width > execution_width:
            raise AssertionError(
                f"batch width {width} exceeds "
                f"execution width {execution_width}"
            )

        padded_inputs = np.zeros(
            (batch_size, execution_width),
            dtype=np.int32,
        )
        padded_targets = np.full(
            (batch_size, execution_width),
            -1,
            dtype=np.int32,
        )

        padded_inputs[:, :width] = inputs
        padded_targets[:, :width] = targets

        padded_batch = dict(batch)
        padded_batch["inputs"] = padded_inputs
        padded_batch["targets"] = padded_targets

        # rollout_sequences、masks、rewards、advantages
        # 保持原对象内容，不修改 CPU reference 数据。
        padded_batches.append(padded_batch)

    return padded_batches

def build_expected_bucket_schedule(
    batches,
    production_batches,
    observed_train_plans,
    batch_size,
):
    """
    根据生产 choose_train_bucket() 的规则，
    构造验证器期望的 train 输入形状。

    CPU reference 不参与修改。
    production_batches 只是补齐后的 Metal 输入副本。
    """
    expected_calls = []

    for example_index, batch in enumerate(batches):
        production_batch = production_batches[example_index]

        sequences = batch["rollout_sequences"]
        inputs = production_batch["inputs"]
        targets = production_batch["targets"]

        num_passes = inputs.shape[0] // batch_size

        for pass_index in range(num_passes):
            b0 = pass_index * batch_size
            b1 = b0 + batch_size

            required_length = 1
            for row in range(b0, b1):
                required = len(sequences[row]) - 1
                if required > required_length:
                    required_length = required

            selected_plan = observed_train_plans[-1]

            for candidate in observed_train_plans:
                if required_length <= candidate.seq_len:
                    selected_plan = candidate
                    break

            bucket_length = int(selected_plan.seq_len)

            inputs_pass = inputs[
                b0:b1,
                :bucket_length,
            ]
            targets_pass = targets[
                b0:b1,
                :bucket_length,
            ]

            # 与生产训练循环一致：
            # 全部 target 都被 mask 时不执行 backward。
            has_valid_target = False
            for value in targets_pass.reshape(-1):
                if int(value) >= 0:
                    has_valid_target = True
                    break

            if not has_valid_target:
                continue

            expected_calls.append(
                (
                    inputs_pass.copy(),
                    targets_pass.copy(),
                    bool(len(expected_calls)),
                )
            )

    return expected_calls

def compare_bucket_dlogits(
    actual_dlogits,
    reference_dlogits,
    bucket_targets,
    audit_close,
    audit_equal,
    label,
):
    """
    比较固定桶 Metal dlogits 与原始 CPU reference dlogits。

    允许两种情况：

      1. Metal bucket T 小于 CPU reference T：
         CPU 被截掉的尾部必须全部是 padding，dlogits 必须为 0。

      2. Metal bucket T 大于 CPU reference T：
         Metal 多出来的固定桶尾部必须全部是 padding，dlogits 必须为 0。

    CPU reference 本身不被修改。
    """
    actual_dlogits = np.asarray(
        actual_dlogits,
        dtype=np.float32,
    )
    reference_dlogits = np.asarray(
        reference_dlogits,
        dtype=np.float32,
    )
    bucket_targets = np.asarray(bucket_targets)

    if actual_dlogits.ndim != 3:
        raise AssertionError(
            f"{label}: actual dlogits must be rank 3, "
            f"got {actual_dlogits.shape}"
        )

    if reference_dlogits.ndim != 3:
        raise AssertionError(
            f"{label}: reference dlogits must be rank 3, "
            f"got {reference_dlogits.shape}"
        )

    if bucket_targets.ndim != 2:
        raise AssertionError(
            f"{label}: bucket targets must be rank 2, "
            f"got {bucket_targets.shape}"
        )

    if actual_dlogits.shape[:2] != bucket_targets.shape:
        raise AssertionError(
            f"{label}: actual dlogits shape "
            f"{actual_dlogits.shape[:2]} does not match "
            f"bucket targets {bucket_targets.shape}"
        )

    if (
        actual_dlogits.shape[0] != reference_dlogits.shape[0]
        or actual_dlogits.shape[2] != reference_dlogits.shape[2]
    ):
        raise AssertionError(
            f"{label}: batch/vocab shape mismatch: "
            f"actual={actual_dlogits.shape}, "
            f"reference={reference_dlogits.shape}"
        )

    actual_width = actual_dlogits.shape[1]
    reference_width = reference_dlogits.shape[1]
    common_width = min(actual_width, reference_width)

    # 固定桶与 CPU reference 的共同有效前缀。
    audit_close(
        label,
        actual_dlogits[:, :common_width, :],
        reference_dlogits[:, :common_width, :],
    )

    # Metal bucket 比 CPU reference 更长：
    # 多出来的固定桶位置必须是 padding，dlogits 必须为零。
    if actual_width > reference_width:
        actual_tail = actual_dlogits[
            :,
            reference_width:,
            :,
        ]

        actual_tail_targets = bucket_targets[
            :,
            reference_width:,
        ]

        if np.any(actual_tail_targets >= 0):
            raise AssertionError(
                f"{label}: fixed-bucket tail contains "
                "valid targets"
            )

        audit_equal(
            f"{label} fixed-bucket tail",
            actual_tail,
            np.zeros_like(actual_tail),
        )

    # CPU reference 比 Metal bucket 更长：
    # 被 Metal bucket 截掉的 CPU 尾部必须全部是 padding。
    if reference_width > actual_width:
        reference_tail = reference_dlogits[
            :,
            actual_width:,
            :,
        ]

        audit_equal(
            f"{label} omitted CPU tail",
            reference_tail,
            np.zeros_like(reference_tail),
        )

def audit_train_step(model, runtime, cpu, batches, batch_size, reference_batches=None, poison_previous_gradients=False, observation_sink=None, clear_initial_gradients=True, require_multichunk=False):
    """Execute the production grouping loop; observe actual GPU CE chunks and grads."""
    checks = AuditChecks('production training / all microbatches and parameters')
    audit_equal = checks.wrap(globals()['audit_equal'])
    audit_close = checks.wrap(globals()['audit_close'])
    audit_masked_dlogits = checks.wrap(globals()['audit_masked_dlogits'])
    import time
    from types import SimpleNamespace
    from runtime import build_train_plan
    _, train_code = audit_production_code()
    # Adapt the Metal graph to CPU reference shapes; never pad/mutate CPU batches.
    real_train_plans = build_audit_train_plans(
        model=model,
        runtime=runtime,
        batches=batches,
        batch_size=batch_size,
    )
    plans = {
        int(plan.seq_len): plan
        for plan in real_train_plans
    }
    execution_width = max(int(plan.seq_len) for plan in real_train_plans)
    production_batches = pad_audit_batches_for_train_buckets(
        batches,
        execution_width,
    )
    backend = runtime.backend
    if any(plan.runtime.backend is not backend for plan in plans.values()):
        raise AssertionError('Expected shared Metal backend for observed plans')
    parameters = {name: t for name, t in named_tensors(model).items() if t.requires_grad}
    batch_fingerprint = audit_digest(batches)
    reference_fingerprint = audit_digest(reference_batches)
    if clear_initial_gradients:
        zero_parameter_gradients(backend, list(parameters.values()))
    if poison_previous_gradients:
        for parameter in parameters.values():
            backend.fill(parameter.grad, 1.0)  # independent test input: stale previous-step gradients
    expected = evaluate_cpu_nanochat_rl_step(
        cpu, batches if reference_batches is None else reference_batches,
        batch_size, len(batches))
    # A separate unchanged CPU model instance checks every accumulated prefix.
    # It cannot overwrite the original full-step reference gradients/results.
    prefix_cpu = audit_clone_cpu_state(cpu)
    prefix_cpu.zero_grad()
    expected_fingerprint = audit_digest(expected)
    captures = []
    ce_chunks = []
    micro_dlogits = {}
    micro_loss = {}
    # Observe the actual chunked GPU dlogits immediately after the CE dispatch.
    original_ctx = backend.ctx
    class ContextObserver:
        def __getattr__(self, name):
            return getattr(original_ctx, name)
        def dispatch_with_args_3d(self, kernel, buffers, offsets, args, *grid):
            result = original_ctx.dispatch_with_args_3d(kernel, buffers, offsets, args, *grid)
            if kernel in ('fused_linear_ce_logits_to_dlogits', "fused_linear_ce_logits_to_dlogits_row_scaled"):
                backend.end_pass()
                backend.synchronize()
                rows, vocab, base = map(int, args[:3])
                array = original_ctx.copy_to_host(buffers[0], offsets[0], (rows, vocab))
                captures.append((base, np.array(array, dtype=np.float32, copy=True)))
                backend.begin_pass()
            return result
    backend.ctx = ContextObserver()
    loss_sum = [0.0]
    calls = []
    executed_microbatches = []
    class ObservedBucketPlan:
	    def __init__(self, observer, real_plan):
	        self.observer = observer
	        self.real_plan = real_plan
	        self.seq_len = int(real_plan.seq_len)

	    def set_inputs(self, value):
	        self.observer.plan = self.real_plan
	        self.observer.set_inputs(value)

	    def set_targets(self, value):
	        self.observer.plan = self.real_plan
	        self.observer.set_targets(value)

	    def set_row_scale(self, value):
	        self.observer.plan = self.real_plan
	        self.observer.set_row_scale(value)

	    def backward(
	        self,
	        grad_scale=1.0,
	        accumulate_parameter_grads=False,
	    ):
	        self.observer.plan = self.real_plan
	        self.observer.backward(
	            grad_scale=grad_scale,
	            accumulate_parameter_grads=(
	                accumulate_parameter_grads
	            ),
	        )

    class ObservedPlan:
        def __init__(self):
        	self.inputs = None
        	self.targets = None
        	self.row_scale = None
        	self.plan = None

        def set_inputs(self, value):
            self.inputs = value.copy()
            width = int(value.shape[1])
            if width not in plans:
            	raise AssertionError(f"production selected unprepared train bucket T={width}")
            self.plan = plans[width]
            # self.plan = plans[int(value.shape[1])]
            self.plan.set_inputs(value)
        def set_targets(self, value):
            self.targets = value.copy()
            self.plan.set_targets(value)
        def set_row_scale(self, value):
        	self.row_scale = value.copy()
        	self.plan.set_row_scale(value)
        def backward(
            self, grad_scale=1.0, accumulate_parameter_grads=False
        ):
            captures.clear()
            self.plan.backward(
                grad_scale=grad_scale,
                accumulate_parameter_grads=accumulate_parameter_grads,
            )
            backend.synchronize()

            raw_loss = float(
                np.asarray(
                    backend.copy_to_host(self.plan.output)
                ).reshape(-1)[0]
            )

            example_index = int(env["example_step"])
            pass_index = int(env["pass_idx"])
            reference_index = pass_index

            for previous_batch in batches[:example_index]:
            	reference_index += (len(previous_batch["inputs"]) // batch_size)

            executed_microbatches.append((example_index, pass_index, reference_index))

            reference_microbatch = expected["microbatches"][reference_index]
            reference_dlogits = np.asarray(reference_microbatch["dlogits"], dtype=np.float32)
            reference_loss = float(reference_microbatch["loss"])

            ce_chunks.append([
                (int(base), int(len(value)))
                for base, value in captures
            ])
            if require_multichunk:
                audit_equal(
                    f"backward {len(calls)} multiple CE chunks executed",
                    len(captures) > 1,
                    True,
                )
            metal_shape = (self.targets.shape[0], self.targets.shape[1], int(model.lm_head.weight.shape[0]))
            observed = audit_assemble_dlogits(
                captures, metal_shape,
            )
            audit_masked_dlogits(observed, self.targets)
            compare_bucket_dlogits(actual_dlogits=observed,
            	                   reference_dlogits=reference_dlogits,
            	                   bucket_targets=self.targets,
            	                   audit_close=audit_close,
            	                   audit_equal=audit_equal,
            	                   label="GPU grouped dlogits")
            audit_close("GPU row-scaled loss", raw_loss, reference_loss)

            loss_sum[0] += raw_loss

            key = (example_index, pass_index)
            if key not in micro_dlogits:
                micro_dlogits[key] = np.zeros_like(observed)
                micro_loss[key] = 0.0

            micro_dlogits[key] += observed
            micro_loss[key] += raw_loss

            # _, prefix_context = prefix_cpu.forward(self.inputs)
            _, prefix_context = prefix_cpu.forward(reference_microbatch["inputs"])
            prefix_cpu.backward(
                context=prefix_context,
                dlogits=reference_dlogits,
            )
            prefix_gradients = prefix_cpu.gradients()

            for parameter_name, parameter in parameters.items():
                label = (
                    f"backward prefix {len(calls)} "
                    f"gradient {parameter_name}"
                )
                checks.check(
                    label,
                    audit_gpu_gradient,
                    label,
                    np.array(
                        backend.copy_to_host(parameter.grad),
                        copy=True,
                    ),
                    prefix_gradients[parameter_name],
                )

            calls.append((
                self.inputs.copy(),
                self.targets.copy(),
                self.row_scale.copy(),
                grad_scale,
                accumulate_parameter_grads,
            ))

    # values = [(b['rollout_sequences'], b['inputs'], b['targets'], b['rewards'], b['advantages']) for b in batches]
    values = [(b["rollout_sequences"], b["inputs"], b["targets"], b["rewards"], b["advantages"]) for b in production_batches]

    observer = ObservedPlan()
    observed_train_plans = tuple(ObservedBucketPlan(observer=observer, real_plan=plans[length]) for length in sorted(plans))
    audit_engine = SimpleNamespace(enable_debug=False)
    env = dict(np=np, time=time, args=SimpleNamespace(device_batch_size=batch_size),
               examples_per_rank=len(batches), batch_iterator=iter(values),
               train_plans=observed_train_plans,
               train_plan=observer, 
               engine=audit_engine,
               first_backward=True, rewards_list=[], sequence_lengths=[],
               print0=lambda *a, **k: None, step=0, num_steps=1)
    try:
        exec(train_code, env)
    finally:
        backend.ctx = original_ctx
    # Independent expected group schedule checks denominator and accumulation across examples.
    expected_calls = build_expected_bucket_schedule(batches=batches,
    	                                            production_batches=production_batches,
    	                                            observed_train_plans=observed_train_plans,
    	                                            batch_size=batch_size)
    audit_equal('group schedule length', len(calls), len(expected_calls))
    for k, (a, b) in enumerate(zip(calls, expected_calls)):
        audit_equal(f'group inputs {k}', a[0], b[0])
        audit_equal(f'group targets {k}', a[1], b[1])
        audit_equal(f"group accumulation {k}", a[4], b[2])

    audit_equal("executed microbatch count", len(calls), len(expected_calls))
    audit_equal("observed microbatch count", len(executed_microbatches), len(calls))
    for cursor, (example_index, pass_index, reference_index) in enumerate(executed_microbatches):
    	ref = expected["microbatches"][reference_index]
    	key = (example_index, pass_index)
    	actual_dlogits = micro_dlogits.get(key)
    	if actual_dlogits is None:
    		expected_width = expected_calls[cursor][0].shape[1]
    		actual_dlogits = np.zeros((batch_size, expected_width, model.tok_emb.weight.shape[0]), dtype=np.float32)

    	bucket_targets = expected_calls[cursor][1]
    	audit_masked_dlogits(actual_dlogits, bucket_targets)
    	compare_bucket_dlogits(actual_dlogits=actual_dlogits,
    		                   reference_dlogits=ref["dlogits"],
    		                   bucket_targets=bucket_targets,
    		                   audit_close=audit_close,
    		                   audit_equal=audit_equal,
    		                   label="microbatch dlogits {cursor}")
    	audit_close(f"microbatch loss {cursor}", micro_loss.get(key, 0.0), ref["loss"])
    audit_equal('CPU batch values untouched', audit_digest(batches), batch_fingerprint)
    audit_equal('CPU reference batches untouched', audit_digest(reference_batches), reference_fingerprint)
    actual = dict(loss=loss_sum[0], gradients={n: np.array(backend.copy_to_host(t.grad), copy=True) for n, t in parameters.items()})
    audit_equal('original full-step CPU result immutable',audit_digest(expected),expected_fingerprint)
    audit_training_outputs_all(expected, actual, checks, atol=3e-5, rtol=3e-5)
    if observation_sink is not None:
        observation_sink.update(expected=expected,actual=actual,ce_chunks=ce_chunks)
    checks.finish()
    return expected, actual


def run_integrated_rl_audit(model, runtime, arena, examples, batch_size, capacity,
                            max_tokens, temperature, top_k, seed, num_samples, step, baseline):
    """Run all independent cases and report failures together; never fabricate data."""
    import copy
    import json
    tokenizer = SyntheticTokenizer()
    cpu = clone_model_to_cpu_reference(model, runtime.backend)
    reports = []
    def case(name, function):
        return audit_case(reports, name, function)
    case('existing checker baseline', baseline)
    native_observed = {}
    native = case('production get_batch / all examples / rewards / padding / sampling seeds',
        lambda: audit_native_batches(model, runtime, arena, cpu, tokenizer, examples,
            num_samples, batch_size, max_tokens, temperature, top_k, step, capacity,
            observation_sink=native_observed))
    # A recorded comparison FAIL remains FAIL. Complete observations still
    # permit downstream comparisons; never substitute CPU data for GPU data.
    if 'batches' in native_observed:
        native = native_observed['batches']
    # Training on known CPU batches still runs if generation/batch comparison failed.
    def prepare_training():
        training = []
        for index, example in enumerate(examples):
            conversation = example['conversation']
            prompt = tokenizer.render_for_completion(conversation)
            b = build_cpu_nanochat_get_batch_reference(cpu, tokenizer, conversation, prompt,
                num_samples, batch_size, max_tokens, temperature, top_k, step, index,
                make_checker_reward(tokenizer), EOS_TOKEN)
            training.append(b)
        return training
    training = case('CPU training fixture preparation', prepare_training)
    if training is not None:
        if native is not None:
            case('end-to-end native Metal batches vs untouched CPU batches / loss / gradients',
                 lambda: audit_train_step(model, runtime, cpu, native, batch_size,
                                         reference_batches=training))
        else:
            reports.append(dict(case='end-to-end native Metal training', status='BLOCKED',
                                error='complete native batch observations unavailable; matched-input training is tested separately'))
        training_observed = {}
        train_result = case('production training / loss / GPU dlogits / all gradients / accumulation',
             lambda: audit_train_step(model, runtime, cpu, training, batch_size,
                                     observation_sink=training_observed))
        if all(key in training_observed for key in ('expected','actual')):
            train_result = (training_observed['expected'],training_observed['actual'])
        if train_result is not None:
            case('GPU AdamW / clipping / two updates / restore weights',
                 lambda: audit_optimizer_update(model, runtime, *train_result))
        else:
            reports.append(dict(case='GPU AdamW', status='BLOCKED', error='complete CPU/Metal training observations unavailable'))
        for kind in ('mixed', 'zero_advantage', 'all_masked'):
            # Separate synthetic inputs exercise edge cases using the unchanged CPU builder.
            # Do not overwrite returned CPU advantages, inputs, targets, or original rewards.
            batches = []
            for source in training:
                sequences = copy.deepcopy(source['rollout_sequences'])
                masks = copy.deepcopy(source['rollout_masks'])
                fixture_rewards = (np.ones(len(sequences), np.float32) if kind == 'zero_advantage'
                                   else np.resize(np.array([0.,2.,1.,1.], np.float32), len(sequences)))
                if kind == 'all_masked':
                    masks = [[0] * len(row) for row in sequences]
                fixture = build_nanochat_batch_reference(sequences, masks, fixture_rewards, EOS_TOKEN)
                fixture['rollout_sequences'] = sequences
                fixture['rollout_masks'] = masks
                batches.append(fixture)
            case(f'production training stress {kind}', lambda batches=batches: audit_train_step(model, runtime, cpu, batches, batch_size))
            if kind in ('zero_advantage','all_masked'):
                case(f'new optimization step with stale gradients: {kind}',
                     lambda batches=batches: audit_train_step(model,runtime,cpu,batches,batch_size,
                                                              poison_previous_gradients=True))
    else:
        reports.append(dict(case='reference-dependent training and optimizer', status='BLOCKED',
                            error='CPU training fixture preparation did not complete'))
    audit_independent_optimizer_cases(model,runtime,case)
    audit_extra_training_cases(model,runtime,cpu,batch_size,case)
    prompt = tokenizer.render_for_completion(examples[0]['conversation'])
    configurations = [(temperature, batch_size, max_tokens, top_k, seed),
        (0.,2,1,1,seed), (0.,2,8,top_k,seed), (1.,2,8,top_k,seed+1),
        (.7,4,8,top_k,seed+2), (1.,2,0,top_k,seed), (1.,3,8,top_k,seed+3),
        (1.,2,min(8,capacity-len(prompt)),K_MAX,seed+4),
        (0.,2,capacity-len(prompt),top_k,seed+5)]
    for temp, batch, limit, k, rng_seed in configurations:
        case(f'rollout T={temp} B={batch} tokens={limit} K={k} seed={rng_seed}',
             lambda temp=temp,batch=batch,limit=limit,k=k,rng_seed=rng_seed:
             audit_rollout_case(model,runtime,arena,cpu,tokenizer,prompt,batch,capacity,limit,temp,k,rng_seed))
    case('strict FP32 bitwise logits (separate from tolerance equivalence)',
         lambda: audit_rollout_case(model,runtime,arena,cpu,tokenizer,prompt,batch_size,
             capacity,max_tokens,temperature,top_k,seed,compare_bitwise=True))
    scripts = {
        'immediate EOS': [[EOS_TOKEN,EOS_TOKEN]],
        'immediate BOS': [[BOS_TOKEN,BOS_TOKEN]],
        'different row termination': [[7,7],[EOS_TOKEN,8],[9,9],[9,EOS_TOKEN]],
        'calculator / forced queue': [[PYTHON_START]*2,[2]*2,[PYTHON_END]*2,[EOS_TOKEN]*2],
    }
    for name, schedule in scripts.items():
        case(name, lambda schedule=schedule: audit_rollout_case(model,runtime,arena,cpu,tokenizer,prompt,2,capacity,8,0.,top_k,seed,schedule))
    audit_additional_cases(model,runtime,arena,cpu,tokenizer,examples,capacity,seed,top_k,case)
    audit_control_and_boundary_cases(model,runtime,arena,cpu,tokenizer,examples,capacity,seed,top_k,case)
    case('two-round independent rollout / training / production optimizer closed loop',
         lambda: audit_closed_loop(model,runtime,arena,examples,batch_size,capacity,max_tokens,
                                   temperature,top_k,num_samples,step))
    audit_real_branch_cases(model,runtime,cpu,case)
    # One line per module; show the first failed case and its first detail.
    modules = {}
    for result in reports:
        name = result['case'].lower()
        if 'closed loop' in name: module = '两轮 RL 闭环'
        elif 'optimizer' in name or 'adamw' in name or 'nonfinite' in name: module = '优化器 / 裁剪 / 跳步'
        elif 'training' in name or 'gradient' in name or 'ce multiple' in name: module = 'RL loss / 梯度'
        elif 'get_batch' in name or 'production batch' in name: module = 'batch / reward / advantage'
        elif 'tool ' in name or 'calculator' in name: module = '工具调用'
        elif 'termination' in name or 'immediate ' in name: module = '终止状态'
        elif 'invalid request' in name: module = '非法输入'
        elif 'reuse' in name: module = 'Engine 复用'
        elif 'bitwise' in name: module = '逐位比较'
        elif 'baseline' in name: module = '原基线检查'
        else: module = 'Rollout / 采样 / TopK'
        modules.setdefault(module, []).append(result)
    for module, items in modules.items():
        bad = [item for item in items if item['status'] != 'PASS']
        if not bad:
            print(f'[通过] {module}', flush=True)
        else:
            first = bad[0]
            detail = [line.strip() for line in first.get('error', '').splitlines() if line.strip()]
            detail = [line for line in detail if 'failed comparisons;' not in line]
            location = detail[0] if detail else '缺少完整结果'
            print(f'[未通过] {module}（{len(bad)}/{len(items)} 项）：{first["case"]}；{location[:180]}', flush=True)
    failures = [r for r in reports if r['status'] != 'PASS']
    if failures:
        raise AssertionError(f'RL audit: {len(failures)}/{len(reports)} cases not passed; details are printed above')
    print('[通过] 全部验证模块', flush=True)


def audit_optimizer_update(model, runtime, cpu_result, metal_result, gradient_pairs=None):
    """Two actual GPU AdamW updates vs an independent NumPy FP32 oracle.

    This is an optimizer-contract test with explicit hyperparameters; it does
    not claim coverage of every production LR scheduler or optimizer mode.
    Model weights and gradients are restored even when a comparison fails.
    """
    checks = AuditChecks('GPU optimizer / both steps and all parameter states')
    audit_close = checks.wrap(globals()['audit_close'])
    if gradient_pairs is not None and len(gradient_pairs) != 2:
        raise ValueError('Expected two optimizer input pairs')
    from optimizer import AdamW
    backend = runtime.backend
    params = {n: t for n, t in named_tensors(model).items() if t.requires_grad}
    before = {n: np.array(backend.copy_to_host(t), copy=True) for n, t in params.items()}
    grads = {n: np.array(backend.copy_to_host(t.grad), copy=True) for n, t in params.items()}
    expected = {n: x.copy() for n, x in before.items()}
    m = {n: np.zeros_like(x) for n, x in before.items()}
    v = {n: np.zeros_like(x) for n, x in before.items()}
    lr, b1, b2, eps, wd = 1e-3, .9, .999, 1e-8, .01
    saved = Runtime._default_runtime
    Runtime._default_runtime = runtime
    try:
        optimizer = AdamW(list(params.values()), lr=lr, betas=(b1,b2), eps=eps, weight_decay=wd, state_mode='fp32')
        gpu_clip = optimizer.build_grad_clip()
        for step in (1,2):
            if gradient_pairs is not None:
                cpu_result, metal_result = gradient_pairs[step-1]
            cpu_norm = np.sqrt(sum(float(np.sum(np.asarray(g, np.float64)**2)) for g in cpu_result['gradients'].values()))
            metal_norm = np.sqrt(sum(float(np.sum(np.asarray(g, np.float64)**2)) for g in metal_result['gradients'].values()))
            cpu_scale = min(1., 1. / (cpu_norm+1e-6))
            metal_scale = min(1., 1. / (metal_norm+1e-6))
            audit_close('gradient clipping norm', metal_norm, cpu_norm)
            for n,t in params.items():
                backend.copy_from_host(t.grad, metal_result['gradients'][n])
                g = np.asarray(cpu_result['gradients'][n] * cpu_scale, np.float32)
                m[n] = np.asarray(b1*m[n]+(1-b1)*g, np.float32)
                v[n] = np.asarray(b2*v[n]+(1-b2)*g*g, np.float32)
                decay = wd if len(t.shape)>1 else 0.
                expected[n] = np.asarray(expected[n]*(1-lr*decay) -
                    (lr*np.sqrt(1-b2**step)/(1-b1**step))*m[n]/(np.sqrt(v[n])+eps), np.float32)
            observed_norm, observed_scale = gpu_clip(1.0)
            audit_close('GPU clipping norm', observed_norm, cpu_norm)
            audit_close('GPU clipping coefficient', observed_scale, cpu_scale)
            optimizer.step(grad_scale=observed_scale)
            backend.synchronize()
            for i,(n,t) in enumerate(params.items()):
                for label, storage, ref in (('m', optimizer.m[i], m[n]), ('v', optimizer.v[i], v[n])):
                    state = backend.ctx.copy_to_host(storage.ptr.buffer_id, storage.ptr.offset, storage.shape)
                    audit_close(f'AdamW step {step} {label} {n}', state, ref)
                actual = np.array(backend.copy_to_host(t), copy=True)
                audit_close(f'AdamW step {step} parameter {n}', actual, expected[n])
                audit_close(f'AdamW step {step} delta {n}', actual-before[n], expected[n]-before[n])
    finally:
        for n,t in params.items():
            backend.copy_from_host(t, before[n])
            backend.copy_from_host(t.grad, grads[n])
        backend.synchronize()
        Runtime._default_runtime = saved

    checks.finish()


def audit_digest(value):
    """Fingerprint reference values without changing arrays or flags."""
    import hashlib
    import pickle
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def audit_reuse_cases(model,runtime,arena,cpu,tokenizer,prompt,capacity,seed,top_k):
    """All reuse calls execute even if an earlier comparison failed."""
    engine=audit_new_engine(model,runtime,arena,tokenizer,2,capacity)
    checks=AuditChecks('Engine A-B-zero-A reset/reuse')
    observations=[]
    shorter=prompt[:max(1,len(prompt)-2)]
    for label,tokens,limit,temp,k,rng in (
        ('A1',prompt,4,1.,top_k,seed),('B',shorter,2,0.,1,seed+1),
        ('zero',prompt,0,1.,top_k,seed),('A2',prompt,4,1.,top_k,seed)):
        observed={}
        checks.check(label,audit_rollout_case,model,runtime,arena,cpu,tokenizer,tokens,2,
                     capacity,limit,temp,k,rng,reuse_engine=engine,observation_sink=observed)
        if not all(key in observed for key in ('sequences','masks','states')):
            raise RuntimeError(f'{label}: rollout observation unavailable; reuse validation incomplete')
        observations.append(observed)
    first,second=observations[0],observations[-1]
    for field in ('sequences','masks'):
        checks.check(f'A1/A2 {field}',audit_equal,field,first[field],second[field])
    checks.check('A1/A2 state count',audit_equal,'states',len(first['states']),len(second['states']))
    for step,(left,right) in enumerate(zip(first['states'],second['states'])):
        for field in ('next_tokens','finished','lengths','position','step','generated_ids','rng_state'):
            checks.check(f'A1/A2 step={step} {field}',audit_equal,field,left[field],right[field])
    checks.finish()



def audit_invalid_request(model,runtime,arena,cpu,tokenizer,prompt,capacity,kind):
    """Use unchanged CPU exception type as oracle; no fabricated reference error."""
    options=dict(prompt_tokens=prompt,num_samples=2,max_tokens=1,temperature=1.,top_k=50,seed=1234)
    if kind=='empty_prompt': options['prompt_tokens']=[]
    elif kind=='zero_samples': options['num_samples']=0
    elif kind=='negative_limit': options['max_tokens']=-1
    elif kind=='negative_temperature': options['temperature']=-1.
    else: raise ValueError(kind)
    try:
        generate_cpu_nanochat_reference(cpu_model=cpu,tokenizer=tokenizer,**options)
    except (ValueError,TypeError) as exc:
        expected_type=type(exc)
    else:
        raise AssertionError('Fixture did not exercise an invalid CPU request')
    engine=audit_new_engine(model,runtime,arena,tokenizer,2,capacity)
    try:
        engine.generate_batch_new(tokens=options['prompt_tokens'],
            num_samples=options['num_samples'],max_tokens=options['max_tokens'],
            temperature=options['temperature'],top_k=options['top_k'],seed=options['seed'])
    except Exception as exc:
        if not isinstance(exc,expected_type):
            raise AssertionError(f'CPU {expected_type.__name__}; Metal {type(exc).__name__}: {exc}') from exc
    else:
        raise AssertionError(f'Metal accepted request rejected by CPU ({expected_type.__name__})')


def audit_sampling_logits(step,rows,vocab,kind):
    """Fixture inputs only; not a replacement CPU policy or reference value."""
    if kind=='ties':
        value=np.zeros((rows,vocab),np.float32)
    elif kind=='extreme':
        value=np.linspace(-80.,80.,vocab,dtype=np.float32)[None,:].repeat(rows,axis=0)
    elif kind=='close':
        value=np.linspace(-1e-5,1e-5,vocab,dtype=np.float32)[None,:].repeat(rows,axis=0)
    else:
        raise ValueError(kind)
    # Keep terminal/tool tokens out of ordinary positive-K sampler tests.
    if kind!='ties':
        value[:,[EOS_TOKEN,BOS_TOKEN,PYTHON_START,PYTHON_END,OUTPUT_START,OUTPUT_END]]=-100.
    return value


def audit_additional_cases(model,runtime,arena,cpu,tokenizer,examples,capacity,seed,top_k,case):
    prompt=tokenizer.render_for_completion(examples[0]['conversation'])
    case('Engine A-B-A reset/reuse',lambda:audit_reuse_cases(model,runtime,arena,cpu,tokenizer,prompt,capacity,seed,top_k))
    for kind in ('empty_prompt','zero_samples','negative_limit','negative_temperature'):
        case(f'invalid request {kind}',lambda kind=kind:audit_invalid_request(model,runtime,arena,cpu,tokenizer,prompt,capacity,kind))
    # Exact edge-case seeds cross Philox block/64-bit boundaries with odd row counts.
    for rng_seed in (0,(1<<32)+1234,(1<<63)+17,(1<<64)-1):
        case(f'64-bit Philox stream B=3 seed={rng_seed}',lambda rng_seed=rng_seed:
            audit_rollout_case(model,runtime,arena,cpu,tokenizer,prompt,3,capacity,7,1.,top_k,rng_seed,
                              logits_factory=lambda s,b,v:audit_sampling_logits(s,b,v,'close')))
    vocab=int(model.tok_emb.weight.shape[0])
    for kind,k,temp in (('ties',1,0.),('ties',50,1.),('close',50,.01),
                        ('extreme',50,2.),('close',0,1.),('close',-1,1.),
                        ('close',K_MAX+1,1.),('close',vocab+7,1.)):
        case(f'sampling {kind} K={k} T={temp}',lambda kind=kind,k=k,temp=temp:
             audit_rollout_case(model,runtime,arena,cpu,tokenizer,prompt,2,capacity,7,temp,k,seed,
                               logits_factory=lambda s,b,v:audit_sampling_logits(s,b,v,kind)))
    p,e=PYTHON_START,PYTHON_END
    scripts={
        'forced queue truncated at start':([[p,p],[2,2],[e,e],[EOS_TOKEN,EOS_TOKEN]],4),
        'forced queue truncated at result':([[p,p],[2,2],[e,e],[EOS_TOKEN,EOS_TOKEN]],5),
        'forced queue consumes sampled EOS':([[p,p],[2,2],[e,e],[EOS_TOKEN,EOS_TOKEN]],8),
        'repeated calculator':([[p,p],[2,2],[e,e],[7,7],[7,7],[7,7],[p,p],[3,3],[e,e],[7,7],[7,7],[7,7],[EOS_TOKEN,EOS_TOKEN]],13),
        'different row tools':([[7,7],[p,8],[2,9],[e,EOS_TOKEN],[7,10],[7,11],[7,12],[EOS_TOKEN,13]],8),
        'empty calculator body':([[p,p],[e,e],[EOS_TOKEN,EOS_TOKEN]],3),
        'invalid expression':([[p,p],[2,2],[3,3],[e,e],[EOS_TOKEN,EOS_TOKEN]],5),
        'python end without start':([[e,e],[EOS_TOKEN,EOS_TOKEN]],2),
        'nested python start':([[p,p],[p,p],[2,2],[e,e],[7,7],[7,7],[7,7],[EOS_TOKEN,EOS_TOKEN]],8),
    }
    for name,(script,limit) in scripts.items():
        for temp in (0.,1.):
            case(f'tool {name} T={temp}',lambda script=script,limit=limit,temp=temp:
                 audit_rollout_case(model,runtime,arena,cpu,tokenizer,prompt,2,capacity,limit,temp,top_k,seed,schedule=script))
    # Prompt-length variants exercise prefill bucket selection and decode offsets.
    for length in (1,3,min(17,capacity-2)):
        variant=([BOS_TOKEN]+[7]*max(0,length-1))
        case(f'prefill/decode prompt length={length}',lambda variant=variant:
             audit_rollout_case(model,runtime,arena,cpu,tokenizer,variant,2,capacity,2,1.,top_k,seed))
    # Independent cases keep later examples observable when an earlier one fails.
    for example_index, example in enumerate(examples):
        example_prompt = tokenizer.render_for_completion(example['conversation'])
        case(f'independent example trace {example_index}',lambda example_prompt=example_prompt,example_index=example_index:
             audit_rollout_case(model,runtime,arena,cpu,tokenizer,example_prompt,2,capacity,4,1.,top_k,
                                hash((17,example_index,0)) & 0x7fffffff))
    # Both num_samples remainders and distinct optimization steps must use real get_batch.
    for samples,batch,step in ((5,2,1),(6,3,17)):
        case(f'production batch samples={samples} B={batch} step={step}',lambda samples=samples,batch=batch,step=step:
             audit_native_batches(model,runtime,arena,cpu,tokenizer,examples,samples,batch,4,1.,top_k,step,capacity))




def run_single_engine_case(model, runtime, arena, examples, batch_size,
                           max_seq_len, max_new_tokens, top_k, seed, temperature,
                           num_samples, examples_per_rank, optimization_step=0):
    def baseline():
        tokenizer = SyntheticTokenizer()
        engine = audit_new_engine(model, runtime, arena, tokenizer, batch_size, max_seq_len)
        if engine.decode_steps > 1:
            cpu = clone_model_to_cpu_reference(model, runtime.backend)
            prompt = tokenizer.render_for_completion(examples[0]['conversation'])
            for value in (seed, seed + 1) if temperature > 0 else (seed,):
                audit_rollout_case(model, runtime, arena, cpu, tokenizer, prompt,
                    batch_size, max_seq_len, max_new_tokens, temperature, top_k, value,
                    reuse_engine=engine)
            return {"baseline": "passed", "decode_steps": engine.decode_steps}
        return run_baseline_engine_case(model=model, runtime=runtime, arena=arena,
            examples=examples, batch_size=batch_size, max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens, top_k=top_k, seed=seed,
            temperature=temperature, num_samples=num_samples,
            examples_per_rank=examples_per_rank, optimization_step=optimization_step)
    return run_integrated_rl_audit(model, runtime, arena, examples, batch_size,
        max_seq_len, max_new_tokens, temperature, top_k, seed, num_samples,
        optimization_step, baseline=baseline)


def run_baseline_engine_case(
    model,
    runtime,
    arena,
    examples,
    batch_size,
    max_seq_len,
    max_new_tokens,
    top_k,
    seed,
    temperature,
    num_samples,
    examples_per_rank,
    optimization_step=0,
):
    batch_size = int(batch_size)
    num_samples = int(num_samples)
    examples_per_rank = int(examples_per_rank)

    if batch_size < 2:
        raise AssertionError(
            "batch_size must be at least 2"
        )

    if len(examples) != examples_per_rank:
        raise AssertionError(
            "examples count mismatch"
        )

    tokenizer = SyntheticTokenizer()

    prompt_tokens = tokenizer.render_for_completion(
        examples[0]["conversation"]
    )

    cpu_reference_model = clone_model_to_cpu_reference(
        metal_model=model,
        metal_backend=runtime.backend,
    )

    reward_fn = make_checker_reward(tokenizer)

    cpu_reference_result = (
        run_cpu_nanochat_rl_reference(
            cpu_model=cpu_reference_model,
            tokenizer=tokenizer,
            examples=examples,
            num_samples=num_samples,
            device_batch_size=batch_size,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            optimization_step=optimization_step,
            examples_per_rank=examples_per_rank,
            reward_fn=reward_fn,
            assistant_end=EOS_TOKEN,
        )
    )

    run_cpu_forced_token_semantics_case()

    print(
        "[CPU-NANOCHAT-RL-PASS]",
        f"loss="
        f"{cpu_reference_result['actual_step']['loss']:.8f}",
        f"microbatches="
        f"{len(cpu_reference_result['actual_step']['microbatches'])}",
        flush=True,
    )

    trace_engine = Engine(
        model=model,
        tokenizer=tokenizer,
        device="mps",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        parameter_runtime=runtime,
        arena=arena,
        max_top_k=K_MAX,
    )

    metal_logits_trace = []
    metal_input_trace = []
    metal_topk_trace = []
    metal_state_trace = []

    pending_capture = capture_logits_wrapper(
        engine=trace_engine,
        prompt_len=len(prompt_tokens),
        captured_logits=metal_logits_trace,
        captured_inputs=metal_input_trace,
        captured_topk=metal_topk_trace,
    )

    wrap_engine_rollout_methods(
        engine=trace_engine,
        state_trace=metal_state_trace,
        captured_logits=metal_logits_trace,
        captured_inputs=metal_input_trace,
        captured_topk=metal_topk_trace,
        pending=pending_capture,
    )

    trace_metal_sequences, trace_metal_masks = (
        trace_engine.generate_batch_new(
            tokens=prompt_tokens,
            num_samples=batch_size,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
    )

    trace_cpu_rollout = (
        generate_cpu_nanochat_reference(
            cpu_model=cpu_reference_model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            num_samples=batch_size,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
    )

    trace_cpu_sequences = (
        trace_cpu_rollout["sequences"]
    )
    trace_cpu_masks = (
        trace_cpu_rollout["masks"]
    )
    trace_cpu_steps = (
        trace_cpu_rollout["trace"]
    )

    check_equal(
        "TRACE_SEQUENCE_MISMATCH",
        np.asarray(
            trace_metal_sequences,
            dtype=object,
        ),
        np.asarray(
            trace_cpu_sequences,
            dtype=object,
        ),
    )

    check_equal(
        "TRACE_MASK_MISMATCH",
        np.asarray(
            trace_metal_masks,
            dtype=object,
        ),
        np.asarray(
            trace_cpu_masks,
            dtype=object,
        ),
    )

    if len(metal_logits_trace) != len(trace_cpu_steps):
        raise AssertionError(
            "TRACE_STEP_COUNT_MISMATCH: "
            f"metal={len(metal_logits_trace)} "
            f"cpu={len(trace_cpu_steps)}"
        )

    compare_steps = min(
        len(metal_logits_trace),
        len(trace_cpu_steps),
    )

    for step_index in range(compare_steps):
        mode = (
            "prefill"
            if step_index == 0
            else "decode"
        )

        metal_logits = select_sampling_logits(
            metal_logits_trace[step_index]["logits"],
            mode=mode,
            prompt_len=len(prompt_tokens),
        )

        cpu_logits = np.asarray(
            trace_cpu_steps[step_index]["logits"],
            dtype=np.float32,
        )

        metal_logits = _align_rows(
            metal_logits,
            cpu_logits.shape[0],
        )

        check_close(
            f"TRACE_LOGITS_MISMATCH step={step_index}",
            metal_logits,
            cpu_logits,
            atol=1e-5,
            rtol=1e-5,
        )

        expected_values, expected_indices, count = (
            numpy_topk_reference(
                cpu_logits,
                top_k,
            )
        )

        actual_values = _align_rows(
            metal_topk_trace[step_index]["values"],
            cpu_logits.shape[0],
        )

        actual_indices = _align_rows(
            metal_topk_trace[step_index]["indices"],
            cpu_logits.shape[0],
        )

        check_close(
            f"TRACE_TOPK_VALUES_MISMATCH step={step_index}",
            actual_values[:, :count],
            expected_values[:, :count],
            atol=1e-5,
            rtol=1e-5,
        )

        check_equal(
            f"TRACE_TOPK_INDICES_MISMATCH step={step_index}",
            actual_indices[:, :count],
            expected_indices[:, :count],
        )

    if temperature > 0.0:
        run_random_sampling_semantics_case(
            model=model,
            runtime=runtime,
            arena=arena,
            cpu_reference_model=cpu_reference_model,
            prompt_tokens=prompt_tokens,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            seed=seed + 1,
        )

    validate_metal_cpu_rl_equivalence(
	    metal_state_trace=metal_state_trace,
	    metal_logits_trace=metal_logits_trace,
	    metal_topk_trace=metal_topk_trace,
	    metal_input_trace=metal_input_trace,
	    metal_sequences=trace_metal_sequences,
	    metal_masks=trace_metal_masks,
	    cpu_rollout=trace_cpu_rollout,
	    prompt_tokens=prompt_tokens,
	    num_samples=batch_size,
	    max_tokens=max_new_tokens,
	    top_k=top_k,
	    temperature=temperature,
	    seed=seed,
	    assistant_end=EOS_TOKEN,
	)

    run_complete_metal_cpu_rl_validation(
	    metal_state_trace=metal_state_trace,
	    metal_logits_trace=metal_logits_trace,
	    metal_topk_trace=metal_topk_trace,
	    metal_input_trace=metal_input_trace,
	    metal_sequences=trace_metal_sequences,
	    metal_masks=trace_metal_masks,
	    cpu_rollout=trace_cpu_rollout,
	    prompt_tokens=prompt_tokens,
	    num_samples=batch_size,
	    top_k=top_k,
	    assistant_end=EOS_TOKEN,
	    tokenizer=tokenizer,
	)

    return {"baseline": "passed"}

def initialize_tiny_model(
    vocab_size,
    embd,
    n_head,
    n_layer,
    sequence_len,
):
    """
    Build a small real GPT2 graph using the current project implementation.

    This does not implement GPT2 in the checker. It instantiates the actual
    GPT2 class from the current directory.
    """
    Runtime._default_runtime = None

    runtime = Runtime(
        device="mps",
        forward_only=False,
        owns_params=True,
    )
    Runtime._default_runtime = runtime

    model = GPT2(
        V=vocab_size,
        C=embd,
        max_T=sequence_len,
        n_head=n_head,
        batch_size=4,
        n_layer=n_layer,
        dtype=np.float32,
        device="mps",
        rollout_max_topk=K_MAX,
    )

    input_ids = Tensor(
        shape=(4, sequence_len),
        dtype=np.int32,
        device="mps",
        requires_grad=False,
        init_val=None,
        name="check_engine_input",
    )
    target_ids = Tensor(
        shape=(4, sequence_len),
        dtype=np.int32,
        device="mps",
        requires_grad=False,
        init_val=None,
        name="check_engine_target",
    )

    loss = model(
        input_ids,
        targets=target_ids,
        mode="train",
    )

    runtime.loss = loss
    runtime.build_topo(loss)
    runtime.plan_memory()
    runtime.is_compiled = True
    runtime.bind_memory()

    return runtime, model, runtime.unified_arena

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--examples-per-step",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--embd",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--optimization-step",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="run only the NumPy nanochat-policy reference checks",
    )

    args = parser.parse_args()

    if args.batch_size < 2:
        parser.error(
            "batch-size must be at least 2"
        )

    if args.num_samples < args.batch_size:
        parser.error(
            "num-samples must be at least batch-size"
        )

    if args.examples_per_step <= 0:
        parser.error(
            "examples-per-step must be positive"
        )

    if args.temperature < 0.0:
        parser.error(
            "temperature must be non-negative"
        )

    if args.embd % args.n_head != 0:
        parser.error(
            "embd must be divisible by n-head"
        )

    tokenizer = SyntheticTokenizer()

    # These checks do not instantiate or import a second model.  They verify
    # the CPU policy itself against an independent transcription of
    # nanochat's Engine.generate/generate_batch state machine.
    run_cpu_forced_token_semantics_case()
    run_cpu_nanochat_policy_equivalence_case()
    run_cpu_philox_stream_case()
    run_cpu_chat_rl_contract_case()

    if args.cpu_only:
        print(
            "[CPU-REFERENCE-PASS] nanochat prompt, Philox state contract, "
            "forced-token state machine, masks, terminal handling, and rollout policy verified",
            flush=True,
        )
        return

    raw_conversations = []

    base_prompts = [
        "7 11 19 23 31",
        "13 17 29 37 43",
        "5 9 15 27 33",
        "6 14 22 38 46",
    ]

    for index in range(args.examples_per_step):
        raw_conversations.append(
            {
                "example_index": index,
                "messages": [
                    {
                        "role": "user",
                        "content": base_prompts[
                            index % len(base_prompts)
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "",
                    },
                ],
            }
        )

    examples = [
        {
            "conversation": conversation,
            "prompt_tokens": tokenizer.render_for_completion(
                conversation
            ),
        }
        for conversation in raw_conversations
    ]

    max_prompt_len = max(
        len(example["prompt_tokens"])
        for example in examples
    )

    if (
        max_prompt_len + args.max_new_tokens
        > args.max_seq_len
    ):
        parser.error(
            "prompt plus max-new-tokens exceeds max-seq-len"
        )

    runtime, model, arena = initialize_tiny_model(
        vocab_size=args.vocab_size,
        embd=args.embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        sequence_len=args.max_seq_len,
    )

    run_single_engine_case(
        model=model,
        runtime=runtime,
        arena=arena,
        examples=examples,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        seed=args.seed,
        temperature=args.temperature,
        num_samples=args.num_samples,
        examples_per_rank=args.examples_per_step,
        optimization_step=args.optimization_step,
    )

if __name__ == "__main__":
    run()