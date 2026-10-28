"""
Reinforcement learning on GSM8K via GRPO.

Run:

    python chat_rl_autograd.py \
        --model-in checkpoints/base_d6/model_000000.gguf \
        --data-dir sft_data
"""

import argparse
import gc
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from tensor import Tensor
from module import GPT2
from runtime import Runtime
from runtime import (
    ExecutionPlan,
    build_prefill_plan,
    build_decode_plan,
    build_train_plan,
)
from optimizer import AdamW
from gguf_checkpoint import load_gguf, save_gguf, _read_exact, _read_string, _read_value
import time

# Engine is intentionally implemented in another script.
# Its public interface should match nanochat's Engine.
from engine import Engine

# nanochat is used only for tokenizer and task definitions.
# The model, autograd, optimizer and memory management come from this directory.
NANOCHAT_DIR = os.path.expanduser("~/Documents/code/nanochat")
if NANOCHAT_DIR not in sys.path:
    sys.path.insert(0, NANOCHAT_DIR)

from nanochat.tokenizer import RustBPETokenizer
from tasks.gsm8k import GSM8K, extract_answer


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="mps", choices=["mps", "cpu"], help="mps|cpu",)
# Model loading
parser.add_argument("--model-in", type=str, required=True, help="input GGUF checkpoint")
parser.add_argument("--model-out", type=str, default=None, help="output checkpoint template, e.g. model_{step:06d}.gguf")
parser.add_argument("--model-tag", type=str, default=None, help="model tag used by the output path")
parser.add_argument("--model-step", type=int, default=None, help="kept for compatibility with chat_rl.py")
# Data
parser.add_argument("--data-dir", type=str, default="sft_data",)
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K",)
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass",)
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step",)
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question",)
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample",)
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature",)
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling, 0 disables top-k",)
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="kept for compatibility with chat_rl.py",)
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="kept for compatibility with chat_rl.py",)
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for AdamW",)
parser.add_argument("--weight-decay", type=float, default=0.0,)
parser.add_argument("--init-lr-frac", type=float, default=0.05,)
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60,)
parser.add_argument("--eval-examples", type=int, default=400,)
parser.add_argument("--save-every", type=int, default=60,)

args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Small compatibility helpers

class DummyWandb:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def print0(*values, **kwargs):
    print(*values, **kwargs)


def compute_cleanup():
    backend = getattr(Runtime.get_default(), "backend", None)
    if backend is not None:
        backend.synchronize()

def read_gguf_metadata(path):
    with open(path, "rb") as f:
        if _read_exact(f, 4) != b"GGUF":
            raise ValueError(f"not a GGUF checkpoint: {path}")

        version, _, metadata_count = __import__("struct").unpack("<IQQ", _read_exact(f, 20),)

        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {version}")

        metadata = {}
        for _ in range(metadata_count):
            name = _read_string(f)
            kind = __import__("struct").unpack("<I", _read_exact(f, 4),)[0]
            metadata[name] = _read_value(f, kind)

    if metadata.get("general.architecture") != "autograd-gpt2":
        raise ValueError("the checkpoint must be produced by this repository")

    return metadata

def build_graph(config, batch_size, device):
    runtime = Runtime(device=device)
    Runtime._default_runtime = runtime

    model = GPT2(
        config["vocab_size"],
        config["n_embd"],
        config["sequence_len"],
        config["n_head"],
        batch_size,
        config["n_layer"],
        np.float32,
        device,
    )

    input_shape = (batch_size, config["sequence_len"],)
    input_ids = Tensor(input_shape, np.int32, device=device, requires_grad=False, init_val=None, name="input_ids",)
    target_ids = Tensor(input_shape, np.int32, device=device, requires_grad=False, init_val=None, name="target_ids",)
    row_scale = Tensor(shape=(batch_size*config["sequence_len"], 1), 
                       dtype=np.float32, device=device, requires_grad=False, init_val=None, name="row_scale")
    loss = model(input_ids, targets=target_ids, row_scale=row_scale)

    runtime.loss = loss
    runtime.build_topo(loss)
    runtime.bind_memory()
    runtime.is_compiled = True

    return runtime, model, input_ids, target_ids, row_scale, loss


def load_local_model(model_path, device, batch_size, sequence_len):
    metadata = read_gguf_metadata(model_path)
    saved_config = json.loads(metadata["autograd.model_config"])
    config = dict(saved_config)
    config["sequence_len"] = sequence_len
    runtime, model, input_ids, target_ids, row_scale, loss = build_graph(config, batch_size, device)
    load_gguf(model_path, model, runtime.backend, saved_config)

    return runtime, model, input_ids, target_ids, row_scale, loss, config, saved_config

def build_train_buckets(model, base_plan, batch_size, device):
    arena = base_plan.runtime.unified_arena
    max_length = base_plan.seq_len
    plans = []
    for length in (128, 192, 256, 320, 384, 448, 512):
        if length >= max_length:
            break
        plan = build_train_plan(model=model, batch_size=batch_size, seq_len=length, device=device, owns_params=False)

        if plan.runtime.arena_bytes > arena.nbytes:
            raise RuntimeError(f"train bucket T={length} requires {plan.runtime.arena_bytes} over {arena.nbytes}")

        plan.bind_to(arena)
        plans.append(plan)

    plans.append(base_plan)
    return tuple(plans)

def choose_train_bucket(plans, sequences, b0, b1):
    required_length = 1
    for row in range(b0, b1):
        length = len(sequences[row]) - 1
        if length > required_length:
            required_length = length

    for plan in plans:
        if required_length <= plan.seq_len:
            return plan

    return plans[-1]

# -----------------------------------------------------------------------------
# Runtime / precision initialization
device = args.device_type

# The current framework is single-process.
ddp = False
ddp_rank = 0
ddp_local_rank = 0
ddp_world_size = 1
master_process = True

# -----------------------------------------------------------------------------
# Wandb initialization
use_dummy_wandb = args.run == "dummy"
if use_dummy_wandb:
    wandb_run = DummyWandb()
else:
    import wandb

    wandb_run = wandb.init(
        project="nanochat-rl",
        name=args.run,
        config=user_config,
    )


# -----------------------------------------------------------------------------
# Tokenizer and model
data_dir = Path(args.data_dir).expanduser().resolve()
tokenizer = RustBPETokenizer.from_directory(str(data_dir / "tokenizer"))
rollout_seq_len = args.max_new_tokens + 300
runtime, model, input_ids, target_ids, row_scale, loss, model_config, saved_model_config = load_local_model(args.model_in, device, 
    args.device_batch_size, rollout_seq_len)

train_plan = ExecutionPlan(runtime=runtime, input_ids=input_ids, target_ids=target_ids, row_scale=row_scale, output=loss, seq_len=rollout_seq_len, mode="train")
train_plans = build_train_buckets(model=model, base_plan=train_plan, batch_size=args.device_batch_size, device=device)

backend = runtime.backend

# Engine is implemented separately and receives the custom model.
engine = Engine(model, tokenizer, 
                device=device, 
                batch_size=args.device_batch_size, 
                max_seq_len=rollout_seq_len, 
                parameter_runtime=runtime, 
                arena=runtime.unified_arena,
                max_top_k=64)

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop
train_task = GSM8K(subset="main", split="train",)
val_task = GSM8K(subset="main", split="test",)
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        # Keep the same completion-prompt construction as chat_rl.py.
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF

            if engine.enable_debug:
                engine.decode_slots = 0
                engine.tool_waste = 0
                engine.eos_bos_waste = 0

            sample_start = time.perf_counter()
            generated_token_sequences_batch, masks_batch = (
                # engine.generate_batch(
                engine.generate_batch_new(
                    tokens,
                    num_samples=args.device_batch_size,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=seed,
                )
            )
            print0(
                f"[RL-GENERATE-DONE] "
                f"step={step} "
                f"example={example_idx} "
                f"sample={sampling_step + 1}/{num_sampling_steps} "
                f"num_samples={args.device_batch_size} "
                f"max_tokens={args.max_new_tokens} "
                f"dt={time.perf_counter() - sample_start:.3f}s",
                flush=True,
            )
            if engine.enable_debug:
                denominator = max(engine.decode_slots, 1)
                print0(
                    f"[RL-DECODE-REDUNDANCY] "
                    f"step={step} "
                    f"example={example_idx} "
                    f"batch={sampling_step + 1}/{num_sampling_steps} | "
                    f"tool_after="
                    f"{100.0 * engine.tool_waste / denominator:.2f}% "
                    f"({engine.tool_waste}/{engine.decode_slots}) | "
                    f"eos_bos_after="
                    f"{100.0 * engine.eos_bos_waste / denominator:.2f}% "
                    f"({engine.eos_bos_waste}/{engine.decode_slots})",
                    flush=True,
                )

            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        rewards = []
        debug_reward = (step < 10 and example_idx < 10)
        for sample_index, sample_tokens in enumerate(generated_token_sequences):
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text,)
            rewards.append(reward)

        fixed_length = rollout_seq_len + 1
        padded_generated_token_sequences = [seq[:fixed_length] + [assistant_end] * max(0, fixed_length - len(seq),) for seq in generated_token_sequences]
        padded_masks = [mask[:fixed_length] + [0] * max(0, fixed_length - len(mask),) for mask in masks]

        ids = np.asarray(padded_generated_token_sequences, dtype=np.int32,)
        mask_ids = np.asarray(padded_masks, dtype=np.int32,)

        # Same autoregressive shift as chat_rl.py.
        inputs = ids[:, :-1].copy()
        targets = ids[:, 1:].copy()

        targets[mask_ids[:, 1:] == 0] = -1
        rewards = np.asarray(rewards, dtype=np.float32,)

        # DAPO-style advantage: reward - batch mean.
        advantages = rewards - rewards.mean()

        yield (
            generated_token_sequences,
            inputs,
            targets,
            rewards,
            advantages,
        )

# -----------------------------------------------------------------------------
# Evaluation loop
def run_gsm8k_eval(
    task,
    tokenizer,
    engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50,
):
    """
    Same evaluation structure as chat_rl.py.

    In the current single-process backend, ddp_rank=0 and
    ddp_world_size=1.
    """

    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)

    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]

        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        assert num_samples <= args.device_batch_size

        generated_token_sequences, masks = engine.generate_batch_new(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        outcomes = []

        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)

            is_correct = task.evaluate(conversation, generated_text)

            outcomes.append({"is_correct": is_correct,})

        yield {"idx": idx, "outcomes": outcomes,}

# -----------------------------------------------------------------------------
# Training setup
optimizer = AdamW(
    model.parameters(),
    lr=args.matrix_lr * args.init_lr_frac,
    weight_decay=args.weight_decay,
    state_mode="fp32",
)
clip_grad_norm = optimizer.build_grad_clip()

# Keep the original learning-rate convention.
initial_lr = args.matrix_lr * args.init_lr_frac
optimizer.lr = initial_lr

examples_per_rank = args.examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")
print0(f"Calculated examples per rank: {examples_per_rank}")

def get_lr_multiplier(step):
    return max(0.0, 1.0 - step / max(1, num_steps),)

def save_model_checkpoint(step):
    if args.model_out is not None:
        output_path = args.model_out.format(step=step)
    else:
        output_dir = (
            data_dir / "chatrl_checkpoints" / (args.model_tag if args.model_tag is not None else f"d{model_config['n_layer']}")
        )

        output_path = output_dir / (f"model_{step:06d}.gguf")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_gguf(
        str(output_path),
        model,
        backend,
        model_config,
        step,
    )

    print(f"Saved model checkpoint to {output_path}")


# -----------------------------------------------------------------------------
# Training loop
engine.enable_debug = True
train_slots = 0
train_padding_slots = 0

batch_iterator = get_batch()
for step in range(num_steps):
    rollout_dt = 0
    optimizer_dt = 0

    step_start = time.perf_counter()
    # Evaluation
    if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
        eval_start = time.perf_counter()
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine,
            num_samples=args.device_batch_size,
            max_examples=args.eval_examples,
            temperature=1.0,
            top_k=args.top_k,
        )
        records = list(records_iter)
        eval_dt = time.perf_counter() - eval_start
        passk = np.zeros(args.device_batch_size, dtype=np.float32,)
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(outcome["is_correct"] for outcome in r["outcomes"][:k]) for r in records)
        num_records = len(records)
        if num_records > 0:
            passk /= num_records

        print0(f"[RL-EVAL-DONE] step={step}, records={num_records}, dt={eval_dt:.2f}s", flush=True)
        print0(f"Step {step} | " + ", ".join(f"Pass@{k}: {passk[k - 1]:.4f}" for k in range(1, args.device_batch_size + 1)))
        wandb_run.log({"step": step, **{f"pass@{k}": float(passk[k - 1]) for k in range(1, args.device_batch_size + 1)}})

    # Forward/backward over rollouts
    rollout_start = time.perf_counter()
    rewards_list = []
    sequence_lengths = []
    # optimizer.zero_grad()
    first_backward = True
    for example_step in range(examples_per_rank):
        if example_step == 0:
            # A new optimization step owns fresh gradients even if no group
            # executes backward. Runtime is the same owner used by AdamW.
            from runtime import Runtime as _RLRuntime
            _RLRuntime.get_default().zero_grad()
        example_start = time.perf_counter()

        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

        assert (inputs_all.shape[0] % args.device_batch_size == 0)
        num_passes = inputs_all.shape[0] // args.device_batch_size

        # The local fused CE returns a scalar: mean(token_nll over B*T)
        # The reference objective is: sum(logp * advantage) / (valid_tokens * num_passes * examples_per_rank)
        # We therefore group rows by equal advantage and use grad_scale: B*T*advantage / (valid_tokens*num_passes*examples_per_rank)
        # Rows outside the group receive target=-1.

        # unique_advantages = sorted(set(float(value) for value in advantages_all))

        for pass_idx in range(num_passes):
            b0 = pass_idx * args.device_batch_size
            b1 = b0 + args.device_batch_size

            curr_train_plan = choose_train_bucket(train_plans, sequences_all, b0, b1)
            bucket_length = curr_train_plan.seq_len

            inputs_pass = inputs_all[b0:b1, :bucket_length]
            targets_pass = targets_all[b0:b1, :bucket_length]
            advantages_pass = advantages_all[b0:b1]

            valid_tokens_pass = np.count_nonzero(targets_pass >= 0)
            if valid_tokens_pass <= 0:
                continue

            denominator = float(valid_tokens_pass * num_passes * examples_per_rank)
            B, T = targets_pass.shape
            row_scale_host = np.broadcast_to(np.asarray(advantages_pass, dtype=np.float32).reshape(B, 1), (B, T),).copy()
            row_scale_host[targets_pass < 0] = 0.0
            row_scale_host /= np.float32(denominator)

            curr_train_plan.set_inputs(inputs_pass)
            curr_train_plan.set_targets(targets_pass)
            curr_train_plan.set_row_scale(row_scale_host.reshape(B*T, 1))
            curr_train_plan.backward(grad_scale=1.0, accumulate_parameter_grads=(not first_backward))

            if engine.enable_debug:
                train_slots = B*T
                train_padding_slots = 0
                for row in range(b0, b1):
                    padding = T - len(sequences_all[row])
                    if padding > 0:
                        train_padding_slots += padding
                example_idx = ddp_rank + (
                    (step * examples_per_rank + example_step)
                    % len(range(ddp_rank, len(train_task), ddp_world_size))
                ) * ddp_world_size
                print0(
                    f"[RL-TRAIN-REDUNDANCY] "
                    f"step={step} "
                    f"example={example_idx} "
                    f"batch={pass_idx + 1}/{num_passes} | "
                    f"train_T={T}/{rollout_seq_len} | "
                    f"padding="
                    f"{100.0 * train_padding_slots / max(train_slots, 1):.2f}% "
                    f"({train_padding_slots}/{train_slots})",
                    flush=True,
                )

            first_backward = False
            print0(
                f"Step {step}/{num_steps} | "
                f"Example step {example_step} | "
                f"Pass {pass_idx} | "
                f"row_scale_abs_mean: "
                f"{float(np.abs(row_scale_host).mean()):.6e}"
            )

        print0(
            f"Step {step}/{num_steps} | "
            f"Example step {example_step} | "
            f"Average reward: "
            f"{float(rewards_all.mean()):.6f}"
        )

        rewards_list.append(float(rewards_all.mean()))
        sequence_lengths.extend(len(sequence) for sequence in sequences_all)

    # Rollout statistics
    mean_reward = sum(rewards_list) / max(1, len(rewards_list))
    mean_sequence_length = sum(sequence_lengths) / max(1, len(sequence_lengths))
    print0(
        f"Step {step}/{num_steps} | "
        f"Average reward: {mean_reward:.6f} | "
        f"Average sequence length: "
        f"{mean_sequence_length:.2f}"
    )
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })
    rollout_dt += (time.perf_counter() - rollout_start)

    # Gradient clipping, learning-rate rampdown and optimizer step
    optimizer_start = time.perf_counter()
    grad_norm, clip_coef = clip_grad_norm(1.0)

    if not math.isfinite(grad_norm):
        print0(
            f"Skipping step {step}: "
            f"non-finite gradient norm {grad_norm}"
        )
        # optimizer.zero_grad()
        continue
    lrm = get_lr_multiplier(step)
    optimizer.lr = initial_lr * lrm

    optimizer.step(grad_scale=clip_coef)

    optimizer_dt = (time.perf_counter() - optimizer_start)
    # optimizer.zero_grad()
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "grad_norm": float(grad_norm),
        "clip_coef": float(clip_coef),
    })

    print0(
        f"[RL-STEP] "
        f"step={step}/{num_steps} "
        f"rollout={rollout_dt:.3f}s "
        f"optimizer={optimizer_dt:.3f}s "
        f"total={(time.perf_counter() - step_start):.3f}s "
        f"reward={mean_reward:.4f} "
        f"grad_norm={grad_norm:.6f} "
        f"seq_len={mean_sequence_length:.1f}",
        flush=True,
    )

    # Checkpointing
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        save_model_checkpoint(step)

# -----------------------------------------------------------------------------
# Cleanup

try:
    from nanochat.report import get_report
    get_report().log(
        section="Chat RL",
        data=[user_config],
    )
except Exception as exc:
    print0(f"Report logging skipped: {exc}")

wandb_run.finish()
compute_cleanup()