"""
Train GPT2 using the custom autograd/Metal framework while matching the
nanochat d6 training configuration and data pipeline.

Run from /Users/a58/Documents/code/autograd.
"""

import os
import gc
import json
import math
import time
import pickle
import argparse

import numpy as np
import pyarrow.parquet as pq

try:
    import wandb
except ImportError:
    wandb = None

from tensor import Tensor
from module import GPT2
from runtime import Runtime
from optimizer import AdamW
from gguf_checkpoint import load_gguf, save_gguf


# -----------------------------------------------------------------------------
# Logging


class DummyWandb:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


# -----------------------------------------------------------------------------
# Arguments: defaults reproduce nanogpt_0520_train_log configuration


parser = argparse.ArgumentParser(
    description="Train GPT2 with the custom Autograd Metal backend"
)

parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--device-type", type=str, default="mps")

parser.add_argument("--depth", type=int, default=6)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=64)
parser.add_argument("--max-seq-len", type=int, default=512)
parser.add_argument("--window-pattern", type=str, default="L")

parser.add_argument("--num-iterations", type=int, default=5000)
parser.add_argument("--target-param-data-ratio", type=float, default=12.0)

parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=16384)

# base_train d6 uses matrix_lr=0.02, subsequently scaled by sqrt(B / 524288)
parser.add_argument("--matrix-lr", type=float, default=0.02)

# The original base_train command specified 0.28, which was internally scaled
# to 0.234903 for the d6 run. Autograd receives the effective value directly.
parser.add_argument("--weight-decay", type=float, default=0.234903)

parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)

parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--eval-tokens", type=int, default=524288)
parser.add_argument("--core-metric-every", type=int, default=-1)
parser.add_argument("--sample-every", type=int, default=-1)
parser.add_argument("--save-every", type=int, default=-1)
parser.add_argument("--model-in", type=str, default=None)
parser.add_argument("--model-out", type=str, default=None)

args = parser.parse_args()
if args.save_every > 0 and not args.model_out:
    parser.error("--model-out is required when --save-every is positive")
user_config = vars(args).copy()


# -----------------------------------------------------------------------------
# nanochat tokenizer adapter


class NanochatTokenizer:
    def __init__(self, base_dir):
        tokenizer_path = os.path.join(
            base_dir, "tokenizer", "tokenizer.pkl"
        )

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"nanochat tokenizer not found: {tokenizer_path}"
            )

        with open(tokenizer_path, "rb") as f:
            self.enc = pickle.load(f)

        self.vocab_size = self.enc.n_vocab
        self.bos_id = self.enc.encode_single_token("<|bos|>")

        special_ids = set()
        for token in self.enc.special_tokens_set:
            special_ids.add(self.enc.encode_single_token(token))
        self.special_ids = special_ids

        # Number of bytes represented by each token. Special tokens contribute
        # zero bytes, matching nanochat loss_eval.py.
        self.token_bytes = np.zeros(self.vocab_size, dtype=np.int64)
        for token_id in range(self.vocab_size):
            if token_id not in self.special_ids:
                self.token_bytes[token_id] = len(
                    self.enc.decode_single_token_bytes(token_id)
                )

    def encode_batch(self, texts, num_threads=4):
        return self.enc.encode_ordinary_batch(
            texts, num_threads=num_threads
        )


# -----------------------------------------------------------------------------
# nanochat-compatible BOS-aligned best-fit dataloader


def list_parquet_files(data_dir):
    files = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".parquet") and not name.endswith(".tmp")
    )

    if len(files) < 2:
        raise RuntimeError(
            f"Expected train and validation parquet files in {data_dir}"
        )

    return files


def document_batches(
    data_dir,
    split,
    tokenizer_batch_size=128,
):
    parquet_files = list_parquet_files(data_dir)

    if split == "train":
        parquet_files = parquet_files[:-1]
    elif split == "val":
        parquet_files = parquet_files[-1:]
    else:
        raise ValueError(f"Invalid split: {split}")

    epoch = 1

    while True:
        for pq_idx, filepath in enumerate(parquet_files):
            parquet_file = pq.ParquetFile(filepath)

            for rg_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(rg_idx)
                texts = row_group.column("text").to_pylist()

                for start in range(0, len(texts), tokenizer_batch_size):
                    yield (
                        texts[start:start + tokenizer_batch_size],
                        {
                            "epoch": epoch,
                            "pq_idx": pq_idx,
                            "rg_idx": rg_idx,
                        },
                    )

        epoch += 1


def bestfit_dataloader(
    data_dir,
    tokenizer,
    batch_size,
    sequence_len,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    buffer_size=1000,
):
    row_capacity = sequence_len + 1

    source = document_batches(
        data_dir=data_dir,
        split=split,
        tokenizer_batch_size=tokenizer_batch_size,
    )

    doc_buffer = []
    state = {
        "epoch": 1,
        "pq_idx": 0,
        "rg_idx": 0,
    }

    row_buffer = np.empty(
        (batch_size, row_capacity),
        dtype=np.int32,
    )

    def refill_buffer():
        nonlocal state

        texts, state = next(source)
        encoded = tokenizer.encode_batch(
            texts,
            num_threads=tokenizer_threads,
        )

        for tokens in encoded:
            doc_buffer.append([tokenizer.bos_id] + tokens)

    while True:
        for row_idx in range(batch_size):
            position = 0

            while position < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - position

                # Select the largest document that fits.
                best_idx = -1
                best_len = 0

                for idx, tokens in enumerate(doc_buffer):
                    token_count = len(tokens)

                    if token_count <= remaining and token_count > best_len:
                        best_idx = idx
                        best_len = token_count

                if best_idx >= 0:
                    tokens = doc_buffer.pop(best_idx)
                    token_count = len(tokens)

                    row_buffer[
                        row_idx,
                        position:position + token_count,
                    ] = tokens

                    position += token_count
                else:
                    # Crop the shortest document to fill the remaining space.
                    shortest_idx = min(
                        range(len(doc_buffer)),
                        key=lambda idx: len(doc_buffer[idx]),
                    )

                    tokens = doc_buffer.pop(shortest_idx)

                    row_buffer[
                        row_idx,
                        position:position + remaining,
                    ] = tokens[:remaining]

                    position += remaining

        x = row_buffer[:, :-1].copy()
        y = row_buffer[:, 1:].copy()

        yield x, y, state.copy()


# -----------------------------------------------------------------------------
# nanochat-compatible schedules


def get_lr_multiplier(step, num_iterations):
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps

    warmdown_steps = round(
        args.warmdown_ratio * num_iterations
    )
    warmdown_start = num_iterations - warmdown_steps

    if step <= warmdown_start:
        return 1.0

    progress = (num_iterations - step) / warmdown_steps

    return (
        progress
        + (1.0 - progress) * args.final_lr_frac
    )


def get_weight_decay(step, num_iterations):
    return args.weight_decay * 0.5 * (
        1.0 + math.cos(math.pi * step / num_iterations)
    )


# -----------------------------------------------------------------------------
# Runtime and tokenizer


np.random.seed(42)

device = args.device_type

Runtime._default_runtime = Runtime(device=device)
runtime = Runtime.get_default()
backend = runtime.backend

base_dir = os.environ.get(
    "NANOCHAT_BASE_DIR",
    os.path.expanduser("~/.cache/nanochat"),
)

data_dir = os.path.join(base_dir, "base_data_climbmix")

if not os.path.isdir(data_dir):
    raise FileNotFoundError(
        f"ClimbMix data directory not found: {data_dir}"
    )

tokenizer = NanochatTokenizer(base_dir)
vocab_size = tokenizer.vocab_size

print(f"Configured device type: {device}")
print("Distributed world size: 1")
if device == "mps":
    print("COMPUTE_DTYPE: float32(custom Autograd Metal backend)")
    print("Using custom fused LayerNorm/FlashAttention/MLP/CE kernels")
else:
    print("COMPUTE_DTYPE: float32(custom Autograd CPU backend)")
    print("Using custom CPU training kernels")

print(f"Vocab size: {vocab_size:,}")


# -----------------------------------------------------------------------------
# Model configuration


base_dim = args.depth * args.aspect_ratio

model_dim = (
    (base_dim + args.head_dim - 1)
    // args.head_dim
    * args.head_dim
)

num_heads = model_dim // args.head_dim

model_config = {
    "sequence_len": args.max_seq_len,
    "vocab_size": vocab_size,
    "n_layer": args.depth,
    "n_head": num_heads,
    "n_kv_head": num_heads,
    "n_embd": model_dim,
    "window_pattern": args.window_pattern,
}

print(
    "Model config:\n"
    + json.dumps(model_config, indent=2)
)

model = GPT2(
    vocab_size,
    model_dim,
    args.max_seq_len,
    num_heads,
    args.device_batch_size,
    args.depth,
    np.float32,
    device,
)

num_params = sum(param.numel for param in model.parameters())

# Matrix parameters used for FLOP estimation:
# - Q/K/V/output projections: 4*C*C
# - MLP projections: 8*C*C
# - LM head: V*C
matrix_params = (
    args.depth * 12 * model_dim * model_dim
    + vocab_size * model_dim
)

num_scaling_params = matrix_params

# Same FLOP accounting convention as nanochat:
# 6 FLOPs per matrix parameter plus causal attention FLOPs.
num_flops_per_token = (
    6 * matrix_params
    + 12
    * args.depth
    * num_heads
    * args.head_dim
    * args.max_seq_len
)

print("Parameter counts:")
print(f"{'total':24s}: {num_params:,}")
print(f"{'matrix parameters':24s}: {matrix_params:,}")
print(
    f"Estimated FLOPs per token: {num_flops_per_token:e}"
)


# -----------------------------------------------------------------------------
# Batch and training horizon


tokens_per_microbatch = (
    args.device_batch_size * args.max_seq_len
)

if args.total_batch_size % tokens_per_microbatch != 0:
    raise ValueError(
        f"total batch {args.total_batch_size} must be divisible by "
        f"B*T={tokens_per_microbatch}"
    )

grad_accum_steps = (
    args.total_batch_size // tokens_per_microbatch
)

if grad_accum_steps > 1 and device != "mps":
    raise ValueError(
        "gradient accumulation > 1 is currently supported only on MPS"
    )

num_iterations = args.num_iterations

if num_iterations <= 0:
    target_tokens = int(
        args.target_param_data_ratio * num_scaling_params
    )
    num_iterations = target_tokens // args.total_batch_size

total_tokens = args.total_batch_size * num_iterations

batch_lr_scale = math.sqrt(
    args.total_batch_size / 524288
)

effective_lr = args.matrix_lr * batch_lr_scale

print(
    f"Scaling LRs by {batch_lr_scale:.4f} for batch size "
    f"{args.total_batch_size:,} (reference: 524,288)"
)
print(
    f"Using effective AdamW matrix LR: {effective_lr:.8f}"
)
print(
    f"Using effective weight decay: {args.weight_decay:.6f}"
)
print(
    f"Using user-provided number of iterations: "
    f"{num_iterations:,}"
)
print(
    f"Total number of training tokens: {total_tokens:,}"
)
print(
    "Tokens : Scaling params ratio: "
    f"{total_tokens / num_scaling_params:.2f}"
)
print(
    "Total training FLOPs estimate: "
    f"{num_flops_per_token * total_tokens:e}"
)
print(
    f"Tokens / micro-batch / rank: "
    f"{args.device_batch_size} x {args.max_seq_len} "
    f"= {tokens_per_microbatch:,}"
)
print(
    f"Tokens / micro-batch: {tokens_per_microbatch:,}"
)
print(
    f"Total batch size {args.total_batch_size:,} "
    f"=> gradient accumulation steps: {grad_accum_steps}"
)


# -----------------------------------------------------------------------------
# Static graph


B = args.device_batch_size
T = args.max_seq_len

input_ids = Tensor(
    (B, T),
    np.int32,
    device=device,
    requires_grad=False,
    init_val=None,
)

# position_ids = Tensor(
#     (B, T),
#     np.int32,
#     device=device,
#     requires_grad=False,
#     init_val=np.tile(
#         np.arange(T, dtype=np.int32),
#         (B, 1),
#     ),
# )

labels = Tensor(
    (B, T),
    np.int32,
    device=device,
    requires_grad=False,
    init_val=None,
)

# Use the fused LM-head + cross-entropy path. It returns mean NLL and avoids
# materializing a B*T*V logits tensor.
loss = model(
    input_ids,
    # position_ids,
    targets=labels,
)

# Compile and allocate the static graph before the initial validation pass.
runtime.loss = loss
runtime.build_topo(loss)
runtime.bind_memory()
runtime.is_compiled = True

if args.model_in:
    checkpoint_metadata = load_gguf(
        args.model_in,
        model,
        backend,
        model_config,
    )
    print(
        f"Loaded {checkpoint_metadata['autograd.loaded_tensor_count']} "
        f"tensors from {args.model_in} "
        f"(saved step: {checkpoint_metadata.get('autograd.training_step', 'unknown')})"
    )

optimizer = AdamW(
    model.parameters(),
    lr=effective_lr,
    weight_decay=args.weight_decay,
    state_mode="fp32",
)


# -----------------------------------------------------------------------------
# Data iterators


train_loader = bestfit_dataloader(
    data_dir=data_dir,
    tokenizer=tokenizer,
    batch_size=B,
    sequence_len=T,
    split="train",
)

x_data, y_data, dataloader_state = next(train_loader)


# -----------------------------------------------------------------------------
# Validation BPB


def evaluate_bpb():
    val_loader = bestfit_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=B,
        sequence_len=T,
        split="val",
    )

    eval_steps = args.eval_tokens // (B * T)
    eval_steps = max(eval_steps, 1)

    total_nats = 0.0
    total_bytes = 0

    for _ in range(eval_steps):
        x_val, y_val, _ = next(val_loader)

        byte_counts = tokenizer.token_bytes[y_val]
        valid = byte_counts > 0

        # Fused CE treats -1 as ignore_index. Its output is divided by B*T,
        # so multiplying by B*T recovers the sum over valid target tokens.
        masked_targets = y_val.copy()
        masked_targets[~valid] = -1

        backend.copy_from_host(input_ids, x_val)
        backend.copy_from_host(labels, masked_targets)

        runtime.forward()
        backend.synchronize()

        mean_nll = float(
            np.asarray(
                backend.copy_to_host(loss)
            ).reshape(-1)[0]
        )

        total_nats += mean_nll * B * T
        total_bytes += int(byte_counts.sum())

    if total_bytes == 0:
        return float("inf")

    return total_nats / (math.log(2.0) * total_bytes)


# -----------------------------------------------------------------------------
# Approximate allocated-memory reporting


def approximate_allocated_bytes():
    storages = {}

    def add_storage(storage):
        if storage is None or isinstance(storage, tuple):
            return

        if hasattr(storage.ptr, "buffer_id"):
            key = ("mps", storage.ptr.buffer_id)
            extent = storage.ptr.offset + storage.nbytes
        else:
            key = ("cpu", id(storage.ptr))
            extent = storage.nbytes

        storages[key] = max(
            storages.get(key, 0),
            extent,
        )

    add_storage(runtime.unified_arena)
    add_storage(input_ids.storage)
    # add_storage(position_ids.storage)
    add_storage(labels.storage)

    for param in model.parameters():
        add_storage(param.storage)

        if param.grad is not None:
            add_storage(param.grad.storage)

    for storage in optimizer.m:
        add_storage(storage)

    for storage in optimizer.v:
        add_storage(storage)

    add_storage(optimizer.qmap_m)
    add_storage(optimizer.qmap_v)
    if optimizer.state_mode in ("8bit", "mini") and isinstance(optimizer.dispatches, tuple):
        for item in optimizer.dispatches:
            if item[0] == "q8":
                add_storage(item[5]) # absmax_m
                add_storage(item[6]) # absmax_v
            elif item[0] == "mini8bit":
                add_storage(item[4]) # absmax_m


    return sum(storages.values())


# -----------------------------------------------------------------------------
# Training loop


if args.run == "dummy":
    wandb_run = DummyWandb()
else:
    if wandb is None:
        raise RuntimeError("wandb is not installed")

    wandb_run = wandb.init(
        project="nanochat-autograd",
        name=args.run,
        config=user_config,
    )

step = 0
smooth_train_loss = 0.0
total_training_time = 0.0
val_bpb = None
min_val_bpb = float("inf")
mfu = 0.0
flops_so_far = 0.0

while True:
    last_step = step == num_iterations

    flops_so_far = (
        num_flops_per_token
        * args.total_batch_size
        * step
    )

    if (
        args.eval_every > 0
        and (last_step or step % args.eval_every == 0)
    ):
        val_bpb = evaluate_bpb()
        min_val_bpb = min(min_val_bpb, val_bpb)

        print(
            f"Step {step:05d} | "
            f"Validation bpb: {val_bpb:.6f}"
        )

        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })

    if last_step:
        break

    backend.synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        backend.copy_from_host(input_ids, x_data)
        backend.copy_from_host(labels, y_data)

        loss.backward(
            grad_scale=1.0 / grad_accum_steps,
            accumulate_parameter_grads=(micro_step > 0),
        )

        x_data, y_data, dataloader_state = next(train_loader)

    train_loss = float(
        np.asarray(
            backend.copy_to_host(loss)
        ).reshape(-1)[0]
    )

    lrm = get_lr_multiplier(step, num_iterations)

    optimizer.lr = effective_lr * lrm
    optimizer.weight_decay = get_weight_decay(
        step,
        num_iterations,
    )

    # 每个完整的逻辑 batch 只更新一次。
    optimizer.step()

    backend.synchronize()
    t1 = time.time()

    dt = t1 - t0

    ema_beta = 0.9

    smooth_train_loss = (
        ema_beta * smooth_train_loss
        + (1.0 - ema_beta) * train_loss
    )

    debiased_smooth_loss = (
        smooth_train_loss
        / (1.0 - ema_beta ** (step + 1))
    )

    pct_done = 100.0 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)

    flops_per_sec = (
        num_flops_per_token
        * args.total_batch_size
        / dt
    )

    # Keep the existing project convention. This is an approximate peak value
    # for calculating a comparable utilization indicator.
    gpu_peak_flops = 10.4e12
    mfu = 100.0 * flops_per_sec / gpu_peak_flops

    if step > 10:
        total_training_time += dt

    steps_done = step - 10

    if steps_done > 0:
        avg_time_per_step = (
            total_training_time / steps_done
        )

        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""

    epoch = (
        f"{dataloader_state['epoch']} "
        f"pq: {dataloader_state['pq_idx']} "
        f"rg: {dataloader_state['rg_idx']}"
    )

    print(
        f"step {step:05d}/{num_iterations:05d} "
        f"({pct_done:.2f}%) | "
        f"loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | "
        f"dt: {dt * 1000:.2f}ms | "
        f"tok/sec: {tok_per_sec:,} | "
        f"bf16_mfu: {mfu:.2f} | "
        f"epoch: {epoch} | "
        f"total time: {total_training_time / 60:.2f}m"
        f"{eta_str}"
    )

    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        })

    first_step_of_run = step == 0
    step += 1

    if args.save_every > 0 and step % args.save_every == 0:
        checkpoint_path = args.model_out.format(step=step)
        save_gguf(
            checkpoint_path,
            model,
            backend,
            model_config,
            step,
        )
        print(f"Saved GGUF checkpoint: {checkpoint_path}")

    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

if args.save_every > 0 and step % args.save_every != 0:
    checkpoint_path = args.model_out.format(step=step)
    save_gguf(
        checkpoint_path,
        model,
        backend,
        model_config,
        step,
    )
    print(f"Saved final GGUF checkpoint: {checkpoint_path}")


peak_memory = approximate_allocated_bytes()

print(
    f"Peak memory usage: "
    f"{peak_memory / 1024 / 1024:.2f}MiB"
)
print(
    f"Total training time: "
    f"{total_training_time / 60:.2f}m"
)

if val_bpb is not None:
    print(
        f"Minimum validation bpb: "
        f"{min_val_bpb:.6f}"
    )

wandb_run.finish()