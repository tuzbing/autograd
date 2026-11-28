"""
Train larger GPT2 models on PG-19 (or mixed datasets) with the custom Autograd Metal backend.

Example (d12 = 135.3M params, T=1024, AdamW-mini, ~1.6 GiB total memory on M1):
/Users/a58/Documents/code/nanochat/.venv/bin/python -B train_gpt2.py \
  --data-dir ~/Documents/datasets/p19 \
  --tokenizer-dir ~/.cache/nanochat/tokenizer \
  --depth 12 \
  --aspect-ratio 64 \
  --head-dim 64 \
  --max-seq-len 1024 \
  --device-batch-size 8 \
  --total-batch-size 32768 \
  --optimizer-state mini \
  --grad-dtype fp32 \
  --matrix-lr 0.0024 \
  --warmup-steps 200\
  --max-norm 1.0 \
  --eval-every 200 \
  --eval-tokens 524288 \
  --num-iterations 10000 \
  --save-every 1000 \
  --model-out '/Users/a58/Documents/code/autograd/checkpoints/gpt2_d12/model_{step:06d}.gguf'
"""

import os
import gc
import json
import math
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from tensor import Tensor
from module import GPT2
from runtime import Runtime
from optimizer import AdamW
from gguf_checkpoint import load_gguf, save_gguf, save_optimizer, load_optimizer
from text_dataset import universal_bestfit_dataloader, parse_data_specs, resolve_split_files


class DummyWandb:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


class NanochatTokenizer:
    def __init__(self, tokenizer_dir):
        tokenizer_path = os.path.join(os.path.expanduser(tokenizer_dir), "tokenizer.pkl")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"nanochat tokenizer not found: {tokenizer_path}")

        with open(tokenizer_path, "rb") as f:
            self.enc = pickle.load(f)

        self.vocab_size = self.enc.n_vocab
        self.bos_id = self.enc.encode_single_token("<|bos|>")

        self.special_ids = {
            self.enc.encode_single_token(token)
            for token in self.enc.special_tokens_set
        }
        self.token_bytes = np.zeros(self.vocab_size, dtype=np.int64)
        for token_id in range(self.vocab_size):
            if token_id not in self.special_ids:
                self.token_bytes[token_id] = len(
                    self.enc.decode_single_token_bytes(token_id)
                )

    def encode_batch(self, texts, num_threads=4):
        return self.enc.encode_ordinary_batch(texts, num_threads=num_threads)


def parse_args():
    p = argparse.ArgumentParser(description="Train larger GPT2 on PG-19 / custom datasets")
    p.add_argument("--run", type=str, default="dummy")
    p.add_argument("--device-type", type=str, default="mps")

    p.add_argument(
        "--data-dir",
        type=str,
        default=os.path.expanduser("~/Documents/datasets/p19"),
        help="Single dataset dir or weighted specs 'path1:0.3,path2:0.7'",
    )
    p.add_argument(
        "--val-data-dir",
        type=str,
        default=None,
        help="Optional separate validation dir (defaults to first path in --data-dir)",
    )
    p.add_argument(
        "--tokenizer-dir",
        type=str,
        default=os.path.join(
            os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat")),
            "tokenizer",
        ),
    )
    p.add_argument("--text-column", type=str, default=None)

    # Model architecture: d12 (135.3M) by default; use depth=15 (228.9M) or depth=18 (362.2M)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--window-pattern", type=str, default="L")

    # Hardware & memory efficiency options
    p.add_argument("--use-fp16-gemm", type=int, choices=[0, 1], default=1)
    p.add_argument("--grad-dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--optimizer-state", choices=["fp32", "8bit", "mini"], default="mini")
    p.add_argument("--max-norm", type=float, default=1.0)

    # Batch & optimization schedule
    p.add_argument("--num-iterations", type=int, default=-1, help="Max steps (-1 = train full dataset for --epochs)")
    p.add_argument("--epochs", type=int, default=1, help="Number of epochs when --num-iterations == -1")
    p.add_argument("--target-param-data-ratio", type=float, default=16.0)
    p.add_argument("--device-batch-size", type=int, default=8)
    p.add_argument("--total-batch-size", type=int, default=32768)
    p.add_argument("--matrix-lr", type=float, default=0.015)
    p.add_argument("--weight-decay", type=float, default=0.2)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--warmdown-ratio", type=float, default=0.65)
    p.add_argument("--final-lr-frac", type=float, default=0.05)

    # Dataloader tuning
    p.add_argument("--parquet-batch-size", type=int, default=16)
    p.add_argument("--buffer-size", type=int, default=256)

    # Eval & checkpointing
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-tokens", type=int, default=524288)
    p.add_argument("--save-every", type=int, default=-1)
    p.add_argument("--model-in", type=str, default=None)
    p.add_argument("--load-optimizer", type=int, choices=[0, 1], default=0)
    p.add_argument("--model-out", type=str, default=None)

    args = p.parse_args()
    if args.save_every > 0 and not args.model_out:
        p.error("--model-out is required when --save-every is positive")
    return args


def approximate_allocated_bytes(runtime, model, optimizer, input_ids, labels):
    storages = {}

    def add_storage(storage):
        if storage is None or isinstance(storage, tuple):
            return
        if hasattr(storage.ptr, "buffer_id"):
            # key = ("mps", storage.ptr.buffer_id)
            key = (storage.device, storage.ptr.buffer_id)
            extent = storage.ptr.offset + storage.nbytes
        else:
            key = ("cpu", id(storage.ptr))
            extent = storage.nbytes
        storages[key] = max(storages.get(key, 0), extent)

    add_storage(runtime.unified_arena)
    add_storage(input_ids.storage)
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
                add_storage(item[5])
                add_storage(item[6])
            elif item[0] == "mini8bit":
                add_storage(item[4])
    return sum(storages.values())


def main():
    args = parse_args()
    user_config = vars(args).copy()
    np.random.seed(42)

    device = args.device_type
    use_fp16_gemm = bool(args.use_fp16_gemm) and device == "mps"
    Runtime._default_runtime = Runtime(
        device=device,
        use_fp16_gemm=use_fp16_gemm,
    )
    runtime = Runtime.get_default()
    backend = runtime.backend

    # Verify dataset split resolution upfront
    specs = parse_data_specs(args.data_dir)
    for path, weight in specs:
        train_files = resolve_split_files(path, "train")
        print(f"Dataset source ({weight * 100:.1f}%): {path} -> {len(train_files)} train file(s)")
    val_spec = args.val_data_dir or specs[0][0]
    val_files = resolve_split_files(val_spec, "val")
    print(f"Validation source: {val_spec} -> {len(val_files)} val file(s)")

    tokenizer = NanochatTokenizer(args.tokenizer_dir)
    vocab_size = tokenizer.vocab_size

    base_dim = args.depth * args.aspect_ratio
    model_dim = (base_dim + args.head_dim - 1) // args.head_dim * args.head_dim
    num_heads = model_dim // args.head_dim
    mlp_hidden = (8 * model_dim) // 3
    if mlp_hidden % 64 != 0:
        print(
            f"WARNING: MLP hidden dim H={mlp_hidden} is not a multiple of 64 "
            f"(consider depth divisible by 3, e.g. 12, 15, 18, for optimal FP16 GEMM alignment)"
        )

    model_config = {
        "sequence_len": args.max_seq_len,
        "vocab_size": vocab_size,
        "n_layer": args.depth,
        "n_head": num_heads,
        "n_kv_head": num_heads,
        "n_embd": model_dim,
        "window_pattern": args.window_pattern,
    }
    print("Model config:\n" + json.dumps(model_config, indent=2))

    B = args.device_batch_size
    T = args.max_seq_len
    tokens_per_microbatch = B * T
    if args.total_batch_size % tokens_per_microbatch != 0:
        raise ValueError(
            f"total_batch_size {args.total_batch_size} must be divisible by B*T={tokens_per_microbatch}"
        )
    grad_accum_steps = args.total_batch_size // tokens_per_microbatch
    if grad_accum_steps > 1 and device not in ("mps", "cuda"):
        raise ValueError("gradient accumulation > 1 is supported only on MPS/CUDA")

    model = GPT2(
        vocab_size,
        model_dim,
        T,
        num_heads,
        B,
        args.depth,
        np.float32,
        device,
    )
    param_grad_dtype = {"fp32": np.float32, "bf16": "bf16"}[args.grad_dtype]
    for param in model.parameters():
        param.grad_dtype = param_grad_dtype

    num_params = sum(param.numel for param in model.parameters())
    matrix_params = (
        args.depth * (4 * model_dim * model_dim + 3 * model_dim * mlp_hidden)
        + vocab_size * model_dim
    )
    num_flops_per_token = (
        6 * matrix_params
        + 12 * args.depth * num_heads * args.head_dim * T
    )

    num_iterations = args.num_iterations
    batch_lr_scale = math.sqrt(args.total_batch_size / 524288)
    effective_lr = args.matrix_lr * batch_lr_scale

    print(f"Total parameters        : {num_params:,}")
    print(f"Matrix parameters       : {matrix_params:,} (MLP H={mlp_hidden})")
    print(f"Tokens / micro-batch    : {B} x {T} = {tokens_per_microbatch:,} (accum={grad_accum_steps})")
    if num_iterations > 0:
        total_tokens = args.total_batch_size * num_iterations
        print(f"Planned training tokens : {total_tokens:,} ({total_tokens / matrix_params:.2f} tok/matrix_param)")
    else:
        print(f"Planned training        : {args.epochs} full epoch(s) over dataset")
    print(f"Optimizer & precision   : state={args.optimizer_state}, grad={args.grad_dtype}, fp16_gemm={use_fp16_gemm}")

    input_ids = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=None)
    labels = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=None)
    loss = model(input_ids, targets=labels)

    runtime.loss = loss
    runtime.build_topo(loss)
    runtime.bind_memory()
    runtime.is_compiled = True

    start_step = 0
    if args.model_in:
        ckpt_meta = load_gguf(args.model_in, model, backend, model_config)
        start_step = int(ckpt_meta.get("autograd.training_step", 0))
        print(
            f"Loaded {ckpt_meta['autograd.loaded_tensor_count']} tensors from {args.model_in} "
            f"(step: {ckpt_meta.get('autograd.training_step', 'unknown')})"
        )
    else:
        resid_scale = np.float32(1.0 / math.sqrt(2.0 * args.depth))
        for blk in model.blocks:
            for w in (blk.attn.proj.weight, blk.mlp.fc2.weight):
                if device == "mps":
                    backend.numpy_view(w)[:] *= resid_scale
                elif device == "cuda":
                    backend.copy_from_host(w, backend.copy_to_host(w) * resid_scale)
                else:
                    w.storage.data[:] *= resid_scale

    optimizer = AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=args.weight_decay,
        state_mode=args.optimizer_state,
    )
    clip_grad_norm = optimizer.build_grad_clip() if args.max_norm > 0 else None

    resume_state = None
    if args.model_in and args.load_optimizer:
        opt_meta = load_optimizer(args.model_in + ".optim.npz", optimizer, model, backend)
        resume_state = opt_meta.get("dataloader_state")

    def get_lr_multiplier(step, progress):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        if args.warmdown_ratio <= 0 or progress <= 1.0 - args.warmdown_ratio:
            return 1.0
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1.0 - decay) + decay * args.final_lr_frac

    def get_weight_decay(progress):
        return args.weight_decay * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def evaluate_bpb():
        val_loader = universal_bestfit_dataloader(
            data_spec=val_spec,
            tokenizer=tokenizer,
            batch_size=B,
            sequence_len=T,
            split="val",
            text_column=args.text_column,
            parquet_batch_size=args.parquet_batch_size,
            buffer_size=args.buffer_size,
        )
        eval_steps = max(1, args.eval_tokens // (B * T))
        total_nats = 0.0
        total_bytes = 0

        for _ in range(eval_steps):
            x_val, y_val, _ = next(val_loader)
            byte_counts = tokenizer.token_bytes[y_val]
            masked_targets = y_val.copy()
            masked_targets[byte_counts <= 0] = -1

            backend.copy_from_host(input_ids, x_val)
            backend.copy_from_host(labels, masked_targets)
            runtime.forward()
            backend.synchronize()

            mean_nll = float(np.asarray(backend.copy_to_host(loss)).reshape(-1)[0])
            total_nats += mean_nll * B * T
            total_bytes += int(byte_counts.sum())

        return float("inf") if total_bytes == 0 else total_nats / (math.log(2.0) * total_bytes)

    train_loader = universal_bestfit_dataloader(
        data_spec=args.data_dir,
        tokenizer=tokenizer,
        batch_size=B,
        sequence_len=T,
        split="train",
        text_column=args.text_column,
        parquet_batch_size=args.parquet_batch_size,
        buffer_size=args.buffer_size,
        max_epochs=args.epochs if num_iterations <= 0 else None,
        start_state=resume_state,
    )
    dataloader_state = resume_state or {"epoch": 1, "pq_idx": 0, "rg_idx": 0, "progress": 0.0}
    data_exhausted = False
    try:
        x_data, y_data, dataloader_state = next(train_loader)
    except StopIteration:
        data_exhausted = True

    if args.run == "dummy":
        wandb_run = DummyWandb()
    else:
        if wandb is None:
            raise RuntimeError("wandb is not installed")
        wandb_run = wandb.init(project="nanochat-autograd-pg19", name=args.run, config=user_config)

    step = start_step
    base_prog = None
    smooth_train_loss = 0.0
    total_training_time = 0.0
    val_bpb = None
    min_val_bpb = float("inf")
    gc_frozen = False

    try:
        while True:
            last_step = (num_iterations > 0 and step >= num_iterations) or data_exhausted
            flops_so_far = num_flops_per_token * args.total_batch_size * step

            if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
                val_bpb = evaluate_bpb()
                min_val_bpb = min(min_val_bpb, val_bpb)
                print(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
                wandb_run.log({
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "val/bpb": val_bpb,
                })

            if last_step:
                break

            # backend.synchronize()
            t0 = time.perf_counter()
            micro_losses = []

            for micro_step in range(grad_accum_steps):
                backend.copy_from_host(input_ids, x_data)
                backend.copy_from_host(labels, y_data)

                loss.backward(
                    grad_scale=1.0 / grad_accum_steps,
                    accumulate_parameter_grads=(micro_step > 0),
                )
                loss_val = float(backend.numpy_view(loss).reshape(-1)[0]) if device == "mps" else float(np.asarray(backend.copy_to_host(loss)).reshape(-1)[0])
                micro_losses.append(loss_val)
                try:
                    x_data, y_data, dataloader_state = next(train_loader)
                except StopIteration:
                    data_exhausted = True
                    break

            clip_coef = 1.0
            grad_norm = 0.0
            if clip_grad_norm is not None:
                grad_norm, clip_coef = clip_grad_norm(args.max_norm)
                if not math.isfinite(grad_norm):
                    print(f"[GRAD-SKIP] step={step + 1} non-finite grad_norm={grad_norm}", flush=True)
                    step += 1
                    continue

            progress = (
                min(1.0, (step + 1) / num_iterations)
                if num_iterations > 0
                else min(1.0, dataloader_state.get("progress", 0.0))
            )
            lrm = get_lr_multiplier(step, progress)
            optimizer.lr = effective_lr * lrm
            optimizer.weight_decay = get_weight_decay(progress)
            optimizer.step(grad_scale=clip_coef)
            backend.synchronize()
            dt = time.perf_counter() - t0

            ema_steps = step - start_step + 1
            train_loss = float(np.mean(micro_losses))
            smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * train_loss
            debiased_loss = smooth_train_loss / (1.0 - 0.9 ** ema_steps)

            actual_tokens = (micro_step + 1) * tokens_per_microbatch
            tok_per_sec = int(actual_tokens / dt)
            mfu = 100.0 * (num_flops_per_token * args.total_batch_size / dt) / 10.4e12
            if ema_steps == 10:
                base_prog = progress
            elif ema_steps > 10:
                total_training_time += dt

            steps_done = ema_steps - 10
            eta_str = ""
            if num_iterations > 0 and steps_done > 0:
                eta_str = f" | eta: {(num_iterations - step) * (total_training_time / steps_done) / 60:.1f}m"
            elif num_iterations <= 0 and base_prog is not None and progress > base_prog:
                eta_str = f" | eta: {(1.0 - progress) * (total_training_time / (progress - base_prog)) / 60:.1f}m"

            epoch_info = (
                f"{dataloader_state['epoch']} pq:{dataloader_state['pq_idx']} rg:{dataloader_state['rg_idx']}"
            )
            norm_str = f" | gnorm: {grad_norm:.2f}" if clip_grad_norm is not None else ""
            step_str = f"{step:06d}/{num_iterations:06d}" if num_iterations > 0 else f"{step:06d}"
            print(
                f"step {step_str} ({100.0 * progress:.2f}%) | "
                f"loss: {debiased_loss:.6f}{norm_str} | lrm: {lrm:.2f} | "
                f"dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
                f"bf16_mfu: {mfu:.2f} | epoch: {epoch_info} | "
                f"total time: {total_training_time / 60:.2f}m{eta_str}"
            )

            if step % 50 == 0:
                wandb_run.log({
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_loss,
                    "train/grad_norm": grad_norm,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                })

            first_step = step == start_step
            step += 1

            if args.save_every > 0 and step % args.save_every == 0:
                ckpt_path = Path(args.model_out.format(step=step)).expanduser().resolve()
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                save_gguf(str(ckpt_path), model, backend, model_config, step)
                save_optimizer(str(ckpt_path) + ".optim.npz", optimizer, model, backend, extra_meta={"dataloader_state": dataloader_state})
                print(f"Saved checkpoint & optimizer: {ckpt_path}")

            if first_step:
                gc.collect()
                gc.freeze()
                gc.disable()
                gc_frozen = True
            elif step % 5000 == 0:
                gc.collect()

        if args.save_every > 0 and step % args.save_every != 0:
            ckpt_path = Path(args.model_out.format(step=step)).expanduser().resolve()
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_gguf(str(ckpt_path), model, backend, model_config, step)
            save_optimizer(str(ckpt_path) + ".optim.npz", optimizer, model, backend, extra_meta={"dataloader_state": dataloader_state})
            print(f"Saved final checkpoint & optimizer: {ckpt_path}")

        peak_mem = approximate_allocated_bytes(runtime, model, optimizer, input_ids, labels)
        print(f"Peak memory usage: {peak_mem / 1024 / 1024:.2f} MiB")
        print(f"Total training time: {total_training_time / 60:.2f}m")
        if val_bpb is not None:
            print(f"Minimum validation bpb: {min_val_bpb:.6f}")
    finally:
        backend.synchronize()
        wandb_run.finish()
        if gc_frozen:
            gc.unfreeze()
            gc.enable()


if __name__ == "__main__":
    main()
