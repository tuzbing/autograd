"""
Train model using the custom autograd framework.
Usage:
python train_autograd.py --depth=24 --target-param-data-ratio=8 --device-batch-size=2
"""
import os
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager
import wandb
import numpy as np
# Autograd framework imports
from tensor import Tensor
from module import GPT2
from runtime import Runtime
from optimizer import AdamW, get_lr_cosine_schedule, clip_grad_norm
from tokenizer import Tokenizer
from dataloader import get_dataloader_batches
# Try to import nanochat common utilities to match base_train.py exactly
try:
    from nanochat.common import DummyWandb
except ImportError:
    class DummyWandb:
        def log(self, *args, **kwargs): pass
        def finish(self): pass
# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model with Autograd")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--device-type", type=str, default="mps", help="cuda|cpu|mps")
parser.add_argument("--fp8", action="store_true", help="enable FP8 training")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"])
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters")
parser.add_argument("--weight-decay", type=float, default=0.28, help="weight decay")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step")
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps")
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------
# wandb logging init
use_dummy_wandb = args.run == "dummy"
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)
# -----------------------------------------------------------------------------
# Helper functions for Static Graph Gradient Management
def zero_param_grads(model, backend):
    for p in model.parameters():
        if p.requires_grad and getattr(p, 'grad', None) is not None and getattr(p.grad, 'storage', None) is not None:
            backend.fill_storage(p.grad.storage, 0.0)
def zero_intermediate_grads(loss_tensor, backend, model):
    param_set = set(id(p) for p in model.parameters())
    visited = set()
    def traverse(t):
        if id(t) in visited: return
        visited.add(id(t))
        if id(t) not in param_set and t.requires_grad and getattr(t, 'grad', None) is not None and getattr(t.grad, 'storage', None) is not None:
            backend.fill_storage(t.grad.storage, 0.0)
        if getattr(t, 'op', None) is not None:
            for inp in t.op.inputs:
                traverse(inp)
    traverse(loss_tensor)
# -----------------------------------------------------------------------------
# Setup Runtime and Device
device = args.device_type
Runtime._default_runtime = Runtime(device=device)
rt = Runtime.get_default()
backend = rt.backend
allocator = rt.allocator
ddp_world_size = 1 
gpu_peak_flops = 10.4e12 
tokenizer = Tokenizer()
vocab_size = 50304
print(f"Vocab size: {vocab_size:,}")
# -----------------------------------------------------------------------------
# Model Configuration
base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
n_layer = args.depth
model = GPT2(vocab_size, model_dim, args.max_seq_len, num_heads, args.device_batch_size, n_layer, np.float32, device)
num_params = sum(p.numel for p in model.parameters())
num_scaling_params = num_params 
num_flops_per_token = 6 * num_params + 12 * n_layer * args.head_dim * model_dim
print(f"Model config: B={args.device_batch_size}, T={args.max_seq_len}, V={vocab_size}, C={model_dim}, n_head={num_heads}, n_layer={n_layer}")
print(f"Number of parameters: {num_params:,}")
# -----------------------------------------------------------------------------
# Calculate Training Horizon & Batch Size
total_batch_size = args.total_batch_size
if total_batch_size == -1:
    total_batch_size = 524288 
grad_accum_steps = total_batch_size // (args.device_batch_size * args.max_seq_len * ddp_world_size)
if grad_accum_steps < 1:
    grad_accum_steps = 1
    total_batch_size = args.device_batch_size * args.max_seq_len * ddp_world_size
if args.num_iterations > 0:
    num_iterations = args.num_iterations
elif args.target_flops > 0:
    num_iterations = int(args.target_flops / (num_flops_per_token * total_batch_size))
elif args.target_param_data_ratio > 0:
    total_tokens = args.target_param_data_ratio * num_scaling_params
    num_iterations = int(total_tokens / total_batch_size)
else:
    num_iterations = 100 
print(f"Calculated num_iterations: {num_iterations}")
print(f"Gradient Accumulation steps: {grad_accum_steps}")
print(f"Total batch size (tokens): {total_batch_size}")
# -----------------------------------------------------------------------------
# Setup inputs and loss graph
B = args.device_batch_size
T = args.max_seq_len
input_ids = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=None)
pos_ids = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=None)
labels = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=None)
# logits = model(input_ids, pos_ids)
# token_ce = logits.cross_entropy(labels)
token_ce = model(input_ids, pos_ids, targets=labels)
inv_n = Tensor((B, T), np.float32, device=device, requires_grad=False, init_val=1.0 / (B * T * grad_accum_steps))
loss = token_ce * inv_n
input_ids.storage = allocator.alloc(input_ids.shape, input_ids.dtype)
labels.storage = allocator.alloc(labels.shape, labels.dtype)
pos_ids.storage = allocator.alloc(pos_ids.shape, pos_ids.dtype)
initial_pos = np.tile(np.arange(T, dtype=np.int32), (B, 1))
backend.copy_from_host(pos_ids, initial_pos)
optimizer = AdamW(model.parameters(), lr=args.matrix_lr, weight_decay=args.weight_decay)
# -----------------------------------------------------------------------------
# Dataloader wrapper
data_dir = os.path.expanduser("~/.cache/nanochat/base_data_climbmix")
def get_stateful_dataloader():
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory {data_dir} not found. Using dummy data.")
        while True:
            yield np.random.randint(0, vocab_size, (B, T)).astype(np.int32), np.random.randint(0, vocab_size, (B, T)).astype(np.int32), {'epoch': 0, 'pq_idx': 0, 'rg_idx': 0}
    else:
        base_loader = get_dataloader_batches(data_dir, tokenizer, B, T)
        for i, (x, y) in enumerate(base_loader):
            yield x, y, {'epoch': 0, 'pq_idx': i, 'rg_idx': 0}
train_loader = get_stateful_dataloader()
# -----------------------------------------------------------------------------
# Training Loop
total_training_time = 0.0
smooth_train_loss = 0.0
ema_beta = 0.9
step = 0
x_data, y_data, dataloader_state_dict = next(train_loader)
while step < num_iterations:
    last_step = (step == num_iterations - 1)
    flops_so_far = num_flops_per_token * total_batch_size * step
    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        # TODO: Implement evaluate_bpb for autograd framework
        pass
    # once in a while: estimate the CORE metric (all ranks participate)
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        # TODO: Implement evaluate_core for autograd framework
        pass
    # once in a while: sample from the model (only on master process)
    if args.sample_every > 0 and (last_step or (step > 0 and step % args.sample_every == 0)):
        # TODO: Implement generation/sampling for autograd framework
        pass
    # save checkpoint: at the end of the run, or every save_every steps
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        # TODO: Implement save_checkpoint for autograd framework
        pass
    if last_step:
        break
    # -------------------------------------------------------------------------
    # single training step
    rt.backend.synchronize()
    t0 = time.time()
    
    zero_param_grads(model, backend)
    train_loss_accum = 0.0
    
    for micro_step in range(grad_accum_steps):
        backend.copy_from_host(input_ids, x_data)
        backend.copy_from_host(labels, y_data)
        
        zero_intermediate_grads(loss, backend, model)
        loss.backward()
        
        loss_val = float(np.sum(backend.copy_to_host(loss))) * grad_accum_steps
        train_loss_accum += loss_val / grad_accum_steps
        
        x_data, y_data, dataloader_state_dict = next(train_loader)
    lrm = get_lr_cosine_schedule(step, num_iterations, warmup_steps=args.warmup_steps, max_lr=1.0, min_lr=args.final_lr_frac)
    optimizer.lr = args.matrix_lr * lrm
    
    clip_grad_norm(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    rt.backend.synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------
    # logging (CPU action only)
    if step == 0:
        smooth_train_loss = train_loss_accum
    else:
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_accum
        
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    
    if step > 10:
        total_training_time += dt
        
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
        
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    
    print(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    
    # EXACT MATCH: wandb logging block from base_train.py
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)
    step += 1
    
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()
print(f"Total training time: {total_training_time/60:.2f}m")
# -----------------------------------------------------------------------------
# Log to report (matching base_train.py)
try:
    from nanochat.report import get_report
    get_report().log(section="Base model training (Autograd)", data=[
        user_config,
        {
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "DDP world size": ddp_world_size,
            "warmup_steps": args.warmup_steps,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_frac": args.final_lr_frac,
        },
        {
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time/60:.2f}m",
        }
    ])
except ImportError:
    pass
wandb_run.finish()