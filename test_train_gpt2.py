import time
import numpy as np
import os

from tensor import Tensor
from module import GPT2
from storage import CPUAllocator, MetalAllocator
from optimizer import get_lr_cosine_schedule, clip_grad_norm, AdamW
from tokenizer import Tokenizer
from dataloader import get_dataloader_batches
from runtime import Runtime

PRESET = "bench"

CONFIGS = {
    "mini": dict(B=2, T=8, V=50304, C=32, n_head=4, n_layer=2, lr=1e-3, steps=20),
    "bench": dict(B=4, T=512, V=50304, C=512, n_head=8, n_layer=6, lr=3e-4, steps=50),
}

def fill_inputs(backend, input_ids, pos_ids, labels, ids, pos, lab):
    backend.copy_from_host(input_ids, ids)
    backend.copy_from_host(pos_ids, pos)
    backend.copy_from_host(labels, lab)

def count_params(params):
    return sum(p.numel for p in params)

def train_gpt2():
    cfg = CONFIGS[PRESET]
    B, T = cfg["B"], cfg["T"]
    V, C, n_head, n_layer = cfg["V"], cfg["C"], cfg["n_head"], cfg["n_layer"]
    lr, steps = cfg["lr"], cfg["steps"]
    dtype = np.float32
    device = "mps"

    Runtime._default_runtime = Runtime(device=device)
    backend = Runtime.get_default().backend

    tokenizer = Tokenizer()
    data_dir = os.path.expanduser("~/.cache/nanochat/base_data_climbmix")
    dataloader = get_dataloader_batches(data_dir, tokenizer, B, T)
    model = GPT2(V, C, T, n_head, B, n_layer, dtype, device)
    input_ids = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=None)
    pos_ids = Tensor(
        (B, T), np.int32, device=device, requires_grad=False,
        init_val=np.tile(np.arange(T, dtype=np.int32), (B, 1)),
    )
    labels = Tensor((B, T), np.int32, device=device, requires_grad=False, init_val=None)
    token_ce = model(input_ids, pos_ids, targets=labels)
    inv_n = Tensor((B, T), dtype, device=device, requires_grad=False, init_val=1.0 / (B * T))
    loss = token_ce * inv_n
    params = model.parameters()

    allocator = Runtime.get_default().allocator
    input_ids.storage = allocator.alloc(input_ids.shape, input_ids.dtype)
    labels.storage = allocator.alloc(labels.shape, labels.dtype)
    pos_ids.storage = allocator.alloc(pos_ids.shape, pos_ids.dtype)
    initial_pos = np.tile(np.arange(T, dtype=np.int32), (B, 1))
    backend.copy_from_host(pos_ids, initial_pos)

    n_params = count_params(params)
    print(f"preset={PRESET} | B={B} T={T} V={V} C={C} n_head={n_head} n_layer={n_layer}")
    print(f"params={n_params:,} | steps={steps} | lr={lr}")
    print(f"{'step':>6} | {'loss':>10} | {'total_ms':>10} | {'bwd_ms':>10} | {'sgd_ms':>10} | note")
    print("-" * 62)

    step_times_ms = []
    bwd_times_ms = []

    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.1)
    
    for step, (ids, lab) in enumerate(dataloader):
        if step >= steps:
            break
            
        pos = np.tile(np.arange(T, dtype=np.int32), (B, 1))
        
        fill_inputs(backend, input_ids, pos_ids, labels, ids, pos, lab)
        
        t0 = time.perf_counter()
        t_bwd = time.perf_counter()
        loss.backward()
        bwd_ms = (time.perf_counter() - t_bwd) * 1000.0
        t_sgd = time.perf_counter()
		
        lr = get_lr_cosine_schedule(step, steps, warmup_steps=10, max_lr=cfg['lr'], min_lr=cfg['lr']/10)
        optimizer.lr = lr
        
        grad_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        sgd_ms = (time.perf_counter() - t_sgd) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0
        step_times_ms.append(total_ms)
        bwd_times_ms.append(bwd_ms)

        note = "compile+bwd" if step == 0 else ""
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            mean_loss = float(np.sum(backend.copy_to_host(loss)))
            print(f"{step:6d} | {mean_loss:10.4f} | {total_ms:10.2f} | {bwd_ms:10.2f} | {sgd_ms:10.2f} | {note}")

    steady_total = step_times_ms[1:] if len(step_times_ms) > 1 else step_times_ms
    steady_bwd = bwd_times_ms[1:] if len(bwd_times_ms) > 1 else bwd_times_ms
    print("-" * 62)
    print(f"step0 total={step_times_ms[0]:.2f}ms (含首次 compile+allocate)")
    if steady_total:
        print(
            f"steady-state (step1..{steps - 1}): "
            f"total avg={np.mean(steady_total):.2f}ms "
            f"min={np.min(steady_total):.2f}ms max={np.max(steady_total):.2f}ms | "
            f"bwd avg={np.mean(steady_bwd):.2f}ms"
        )

if __name__ == "__main__":
    train_gpt2()
