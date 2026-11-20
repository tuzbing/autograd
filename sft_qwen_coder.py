"""SFT Qwen2.5-Coder with the custom Autograd Metal backend.

/Users/a58/Documents/code/nanochat/.venv/bin/python -B sft_qwen_coder.py \
  --nanochat-dir /Users/a58/Documents/code/nanochat \
  --model-dir /Users/a58/Documents/Qwen2.5-Coder-1.5B \
  --data-dir /Users/a58/Documents/code/autograd/sft_data \
  --model-out '/Users/a58/Documents/code/autograd/checkpoints/sft_qwen_coder/model_{step:06d}.gguf' \
  --load-optimizer 0 \
  --max-seq-len 512 \
  --device-batch-size 4 \
  --total-batch-size 16384 \
  --matrix-lr 0.02 \
  --num-iterations -1 \
  --eval-every 200 \
  --eval-tokens 524288

"""
import os
import sys
import gc
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np

from tensor import Tensor
from module import QwenCoder
from runtime import Runtime
from optimizer import AdamW
from gguf_checkpoint import named_parameters
from safetensors_loader import load_safetensors

from sft_autograd import (
    PackedSFT,
    DummyWandb,
    lr_multiplier,
    optimizer_storages,
    storage_view,
    optimizer_signature,
    save_optimizer,
    load_optimizer,
)


class QwenTokenizer:
    def __init__(self, path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Qwen tokenizer requires `tokenizers`") from exc
        self._tokenizer = Tokenizer.from_file(str(path))
        self._special = {}
        for token_id, token in self._tokenizer.get_added_tokens_decoder().items():
            if token.special:
                self._special[token.content] = int(token_id)
        self._bos = self._special["<|endoftext|>"]

    def get_vocab_size(self):
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def get_bos_token_id(self):
        return self._bos

    def get_special_tokens(self):
        return set(self._special)

    def encode_special(self, text):
        return self._special[text]

    def encode(self, text):
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def render_conversation(self, conversation, max_tokens=2048):
        ids, mask = [], []

        def add(value, supervised):
            if isinstance(value, int):
                value = [value]
            ids.extend(value)
            mask.extend([supervised] * len(value))

        messages = conversation["messages"]
        for message in messages:
            role = message["role"]
            content = message["content"]
            add(self.encode_special("<|im_start|>"), False)
            add(self.encode(role), False)
            add(self.encode("\n"), False)
            if isinstance(content, str):
                add(self.encode(content), role == "assistant")
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        raise ValueError(f"Invalid message part: {part!r}")
                    part_type = part.get("type")
                    part_text = part.get("text")
                    if not isinstance(part_text, str):
                        raise ValueError(f"Invalid message part text: {part!r}")
                    supervised = (role == "assistant" and part_type != "python_output")
                    add(self.encode(part_text), supervised)
            else:
                raise ValueError(f"Unsupported message content type: {type(content).__name__}")
            add(self.encode_special("<|im_end|>"), role == "assistant")
            add(self.encode("\n"), False)

        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, conversation):
        messages = conversation["messages"][:-1]
        ids, _ = self.render_conversation({"messages": messages})
        ids.append(self.encode_special("<|im_start|>"))
        ids.extend(self.encode("assistant"))
        ids.extend(self.encode("\n"))
        return ids


def parse_args():
    p = argparse.ArgumentParser(description="Qwen2.5-Coder SFT")
    p.add_argument("--model-dir", default=os.path.expanduser("~/Documents/Qwen2.5-Coder-1.5B"))
    p.add_argument("--nanochat-dir", default=os.path.expanduser("~/Documents/code/nanochat"))
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "sft_data"))
    p.add_argument("--device-type", choices=["mps"], default="mps")
    p.add_argument("--run", default="dummy")
    p.add_argument("--num-iterations", type=int, default=-1)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--device-batch-size", type=int, default=1)
    p.add_argument("--total-batch-size", type=int, default=128)
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--init-lr-frac", type=float, default=0.8)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--warmdown-ratio", type=float, default=0.5)
    p.add_argument("--final-lr-frac", type=float, default=0.0)
    p.add_argument("--buffer-size", type=int, default=100)
    p.add_argument("--mmlu-epochs", type=int, default=3)
    p.add_argument("--gsm8k-epochs", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-tokens", type=int, default=40 * 524288)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--model-out", default=None)
    p.add_argument("--max-norm", type=float, default=2.5)
    p.add_argument("--load-optimizer", type=int, choices=[0, 1], default=0)
    p.add_argument("--grad-dtype", choices=["fp32", "bf16"], default="bf16")
    args = p.parse_args()
    if args.total_batch_size % (args.device_batch_size * args.max_seq_len):
        p.error("total_batch_size must be divisible by device_batch_size*max_seq_len")
    return args


def build_graph(config, batch_size, device, grad_dtype="fp32"):
    runtime = Runtime(device=device)
    Runtime._default_runtime = runtime
    model = QwenCoder(
        V=config["vocab_size"], C=config["n_embd"],
        max_T=config["sequence_len"], n_head=config["n_head"],
        n_layer=config["n_layer"], dtype=np.float32, device=device,
        H=config["intermediate_size"], n_kv_head=config["n_kv_head"],
        eps=config["rms_norm_eps"], rope_theta=config["rope_theta"],
    )
    parameter_grad_dtype = {"fp32": np.float32, "bf16": "bf16"}[grad_dtype]
    for parameter in model.parameters():
        parameter.grad_dtype = parameter_grad_dtype

    shape = (batch_size, config["sequence_len"])
    x = Tensor(shape, np.int32, device=device, requires_grad=False, init_val=None)
    y = Tensor(shape, np.int32, device=device, requires_grad=False, init_val=None)
    loss = model(x, targets=y)
    runtime.loss = loss
    runtime.build_topo(loss)
    runtime.bind_memory()
    runtime.is_compiled = True
    return runtime, model, x, y, loss


def load_qwen_config(model_dir, sequence_len):
    with open(model_dir / "config.json", encoding="utf-8") as f:
        hf = json.load(f)
    return {
        "vocab_size": hf["vocab_size"],
        "n_embd": hf["hidden_size"],
        "sequence_len": sequence_len,
        "n_head": hf["num_attention_heads"],
        "n_kv_head": hf["num_key_value_heads"],
        "n_layer": hf["num_hidden_layers"],
        "intermediate_size": hf["intermediate_size"],
        "rms_norm_eps": hf["rms_norm_eps"],
        "rope_theta": hf["rope_theta"],
    }, hf


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    os.environ.update({
        "NANOCHAT_BASE_DIR": str(data_dir),
        "HF_HOME": str(data_dir / "huggingface"),
        "HF_DATASETS_CACHE": str(data_dir / "huggingface" / "datasets"),
        "HF_HUB_CACHE": str(data_dir / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })
    sys.path.append(os.path.abspath(args.nanochat_dir))

    from tasks.common import TaskMixture
    from tasks.smoltalk import SmolTalk
    from tasks.customjson import CustomJSON
    from tasks.mmlu import MMLU
    from tasks.gsm8k import GSM8K
    from tasks.spellingbee import SimpleSpelling, SpellingBee

    tokenizer = QwenTokenizer(model_dir / "tokenizer.json")
    config, hf_config = load_qwen_config(model_dir, args.max_seq_len)
    # if tokenizer.get_vocab_size() != config["vocab_size"]:
    #     raise ValueError("Qwen tokenizer/model vocabulary mismatch")
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    model_vocab_size = config["vocab_size"]

    if tokenizer_vocab_size > model_vocab_size:
        raise ValueError(
            "Qwen tokenizer vocabulary is larger than model vocabulary: "
            f"{tokenizer_vocab_size} > {model_vocab_size}"
        )

    print(
        f"Qwen vocabulary: tokenizer={tokenizer_vocab_size}, "
        f"model={model_vocab_size}"
    )

    B, T = args.device_batch_size, args.max_seq_len
    accumulation = args.total_batch_size // (B * T)
    runtime, model, x, y, loss = build_graph(config, B, args.device_type, args.grad_dtype)
    backend = runtime.backend

    load_safetensors(
        str(model_dir / "model.safetensors"),
        model,
        backend,
        hf_config,
        strict=True,
    )

    effective_lr = args.matrix_lr * math.sqrt(args.total_batch_size / 524288)
    optimizer = AdamW(
        model.parameters(),
        lr=effective_lr * args.init_lr_frac,
        weight_decay=0.0,
        state_mode="8bit", # fp32/8bit/mini
    )
    clip_grad_norm = optimizer.build_grad_clip()

    if args.load_optimizer:
        load_optimizer(
            str(model_dir / "model.safetensors") + ".optim.npz",
            optimizer, model, backend,
        )

    identity = data_dir / "identity_conversations.jsonl"
    train_dataset = TaskMixture([
        SmolTalk(split="train"),
        CustomJSON(filepath=str(identity)),
        CustomJSON(filepath=str(identity)),
        *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
        *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ])
    train_loader = PackedSFT(
        train_dataset, tokenizer, B, T, args.buffer_size,
        repeat=args.num_iterations > 0,
    )

    if args.run == "dummy":
        wandb_run = DummyWandb()
    else:
        import wandb
        wandb_run = wandb.init(project="qwen-coder-sft", name=args.run, config=vars(args))

    step = 0
    ema = 0.0
    total_time = 0.0
    gc_frozen = False

    def read_loss():
        backend.synchronize()
        return float(np.asarray(backend.copy_to_host(loss)).reshape(-1)[0])

    try:
        while args.num_iterations < 0 or step < args.num_iterations:
            batches = []
            for _ in range(accumulation):
                try:
                    batches.append(next(train_loader))
                except StopIteration:
                    break
            if not batches:
                break

            t0 = time.perf_counter()
            micro_losses = []
            actual_accumulation = len(batches)
            for micro_step, (inputs, targets) in enumerate(batches):
                supervised = int(np.count_nonzero(targets >= 0))
                if supervised == 0:
                    continue
                backend.copy_from_host(x, inputs)
                backend.copy_from_host(y, targets)
                normalization = (B * T) / supervised
                loss.backward(
                    grad_scale=normalization / actual_accumulation,
                    accumulate_parameter_grads=micro_step > 0,
                )
                micro_losses.append(read_loss() * normalization)

            if not micro_losses:
                continue
            progress = ((step + 1) / args.num_iterations
                        if args.num_iterations > 0 else train_loader.progress)
            grad_norm, clip_coef = clip_grad_norm(args.max_norm)
            if not math.isfinite(grad_norm):
                step += 1
                continue

            optimizer.lr = effective_lr * args.init_lr_frac * lr_multiplier(progress, args)
            optimizer.step(grad_scale=clip_coef)
            backend.synchronize()
            dt = time.perf_counter() - t0
            step += 1
            total_time += dt

            if args.save_every > 0 and step % args.save_every == 0:
                output = Path(args.model_out or f"qwen_coder_sft_{step:06d}.gguf")
                output.parent.mkdir(parents=True, exist_ok=True)
                from gguf_checkpoint import save_gguf
                save_gguf(str(output), model, backend, config, step)
                save_optimizer(str(output) + ".optim.npz", optimizer, model, backend)

            train_loss = float(np.mean(micro_losses))
            ema = 0.9 * ema + 0.1 * train_loss
            smooth = ema / (1 - 0.9 ** step)
            print(
                f"step {step:05d} ({100 * progress:.2f}%) | "
                f"loss: {smooth:.6f} | dt: {1000 * dt:.2f}ms | "
                f"epoch: {train_loader.epoch}"
            )
            if step % 10 == 0:
                wandb_run.log({"step": step, "train/loss": smooth})
            if step == 1:
                gc.collect()
                gc.freeze()
                gc.disable()
                gc_frozen = True

        if step == 0:
            raise RuntimeError("No supervised training batches were produced")
    finally:
        backend.synchronize()
        wandb_run.finish()
        if gc_frozen:
            gc.unfreeze()


if __name__ == "__main__":
    main()
