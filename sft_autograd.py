"""
SFT using this repository's GPT2 / Runtime / AdamW-mini.

/Users/a58/Documents/code/nanochat/.venv/bin/python -B sft_autograd.py \
  --nanochat-dir /Users/a58/Documents/code/nanochat \
  --model-in /Users/a58/Documents/code/autograd/checkpoints/base_d6/fp32_model_005000.gguf \
  --model-out '/Users/a58/Documents/code/autograd/checkpoints/sft_d6/model_{step:06d}.gguf' \
  --load-optimizer 0 \
  --max-seq-len 512 \
  --device-batch-size 32 \
  --total-batch-size 16384 \
  --matrix-lr 0.02 \
  --num-iterations -1 \
  --eval-every 200 \
  --eval-tokens 524288 \
  --chatcore-every -1

"""

import os
import sys
import gc
import json
import math
import time
import struct
import argparse
from collections import deque
from functools import partial
from pathlib import Path

import numpy as np

from tensor import Tensor
from module import GPT2
from runtime import Runtime
from optimizer import AdamW
from gguf_checkpoint import (
    load_gguf,
    save_gguf,
    named_parameters,
    _read_exact,
    _read_string,
    _read_value,
)


class DummyWandb:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="SFT with the custom Autograd Metal backend"
    )
    p.add_argument(
        "--nanochat-dir",
        default=os.path.expanduser("~/Documents/code/nanochat"),
    )
    p.add_argument("--run", default="dummy")
    p.add_argument("--device-type", choices=["mps"], default="mps")

    p.add_argument("--model-in", required=True)
    p.add_argument("--model-out", default=None)
    p.add_argument("--model-tag", default=None)
    p.add_argument("--load-optimizer", type=int, choices=[0, 1], default=1)

    p.add_argument("--num-iterations", type=int, default=-1)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=None)
    p.add_argument("--total-batch-size", type=int, default=None)

    # Same optimizer convention as train_autograd.py:
    # one AdamW-mini LR, scaled by sqrt(total_batch_size / 524288).
    p.add_argument("--matrix-lr", type=float, default=None)
    p.add_argument("--init-lr-frac", type=float, default=0.8)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--warmdown-ratio", type=float, default=0.5)
    p.add_argument("--final-lr-frac", type=float, default=0.0)

    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-tokens", type=int, default=40 * 524288)
    p.add_argument("--chatcore-every", type=int, default=200)
    p.add_argument("--chatcore-max-cat", type=int, default=-1)
    p.add_argument("--chatcore-max-sample", type=int, default=24)

    p.add_argument("--mmlu-epochs", type=int, default=3)
    p.add_argument("--gsm8k-epochs", type=int, default=4)
    p.add_argument("--buffer-size", type=int, default=100)
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "sft_data"))
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--max-norm", type=float, default=2.5)
    p.add_argument("--grad-dtype", choices=["fp32", "bf16"], default="fp32")

    args = p.parse_args()

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("The current custom backend supports single-process SFT only")
    if args.num_iterations != -1 and args.num_iterations < 1:
        p.error("--num-iterations must be -1 or a positive integer")
    if args.buffer_size < 1:
        p.error("--buffer-size must be positive")
    if args.mmlu_epochs < 0 or args.gsm8k_epochs < 0:
        p.error("Dataset epoch counts must be nonnegative")
    if args.eval_tokens < 1:
        p.error("--eval-tokens must be positive")
    if args.chatcore_max_cat == 0 or args.chatcore_max_sample == 0:
        p.error("ChatCORE limits must be negative or positive, not zero")
    if not 0 <= args.warmup_ratio <= 1:
        p.error("--warmup-ratio must be in [0, 1]")
    if not 0 <= args.warmdown_ratio <= 1:
        p.error("--warmdown-ratio must be in [0, 1]")
    if args.warmup_ratio + args.warmdown_ratio > 1:
        p.error("warmup_ratio + warmdown_ratio must be <= 1")
    if not 0 <= args.final_lr_frac <= 1 or args.init_lr_frac <= 0:
        p.error("Invalid LR fractions")

    return args


def read_gguf_metadata(path):
    """Read metadata only; do not allocate model tensors."""
    with open(path, "rb") as f:
        if _read_exact(f, 4) != b"GGUF":
            raise ValueError(f"Not a GGUF checkpoint: {path}")
        version, _, count = struct.unpack("<IQQ", _read_exact(f, 20))
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        metadata = {}
        for _ in range(count):
            name = _read_string(f)
            kind = struct.unpack("<I", _read_exact(f, 4))[0]
            metadata[name] = _read_value(f, kind)

    if metadata.get("general.architecture") != "autograd-gpt2":
        raise ValueError("Expected this repository's autograd-gpt2 checkpoint")
    return metadata


def build_graph(config, batch_size, device, grad_dtype="fp32"):
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
    parameter_grad_dtype = {"fp32": np.float32, "bf16": "bf16"}[grad_dtype]
    for parameter in model.parameters():
        parameter.grad_dtype = parameter_grad_dtype

    shape = (batch_size, config["sequence_len"])
    x = Tensor(
        shape, np.int32, device=device,
        requires_grad=False, init_val=None,
    )
    y = Tensor(
        shape, np.int32, device=device,
        requires_grad=False, init_val=None,
    )
    loss = model(x, targets=y)

    runtime.loss = loss
    runtime.build_topo(loss)
    runtime.bind_memory()
    runtime.is_compiled = True
    return runtime, model, x, y, loss


class PackedSFT:
    """
    BOS-aligned best-fit packing.

    A finite loader drains its buffer before stopping, so an epoch consumes
    every conversation exactly once. Infinite loaders repeat the same
    deterministically shuffled TaskMixture, as in nanochat.
    """

    def __init__(
        self, dataset, tokenizer, batch_size, sequence_len,
        buffer_size=100, repeat=False,
    ):
        if len(dataset) == 0:
            raise ValueError("Empty SFT dataset")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.capacity = sequence_len + 1
        self.buffer_size = buffer_size
        self.repeat = repeat
        self.buffer = []
        self.cursor = 0
        self.consumed = 0
        self.eof = False
        self.empty_batches = 0

    def __iter__(self):
        return self

    def refill(self):
        while len(self.buffer) < self.buffer_size and not self.eof:
            if self.cursor == len(self.dataset):
                if not self.repeat:
                    self.eof = True
                    break
                self.cursor = 0

            conversation = self.dataset[self.cursor]
            self.cursor += 1

            # Reference tokenizer truncates conversations to 2048 tokens.
            # Also constrain to row capacity when using a shorter context;
            # otherwise an oversized conversation can block best-fit packing.
            ids, mask = self.tokenizer.render_conversation(
                conversation,
                max_tokens=min(2048, self.capacity),
            )
            if not ids or len(ids) != len(mask):
                raise ValueError("Invalid tokenized conversation")
            self.buffer.append((ids, mask))

    def __next__(self):
        while True:
            bos = self.tokenizer.get_bos_token_id()
            rows = np.full(
                (self.batch_size, self.capacity), bos, dtype=np.int32
            )
            masks = np.zeros_like(rows, dtype=np.bool_)
            used = False

            for row_index in range(self.batch_size):
                pos = 0
                while pos < self.capacity:
                    self.refill()
                    remaining = self.capacity - pos
                    best_index = -1
                    best_length = 0

                    for i, (ids, _) in enumerate(self.buffer):
                        if best_length < len(ids) <= remaining:
                            best_index = i
                            best_length = len(ids)

                    if best_index < 0:
                        break

                    ids, mask = self.buffer.pop(best_index)
                    end = pos + len(ids)
                    rows[row_index, pos:end] = ids
                    masks[row_index, pos:end] = mask
                    pos = end
                    self.consumed += 1
                    used = True

            if not used:
                raise StopIteration

            x = rows[:, :-1].copy()
            y = rows[:, 1:].copy()
            y[~masks[:, 1:]] = -1

            if np.any(y >= 0):
                self.empty_batches = 0
                return x, y

            # Truncation can leave a conversation containing only a prompt.
            # Such batches must not create NaN losses or optimizer updates.
            self.empty_batches += 1
            if self.repeat and self.empty_batches >= 1000:
                raise RuntimeError(
                    "1000 batches contain no supervised tokens; "
                    "increase --max-seq-len or inspect the dataset"
                )

    @property
    def progress(self):
        return min(1.0, self.consumed / len(self.dataset))

    @property
    def epoch(self):
        return 1 + max(0, self.consumed - 1) // len(self.dataset)


def lr_multiplier(progress, args):
    progress = min(1.0, max(0.0, progress))
    if args.warmup_ratio and progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    if progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    if args.warmdown_ratio == 0:
        return 1.0
    decay = (
        progress - (1.0 - args.warmdown_ratio)
    ) / args.warmdown_ratio
    return (1.0 - decay) + decay * args.final_lr_frac


def optimizer_storages(optimizer):
    if not isinstance(optimizer.dispatches, tuple):
        raise TypeError(
            "Optimizer checkpoint requires Metal tuple dispatches"
        )

    result = {}
    for i, storage in enumerate(optimizer.m):
        result[f"m_{i}"] = storage
    for i, storage in enumerate(optimizer.v):
        result[f"v_{i}"] = storage

    if optimizer.state_mode != "fp32":
        for i, item in enumerate(optimizer.dispatches):
            kind = item[0]

            if kind == "mini8bit":
                # 保留已有键名，兼容原来的 mini checkpoint。
                result[f"absmax_m_{i}"] = item[4]

            elif kind == "q8":
                result[f"absmax_m_{i}"] = item[5]
                result[f"absmax_v_{i}"] = item[6]

            elif kind != "fp32":
                raise ValueError(
                    f"Unsupported optimizer dispatch: {kind!r}"
                )

    if optimizer.qmap_m is not None:
        result["qmap_m"] = optimizer.qmap_m
    if optimizer.qmap_v is not None:
        result["qmap_v"] = optimizer.qmap_v

    return result


def storage_view(backend, storage):
    tensor = Tensor(
        storage.shape, storage.dtype,
        device="mps", requires_grad=False,
    )
    tensor.storage = storage
    return backend.numpy_view(tensor)


def optimizer_signature(optimizer, model):
    if not isinstance(optimizer.dispatches, tuple):
        raise TypeError(
            "Optimizer checkpoint requires Metal tuple dispatches"
        )

    dispatches = []
    for item in optimizer.dispatches:
        if optimizer.state_mode == "fp32":
            # (p, grad, m, v, numel, decay_numel)
            if len(item) != 6:
                raise ValueError("Unexpected fp32 dispatch layout")
            dispatches.append([
                "fp32_group", int(item[4]), int(item[5])
            ])
            continue

        kind = item[0]
        if kind == "mini8bit":
            # (kind, p, grad, m, meta, v, n, ...)
            if len(item) != 14:
                raise ValueError("Unexpected mini8bit dispatch layout")
            dispatches.append([kind, *map(int, item[6:])])

        elif kind == "q8":
            # (kind, p, grad, qm, qv,
            #  absmax_m, absmax_v, numel, apply_wd)
            if len(item) != 9:
                raise ValueError("Unexpected q8 dispatch layout")
            dispatches.append([
                kind, int(item[7]), int(item[8])
            ])

        elif kind == "fp32":
            # 小参数不量化：
            # (kind, p, grad, m, v, numel, apply_wd)
            if len(item) != 7:
                raise ValueError("Unexpected fp32 dispatch layout")
            dispatches.append([
                kind, int(item[5]), int(item[6])
            ])

        else:
            raise ValueError(f"Unsupported optimizer dispatch: {kind!r}")

    return {
        "state_mode": optimizer.state_mode,
        "parameters": [
            [name, list(tensor.shape)]
            for name, tensor in named_parameters(model)
        ],
        "dispatches": dispatches,
    }

def save_optimizer(path, optimizer, model, backend):
    backend.synchronize()
    metadata = optimizer_signature(optimizer, model)
    metadata.update({
        "t": optimizer.t,
        "betas": list(optimizer.betas),
        "eps": optimizer.eps,
    })
    arrays = {
        name: storage_view(backend, storage).copy()
        for name, storage in optimizer_storages(optimizer).items()
    }
    # Opening explicitly prevents NumPy from changing the supplied filename.
    with open(path, "wb") as f:
        np.savez(f, metadata=np.array(json.dumps(metadata)), **arrays)


def load_optimizer(path, optimizer, model, backend):
    if not os.path.isfile(path):
        print("Optimizer checkpoint not found; starting fresh AdamW-mini")
        return

    backend.synchronize()
    storages = optimizer_storages(optimizer)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        expected = optimizer_signature(optimizer, model)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Incompatible optimizer state: {key}")
        if set(data.files) != set(storages) | {"metadata"}:
            raise ValueError("Optimizer buffer names do not match")

        # Validate all arrays before modifying any optimizer buffer.
        for name, storage in storages.items():
            array = data[name]
            if (
                array.shape != storage.shape
                or array.dtype != np.dtype(storage.dtype)
            ):
                raise ValueError(f"Optimizer buffer mismatch: {name}")

        for name, storage in storages.items():
            storage_view(backend, storage)[:] = data[name]

    optimizer.t = int(metadata["t"])
    optimizer.betas = tuple(metadata["betas"])
    optimizer.eps = float(metadata["eps"])
    # Deliberately retain newly configured SFT LR and zero weight decay.
    print(f"Loaded AdamW-mini state at optimizer step {optimizer.t}")


class ChatEvaluator:
    """
    Separate batch-1 static graph for ChatCORE.

    Transformer forward uses the custom backend. Only the selected final
    hidden vector is projected through the current LM-head weights in NumPy,
    avoiding a full B*T*V logits allocation.

    This is greedy evaluation without KV caching.
    """

    def __init__(self, config, training_runtime, device):
        inference_config = dict(config)

        # render_for_completion: at most 2048 + assistant_start.
        # Before the 512th generated token: at most 2049 + 511 = 2560.
        inference_config["sequence_len"] = 2560
        try:
            (
                self.runtime, self.model, self.x, self.y, self.loss,
            ) = build_graph(inference_config, 1, device)
        finally:
            Runtime._default_runtime = training_runtime

        self.backend = self.runtime.backend
        self.hidden = self.loss.op.inputs[0]
        self.head = None

    def refresh(self, source):
        self.backend.synchronize()
        source_params = dict(named_parameters(source))
        for name, tensor in named_parameters(self.model):
            values = np.asarray(
                self.backend.copy_to_host(source_params[name]),
                dtype=np.float32,
            )
            self.backend.copy_from_host(tensor, values)
        self.head = np.asarray(
            self.backend.copy_to_host(source.lm_head.weight),
            dtype=np.float32,
        ).copy()

    def next_logits(self, tokens, bos):
        if not tokens or len(tokens) > self.x.shape[1]:
            raise ValueError("ChatCORE prompt exceeds inference context")

        padded = np.full(self.x.shape, bos, dtype=np.int32)
        padded[0, :len(tokens)] = tokens
        self.backend.copy_from_host(self.x, padded)

        # Execute the existing transformer graph up to the fused CE node.
        self.backend.begin_pass()
        try:
            for node in self.runtime.topo:
                if node is self.loss.op:
                    break
                node.forward(node.inputs, node.outputs)
        finally:
            self.backend.end_pass()

        self.backend.synchronize()
        hidden = np.asarray(self.backend.copy_to_host(self.hidden))
        return hidden[0, len(tokens) - 1] @ self.head.T

    def generate(self, prompt, tokenizer, use_calculator):
        tokens = list(prompt)
        completion = []
        forced = deque()
        expression = []
        in_python = False

        special = tokenizer.encode_special
        bos = tokenizer.get_bos_token_id()
        assistant_end = special("<|assistant_end|>")
        python_start = special("<|python_start|>")
        python_end = special("<|python_end|>")
        output_start = special("<|output_start|>")
        output_end = special("<|output_end|>")

        # Same default as run_chat_eval: one sample, temperature=0,
        # max_new_tokens=512. Forced calculator tokens count toward the limit.
        for _ in range(512):
            if forced:
                token = forced.popleft()
            else:
                token = int(np.argmax(self.next_logits(tokens, bos)))

            tokens.append(token)
            if token in (assistant_end, bos):
                break
            completion.append(token)

            if token == python_start:
                in_python = True
                expression = []
            elif token == python_end and in_python:
                in_python = False
                if expression:
                    result = use_calculator(tokenizer.decode(expression))
                    if result is not None:
                        forced.append(output_start)
                        forced.extend(tokenizer.encode(str(result)))
                        forced.append(output_end)
                expression = []
            elif in_python:
                expression.append(token)

        return tokenizer.decode(completion)

    def evaluate(self, tokenizer, args):
        from tasks.arc import ARC
        from tasks.mmlu import MMLU
        from tasks.gsm8k import GSM8K
        from tasks.humaneval import HumanEval
        from tasks.spellingbee import SpellingBee
        from nanochat.engine import use_calculator

        factories = {
            "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
            "ARC-Challenge": partial(
                ARC, subset="ARC-Challenge", split="test"
            ),
            "MMLU": partial(MMLU, subset="all", split="test"),
            "GSM8K": partial(GSM8K, subset="main", split="test"),
            "HumanEval": HumanEval,
            "SpellingBee": partial(SpellingBee, size=256, split="test"),
        }
        categorical = {"ARC-Easy", "ARC-Challenge", "MMLU"}
        baseline = {name: 0.25 if name in categorical else 0.0
                    for name in factories}

        results = {}
        letter_ids = {}
        bos = tokenizer.get_bos_token_id()

        for name, factory in factories.items():
            task = factory()
            limit = (
                args.chatcore_max_cat if name in categorical
                else args.chatcore_max_sample
            )
            count = len(task) if limit < 0 else min(limit, len(task))
            if count == 0:
                raise ValueError(f"Empty ChatCORE task: {name}")

            passed = 0
            for index in range(count):
                conversation = task[index]
                prompt = tokenizer.render_for_completion(conversation)

                if name in categorical:
                    logits = self.next_logits(prompt, bos)
                    ids = []
                    for letter in conversation["letters"]:
                        if letter not in letter_ids:
                            encoded = tokenizer.encode(letter)
                            if len(encoded) != 1:
                                raise ValueError("Choice must be one token")
                            letter_ids[letter] = encoded[0]
                        ids.append(letter_ids[letter])

                    prediction = conversation["letters"][
                        int(np.argmax(logits[ids]))
                    ]
                else:
                    prediction = self.generate(
                        prompt, tokenizer, use_calculator
                    )

                passed += int(task.evaluate(conversation, prediction))
                print(
                    f"\r{name}: {index + 1}/{count}, "
                    f"accuracy={passed / (index + 1):.4f}",
                    end="", flush=True,
                )

            print()
            results[name] = passed / count

        def centered(names):
            return sum(
                (results[name] - baseline[name]) / (1 - baseline[name])
                for name in names
            ) / len(names)

        return {
            "chatcore_metric": centered(list(factories)),
            "chatcore_cat": centered(sorted(categorical)),
            **{f"chatcore/{name}": value
               for name, value in results.items()},
        }

def save_checkpoint(
    args,
    step_to_save,
    *,
    model,
    optimizer,
    backend,
    config,
    base_dir,
    input_config,
    effective_lr,
    val_bpb,
    val_bpb_step,
):
    import tempfile

    tag = args.model_tag or f"d{config['n_layer']}"
    template = args.model_out or os.path.join(
        base_dir,
        "chatsft_checkpoints",
        tag,
        "model_{step:06d}.gguf",
    )
    output = Path(
        template.format(step=step_to_save)
    ).expanduser().resolve()

    if args.save_every > 0:
        next_output = Path(
            template.format(step=step_to_save + args.save_every)
        ).expanduser().resolve()
        if next_output == output:
            raise ValueError(
                "--model-out 必须包含 {step} 或 {step:06d}"
            )

    if output == Path(args.model_in).expanduser().resolve():
        raise ValueError("输出 checkpoint 不能覆盖输入模型")

    optimizer_path = Path(str(output) + ".optim.npz")
    metadata_path = Path(str(output) + ".json")

    metadata = {
        "step": step_to_save,
        "val_bpb": val_bpb,
        "val_bpb_step": val_bpb_step,
        "model_config": config,
        "user_config": input_config,
        "resolved_config": vars(args).copy(),
        "max_seq_len": args.max_seq_len,
        "device_batch_size": args.device_batch_size,
        "total_batch_size": args.total_batch_size,
        "optimizer": "AdamW",
        "optimizer_state_mode": optimizer.state_mode,
        "effective_base_lr": effective_lr,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    # 先写完所有临时文件，最后发布 GGUF。
    with tempfile.TemporaryDirectory(
        prefix=".sft-save-", dir=str(output.parent)
    ) as temporary_dir:
        temporary = Path(temporary_dir)
        model_tmp = temporary / output.name
        optimizer_tmp = temporary / "optimizer.npz"
        metadata_tmp = temporary / "metadata.json"

        save_gguf(
            str(model_tmp), model, backend, config, step_to_save
        )
        save_optimizer(
            str(optimizer_tmp), optimizer, model, backend
        )
        with metadata_tmp.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        os.replace(optimizer_tmp, optimizer_path)
        os.replace(metadata_tmp, metadata_path)
        os.replace(model_tmp, output)

    print(f"Saved checkpoint: {output}")
    return str(output)


def main():
    args = parse_args()
    input_config = vars(args).copy()

    data_dir = Path(args.data_dir).expanduser().resolve()
    args.data_dir = str(data_dir)

    # 必须在导入 nanochat / tasks / datasets 之前设置。
    os.environ.update({
        "NANOCHAT_BASE_DIR": str(data_dir),
        "HF_HOME": str(data_dir / "huggingface"),
        "HF_DATASETS_CACHE": str(data_dir / "huggingface" / "datasets"),
        "HF_HUB_CACHE": str(data_dir / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    })

    # 这两个普通文件使用独立下载逻辑，不受 HF 离线开关控制。
    # 提前检查，确保缺失时直接报错，不触发自动下载。
    for relative_path in (
        "tokenizer/tokenizer.pkl",
        "identity_conversations.jsonl",
        "words_alpha.txt",
    ):
        path = data_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"缺少本地 SFT 文件: {path}")

    for relative_path in ("huggingface/datasets", "huggingface/hub"):
        path = data_dir / relative_path
        if not path.is_dir():
            raise FileNotFoundError(f"缺少本地数据缓存目录: {path}")

    # Keep local tensor/module/runtime imports resolved to this repository.
    sys.path.append(os.path.abspath(args.nanochat_dir))

    from nanochat.tokenizer import RustBPETokenizer
    from tasks.common import TaskMixture
    from tasks.smoltalk import SmolTalk
    from tasks.customjson import CustomJSON
    from tasks.mmlu import MMLU
    from tasks.gsm8k import GSM8K
    from tasks.spellingbee import SimpleSpelling, SpellingBee

    base_dir = str(data_dir)
    tokenizer = RustBPETokenizer.from_directory(
        os.path.join(base_dir, "tokenizer")
    )

    gguf_meta = read_gguf_metadata(args.model_in)
    saved_config = json.loads(gguf_meta["autograd.model_config"])
    sidecar_path = args.model_in + ".json"
    pretrain_meta = {}
    if os.path.isfile(sidecar_path):
        with open(sidecar_path, encoding="utf-8") as f:
            pretrain_meta = json.load(f)
    previous = pretrain_meta.get(
        "resolved_config", pretrain_meta.get("user_config", {})
    )

    defaults = {
        "max_seq_len": saved_config["sequence_len"],
        "device_batch_size": 32,
        "total_batch_size": 16384,
        "matrix_lr": 0.02,
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            inherited = pretrain_meta.get(name)
            if inherited is None:
                inherited = previous.get(name)
            value = fallback if inherited is None else inherited
            setattr(args, name, value)
            print(f"Resolved {name}={value}")

    B, T = args.device_batch_size, args.max_seq_len
    if min(B, T, args.total_batch_size) < 1:
        raise ValueError("Batch sizes and context length must be positive")
    if args.total_batch_size % (B * T):
        raise ValueError("total_batch_size must be divisible by B*T")
    if args.matrix_lr <= 0:
        raise ValueError("matrix_lr must be positive")
    accumulation = args.total_batch_size // (B * T)

    config = dict(saved_config)
    config["sequence_len"] = T
    if config["vocab_size"] != tokenizer.get_vocab_size():
        raise ValueError("Checkpoint/tokenizer vocabulary size mismatch")
    if config["n_kv_head"] != config["n_head"]:
        raise ValueError("The current GPT2 implementation does not use GQA")
    if config.get("window_pattern", "L") != "L":
        raise ValueError("The current GPT2 implementation uses full attention")

    np.random.seed(42)
    runtime, model, x, y, loss = build_graph(config, B, args.device_type)
    backend = runtime.backend

    # Validate against the saved configuration. RoPE context length can be
    # changed independently because the checkpoint contains trainable weights.
    load_gguf(args.model_in, model, backend, saved_config)

    effective_lr = args.matrix_lr * math.sqrt(
        args.total_batch_size / 524288
    )
    initial_lr = effective_lr * args.init_lr_frac
    optimizer = AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=0.0,
        state_mode="8bit", # fp32/8bit/mini
    )
    clip_grad_norm = optimizer.build_grad_clip()

    if args.load_optimizer:
        load_optimizer(
            args.model_in + ".optim.npz", optimizer, model, backend
        )

    identity = os.path.join(base_dir, "identity_conversations.jsonl")
    train_dataset = TaskMixture([
        SmolTalk(split="train"),
        CustomJSON(filepath=identity),
        CustomJSON(filepath=identity),
        *[
            MMLU(subset="all", split="auxiliary_train")
            for _ in range(args.mmlu_epochs)
        ],
        *[
            GSM8K(subset="main", split="train")
            for _ in range(args.gsm8k_epochs)
        ],
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ])
    val_dataset = TaskMixture([
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ])

    def make_loader(dataset, repeat):
        return PackedSFT(
            dataset, tokenizer, B, T, args.buffer_size, repeat=repeat
        )

    train_loader = make_loader(
        train_dataset, repeat=args.num_iterations > 0
    )

    token_bytes = np.zeros(config["vocab_size"], dtype=np.int64)
    special_ids = {
        tokenizer.encode_special(token)
        for token in tokenizer.enc.special_tokens_set
    }
    for token_id in range(len(token_bytes)):
        if token_id not in special_ids:
            token_bytes[token_id] = len(
                tokenizer.enc.decode_single_token_bytes(token_id)
            )

    def read_loss():
        backend.synchronize()
        return float(np.asarray(backend.copy_to_host(loss)).reshape(-1)[0])

    def evaluate_bpb():
        loader = make_loader(val_dataset, repeat=True)
        total_nats, total_bytes = 0.0, 0
        for _ in range(max(1, args.eval_tokens // (B * T))):
            inputs, targets = next(loader)
            valid = targets >= 0
            counts = np.zeros_like(targets, dtype=np.int64)
            counts[valid] = token_bytes[targets[valid]]
            targets[counts == 0] = -1

            backend.copy_from_host(x, inputs)
            backend.copy_from_host(y, targets)
            runtime.forward()

            # Fused CE averages over all B*T positions.
            total_nats += read_loss() * B * T
            total_bytes += int(counts.sum())

        if total_bytes == 0:
            raise RuntimeError("Validation contains no supervised text bytes")
        return total_nats / (math.log(2) * total_bytes)

    if args.run == "dummy":
        wandb_run = DummyWandb()
    else:
        import wandb
        wandb_run = wandb.init(
            project="nanochat-sft",
            name=args.run,
            config=vars(args),
        )

    depth, width = config["n_layer"], config["n_embd"]
    matrix_parameters = (
        depth * 12 * width * width + config["vocab_size"] * width
    )
    flops_per_token = 6 * matrix_parameters + 12 * depth * width * T

    print(f"Training mixture: {len(train_dataset):,} rows")
    print("COMPUTE_DTYPE: float32; custom Autograd Metal backend")
    print(f"Gradient accumulation: {accumulation}")
    print(f"Initial AdamW-mini LR: {initial_lr:.8g}; weight decay: 0")
    if T + 1 < 2048:
        print(
            f"NOTE: conversations are truncated to {T + 1} tokens "
            "to fit the configured context"
        )

    step = 0
    progress = 0.0
    ema = 0.0
    total_training_time = 0.0
    min_val_bpb = float("inf")
    val_bpb = None
    chat_evaluator = None
    last_eval_step = None
    last_chat_step = None
    gc_was_enabled = gc.isenabled()
    gc_frozen_here = False

    def evaluate(final=False):
        nonlocal val_bpb, min_val_bpb, chat_evaluator
        nonlocal last_eval_step, last_chat_step

        common = {
            "step": step,
            "total_training_flops":
                flops_per_token * args.total_batch_size * step,
            "total_training_time": total_training_time,
        }

        # As in chat_sft.py, final BPB is evaluated even with eval_every=-1.
        if last_eval_step != step and (
            final or (args.eval_every > 0 and step % args.eval_every == 0)
        ):
            val_bpb = evaluate_bpb()
            min_val_bpb = min(min_val_bpb, val_bpb)
            last_eval_step = step
            print(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
            wandb_run.log({**common, "val/bpb": val_bpb})

        if (
            args.chatcore_every > 0
            and last_chat_step != step
            and (final or (step > 0 and step % args.chatcore_every == 0))
        ):
            if chat_evaluator is None:
                chat_evaluator = ChatEvaluator(
                    config, runtime, args.device_type
                )
            chat_evaluator.refresh(model)
            metrics = chat_evaluator.evaluate(tokenizer, args)
            last_chat_step = step
            print(
                f"Step {step:05d} | "
                f"ChatCORE: {metrics['chatcore_metric']:.4f} | "
                f"ChatCORE_cat: {metrics['chatcore_cat']:.4f}"
            )
            wandb_run.log({**common, **metrics})

    last_saved_step = None

    try:
        while True:
            reached_limit = (
                args.num_iterations > 0 and step >= args.num_iterations
            )
            if reached_limit:
                break

            # Read the logical batch first, so final partial accumulation can
            # be normalized by its actual number of microbatches.
            batches = []
            for _ in range(accumulation):
                try:
                    batches.append(next(train_loader))
                except StopIteration:
                    break

            if not batches:
                break

            evaluate()
            backend.synchronize()
            t0 = time.perf_counter()

            actual_accumulation = len(batches)
            micro_losses = []

            for micro_step, (inputs, targets) in enumerate(batches):
                supervised = int(np.count_nonzero(targets >= 0))
                if supervised == 0:
                    raise RuntimeError("Unexpected empty supervised batch")

                backend.copy_from_host(x, inputs)
                backend.copy_from_host(y, targets)

                backend.debug_rms_step = step + 1

                # Reference loss: mean over valid targets within each
                # microbatch, then average across accumulation steps.
                normalization = (B * T) / supervised
                loss.backward(
                    grad_scale=normalization / actual_accumulation,
                    accumulate_parameter_grads=micro_step > 0,
                )
                micro_losses.append(read_loss() * normalization)

            if args.num_iterations > 0:
                progress = (step + 1) / args.num_iterations
            else:
                progress = train_loader.progress

            grad_norm, clip_coef = clip_grad_norm(args.max_norm)
            if not math.isfinite(grad_norm):
                print(f"[GRAD-SKIP] step={step+1} grad_norm={grad_norm} (non-finite, skip step)", flush=True)
                # optimizer.zero_grad()
                step += 1
                continue
            print(
                f"[GRAD-NORM] step={step + 1} "
                f"grad_norm={grad_norm:.6f} "
                f"clip_coef={clip_coef:.6f}",
                flush=True,
            )

            lrm = lr_multiplier(progress, args)
            optimizer.lr = initial_lr * lrm
            optimizer.step(grad_scale=clip_coef)

            backend.synchronize()
            dt = time.perf_counter() - t0
            step += 1

            if (args.save_every > 0 and step % args.save_every == 0):
                save_checkpoint(
                    args,
                    step,
                    model=model,
                    optimizer=optimizer,
                    backend=backend,
                    config=config,
                    base_dir=base_dir,
                    input_config=input_config,
                    effective_lr=effective_lr,
                    val_bpb=val_bpb,
                    val_bpb_step=last_eval_step,
                )
                last_saved_step = step

            train_loss = float(np.mean(micro_losses))
            ema = 0.9 * ema + 0.1 * train_loss
            smooth_loss = ema / (1 - 0.9 ** step)
            actual_tokens = actual_accumulation * B * T
            tokens_per_second = int(actual_tokens / dt)

            # Same approximate MPS peak convention as train_autograd.py.
            mfu = (
                100 * flops_per_token * actual_tokens / dt / 10.4e12
            )
            if step > 10:
                total_training_time += dt

            print(
                f"step {step:05d} ({100 * progress:.2f}%) | "
                f"loss: {smooth_loss:.6f} | lrm: {lrm:.3f} | "
                f"dt: {1000 * dt:.2f}ms | "
                f"tok/sec: {tokens_per_second:,} | "
                f"mfu(est): {mfu:.2f} | epoch: {train_loader.epoch} | "
                f"total time: {total_training_time / 60:.2f}m"
            )
            if step % 10 == 0:
                wandb_run.log({
                    "step": step,
                    "total_training_time": total_training_time,
                    "total_training_flops":
                        flops_per_token * args.total_batch_size * step,
                    "train/loss": smooth_loss,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/tok_per_sec": tokens_per_second,
                    "train/mfu": mfu,
                    "train/epoch": train_loader.epoch,
                })

            if step == 1:
                gc.collect()
                gc.freeze()
                gc_frozen_here = True
                gc.disable()
            elif step % 5000 == 0:
                gc.collect()

        if step == 0:
            raise RuntimeError("No supervised training batches were produced")

        if last_saved_step != step:
            save_checkpoint(
                args,
                step,
                model=model,
                optimizer=optimizer,
                backend=backend,
                config=config,
                base_dir=base_dir,
                input_config=input_config,
                effective_lr=effective_lr,
                val_bpb=val_bpb,
                val_bpb_step=last_eval_step,
            )
            last_saved_step = step

        evaluate(final=True)

        print(f"Total training time: {total_training_time / 60:.2f}m")
        print(f"Minimum validation bpb: {min_val_bpb:.6f}")

        from nanochat.report import get_report
        get_report().log(section="SFT", data=[
            vars(args),
            {"Number of iterations": step, "DDP world size": 1},
            {"Minimum validation bpb": min_val_bpb},
        ])
    finally:
        backend.synchronize()
        wandb_run.finish()
        if gc_frozen_here:
            gc.unfreeze()
        if gc_was_enabled:
            gc.enable()


if __name__ == "__main__":
    main()