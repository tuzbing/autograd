"""Nanochat-compatible inference engine for the local autograd model."""

from collections import deque
import signal
import warnings
import time
import re

import numpy as np

from runtime import Runtime
from runtime import build_decode_plan, build_prefill_plan
from tensor import Tensor

class KVCache:
    """Compatibility container; local attention currently has no KV API."""

    def __init__(self, batch_size, num_heads, max_seq_len, head_dim, num_layers,
                 device=None, dtype=np.float32):
        self.batch_size = int(batch_size)
        self.num_heads = num_heads
        self.max_seq_len = int(max_seq_len)
        self.head_dim = int(head_dim)
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.k_cache = [Tensor(shape=(self.batch_size, self.max_seq_len, self.num_heads, self.head_dim),
                                dtype=dtype, device=device, requires_grad=False, init_val=0, name=f"k_cache_{layer}")
                        for layer in range(self.num_layers)]
        self.v_cache = [Tensor(shape=(self.batch_size, self.max_seq_len, self.num_heads, self.head_dim),
                                dtype=dtype, device=device, requires_grad=False, init_val=0, name=f"v_cache_{layer}")
                        for layer in range(self.num_layers)]
        self.position = Tensor(shape=(1,), dtype=np.int32, device=device, requires_grad=False, init_val=0, name="kv_cache_position")
        self.position_host = np.zeros((1,), dtype=np.int32)

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def set_position(self, position):
        self.position_host[0] = int(position)

    def reset(self):
        self.position_host.fill(0)

    def advance(self, num_tokens):
        self.position_host[0] += num_tokens

    def position_feeds(self):
        return {
            self.position: self.position_host,
        }

warnings.filterwarnings("ignore", category=SyntaxWarning)
_NUM_CHARS = frozenset("0123456789*+-/.() ")
_ALLOWED_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ ")
_DANGEROUS_RE = re.compile(
    r"__|import|exec|eval|compile|open|file|input|raw_input|"
    r"globals|locals|vars|dir|getattr|setattr|delattr|hasattr"
)
_COUNT_RE = re.compile(r"\.count\(")

def _eval_with_timeout(formula, max_time=3):
    def handler(signum, frame):
        raise TimeoutError(f"'{formula}': timed out after {max_time} seconds")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(max_time)
    try:
        return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

# 是否可用缓存降低耗时?
def use_calculator(expr):
    """Evaluate nanochat's restricted arithmetic and ``.count()`` syntax."""
    if not expr:
        return None
    expr = expr.replace(",", "")
    if set(expr) <= _NUM_CHARS:
        if "**" in expr:
            return None
        return _eval_with_timeout(expr)
    if not (set(expr) <= _ALLOWED_CHARS):
        return None
    if _DANGEROUS_RE.search(expr.lower()) or not _COUNT_RE.search(expr):
        return None
    return _eval_with_timeout(expr)

class RolloutState:
    def __init__(self, batch_size, max_seq_len, device):
        def make(name, shape, dtype=np.int32, value=0):
            return Tensor(shape=shape, dtype=dtype, device=device, requires_grad=False, 
                          init_val=np.full(shape, value, dtype=dtype), name=name)
        self.tokens = make("tokens", (batch_size, max_seq_len))
        self.mask = make("mask", (batch_size, max_seq_len), np.uint8)
        self.metadata = make("metadata", (batch_size, 2))
        self.finished = make("finished", (batch_size,), np.uint8)

        self.generated_ids = self.tokens
        self.resp_mask = self.mask
        self.lengths = make("lengths", (batch_size,))
        self.next_tokens = make("next_tokens", (batch_size,1))
        self.decode_candidates = make("decode_candidates", (batch_size, 1))
        self.decode_position = make("decode_position", (batch_size,))
        self.decode_step = make("decode_step", (batch_size,))
        self.position = make("position", (1,))
        self.step = make("step", (1,))
        self.decode_config = make("decode_config", (3,)) # prompt length, budget, chunk size
        self.rng_state = make("rng_state", (batch_size, 4), np.uint32)
        self.rng_raw = make("unused_rng_raw", (1,), np.uint32)
        self.temperature = make("temperature", (1,), np.float32, 1.0)
        self.top_k = make("top_k", (1,), value=50)

class Engine:
    """Sampler reusing the model's already compiled static autograd graph."""

    def __init__(self, model, tokenizer, device, batch_size, max_seq_len, parameter_runtime, arena,
                 max_top_k=64, decode_steps=8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.bos = int(tokenizer.get_bos_token_id())
        self.max_top_k = max_top_k
        self.decode_steps = decode_steps
        self.assistant_end = tokenizer.encode_special("<|assistant_end|>")

        self.parameter_runtime = parameter_runtime
        self.shared_arena = arena
        # Decode cache: one row per sampled answer
        self.kv_cache = self._create_kv_cache(self.batch_size, self.max_seq_len)
        # Prefill cache: only one prompt row
        # self.prefill_kv_cache = self._create_kv_cache(1, self.max_seq_len)

        self.prefill_plans = {}
        self.prefill_bucket_lens = []
        base = 128
        while base < max_seq_len:
            self.prefill_bucket_lens.append(base)
            base *= 2
        if not self.prefill_bucket_lens or self.prefill_bucket_lens[-1] != max_seq_len:
            self.prefill_bucket_lens.append(max_seq_len)

        self.prefill_last_token = Tensor(shape=(1,), dtype=np.int32, device=device, requires_grad=False, init_val=0, name="prefill_last_token")

        self.rollout_state = RolloutState(batch_size=self.batch_size, max_seq_len=self.max_seq_len, device=self.device)
        self.decode_plan = build_decode_plan(model=self.model, 
                                             batch_size=self.batch_size, 
                                             device=self.device, 
                                             kv_cache=self.kv_cache,
                                             top_k=self.rollout_state.top_k,
                                             temperature=self.rollout_state.temperature,
                                             eos_token=self.assistant_end,
                                             bos_token=self.bos,
                                             rng_state=self.rollout_state.rng_state,
                                             rng_raw=self.rollout_state.rng_raw,
                                             finished=self.rollout_state.finished,
                                             next_tokens=self.rollout_state.next_tokens,
                                             generated_ids=self.rollout_state.generated_ids,
                                             lengths=self.rollout_state.lengths,
                                             decode_position=self.rollout_state.position,
                                             decode_step=self.rollout_state.step,
                                             state=self.rollout_state,
                                             decode_steps=self.decode_steps)
        if self.decode_plan.runtime.arena_bytes > self.shared_arena.nbytes:
            raise RuntimeError(f"decode plan need {self.decode_plan.runtime.arena_bytes} exceeds train arena {self.shared_arena.nbytes}")
        self.decode_plan.bind_to(self.shared_arena)

        self._prepare_rollout_policy_views()

        self.profile = {}
        self.enable_debug = True
        self.decode_slots = 0
        self.tool_waste = 0
        self.eos_bos_waste = 0

    def _sync(self):
        backend = self.decode_plan.runtime.backend
        if hasattr(backend, "synchronize"):
            backend.synchronize()

    def _reset_profile(self):
        self.profile = {
            "sampling_config": 0.0,
            "reset_state": 0.0,
            "policy_init": 0.0,
            "prompt_prepare": 0.0,
            "prefill_plan": 0.0,
            "prefill_prepare": 0.0,
            "prefill_forward": 0.0,
            "kv_copy": 0.0,
            "prefill_policy": 0.0,
            "decode_forward": 0.0,
            "decode_policy": 0.0,
            "decode_p1": 0.0,
            "decode_p2": 0.0,
            "loop_check": 0.0,
            "result_extract": 0.0,
            "decode_count": 0,
            "prefill_cache_miss": False,
            "prefill_bucket": 0,
            "total": 0.0,
            "other": 0.0,
        }

    def _prepare_rollout_policy_views(self):
        backend = self.decode_plan.runtime.backend
        state = self.rollout_state

        for t in vars(state).values():
            if t.storage is None:
                t.storage = self.decode_plan.runtime.allocator.alloc(t.shape, t.dtype)

        self._policy_tokens = self._weight_view(backend, state.next_tokens).reshape(-1)
        self._policy_done = self._weight_view(backend, state.finished).reshape(-1)
        self._policy_generated = self._weight_view(backend, state.tokens)
        self._policy_response_masks = self._weight_view(backend, state.mask)
        self._metadata = self._weight_view(backend, state.metadata)
        self._policy_lengths = self._weight_view(backend, state.lengths).reshape(-1)
        self._policy_position = self._weight_view(backend, state.position).reshape(-1)
        self._policy_step = self._weight_view(backend, state.step).reshape(-1)

        self._decode_config = self._weight_view(backend, state.decode_config)
        self._policy_output_views = {id(self.decode_plan.output): self._weight_view(backend, self.decode_plan.output).reshape(-1)}

    def _choose(self, prompt_len):
        for length in self.prefill_bucket_lens:
            if length >= prompt_len:
                if length in self.prefill_plans:
                    return self.prefill_plans[length]
                else:
                    plan = build_prefill_plan(model=self.model, 
                                              seq_len=length,
                                              device=self.device,
                                              kv_cache=self.kv_cache,
                                              top_k=self.rollout_state.top_k,
                                              temperature=self.rollout_state.temperature,
                                              eos_token=self.assistant_end,
                                              bos_token=self.bos,
                                              rng_state=self.rollout_state.rng_state,
                                              rng_raw=self.rollout_state.rng_raw,
                                              finished=self.rollout_state.finished,
                                              next_tokens=self.rollout_state.next_tokens,
                                              generated_ids=self.rollout_state.generated_ids,
                                              lengths=self.rollout_state.lengths,
                                              decode_position=self.rollout_state.position,
                                              decode_step=self.rollout_state.step,
                                              last_token=self.prefill_last_token)
                    if plan.runtime.arena_bytes > self.shared_arena.nbytes:
                        raise RuntimeError(f"prefill plan T={length} requires {plan.runtime.arena_bytes} bytes, but shared {self.shared_arena.nbytes}")
                    plan.bind_to(self.shared_arena)
                    self._policy_output_views[id(plan.output)] = self._weight_view(plan.runtime.backend, plan.output).reshape(-1)
                    self.prefill_plans[length] = plan
                    return plan
        return None

    def _create_kv_cache(self, batch_size, max_seq_len):
        first_block = self.model.blocks[0]
        return KVCache(batch_size, first_block.attn.n_head, max_seq_len, first_block.attn.head_dim, len(self.model.blocks),
                        device=self.device, dtype=np.float32)

    def _weight_view(self, backend, weight):
        if hasattr(backend, "numpy_view"):
            return backend.numpy_view(weight)
        return backend.copy_to_host(weight)

    def _set_sampling_config(self, temperature, top_k):
        temperature_host = np.asarray([temperature], dtype=np.float32)
        top_k_host = np.asarray([top_k], dtype=np.int32)
        backend = self.decode_plan.runtime.backend
        backend.copy_from_host(self.rollout_state.temperature, temperature_host)
        backend.copy_from_host(self.rollout_state.top_k, top_k_host)

    def _reset_rollout_state(self, seed):
        backend = self.decode_plan.runtime.backend

        for name in ("tokens", "mask", "metadata", "finished", "lengths", "next_tokens", "decode_candidates",
                     "decode_position", "decode_step", "position", "step"):
            self._weight_view(backend, getattr(self.rollout_state, name)).fill(0)

        seed64 = int(seed) & 0xFFFFFFFFFFFFFFFF
        rng = np.zeros(self.rollout_state.rng_state.shape, dtype=np.uint32)
        rng[:, 0] = seed64 & 0xFFFFFFFF
        rng[:, 1] = (seed64 >> 32) & 0xFFFFFFFF
        backend.copy_from_host(self.rollout_state.rng_state, rng)

    def _prefill_rollout(self, prompt_ids):
        num_samples, prompt_len = prompt_ids.shape

        if self.enable_debug:
            cached_count = len(self.prefill_plans)
            t1 = time.perf_counter()
        plan = self._choose(prompt_len)
        if self.enable_debug: 
            t2 = time.perf_counter()
            self.profile["prefill_plan"] += (t2 - t1)
            self.profile["prefill_bucket"] = plan.seq_len
            self.profile["prefill_cache_miss"] = len(self.prefill_plans) > cached_count

        padded_prompt = np.full((1, plan.seq_len), self.bos, dtype=np.int32)
        padded_prompt[0, :prompt_len] = prompt_ids[0]
        # self.prefill_kv_cache.reset()
        self.kv_cache.reset()

        feeds={
            plan.input_ids: padded_prompt,
            self.kv_cache.position: np.asarray([0], dtype=np.int32),
            self.rollout_state.position: np.asarray([prompt_len-1], dtype=np.int32),
        }
        if self.enable_debug:
            t3 = time.perf_counter()
            self.profile["prefill_prepare"] += (t3 - t2)
        outputs = plan.forward(feeds=feeds, return_value=False)
        if self.enable_debug:
            t4 = time.perf_counter()
            self.profile["prefill_forward"] += (t4 - t3)

        # self._broadcast_prefill_kv(prompt_len=prompt_len, num_samples=num_samples)
        if self.enable_debug:
            t5 = time.perf_counter()
            self.profile["kv_copy"] += (t5 - t4)

        res = self._apply_rollout_policy(plan)
        if self.enable_debug:
            t6 = time.perf_counter()
            self.profile["prefill_policy"] += (t6 - t5)
        return res

    def _decode_rollout(self):
        if self.enable_debug:
            t1 = time.perf_counter()
        self.decode_plan.forward(feeds=None, return_value=False)
        if self.enable_debug:
            t2 = time.perf_counter()
            self.profile["decode_count"] += 1
            self.profile["decode_forward"] += (t2 - t1)

        # return self._apply_rollout_policy(self.decode_plan)
        ret = self._apply_rollout_policy(self.decode_plan)
        if self.enable_debug:
            t3 = time.perf_counter()
            self.profile["decode_policy"] += (t3 - t2)
        return ret

    def _start_rollout_policy(self, prompt_tokens, num_samples):
        self._policy_prompt = list(prompt_tokens)
        self.expression_start = np.full(num_samples, -1, np.int32) # ???
        self._policy_special = {
            name: int(self.tokenizer.encode_special('<|' + name + '|>'))
            for name in ('python_start', 'python_end', 'output_start', 'output_end')
        }

    def _apply_rollout_policy(self, plan):
        tape, masks, meta = self._policy_generated, self._policy_response_masks, self._metadata
        special = self._policy_special

        done_buf = self._policy_done
        token_buf = self._policy_tokens
        length_buf = self._policy_lengths

        prefill = plan is not self.decode_plan
        budget = int(self._decode_config[1])
        global_start = int(self._policy_step[0])
        prompt_len = len(self._policy_prompt)
        chunk_size = 1 if prefill else self.decode_steps
        if self.enable_debug:
            if not prefill:
                self.decode_slots += self.batch_size * chunk_size

        max_processed = 0
        all_done = True
        any_active = False

        for row in range(self.batch_size):
            # t0 = time.perf_counter()
            was_done = bool(done_buf[row])
            processed = int(meta[row, 0])

            if self.enable_debug:
                if not prefill and was_done:
                    self.eos_bos_waste += chunk_size

            if was_done and self.decode_steps > 1:
                if processed > max_processed:
                    max_processed = processed
                continue

            forced_end = int(meta[row, 1])
            response_length = int(length_buf[row])
            expression_start = int(self.expression_start[row])
            row_done = was_done

            start = global_start if was_done else processed
            end = min(start+chunk_size, budget)
            last_token = int(token_buf[row])
            if prefill:
                tape[row, start] = last_token
                masks[row, start] = 1
            # t1 = time.perf_counter()
            # self.profile["decode_p2"] += (t1 - t0)

            for i in range(start, end):
                token = int(tape[row, i])
                last_token = token
                terminal = token == self.assistant_end or token == self.bos
                if not was_done:
                    processed = i + 1
                    if terminal:
                        row_done = True
                    else:
                        response_length = i+1

                tool_result = None
                if token == special["python_start"]:
                    expression_start = i+1
                elif token == special["python_end"] and expression_start >= 0:
                    first = expression_start
                    expression_start = -1
                    if first < i:
                        tool_result = use_calculator(self.tokenizer.decode(tape[row, first:i].tolist()))

                if tool_result is not None:
                    result_ids = [int(x) for x in self.tokenizer.encode(str(tool_result))]
                    vocab = int(self.model.tok_emb.weight.shape[0])
                    if any(x < 0 or x >= vocab for x in result_ids):
                        raise ValueError("calculator returned a token outside vocabulary")
                    queued = [special["output_start"], *result_ids, special["output_end"]]
                    at = max(i+1, forced_end)
                    count = max(0, min(len(queued), budget - at))
                    tape[row, at:at+count] = queued[:count]
                    masks[row, at:at+count] = 0
                    forced_end = at + count

                    if self.enable_debug:
                        if not prefill and not was_done:
                            self.tool_waste += chunk_size-(i-start+1)
                    break
                if terminal and not was_done:
                    if self.enable_debug:
                        if not prefill:
                            self.eos_bos_waste += chunk_size - (i-start+1)
                    break

            # t2 = time.perf_counter()
            # self.profile["decode_p1"] += (t2 - t1)

            token_buf[row] = last_token
            meta[row, 0] = processed
            meta[row, 1] = max(processed, forced_end)

            if not was_done:
                done_buf[row] = int(row_done)
                length_buf[row] = response_length
            self.expression_start[row] = expression_start

            if processed > max_processed:
                max_processed = processed

            if not row_done:
                all_done = False
                if processed < budget:
                    any_active = True

        # t3 = time.perf_counter()
        if prefill or self.decode_steps == 1:
            next_step = global_start + 1
        else:
            next_step = max_processed
        self._policy_step[0] = next_step
        self._policy_position[0] = prompt_len + next_step - 1
        self._policy_output_views[id(plan.output)][0] = int(all_done)
        self._policy_has_active = any_active
        # t4 = time.perf_counter()
        # self.profile["decode_p2"] += (t4 - t3)

        return all_done

    def _get_rollout_results(self, prompt_tokens, num_samples):
        sequences, masks = [], []
        for row in range(num_samples):
            n = int(self._policy_lengths[row])
            sequences.append(list(prompt_tokens) + self._policy_generated[row, :n].tolist())
            masks.append([0]*len(prompt_tokens) + self._policy_response_masks[row, :n].tolist())
        return sequences, masks


    def generate_batch_new(self, tokens, num_samples=None, max_tokens=None,
                           temperature=1.0, top_k=50, seed=42):
        if self.enable_debug:
            total_start = time.perf_counter()
            self._reset_profile()
        prompt = [int(token) for token in tokens]
        num_samples = self.batch_size if num_samples is None else int(num_samples)
        max_tokens = self.max_seq_len - len(prompt) if max_tokens is None else int(max_tokens)
        temperature = float(temperature)
        top_k = int(top_k)
        vocab = int(self.model.tok_emb.weight.shape[0])
        if not prompt:
            raise ValueError('prompt_tokens must not be empty')
        if num_samples <= 0 or num_samples > self.batch_size:
            raise ValueError('num_samples must be positive and no greater than engine batch_size')
        if max_tokens < 0:
            raise ValueError('max_tokens must be non-negative')
        capacity = min(self.max_seq_len, int(self.model.rope_cos.shape[0]))
        if len(prompt) + max_tokens > capacity:
            raise ValueError('prompt plus generated tokens exceeds model capacity')
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and non-negative')
        if any(token < 0 or token >= vocab for token in prompt):
            raise ValueError('prompt token is outside vocabulary')
        if num_samples != self.batch_size:
            print(f"num_samples dismatch {num_samples} vs {self.batch_size} init engine again!!!", flush=True)
            # Philox stride is the active row count, not unused engine capacity.
            smaller = Engine(self.model, self.tokenizer, self.device, num_samples,
                             self.max_seq_len, self.parameter_runtime, self.shared_arena,
                             max_top_k=self.max_top_k, decode_steps=self.decode_steps)
            return smaller.generate_batch_new(prompt, num_samples, max_tokens,
                                              temperature, top_k, seed)
        # Preserve no-top-k as zero and clamp only to vocabulary, never to 64.
        top_k = min(top_k, vocab) if top_k > 0 else 0
        if self.enable_debug:
            t1 = time.perf_counter()
        self._set_sampling_config(temperature=temperature, top_k=top_k)
        if self.enable_debug:
            t2 = time.perf_counter()
            self.profile["sampling_config"] += (t2 - t1)
        self._reset_rollout_state(seed)
        if self.enable_debug:
            t3 = time.perf_counter()
            self.profile["reset_state"] += (t3 - t2)
        self._start_rollout_policy(prompt, num_samples)
        if self.enable_debug:
            t4 = time.perf_counter()
            self.profile["policy_init"] += (t4 - t3)

        prompt_ids = np.asarray([prompt] * num_samples, dtype=np.int32)
        self._decode_config[:] = (len(prompt), max_tokens, self.decode_steps)
        if self.enable_debug:
            t5 = time.perf_counter()
            self.profile["prompt_prepare"] += (t5 - t4)
        self._policy_has_active = False
        if max_tokens > 0:
            self._prefill_rollout(prompt_ids)
            while self._policy_has_active:
                self._decode_rollout()
        if self.enable_debug:
            t6 = time.perf_counter()
        result = self._get_rollout_results(prompt, num_samples)
        if self.enable_debug:
            t7 = time.perf_counter()
            self.profile["result_extract"] += (t7 - t6)
        if self.enable_debug:
            p = self.profile
            p["total"] = time.perf_counter() - total_start
            stage_keys = (
                "sampling_config",
                "reset_state",
                "policy_init",
                "prompt_prepare",
                "prefill_plan",
                "prefill_prepare",
                "prefill_forward",
                "kv_copy",
                "prefill_policy",
                "decode_forward",
                "decode_policy",
                "decode_p1",
                "decode_p2",
                "loop_check",
                "result_extract",
            )
            p["other"] = p["total"] - sum(p[key] for key in stage_keys)
            chunks = p["decode_count"]
            avg_chunk_ms = p["decode_forward"] * 1000.0 / chunks if chunks else 0.0
            stages = " ".join(f"{key}={p[key] * 1000.0:.3f}ms" for key in stage_keys)
            print(
                f"[ENGINE-PROFILE] "
                f"batch={num_samples} "
                f"prompt_len={len(prompt)} "
                f"bucket={p['prefill_bucket']} "
                f"cache_miss={int(p['prefill_cache_miss'])} "
                f"decode_steps={self.decode_steps} "
                f"chunks={chunks} "
                f"avg_chunk_forward={avg_chunk_ms:.3f}ms "
                f"{stages} "
                f"other={p['other'] * 1000.0:.3f}ms "
                f"total={p['total'] * 1000.0:.3f}ms",
                flush=True,
            )
        return result

