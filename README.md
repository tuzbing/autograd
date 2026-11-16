# autograd

autograd is a simple framework for training LLM models(like GPT2/3/4, or MoE supported LLM etc). It will try to implemented with low memory usage and trainning time. It also try to support multiple backend devices, like CPU, Metal, CUDA.

Design Philosophy
    To maximize performance, during the compilation of the computation graph(DAG) constructed from tensors, all tensor operations are bound to kernel functions provided by the target backend device.
	At the same time, all memory required for the training process is pre-allocated in advance, aoviding frequent memory allocation and deallocation during training.

Requirements
	tiktoken
	pyarrow
	pybind11

Run
a) compile metal
	python setup.py build_ext --inplace
b) train gpt2
	python test_train_gpt2.py

## Training Logs

```text

pretrain logs
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat" \
/Users/a58/Documents/code/nanochat/.venv/bin/python train_autograd.py \
  --device-type=mps \
  --depth=6 \
  --aspect-ratio=64 \
  --head-dim=64 \
  --window-pattern=L \
  --max-seq-len=512 \
  --device-batch-size=32 \
  --total-batch-size=16384 \
  --matrix-lr=0.02 \
  --weight-decay=0.234903 \
  --warmup-steps=40 \
  --warmdown-ratio=0.65 \
  --final-lr-frac=0.05 \
  --eval-every=100 \
  --eval-tokens=524288 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --num-iterations=5000 \
  --save-every=1000 \
  --model-out='/Users/a58/Documents/code/autograd/checkpoints/base_d6/fp32_model_{step:06d}.gguf' \
  --run=dummy
Configured device type: mps
Distributed world size: 1
COMPUTE_DTYPE: float32(custom Autograd Metal backend)
Using custom fused LayerNorm/FlashAttention/MLP/CE kernels
Vocab size: 32,768
Model config:
{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 6,
  "n_head": 6,
  "n_kv_head": 6,
  "n_embd": 384,
  "window_pattern": "L"
}
Parameter counts:
total                   : 35,791,872
matrix parameters       : 23,199,744
Estimated FLOPs per token: 1.533542e+08
Scaling LRs by 0.1768 for batch size 16,384 (reference: 524,288)
Using effective AdamW matrix LR: 0.00353553
Using effective weight decay: 0.234903
Using user-provided number of iterations: 5,000
Total number of training tokens: 81,920,000
Tokens : Scaling params ratio: 3.53
Total training FLOPs estimate: 1.256278e+16
Tokens / micro-batch / rank: 32 x 512 = 16,384
Tokens / micro-batch: 16,384
Total batch size 16,384 => gradient accumulation steps: 1
[MODEL-WEIGHT-MEMORY] weight_bytes=143167488 weight_grad_bytes=143167488 weight_mib=136.54 weight_grad_mib=136.54 weight_gib=0.1333 weight_grad_gib=0.1333
Training arena need 427819264
[ARENA-TRAIN-ALLOC] bytes=427819264 buffer_id=44994564928
Step 00000 | Validation bpb: 3.207953
step 00000/05000 (0.00%) | loss: 10.427748 | lrm: 0.03 | dt: 2198.72ms | tok/sec: 7,451 | bf16_mfu: 10.99 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/05000 (0.02%) | loss: 10.385238 | lrm: 0.05 | dt: 2232.79ms | tok/sec: 7,337 | bf16_mfu: 10.82 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/05000 (0.04%) | loss: 10.302644 | lrm: 0.07 | dt: 2197.24ms | tok/sec: 7,456 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00003/05000 (0.06%) | loss: 10.151250 | lrm: 0.10 | dt: 2215.94ms | tok/sec: 7,393 | bf16_mfu: 10.90 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00004/05000 (0.08%) | loss: 9.893887 | lrm: 0.12 | dt: 2201.91ms | tok/sec: 7,440 | bf16_mfu: 10.97 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00005/05000 (0.10%) | loss: 9.579199 | lrm: 0.15 | dt: 2222.99ms | tok/sec: 7,370 | bf16_mfu: 10.87 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00006/05000 (0.12%) | loss: 9.363249 | lrm: 0.17 | dt: 2219.59ms | tok/sec: 7,381 | bf16_mfu: 10.88 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00007/05000 (0.14%) | loss: 9.244912 | lrm: 0.20 | dt: 2200.42ms | tok/sec: 7,445 | bf16_mfu: 10.98 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00008/05000 (0.16%) | loss: 9.098057 | lrm: 0.23 | dt: 2219.36ms | tok/sec: 7,382 | bf16_mfu: 10.89 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00009/05000 (0.18%) | loss: 9.060246 | lrm: 0.25 | dt: 2198.25ms | tok/sec: 7,453 | bf16_mfu: 10.99 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00010/05000 (0.20%) | loss: 8.964076 | lrm: 0.28 | dt: 2220.32ms | tok/sec: 7,379 | bf16_mfu: 10.88 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00011/05000 (0.22%) | loss: 8.906764 | lrm: 0.30 | dt: 2195.46ms | tok/sec: 7,462 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.04m | eta: 182.6m
step 00012/05000 (0.24%) | loss: 8.828311 | lrm: 0.33 | dt: 2200.04ms | tok/sec: 7,447 | bf16_mfu: 10.98 | epoch: 1 pq: 0 rg: 1 | total time: 0.07m | eta: 182.7m
step 00013/05000 (0.26%) | loss: 8.755080 | lrm: 0.35 | dt: 2195.64ms | tok/sec: 7,462 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.11m | eta: 182.6m
step 00014/05000 (0.28%) | loss: 8.688195 | lrm: 0.38 | dt: 2223.46ms | tok/sec: 7,368 | bf16_mfu: 10.87 | epoch: 1 pq: 0 rg: 2 | total time: 0.15m | eta: 183.1m
step 00015/05000 (0.30%) | loss: 8.614243 | lrm: 0.40 | dt: 2196.23ms | tok/sec: 7,460 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.18m | eta: 183.0m
step 00016/05000 (0.32%) | loss: 8.530915 | lrm: 0.42 | dt: 2191.86ms | tok/sec: 7,474 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 2 | total time: 0.22m | eta: 182.8m
step 00017/05000 (0.34%) | loss: 8.467483 | lrm: 0.45 | dt: 2218.76ms | tok/sec: 7,384 | bf16_mfu: 10.89 | epoch: 1 pq: 0 rg: 2 | total time: 0.26m | eta: 183.0m
step 00018/05000 (0.36%) | loss: 8.421527 | lrm: 0.47 | dt: 2197.11ms | tok/sec: 7,457 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.29m | eta: 182.9m
step 00019/05000 (0.38%) | loss: 8.378601 | lrm: 0.50 | dt: 2187.13ms | tok/sec: 7,491 | bf16_mfu: 11.05 | epoch: 1 pq: 0 rg: 2 | total time: 0.33m | eta: 182.7m
step 00020/05000 (0.40%) | loss: 8.342422 | lrm: 0.53 | dt: 2217.77ms | tok/sec: 7,387 | bf16_mfu: 10.89 | epoch: 1 pq: 0 rg: 2 | total time: 0.37m | eta: 182.8m
step 00021/05000 (0.42%) | loss: 8.309439 | lrm: 0.55 | dt: 2194.98ms | tok/sec: 7,464 | bf16_mfu: 11.01 | epoch: 1 pq: 0 rg: 2 | total time: 0.40m | eta: 182.7m
step 00022/05000 (0.44%) | loss: 8.268531 | lrm: 0.57 | dt: 2196.26ms | tok/sec: 7,459 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.44m | eta: 182.6m
step 00023/05000 (0.46%) | loss: 8.214435 | lrm: 0.60 | dt: 2222.42ms | tok/sec: 7,372 | bf16_mfu: 10.87 | epoch: 1 pq: 0 rg: 2 | total time: 0.48m | eta: 182.7m
step 00024/05000 (0.48%) | loss: 8.165379 | lrm: 0.62 | dt: 2193.14ms | tok/sec: 7,470 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 2 | total time: 0.51m | eta: 182.6m
step 00025/05000 (0.50%) | loss: 8.110080 | lrm: 0.65 | dt: 2193.12ms | tok/sec: 7,470 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 2 | total time: 0.55m | eta: 182.5m
step 00026/05000 (0.52%) | loss: 8.063911 | lrm: 0.68 | dt: 2229.89ms | tok/sec: 7,347 | bf16_mfu: 10.83 | epoch: 1 pq: 0 rg: 2 | total time: 0.59m | eta: 182.7m
step 00027/05000 (0.54%) | loss: 8.003168 | lrm: 0.70 | dt: 2191.64ms | tok/sec: 7,475 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 2 | total time: 0.62m | eta: 182.6m
step 00028/05000 (0.56%) | loss: 7.956719 | lrm: 0.72 | dt: 2215.54ms | tok/sec: 7,395 | bf16_mfu: 10.90 | epoch: 1 pq: 0 rg: 2 | total time: 0.66m | eta: 182.6m
step 00029/05000 (0.58%) | loss: 7.906668 | lrm: 0.75 | dt: 2205.21ms | tok/sec: 7,429 | bf16_mfu: 10.96 | epoch: 1 pq: 0 rg: 2 | total time: 0.70m | eta: 182.6m
step 00030/05000 (0.60%) | loss: 7.854104 | lrm: 0.78 | dt: 2193.83ms | tok/sec: 7,468 | bf16_mfu: 11.01 | epoch: 1 pq: 0 rg: 2 | total time: 0.73m | eta: 182.5m
step 00031/05000 (0.62%) | loss: 7.800446 | lrm: 0.80 | dt: 2212.37ms | tok/sec: 7,405 | bf16_mfu: 10.92 | epoch: 1 pq: 0 rg: 2 | total time: 0.77m | eta: 182.5m
step 00032/05000 (0.64%) | loss: 7.747600 | lrm: 0.82 | dt: 2200.19ms | tok/sec: 7,446 | bf16_mfu: 10.98 | epoch: 1 pq: 0 rg: 2 | total time: 0.81m | eta: 182.4m
step 00033/05000 (0.66%) | loss: 7.709855 | lrm: 0.85 | dt: 2192.13ms | tok/sec: 7,474 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 2 | total time: 0.84m | eta: 182.4m
step 00034/05000 (0.68%) | loss: 7.719936 | lrm: 0.88 | dt: 2216.61ms | tok/sec: 7,391 | bf16_mfu: 10.90 | epoch: 1 pq: 0 rg: 2 | total time: 0.88m | eta: 182.4m
step 00035/05000 (0.70%) | loss: 7.745278 | lrm: 0.90 | dt: 2193.17ms | tok/sec: 7,470 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 2 | total time: 0.92m | eta: 182.3m
step 00036/05000 (0.72%) | loss: 7.730139 | lrm: 0.93 | dt: 2198.26ms | tok/sec: 7,453 | bf16_mfu: 10.99 | epoch: 1 pq: 0 rg: 2 | total time: 0.95m | eta: 182.2m
step 00037/05000 (0.74%) | loss: 7.705806 | lrm: 0.95 | dt: 2224.34ms | tok/sec: 7,365 | bf16_mfu: 10.86 | epoch: 1 pq: 0 rg: 3 | total time: 0.99m | eta: 182.3m
step 00038/05000 (0.76%) | loss: 7.676370 | lrm: 0.97 | dt: 2201.95ms | tok/sec: 7,440 | bf16_mfu: 10.97 | epoch: 1 pq: 0 rg: 3 | total time: 1.03m | eta: 182.2m
step 00039/05000 (0.78%) | loss: 7.656932 | lrm: 1.00 | dt: 2195.24ms | tok/sec: 7,463 | bf16_mfu: 11.01 | epoch: 1 pq: 0 rg: 3 | total time: 1.06m | eta: 182.2m
step 00040/05000 (0.80%) | loss: 7.619871 | lrm: 1.00 | dt: 2216.35ms | tok/sec: 7,392 | bf16_mfu: 10.90 | epoch: 1 pq: 0 rg: 3 | total time: 1.10m | eta: 182.2m
step 00041/05000 (0.82%) | loss: 7.575768 | lrm: 1.00 | dt: 2194.91ms | tok/sec: 7,464 | bf16_mfu: 11.01 | epoch: 1 pq: 0 rg: 3 | total time: 1.14m | eta: 182.1m
step 00042/05000 (0.84%) | loss: 7.551325 | lrm: 1.00 | dt: 2197.23ms | tok/sec: 7,456 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 3 | total time: 1.18m | eta: 182.1m
step 00043/05000 (0.86%) | loss: 7.507636 | lrm: 1.00 | dt: 2218.90ms | tok/sec: 7,383 | bf16_mfu: 10.89 | epoch: 1 pq: 0 rg: 3 | total time: 1.21m | eta: 182.1m
step 00044/05000 (0.88%) | loss: 7.470548 | lrm: 1.00 | dt: 2192.57ms | tok/sec: 7,472 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 3 | total time: 1.25m | eta: 182.0m
step 00045/05000 (0.90%) | loss: 7.434122 | lrm: 1.00 | dt: 2227.60ms | tok/sec: 7,354 | bf16_mfu: 10.85 | epoch: 1 pq: 0 rg: 3 | total time: 1.29m | eta: 182.0m
step 00046/05000 (0.92%) | loss: 7.402656 | lrm: 1.00 | dt: 2202.64ms | tok/sec: 7,438 | bf16_mfu: 10.97 | epoch: 1 pq: 0 rg: 3 | total time: 1.32m | eta: 182.0m
step 00047/05000 (0.94%) | loss: 7.373384 | lrm: 1.00 | dt: 2192.29ms | tok/sec: 7,473 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 3 | total time: 1.36m | eta: 181.9m
step 00048/05000 (0.96%) | loss: 7.341642 | lrm: 1.00 | dt: 2219.98ms | tok/sec: 7,380 | bf16_mfu: 10.88 | epoch: 1 pq: 0 rg: 3 | total time: 1.40m | eta: 181.9m
step 00049/05000 (0.98%) | loss: 7.344061 | lrm: 1.00 | dt: 2196.71ms | tok/sec: 7,458 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 3 | total time: 1.43m | eta: 181.9m
step 00050/05000 (1.00%) | loss: 7.330882 | lrm: 1.00 | dt: 2194.30ms | tok/sec: 7,466 | bf16_mfu: 11.01 | epoch: 1 pq: 0 rg: 3 | total time: 1.47m | eta: 181.8m
step 00051/05000 (1.02%) | loss: 7.323075 | lrm: 1.00 | dt: 2217.48ms | tok/sec: 7,388 | bf16_mfu: 10.89 | epoch: 1 pq: 0 rg: 3 | total time: 1.51m | eta: 181.8m
step 00052/05000 (1.04%) | loss: 7.302677 | lrm: 1.00 | dt: 2194.76ms | tok/sec: 7,465 | bf16_mfu: 11.01 | epoch: 1 pq: 0 rg: 3 | total time: 1.54m | eta: 181.7m
step 00053/05000 (1.06%) | loss: 7.285698 | lrm: 1.00 | dt: 2189.56ms | tok/sec: 7,482 | bf16_mfu: 11.03 | epoch: 1 pq: 0 rg: 3 | total time: 1.58m | eta: 181.7m
step 00054/05000 (1.08%) | loss: 7.244248 | lrm: 1.00 | dt: 2220.79ms | tok/sec: 7,377 | bf16_mfu: 10.88 | epoch: 1 pq: 0 rg: 3 | total time: 1.62m | eta: 181.7m
step 00055/05000 (1.10%) | loss: 7.223006 | lrm: 1.00 | dt: 2192.59ms | tok/sec: 7,472 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 3 | total time: 1.65m | eta: 181.6m
step 00056/05000 (1.12%) | loss: 7.206285 | lrm: 1.00 | dt: 2201.07ms | tok/sec: 7,443 | bf16_mfu: 10.98 | epoch: 1 pq: 0 rg: 3 | total time: 1.69m | eta: 181.6m
step 00057/05000 (1.14%) | loss: 7.171540 | lrm: 1.00 | dt: 2221.69ms | tok/sec: 7,374 | bf16_mfu: 10.87 | epoch: 1 pq: 0 rg: 3 | total time: 1.73m | eta: 181.6m
step 00058/05000 (1.16%) | loss: 7.140428 | lrm: 1.00 | dt: 2201.59ms | tok/sec: 7,441 | bf16_mfu: 10.97 | epoch: 1 pq: 0 rg: 3 | total time: 1.76m | eta: 181.5m
step 00059/05000 (1.18%) | loss: 7.111346 | lrm: 1.00 | dt: 2196.06ms | tok/sec: 7,460 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 3 | total time: 1.80m | eta: 181.5m
step 00060/05000 (1.20%) | loss: 7.079682 | lrm: 1.00 | dt: 2225.79ms | tok/sec: 7,360 | bf16_mfu: 10.85 | epoch: 1 pq: 0 rg: 4 | total time: 1.84m | eta: 181.5m
step 00061/05000 (1.22%) | loss: 7.058748 | lrm: 1.00 | dt: 2191.94ms | tok/sec: 7,474 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 4 | total time: 1.87m | eta: 181.4m
step 00062/05000 (1.24%) | loss: 7.040439 | lrm: 1.00 | dt: 2195.63ms | tok/sec: 7,462 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 4 | total time: 1.91m | eta: 181.4m
step 00063/05000 (1.26%) | loss: 7.002488 | lrm: 1.00 | dt: 2232.14ms | tok/sec: 7,340 | bf16_mfu: 10.82 | epoch: 1 pq: 0 rg: 4 | total time: 1.95m | eta: 181.4m
step 00064/05000 (1.28%) | loss: 7.003826 | lrm: 1.00 | dt: 2192.59ms | tok/sec: 7,472 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 4 | total time: 1.98m | eta: 181.3m
step 00065/05000 (1.30%) | loss: 7.006284 | lrm: 1.00 | dt: 2192.12ms | tok/sec: 7,474 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 4 | total time: 2.02m | eta: 181.3m
step 00066/05000 (1.32%) | loss: 6.985928 | lrm: 1.00 | dt: 2219.44ms | tok/sec: 7,382 | bf16_mfu: 10.89 | epoch: 1 pq: 0 rg: 4 | total time: 2.06m | eta: 181.3m
step 00067/05000 (1.34%) | loss: 6.964270 | lrm: 1.00 | dt: 2192.26ms | tok/sec: 7,473 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 4 | total time: 2.09m | eta: 181.2m
step 00068/05000 (1.36%) | loss: 6.958048 | lrm: 1.00 | dt: 2198.57ms | tok/sec: 7,452 | bf16_mfu: 10.99 | epoch: 1 pq: 0 rg: 4 | total time: 2.13m | eta: 181.2m
step 00069/05000 (1.38%) | loss: 6.944392 | lrm: 1.00 | dt: 2218.74ms | tok/sec: 7,384 | bf16_mfu: 10.89 | epoch: 1 pq: 0 rg: 4 | total time: 2.17m | eta: 181.1m
step 00070/05000 (1.40%) | loss: 6.943193 | lrm: 1.00 | dt: 2195.90ms | tok/sec: 7,461 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 4 | total time: 2.20m | eta: 181.1m
step 00071/05000 (1.42%) | loss: 6.936025 | lrm: 1.00 | dt: 2194.08ms | tok/sec: 7,467 | bf16_mfu: 11.01 | epoch: 1 pq: 0 rg: 4 | total time: 2.24m | eta: 181.0m
step 00072/05000 (1.44%) | loss: 6.931724 | lrm: 1.00 | dt: 2226.89ms | tok/sec: 7,357 | bf16_mfu: 10.85 | epoch: 1 pq: 0 rg: 4 | total time: 2.28m | eta: 181.0m
step 00073/05000 (1.46%) | loss: 6.914391 | lrm: 1.00 | dt: 2188.60ms | tok/sec: 7,486 | bf16_mfu: 11.04 | epoch: 1 pq: 0 rg: 4 | total time: 2.31m | eta: 181.0m
step 00074/05000 (1.48%) | loss: 6.890650 | lrm: 1.00 | dt: 2189.92ms | tok/sec: 7,481 | bf16_mfu: 11.03 | epoch: 1 pq: 0 rg: 4 | total time: 2.35m | eta: 180.9m
step 00075/05000 (1.50%) | loss: 6.872500 | lrm: 1.00 | dt: 2230.29ms | tok/sec: 7,346 | bf16_mfu: 10.83 | epoch: 1 pq: 0 rg: 4 | total time: 2.39m | eta: 180.9m
step 00076/05000 (1.52%) | loss: 6.869487 | lrm: 1.00 | dt: 2197.83ms | tok/sec: 7,454 | bf16_mfu: 10.99 | epoch: 1 pq: 0 rg: 4 | total time: 2.42m | eta: 180.9m
step 00077/05000 (1.54%) | loss: 6.845004 | lrm: 1.00 | dt: 2202.04ms | tok/sec: 7,440 | bf16_mfu: 10.97 | epoch: 1 pq: 0 rg: 4 | total time: 2.46m | eta: 180.8m
step 00078/05000 (1.56%) | loss: 6.831813 | lrm: 1.00 | dt: 2222.17ms | tok/sec: 7,372 | bf16_mfu: 10.87 | epoch: 1 pq: 0 rg: 4 | total time: 2.50m | eta: 180.8m
step 00079/05000 (1.58%) | loss: 6.822285 | lrm: 1.00 | dt: 2198.06ms | tok/sec: 7,453 | bf16_mfu: 10.99 | epoch: 1 pq: 0 rg: 4 | total time: 2.53m | eta: 180.8m
step 00080/05000 (1.60%) | loss: 6.811089 | lrm: 1.00 | dt: 2198.38ms | tok/sec: 7,452 | bf16_mfu: 10.99 | epoch: 1 pq: 0 rg: 4 | total time: 2.57m | eta: 180.7m
step 00081/05000 (1.62%) | loss: 6.808728 | lrm: 1.00 | dt: 2222.85ms | tok/sec: 7,370 | bf16_mfu: 10.87 | epoch: 1 pq: 0 rg: 4 | total time: 2.61m | eta: 180.7m
step 00082/05000 (1.64%) | loss: 6.804447 | lrm: 1.00 | dt: 2191.51ms | tok/sec: 7,476 | bf16_mfu: 11.02 | epoch: 1 pq: 0 rg: 4 | total time: 2.65m | eta: 180.7m
step 00083/05000 (1.66%) | loss: 6.786666 | lrm: 1.00 | dt: 2219.49ms | tok/sec: 7,381 | bf16_mfu: 10.89 | epoch: 1 pq: 0 rg: 5 | total time: 2.68m | eta: 180.6m
step 00084/05000 (1.68%) | loss: 6.762711 | lrm: 1.00 | dt: 2206.17ms | tok/sec: 7,426 | bf16_mfu: 10.95 | epoch: 1 pq: 0 rg: 5 | total time: 2.72m | eta: 180.6m
step 00085/05000 (1.70%) | loss: 6.747888 | lrm: 1.00 | dt: 2195.62ms | tok/sec: 7,462 | bf16_mfu: 11.00 | epoch: 1 pq: 0 rg: 5 | total time: 2.76m | eta: 180.6m

```
