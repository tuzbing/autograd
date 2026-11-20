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
[ARENA-TRAIN-ALLOC] bytes=427819264 buffer_id=47199512192
Step 00000 | Validation bpb: 3.207952
step 00000/05000 (0.00%) | loss: 10.427742 | lrm: 0.03 | dt: 2015.99ms | tok/sec: 8,127 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/05000 (0.02%) | loss: 10.385231 | lrm: 0.05 | dt: 2021.10ms | tok/sec: 8,106 | bf16_mfu: 11.95 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/05000 (0.04%) | loss: 10.302640 | lrm: 0.07 | dt: 1989.64ms | tok/sec: 8,234 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00003/05000 (0.06%) | loss: 10.151242 | lrm: 0.10 | dt: 2013.60ms | tok/sec: 8,136 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00004/05000 (0.08%) | loss: 9.893878 | lrm: 0.12 | dt: 1990.40ms | tok/sec: 8,231 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00005/05000 (0.10%) | loss: 9.579190 | lrm: 0.15 | dt: 2013.17ms | tok/sec: 8,138 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00006/05000 (0.12%) | loss: 9.363238 | lrm: 0.17 | dt: 2015.84ms | tok/sec: 8,127 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00007/05000 (0.14%) | loss: 9.244887 | lrm: 0.20 | dt: 1993.30ms | tok/sec: 8,219 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00008/05000 (0.16%) | loss: 9.098031 | lrm: 0.23 | dt: 2010.66ms | tok/sec: 8,148 | bf16_mfu: 12.02 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00009/05000 (0.18%) | loss: 9.060223 | lrm: 0.25 | dt: 1994.52ms | tok/sec: 8,214 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00010/05000 (0.20%) | loss: 8.964049 | lrm: 0.28 | dt: 2016.77ms | tok/sec: 8,123 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00011/05000 (0.22%) | loss: 8.906734 | lrm: 0.30 | dt: 1979.19ms | tok/sec: 8,278 | bf16_mfu: 12.21 | epoch: 1 pq: 0 rg: 1 | total time: 0.03m | eta: 164.6m
step 00012/05000 (0.24%) | loss: 8.828270 | lrm: 0.33 | dt: 2007.17ms | tok/sec: 8,162 | bf16_mfu: 12.04 | epoch: 1 pq: 0 rg: 1 | total time: 0.07m | eta: 165.7m
step 00013/05000 (0.26%) | loss: 8.755030 | lrm: 0.35 | dt: 1988.54ms | tok/sec: 8,239 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 1 | total time: 0.10m | eta: 165.5m
step 00014/05000 (0.28%) | loss: 8.688153 | lrm: 0.38 | dt: 2024.00ms | tok/sec: 8,094 | bf16_mfu: 11.94 | epoch: 1 pq: 0 rg: 2 | total time: 0.13m | eta: 166.2m
step 00015/05000 (0.30%) | loss: 8.614209 | lrm: 0.40 | dt: 1989.73ms | tok/sec: 8,234 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 2 | total time: 0.17m | eta: 166.0m
step 00016/05000 (0.32%) | loss: 8.530892 | lrm: 0.42 | dt: 1989.44ms | tok/sec: 8,235 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 2 | total time: 0.20m | eta: 165.8m
step 00017/05000 (0.34%) | loss: 8.467441 | lrm: 0.45 | dt: 2009.89ms | tok/sec: 8,151 | bf16_mfu: 12.02 | epoch: 1 pq: 0 rg: 2 | total time: 0.23m | eta: 166.0m
step 00018/05000 (0.36%) | loss: 8.421451 | lrm: 0.47 | dt: 1990.35ms | tok/sec: 8,231 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 2 | total time: 0.27m | eta: 165.8m
step 00019/05000 (0.38%) | loss: 8.378556 | lrm: 0.50 | dt: 1989.39ms | tok/sec: 8,235 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 2 | total time: 0.30m | eta: 165.7m
step 00020/05000 (0.40%) | loss: 8.342315 | lrm: 0.53 | dt: 2011.37ms | tok/sec: 8,145 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 2 | total time: 0.33m | eta: 165.8m
step 00021/05000 (0.42%) | loss: 8.309185 | lrm: 0.55 | dt: 1999.29ms | tok/sec: 8,194 | bf16_mfu: 12.08 | epoch: 1 pq: 0 rg: 2 | total time: 0.37m | eta: 165.8m
step 00022/05000 (0.44%) | loss: 8.268358 | lrm: 0.57 | dt: 2001.78ms | tok/sec: 8,184 | bf16_mfu: 12.07 | epoch: 1 pq: 0 rg: 2 | total time: 0.40m | eta: 165.8m
step 00023/05000 (0.46%) | loss: 8.214408 | lrm: 0.60 | dt: 2023.87ms | tok/sec: 8,095 | bf16_mfu: 11.94 | epoch: 1 pq: 0 rg: 2 | total time: 0.43m | eta: 165.9m
step 00024/05000 (0.48%) | loss: 8.165227 | lrm: 0.62 | dt: 1993.03ms | tok/sec: 8,220 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 2 | total time: 0.47m | eta: 165.8m
step 00025/05000 (0.50%) | loss: 8.109949 | lrm: 0.65 | dt: 1987.69ms | tok/sec: 8,242 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 2 | total time: 0.50m | eta: 165.7m
step 00026/05000 (0.52%) | loss: 8.063710 | lrm: 0.68 | dt: 2018.83ms | tok/sec: 8,115 | bf16_mfu: 11.97 | epoch: 1 pq: 0 rg: 2 | total time: 0.53m | eta: 165.8m
step 00027/05000 (0.54%) | loss: 8.003073 | lrm: 0.70 | dt: 1995.88ms | tok/sec: 8,208 | bf16_mfu: 12.10 | epoch: 1 pq: 0 rg: 2 | total time: 0.57m | eta: 165.8m
step 00028/05000 (0.56%) | loss: 7.956738 | lrm: 0.72 | dt: 2015.77ms | tok/sec: 8,127 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 2 | total time: 0.60m | eta: 165.8m
step 00029/05000 (0.58%) | loss: 7.906719 | lrm: 0.75 | dt: 1992.43ms | tok/sec: 8,223 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 2 | total time: 0.63m | eta: 165.7m
step 00030/05000 (0.60%) | loss: 7.854174 | lrm: 0.78 | dt: 1992.72ms | tok/sec: 8,221 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 2 | total time: 0.67m | eta: 165.7m
step 00031/05000 (0.62%) | loss: 7.800669 | lrm: 0.80 | dt: 2013.40ms | tok/sec: 8,137 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.70m | eta: 165.7m
step 00032/05000 (0.64%) | loss: 7.749631 | lrm: 0.82 | dt: 1993.60ms | tok/sec: 8,218 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 2 | total time: 0.73m | eta: 165.6m
step 00033/05000 (0.66%) | loss: 7.730531 | lrm: 0.85 | dt: 1989.19ms | tok/sec: 8,236 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 2 | total time: 0.77m | eta: 165.6m
step 00034/05000 (0.68%) | loss: 7.759639 | lrm: 0.88 | dt: 2015.31ms | tok/sec: 8,129 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 2 | total time: 0.80m | eta: 165.6m
step 00035/05000 (0.70%) | loss: 7.735408 | lrm: 0.90 | dt: 1995.36ms | tok/sec: 8,211 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 2 | total time: 0.83m | eta: 165.5m
step 00036/05000 (0.72%) | loss: 7.719984 | lrm: 0.93 | dt: 1987.03ms | tok/sec: 8,245 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 2 | total time: 0.87m | eta: 165.4m
step 00037/05000 (0.74%) | loss: 7.696400 | lrm: 0.95 | dt: 2030.87ms | tok/sec: 8,067 | bf16_mfu: 11.90 | epoch: 1 pq: 0 rg: 3 | total time: 0.90m | eta: 165.5m
step 00038/05000 (0.76%) | loss: 7.648251 | lrm: 0.97 | dt: 1994.68ms | tok/sec: 8,213 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 3 | total time: 0.93m | eta: 165.5m
step 00039/05000 (0.78%) | loss: 7.605126 | lrm: 1.00 | dt: 1986.41ms | tok/sec: 8,248 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 3 | total time: 0.97m | eta: 165.4m
step 00040/05000 (0.80%) | loss: 7.566101 | lrm: 1.00 | dt: 2018.56ms | tok/sec: 8,116 | bf16_mfu: 11.97 | epoch: 1 pq: 0 rg: 3 | total time: 1.00m | eta: 165.4m
step 00041/05000 (0.82%) | loss: 7.527114 | lrm: 1.00 | dt: 1989.73ms | tok/sec: 8,234 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.03m | eta: 165.3m
step 00042/05000 (0.84%) | loss: 7.510663 | lrm: 1.00 | dt: 1989.49ms | tok/sec: 8,235 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.07m | eta: 165.3m
step 00043/05000 (0.86%) | loss: 7.469676 | lrm: 1.00 | dt: 2016.90ms | tok/sec: 8,123 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 3 | total time: 1.10m | eta: 165.3m
step 00044/05000 (0.88%) | loss: 7.434433 | lrm: 1.00 | dt: 1987.31ms | tok/sec: 8,244 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 3 | total time: 1.13m | eta: 165.2m
step 00045/05000 (0.90%) | loss: 7.400616 | lrm: 1.00 | dt: 2019.19ms | tok/sec: 8,114 | bf16_mfu: 11.96 | epoch: 1 pq: 0 rg: 3 | total time: 1.17m | eta: 165.2m
step 00046/05000 (0.92%) | loss: 7.368733 | lrm: 1.00 | dt: 1996.48ms | tok/sec: 8,206 | bf16_mfu: 12.10 | epoch: 1 pq: 0 rg: 3 | total time: 1.20m | eta: 165.2m
step 00047/05000 (0.94%) | loss: 7.337410 | lrm: 1.00 | dt: 1988.20ms | tok/sec: 8,240 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 3 | total time: 1.23m | eta: 165.1m
step 00048/05000 (0.96%) | loss: 7.295033 | lrm: 1.00 | dt: 2005.05ms | tok/sec: 8,171 | bf16_mfu: 12.05 | epoch: 1 pq: 0 rg: 3 | total time: 1.27m | eta: 165.1m
step 00049/05000 (0.98%) | loss: 7.290633 | lrm: 1.00 | dt: 1989.68ms | tok/sec: 8,234 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.30m | eta: 165.0m
step 00050/05000 (1.00%) | loss: 7.304640 | lrm: 1.00 | dt: 1987.03ms | tok/sec: 8,245 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 3 | total time: 1.33m | eta: 165.0m
step 00051/05000 (1.02%) | loss: 7.298804 | lrm: 1.00 | dt: 2017.17ms | tok/sec: 8,122 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 3 | total time: 1.37m | eta: 165.0m
step 00052/05000 (1.04%) | loss: 7.279337 | lrm: 1.00 | dt: 1991.68ms | tok/sec: 8,226 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 3 | total time: 1.40m | eta: 164.9m
step 00053/05000 (1.06%) | loss: 7.275640 | lrm: 1.00 | dt: 1989.63ms | tok/sec: 8,234 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.43m | eta: 164.9m
step 00054/05000 (1.08%) | loss: 7.238672 | lrm: 1.00 | dt: 2020.79ms | tok/sec: 8,107 | bf16_mfu: 11.96 | epoch: 1 pq: 0 rg: 3 | total time: 1.47m | eta: 164.9m
step 00055/05000 (1.10%) | loss: 7.217595 | lrm: 1.00 | dt: 1990.21ms | tok/sec: 8,232 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.50m | eta: 164.8m
step 00056/05000 (1.12%) | loss: 7.203374 | lrm: 1.00 | dt: 1988.36ms | tok/sec: 8,239 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 3 | total time: 1.53m | eta: 164.8m
step 00057/05000 (1.14%) | loss: 7.167904 | lrm: 1.00 | dt: 2020.98ms | tok/sec: 8,106 | bf16_mfu: 11.95 | epoch: 1 pq: 0 rg: 3 | total time: 1.57m | eta: 164.8m
step 00058/05000 (1.16%) | loss: 7.137389 | lrm: 1.00 | dt: 1994.55ms | tok/sec: 8,214 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 3 | total time: 1.60m | eta: 164.7m
step 00059/05000 (1.18%) | loss: 7.108010 | lrm: 1.00 | dt: 1990.37ms | tok/sec: 8,231 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.63m | eta: 164.7m
step 00060/05000 (1.20%) | loss: 7.080744 | lrm: 1.00 | dt: 2027.86ms | tok/sec: 8,079 | bf16_mfu: 11.91 | epoch: 1 pq: 0 rg: 4 | total time: 1.67m | eta: 164.7m
step 00061/05000 (1.22%) | loss: 7.073455 | lrm: 1.00 | dt: 1998.38ms | tok/sec: 8,198 | bf16_mfu: 12.09 | epoch: 1 pq: 0 rg: 4 | total time: 1.70m | eta: 164.7m
step 00062/05000 (1.24%) | loss: 7.073158 | lrm: 1.00 | dt: 1990.18ms | tok/sec: 8,232 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 1.73m | eta: 164.6m
step 00063/05000 (1.26%) | loss: 7.055619 | lrm: 1.00 | dt: 2024.78ms | tok/sec: 8,091 | bf16_mfu: 11.93 | epoch: 1 pq: 0 rg: 4 | total time: 1.77m | eta: 164.6m
step 00064/05000 (1.28%) | loss: 7.048592 | lrm: 1.00 | dt: 1986.56ms | tok/sec: 8,247 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 4 | total time: 1.80m | eta: 164.6m
step 00065/05000 (1.30%) | loss: 7.036817 | lrm: 1.00 | dt: 1992.52ms | tok/sec: 8,222 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 4 | total time: 1.83m | eta: 164.5m
step 00066/05000 (1.32%) | loss: 7.018598 | lrm: 1.00 | dt: 2012.16ms | tok/sec: 8,142 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 4 | total time: 1.87m | eta: 164.5m
step 00067/05000 (1.34%) | loss: 6.995808 | lrm: 1.00 | dt: 1990.03ms | tok/sec: 8,233 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 1.90m | eta: 164.5m
step 00068/05000 (1.36%) | loss: 6.989129 | lrm: 1.00 | dt: 1986.00ms | tok/sec: 8,249 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 4 | total time: 1.93m | eta: 164.4m
step 00069/05000 (1.38%) | loss: 6.970683 | lrm: 1.00 | dt: 2014.66ms | tok/sec: 8,132 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 4 | total time: 1.97m | eta: 164.4m
step 00070/05000 (1.40%) | loss: 6.962532 | lrm: 1.00 | dt: 1994.72ms | tok/sec: 8,213 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 4 | total time: 2.00m | eta: 164.4m
step 00071/05000 (1.42%) | loss: 6.955625 | lrm: 1.00 | dt: 1988.52ms | tok/sec: 8,239 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 4 | total time: 2.03m | eta: 164.3m
step 00072/05000 (1.44%) | loss: 6.950037 | lrm: 1.00 | dt: 2018.80ms | tok/sec: 8,115 | bf16_mfu: 11.97 | epoch: 1 pq: 0 rg: 4 | total time: 2.07m | eta: 164.3m
step 00073/05000 (1.46%) | loss: 6.925325 | lrm: 1.00 | dt: 1988.13ms | tok/sec: 8,240 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 4 | total time: 2.10m | eta: 164.2m
step 00074/05000 (1.48%) | loss: 6.897535 | lrm: 1.00 | dt: 1989.77ms | tok/sec: 8,234 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 2.13m | eta: 164.2m
step 00075/05000 (1.50%) | loss: 6.879212 | lrm: 1.00 | dt: 2018.48ms | tok/sec: 8,117 | bf16_mfu: 11.97 | epoch: 1 pq: 0 rg: 4 | total time: 2.17m | eta: 164.2m
step 00076/05000 (1.52%) | loss: 6.874531 | lrm: 1.00 | dt: 1989.34ms | tok/sec: 8,235 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 2.20m | eta: 164.1m
step 00077/05000 (1.54%) | loss: 6.849498 | lrm: 1.00 | dt: 1988.53ms | tok/sec: 8,239 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 4 | total time: 2.23m | eta: 164.1m
step 00078/05000 (1.56%) | loss: 6.839081 | lrm: 1.00 | dt: 2016.09ms | tok/sec: 8,126 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 4 | total time: 2.27m | eta: 164.1m
step 00079/05000 (1.58%) | loss: 6.829721 | lrm: 1.00 | dt: 1987.72ms | tok/sec: 8,242 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 4 | total time: 2.30m | eta: 164.0m
step 00080/05000 (1.60%) | loss: 6.817215 | lrm: 1.00 | dt: 1989.04ms | tok/sec: 8,237 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 4 | total time: 2.33m | eta: 164.0m
step 00081/05000 (1.62%) | loss: 6.814691 | lrm: 1.00 | dt: 2017.69ms | tok/sec: 8,120 | bf16_mfu: 11.97 | epoch: 1 pq: 0 rg: 4 | total time: 2.37m | eta: 164.0m
step 00082/05000 (1.64%) | loss: 6.809471 | lrm: 1.00 | dt: 1987.69ms | tok/sec: 8,242 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 4 | total time: 2.40m | eta: 163.9m
step 00083/05000 (1.66%) | loss: 6.790530 | lrm: 1.00 | dt: 2021.36ms | tok/sec: 8,105 | bf16_mfu: 11.95 | epoch: 1 pq: 0 rg: 5 | total time: 2.43m | eta: 163.9m
step 00084/05000 (1.68%) | loss: 6.766337 | lrm: 1.00 | dt: 1996.43ms | tok/sec: 8,206 | bf16_mfu: 12.10 | epoch: 1 pq: 0 rg: 5 | total time: 2.47m | eta: 163.9m
step 00085/05000 (1.70%) | loss: 6.752297 | lrm: 1.00 | dt: 1982.89ms | tok/sec: 8,262 | bf16_mfu: 12.18 | epoch: 1 pq: 0 rg: 5 | total time: 2.50m | eta: 163.8m
step 00086/05000 (1.72%) | loss: 6.729776 | lrm: 1.00 | dt: 2012.56ms | tok/sec: 8,140 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 5 | total time: 2.53m | eta: 163.8m
step 00087/05000 (1.74%) | loss: 6.717146 | lrm: 1.00 | dt: 1992.44ms | tok/sec: 8,223 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 5 | total time: 2.57m | eta: 163.8m
step 00088/05000 (1.76%) | loss: 6.698003 | lrm: 1.00 | dt: 1996.98ms | tok/sec: 8,204 | bf16_mfu: 12.10 | epoch: 1 pq: 0 rg: 5 | total time: 2.60m | eta: 163.7m
step 00089/05000 (1.78%) | loss: 6.680960 | lrm: 1.00 | dt: 2014.42ms | tok/sec: 8,133 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 5 | total time: 2.63m | eta: 163.7m
step 00090/05000 (1.80%) | loss: 6.687532 | lrm: 1.00 | dt: 1998.97ms | tok/sec: 8,196 | bf16_mfu: 12.09 | epoch: 1 pq: 0 rg: 5 | total time: 2.67m | eta: 163.7m
step 00091/05000 (1.82%) | loss: 6.683723 | lrm: 1.00 | dt: 1985.62ms | tok/sec: 8,251 | bf16_mfu: 12.17 | epoch: 1 pq: 0 rg: 5 | total time: 2.70m | eta: 163.6m
step 00092/05000 (1.84%) | loss: 6.672331 | lrm: 1.00 | dt: 2011.70ms | tok/sec: 8,144 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 5 | total time: 2.73m | eta: 163.6m
step 00093/05000 (1.86%) | loss: 6.662730 | lrm: 1.00 | dt: 1980.86ms | tok/sec: 8,271 | bf16_mfu: 12.20 | epoch: 1 pq: 0 rg: 5 | total time: 2.77m | eta: 163.6m
step 00094/05000 (1.88%) | loss: 6.652967 | lrm: 1.00 | dt: 1984.00ms | tok/sec: 8,258 | bf16_mfu: 12.18 | epoch: 1 pq: 0 rg: 5 | total time: 2.80m | eta: 163.5m
step 00095/05000 (1.90%) | loss: 6.653304 | lrm: 1.00 | dt: 2009.98ms | tok/sec: 8,151 | bf16_mfu: 12.02 | epoch: 1 pq: 0 rg: 5 | total time: 2.83m | eta: 163.5m
step 00096/05000 (1.92%) | loss: 6.640850 | lrm: 1.00 | dt: 1994.56ms | tok/sec: 8,214 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 5 | total time: 2.87m | eta: 163.4m
step 00097/05000 (1.94%) | loss: 6.635481 | lrm: 1.00 | dt: 1985.92ms | tok/sec: 8,250 | bf16_mfu: 12.17 | epoch: 1 pq: 0 rg: 5 | total time: 2.90m | eta: 163.4m
step 00098/05000 (1.96%) | loss: 6.635180 | lrm: 1.00 | dt: 2017.43ms | tok/sec: 8,121 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 5 | total time: 2.93m | eta: 163.4m
step 00099/05000 (1.98%) | loss: 6.616316 | lrm: 1.00 | dt: 1991.08ms | tok/sec: 8,228 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 5 | total time: 2.97m | eta: 163.3m

```
