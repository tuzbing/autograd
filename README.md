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
[ARENA-TRAIN-ALLOC] bytes=427819264 buffer_id=50062124672
Step 00000 | Validation bpb: 3.207953
step 00000/05000 (0.00%) | loss: 10.427744 | lrm: 0.03 | dt: 2186.67ms | tok/sec: 7,492 | bf16_mfu: 11.05 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/05000 (0.02%) | loss: 10.385235 | lrm: 0.05 | dt: 2144.60ms | tok/sec: 7,639 | bf16_mfu: 11.27 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/05000 (0.04%) | loss: 10.302645 | lrm: 0.07 | dt: 2121.08ms | tok/sec: 7,724 | bf16_mfu: 11.39 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00003/05000 (0.06%) | loss: 10.151248 | lrm: 0.10 | dt: 2028.57ms | tok/sec: 8,076 | bf16_mfu: 11.91 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00004/05000 (0.08%) | loss: 9.893886 | lrm: 0.12 | dt: 2027.44ms | tok/sec: 8,081 | bf16_mfu: 11.92 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00005/05000 (0.10%) | loss: 9.579196 | lrm: 0.15 | dt: 1883.27ms | tok/sec: 8,699 | bf16_mfu: 12.83 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00006/05000 (0.12%) | loss: 9.363248 | lrm: 0.17 | dt: 2008.74ms | tok/sec: 8,156 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00007/05000 (0.14%) | loss: 9.244902 | lrm: 0.20 | dt: 927127.99ms | tok/sec: 17 | bf16_mfu: 0.03 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00008/05000 (0.16%) | loss: 9.098048 | lrm: 0.23 | dt: 2042.33ms | tok/sec: 8,022 | bf16_mfu: 11.83 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00009/05000 (0.18%) | loss: 9.060227 | lrm: 0.25 | dt: 1877.61ms | tok/sec: 8,725 | bf16_mfu: 12.87 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00010/05000 (0.20%) | loss: 8.964053 | lrm: 0.28 | dt: 2008.41ms | tok/sec: 8,157 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00011/05000 (0.22%) | loss: 8.906731 | lrm: 0.30 | dt: 1990.26ms | tok/sec: 8,232 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 1 | total time: 0.03m | eta: 165.5m
step 00012/05000 (0.24%) | loss: 8.828265 | lrm: 0.33 | dt: 2015.02ms | tok/sec: 8,130 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 1 | total time: 0.07m | eta: 166.5m
step 00013/05000 (0.26%) | loss: 8.755027 | lrm: 0.35 | dt: 1983.51ms | tok/sec: 8,260 | bf16_mfu: 12.18 | epoch: 1 pq: 0 rg: 1 | total time: 0.10m | eta: 165.9m
step 00014/05000 (0.28%) | loss: 8.688148 | lrm: 0.38 | dt: 2024.70ms | tok/sec: 8,092 | bf16_mfu: 11.93 | epoch: 1 pq: 0 rg: 2 | total time: 0.13m | eta: 166.5m
step 00015/05000 (0.30%) | loss: 8.614204 | lrm: 0.40 | dt: 1998.18ms | tok/sec: 8,199 | bf16_mfu: 12.09 | epoch: 1 pq: 0 rg: 2 | total time: 0.17m | eta: 166.4m
step 00016/05000 (0.32%) | loss: 8.530887 | lrm: 0.42 | dt: 1981.14ms | tok/sec: 8,269 | bf16_mfu: 12.19 | epoch: 1 pq: 0 rg: 2 | total time: 0.20m | eta: 166.0m
step 00017/05000 (0.34%) | loss: 8.467435 | lrm: 0.45 | dt: 2008.66ms | tok/sec: 8,156 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 2 | total time: 0.23m | eta: 166.1m
step 00018/05000 (0.36%) | loss: 8.421463 | lrm: 0.47 | dt: 1990.83ms | tok/sec: 8,229 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 2 | total time: 0.27m | eta: 166.0m
step 00019/05000 (0.38%) | loss: 8.378566 | lrm: 0.50 | dt: 1985.12ms | tok/sec: 8,253 | bf16_mfu: 12.17 | epoch: 1 pq: 0 rg: 2 | total time: 0.30m | eta: 165.8m
step 00020/05000 (0.40%) | loss: 8.342336 | lrm: 0.53 | dt: 2004.11ms | tok/sec: 8,175 | bf16_mfu: 12.05 | epoch: 1 pq: 0 rg: 2 | total time: 0.33m | eta: 165.8m
step 00021/05000 (0.42%) | loss: 8.309220 | lrm: 0.55 | dt: 1985.45ms | tok/sec: 8,252 | bf16_mfu: 12.17 | epoch: 1 pq: 0 rg: 2 | total time: 0.37m | eta: 165.7m
step 00022/05000 (0.44%) | loss: 8.268379 | lrm: 0.57 | dt: 1987.50ms | tok/sec: 8,243 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 2 | total time: 0.40m | eta: 165.6m
step 00023/05000 (0.46%) | loss: 8.214410 | lrm: 0.60 | dt: 2001.53ms | tok/sec: 8,185 | bf16_mfu: 12.07 | epoch: 1 pq: 0 rg: 2 | total time: 0.43m | eta: 165.6m
step 00024/05000 (0.48%) | loss: 8.165240 | lrm: 0.62 | dt: 1989.12ms | tok/sec: 8,236 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 2 | total time: 0.47m | eta: 165.5m
step 00025/05000 (0.50%) | loss: 8.109954 | lrm: 0.65 | dt: 1987.41ms | tok/sec: 8,243 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 2 | total time: 0.50m | eta: 165.5m
step 00026/05000 (0.52%) | loss: 8.063729 | lrm: 0.68 | dt: 2017.08ms | tok/sec: 8,122 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 2 | total time: 0.53m | eta: 165.5m
step 00027/05000 (0.54%) | loss: 8.003084 | lrm: 0.70 | dt: 1981.97ms | tok/sec: 8,266 | bf16_mfu: 12.19 | epoch: 1 pq: 0 rg: 2 | total time: 0.57m | eta: 165.4m
step 00028/05000 (0.56%) | loss: 7.956735 | lrm: 0.72 | dt: 2012.83ms | tok/sec: 8,139 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.60m | eta: 165.5m
step 00029/05000 (0.58%) | loss: 7.906711 | lrm: 0.75 | dt: 2000.70ms | tok/sec: 8,189 | bf16_mfu: 12.08 | epoch: 1 pq: 0 rg: 2 | total time: 0.63m | eta: 165.5m
step 00030/05000 (0.60%) | loss: 7.854163 | lrm: 0.78 | dt: 1988.70ms | tok/sec: 8,238 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 2 | total time: 0.67m | eta: 165.4m
step 00031/05000 (0.62%) | loss: 7.800707 | lrm: 0.80 | dt: 927025.96ms | tok/sec: 17 | bf16_mfu: 0.03 | epoch: 1 pq: 0 rg: 2 | total time: 16.12m | eta: 3813.4m
step 00032/05000 (0.64%) | loss: 7.750107 | lrm: 0.82 | dt: 2151.17ms | tok/sec: 7,616 | bf16_mfu: 11.23 | epoch: 1 pq: 0 rg: 2 | total time: 16.15m | eta: 3647.4m
step 00033/05000 (0.66%) | loss: 7.734466 | lrm: 0.85 | dt: 2134.45ms | tok/sec: 7,675 | bf16_mfu: 11.32 | epoch: 1 pq: 0 rg: 2 | total time: 16.19m | eta: 3495.8m
step 00034/05000 (0.68%) | loss: 7.760694 | lrm: 0.88 | dt: 2005.18ms | tok/sec: 8,170 | bf16_mfu: 12.05 | epoch: 1 pq: 0 rg: 2 | total time: 16.22m | eta: 3356.4m
step 00035/05000 (0.70%) | loss: 7.733828 | lrm: 0.90 | dt: 2001.26ms | tok/sec: 8,186 | bf16_mfu: 12.07 | epoch: 1 pq: 0 rg: 2 | total time: 16.25m | eta: 3228.1m
step 00036/05000 (0.72%) | loss: 7.718328 | lrm: 0.93 | dt: 1992.97ms | tok/sec: 8,220 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 2 | total time: 16.29m | eta: 3109.6m
step 00037/05000 (0.74%) | loss: 7.695682 | lrm: 0.95 | dt: 2028.10ms | tok/sec: 8,078 | bf16_mfu: 11.91 | epoch: 1 pq: 0 rg: 3 | total time: 16.32m | eta: 3000.1m
step 00038/05000 (0.76%) | loss: 7.648221 | lrm: 0.97 | dt: 1992.61ms | tok/sec: 8,222 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 3 | total time: 16.35m | eta: 2898.2m
step 00039/05000 (0.78%) | loss: 7.605316 | lrm: 1.00 | dt: 2003.93ms | tok/sec: 8,175 | bf16_mfu: 12.06 | epoch: 1 pq: 0 rg: 3 | total time: 16.39m | eta: 2803.4m
step 00040/05000 (0.80%) | loss: 7.566379 | lrm: 1.00 | dt: 2013.73ms | tok/sec: 8,136 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 3 | total time: 16.42m | eta: 2715.0m
step 00041/05000 (0.82%) | loss: 7.525426 | lrm: 1.00 | dt: 1987.87ms | tok/sec: 8,241 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 3 | total time: 16.45m | eta: 2632.2m
step 00042/05000 (0.84%) | loss: 7.506892 | lrm: 1.00 | dt: 1989.92ms | tok/sec: 8,233 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 16.49m | eta: 2554.6m
step 00043/05000 (0.86%) | loss: 7.466387 | lrm: 1.00 | dt: 2011.55ms | tok/sec: 8,144 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 3 | total time: 16.52m | eta: 2481.7m
step 00044/05000 (0.88%) | loss: 7.431544 | lrm: 1.00 | dt: 2004.17ms | tok/sec: 8,174 | bf16_mfu: 12.05 | epoch: 1 pq: 0 rg: 3 | total time: 16.55m | eta: 2413.1m
step 00045/05000 (0.90%) | loss: 7.397911 | lrm: 1.00 | dt: 2019.18ms | tok/sec: 8,114 | bf16_mfu: 11.96 | epoch: 1 pq: 0 rg: 3 | total time: 16.59m | eta: 2348.4m
step 00046/05000 (0.92%) | loss: 7.365649 | lrm: 1.00 | dt: 1987.84ms | tok/sec: 8,242 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 3 | total time: 16.62m | eta: 2287.3m
step 00047/05000 (0.94%) | loss: 7.334270 | lrm: 1.00 | dt: 1995.31ms | tok/sec: 8,211 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 3 | total time: 16.65m | eta: 2229.5m
step 00048/05000 (0.96%) | loss: 7.293590 | lrm: 1.00 | dt: 2018.70ms | tok/sec: 8,116 | bf16_mfu: 11.97 | epoch: 1 pq: 0 rg: 3 | total time: 16.69m | eta: 2174.7m
step 00049/05000 (0.98%) | loss: 7.295524 | lrm: 1.00 | dt: 1983.56ms | tok/sec: 8,259 | bf16_mfu: 12.18 | epoch: 1 pq: 0 rg: 3 | total time: 16.72m | eta: 2122.8m
step 00050/05000 (1.00%) | loss: 7.308799 | lrm: 1.00 | dt: 1982.34ms | tok/sec: 8,264 | bf16_mfu: 12.19 | epoch: 1 pq: 0 rg: 3 | total time: 16.75m | eta: 2073.4m
step 00051/05000 (1.02%) | loss: 7.303832 | lrm: 1.00 | dt: 2011.83ms | tok/sec: 8,143 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 3 | total time: 16.79m | eta: 2026.4m
step 00052/05000 (1.04%) | loss: 7.288399 | lrm: 1.00 | dt: 1987.11ms | tok/sec: 8,245 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 3 | total time: 16.82m | eta: 1981.7m
step 00053/05000 (1.06%) | loss: 7.283172 | lrm: 1.00 | dt: 1988.68ms | tok/sec: 8,238 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 3 | total time: 16.85m | eta: 1939.0m
step 00054/05000 (1.08%) | loss: 7.244623 | lrm: 1.00 | dt: 2017.13ms | tok/sec: 8,122 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 3 | total time: 16.89m | eta: 1898.3m
step 00055/05000 (1.10%) | loss: 7.222910 | lrm: 1.00 | dt: 907044.37ms | tok/sec: 18 | bf16_mfu: 0.03 | epoch: 1 pq: 0 rg: 3 | total time: 32.01m | eta: 3517.0m
step 00056/05000 (1.12%) | loss: 7.210744 | lrm: 1.00 | dt: 2014.65ms | tok/sec: 8,132 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 3 | total time: 32.04m | eta: 3443.5m
step 00057/05000 (1.14%) | loss: 7.178051 | lrm: 1.00 | dt: 2022.74ms | tok/sec: 8,099 | bf16_mfu: 11.94 | epoch: 1 pq: 0 rg: 3 | total time: 32.07m | eta: 3373.1m
step 00058/05000 (1.16%) | loss: 7.147384 | lrm: 1.00 | dt: 1990.52ms | tok/sec: 8,231 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 32.11m | eta: 3305.5m
step 00059/05000 (1.18%) | loss: 7.117085 | lrm: 1.00 | dt: 1982.78ms | tok/sec: 8,263 | bf16_mfu: 12.18 | epoch: 1 pq: 0 rg: 3 | total time: 32.14m | eta: 3240.8m
step 00060/05000 (1.20%) | loss: 7.086779 | lrm: 1.00 | dt: 2017.24ms | tok/sec: 8,121 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 4 | total time: 32.17m | eta: 3178.6m
step 00061/05000 (1.22%) | loss: 7.068400 | lrm: 1.00 | dt: 1983.05ms | tok/sec: 8,262 | bf16_mfu: 12.18 | epoch: 1 pq: 0 rg: 4 | total time: 32.21m | eta: 3118.9m
step 00062/05000 (1.24%) | loss: 7.050259 | lrm: 1.00 | dt: 1985.61ms | tok/sec: 8,251 | bf16_mfu: 12.17 | epoch: 1 pq: 0 rg: 4 | total time: 32.24m | eta: 3061.4m
step 00063/05000 (1.26%) | loss: 7.009080 | lrm: 1.00 | dt: 2011.41ms | tok/sec: 8,145 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 4 | total time: 32.27m | eta: 3006.2m
step 00064/05000 (1.28%) | loss: 7.000410 | lrm: 1.00 | dt: 1986.36ms | tok/sec: 8,248 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 4 | total time: 32.31m | eta: 2952.9m
step 00065/05000 (1.30%) | loss: 6.996325 | lrm: 1.00 | dt: 1985.76ms | tok/sec: 8,250 | bf16_mfu: 12.17 | epoch: 1 pq: 0 rg: 4 | total time: 32.34m | eta: 2901.6m
step 00066/05000 (1.32%) | loss: 6.988550 | lrm: 1.00 | dt: 2007.96ms | tok/sec: 8,159 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 4 | total time: 32.37m | eta: 2852.2m
step 00067/05000 (1.34%) | loss: 6.969863 | lrm: 1.00 | dt: 1990.38ms | tok/sec: 8,231 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 32.40m | eta: 2804.4m
step 00068/05000 (1.36%) | loss: 6.965020 | lrm: 1.00 | dt: 1984.01ms | tok/sec: 8,258 | bf16_mfu: 12.18 | epoch: 1 pq: 0 rg: 4 | total time: 32.44m | eta: 2758.3m
step 00069/05000 (1.38%) | loss: 6.947698 | lrm: 1.00 | dt: 1998.10ms | tok/sec: 8,199 | bf16_mfu: 12.09 | epoch: 1 pq: 0 rg: 4 | total time: 32.47m | eta: 2713.8m
step 00070/05000 (1.40%) | loss: 6.939132 | lrm: 1.00 | dt: 1981.90ms | tok/sec: 8,266 | bf16_mfu: 12.19 | epoch: 1 pq: 0 rg: 4 | total time: 32.50m | eta: 2670.8m
step 00071/05000 (1.42%) | loss: 6.925082 | lrm: 1.00 | dt: 1979.97ms | tok/sec: 8,274 | bf16_mfu: 12.20 | epoch: 1 pq: 0 rg: 4 | total time: 32.54m | eta: 2629.1m
step 00072/05000 (1.44%) | loss: 6.913526 | lrm: 1.00 | dt: 2015.93ms | tok/sec: 8,127 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 4 | total time: 32.57m | eta: 2588.9m
step 00073/05000 (1.46%) | loss: 6.890031 | lrm: 1.00 | dt: 1981.21ms | tok/sec: 8,269 | bf16_mfu: 12.19 | epoch: 1 pq: 0 rg: 4 | total time: 32.60m | eta: 2549.8m
step 00074/05000 (1.48%) | loss: 6.860357 | lrm: 1.00 | dt: 1991.81ms | tok/sec: 8,225 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 4 | total time: 32.64m | eta: 2512.0m
step 00075/05000 (1.50%) | loss: 6.841683 | lrm: 1.00 | dt: 2019.35ms | tok/sec: 8,113 | bf16_mfu: 11.96 | epoch: 1 pq: 0 rg: 4 | total time: 32.67m | eta: 2475.4m
step 00076/05000 (1.52%) | loss: 6.838064 | lrm: 1.00 | dt: 1982.75ms | tok/sec: 8,263 | bf16_mfu: 12.18 | epoch: 1 pq: 0 rg: 4 | total time: 32.70m | eta: 2439.9m
step 00077/05000 (1.54%) | loss: 6.813700 | lrm: 1.00 | dt: 1980.45ms | tok/sec: 8,272 | bf16_mfu: 12.20 | epoch: 1 pq: 0 rg: 4 | total time: 32.74m | eta: 2405.4m
step 00078/05000 (1.56%) | loss: 6.804941 | lrm: 1.00 | dt: 2011.54ms | tok/sec: 8,144 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 4 | total time: 32.77m | eta: 2372.0m
step 00079/05000 (1.58%) | loss: 6.805015 | lrm: 1.00 | dt: 1980.42ms | tok/sec: 8,273 | bf16_mfu: 12.20 | epoch: 1 pq: 0 rg: 4 | total time: 32.80m | eta: 2339.5m
step 00080/05000 (1.60%) | loss: 6.800858 | lrm: 1.00 | dt: 1983.65ms | tok/sec: 8,259 | bf16_mfu: 12.18 | epoch: 1 pq: 0 rg: 4 | total time: 32.84m | eta: 2307.9m
step 00081/05000 (1.62%) | loss: 6.803361 | lrm: 1.00 | dt: 2011.88ms | tok/sec: 8,143 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 4 | total time: 32.87m | eta: 2277.3m
step 00082/05000 (1.64%) | loss: 6.797835 | lrm: 1.00 | dt: 1987.29ms | tok/sec: 8,244 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 4 | total time: 32.90m | eta: 2247.5m
step 00083/05000 (1.66%) | loss: 6.780545 | lrm: 1.00 | dt: 2022.71ms | tok/sec: 8,100 | bf16_mfu: 11.94 | epoch: 1 pq: 0 rg: 5 | total time: 32.94m | eta: 2218.5m
step 00084/05000 (1.68%) | loss: 6.756941 | lrm: 1.00 | dt: 1988.51ms | tok/sec: 8,239 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 5 | total time: 32.97m | eta: 2190.3m
step 00085/05000 (1.70%) | loss: 6.743332 | lrm: 1.00 | dt: 1987.34ms | tok/sec: 8,244 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 5 | total time: 33.00m | eta: 2162.8m
step 00086/05000 (1.72%) | loss: 6.720465 | lrm: 1.00 | dt: 443386.31ms | tok/sec: 36 | bf16_mfu: 0.05 | epoch: 1 pq: 0 rg: 5 | total time: 40.39m | eta: 2611.7m
step 00087/05000 (1.74%) | loss: 6.708947 | lrm: 1.00 | dt: 2410.40ms | tok/sec: 6,797 | bf16_mfu: 10.02 | epoch: 1 pq: 0 rg: 5 | total time: 40.43m | eta: 2579.8m
step 00088/05000 (1.76%) | loss: 6.688838 | lrm: 1.00 | dt: 2159.77ms | tok/sec: 7,585 | bf16_mfu: 11.19 | epoch: 1 pq: 0 rg: 5 | total time: 40.47m | eta: 2548.5m
step 00089/05000 (1.78%) | loss: 6.672294 | lrm: 1.00 | dt: 2136.69ms | tok/sec: 7,667 | bf16_mfu: 11.31 | epoch: 1 pq: 0 rg: 5 | total time: 40.50m | eta: 2517.9m
step 00090/05000 (1.80%) | loss: 6.679676 | lrm: 1.00 | dt: 2048.91ms | tok/sec: 7,996 | bf16_mfu: 11.79 | epoch: 1 pq: 0 rg: 5 | total time: 40.54m | eta: 2488.1m
step 00091/05000 (1.82%) | loss: 6.681839 | lrm: 1.00 | dt: 2034.90ms | tok/sec: 8,051 | bf16_mfu: 11.87 | epoch: 1 pq: 0 rg: 5 | total time: 40.57m | eta: 2458.9m

```
