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
[ARENA-TRAIN-ALLOC] bytes=427819264 buffer_id=36836828800
Step 00000 | Validation bpb: 3.207953
step 00000/05000 (0.00%) | loss: 10.427744 | lrm: 0.03 | dt: 2083.72ms | tok/sec: 7,862 | bf16_mfu: 11.59 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/05000 (0.02%) | loss: 10.385235 | lrm: 0.05 | dt: 2089.99ms | tok/sec: 7,839 | bf16_mfu: 11.56 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/05000 (0.04%) | loss: 10.302645 | lrm: 0.07 | dt: 2057.43ms | tok/sec: 7,963 | bf16_mfu: 11.74 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00003/05000 (0.06%) | loss: 10.151248 | lrm: 0.10 | dt: 2080.88ms | tok/sec: 7,873 | bf16_mfu: 11.61 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00004/05000 (0.08%) | loss: 9.893886 | lrm: 0.12 | dt: 2054.11ms | tok/sec: 7,976 | bf16_mfu: 11.76 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00005/05000 (0.10%) | loss: 9.579196 | lrm: 0.15 | dt: 2075.10ms | tok/sec: 7,895 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00006/05000 (0.12%) | loss: 9.363244 | lrm: 0.17 | dt: 2085.77ms | tok/sec: 7,855 | bf16_mfu: 11.58 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00007/05000 (0.14%) | loss: 9.244897 | lrm: 0.20 | dt: 2060.79ms | tok/sec: 7,950 | bf16_mfu: 11.72 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00008/05000 (0.16%) | loss: 9.098042 | lrm: 0.23 | dt: 2079.64ms | tok/sec: 7,878 | bf16_mfu: 11.62 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00009/05000 (0.18%) | loss: 9.060228 | lrm: 0.25 | dt: 2056.19ms | tok/sec: 7,968 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00010/05000 (0.20%) | loss: 8.964054 | lrm: 0.28 | dt: 2078.14ms | tok/sec: 7,883 | bf16_mfu: 11.63 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00011/05000 (0.22%) | loss: 8.906734 | lrm: 0.30 | dt: 2050.57ms | tok/sec: 7,989 | bf16_mfu: 11.78 | epoch: 1 pq: 0 rg: 1 | total time: 0.03m | eta: 170.5m
step 00012/05000 (0.24%) | loss: 8.828270 | lrm: 0.33 | dt: 2079.41ms | tok/sec: 7,879 | bf16_mfu: 11.62 | epoch: 1 pq: 0 rg: 1 | total time: 0.07m | eta: 171.7m
step 00013/05000 (0.26%) | loss: 8.755034 | lrm: 0.35 | dt: 2053.12ms | tok/sec: 7,980 | bf16_mfu: 11.77 | epoch: 1 pq: 0 rg: 1 | total time: 0.10m | eta: 171.3m
step 00014/05000 (0.28%) | loss: 8.688155 | lrm: 0.38 | dt: 2096.46ms | tok/sec: 7,815 | bf16_mfu: 11.52 | epoch: 1 pq: 0 rg: 2 | total time: 0.14m | eta: 172.0m
step 00015/05000 (0.30%) | loss: 8.614210 | lrm: 0.40 | dt: 2056.37ms | tok/sec: 7,967 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 2 | total time: 0.17m | eta: 171.7m
step 00016/05000 (0.32%) | loss: 8.530892 | lrm: 0.42 | dt: 2061.63ms | tok/sec: 7,947 | bf16_mfu: 11.72 | epoch: 1 pq: 0 rg: 2 | total time: 0.21m | eta: 171.6m
step 00017/05000 (0.34%) | loss: 8.467437 | lrm: 0.45 | dt: 2075.66ms | tok/sec: 7,893 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 2 | total time: 0.24m | eta: 171.7m
step 00018/05000 (0.36%) | loss: 8.421455 | lrm: 0.47 | dt: 2054.64ms | tok/sec: 7,974 | bf16_mfu: 11.76 | epoch: 1 pq: 0 rg: 2 | total time: 0.28m | eta: 171.5m
step 00019/05000 (0.38%) | loss: 8.378560 | lrm: 0.50 | dt: 2059.57ms | tok/sec: 7,955 | bf16_mfu: 11.73 | epoch: 1 pq: 0 rg: 2 | total time: 0.31m | eta: 171.5m
step 00020/05000 (0.40%) | loss: 8.342318 | lrm: 0.53 | dt: 2075.97ms | tok/sec: 7,892 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 2 | total time: 0.34m | eta: 171.5m
step 00021/05000 (0.42%) | loss: 8.309187 | lrm: 0.55 | dt: 2060.73ms | tok/sec: 7,950 | bf16_mfu: 11.72 | epoch: 1 pq: 0 rg: 2 | total time: 0.38m | eta: 171.4m
step 00022/05000 (0.44%) | loss: 8.268356 | lrm: 0.57 | dt: 2056.68ms | tok/sec: 7,966 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 2 | total time: 0.41m | eta: 171.3m
step 00023/05000 (0.46%) | loss: 8.214404 | lrm: 0.60 | dt: 2076.09ms | tok/sec: 7,891 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 2 | total time: 0.45m | eta: 171.4m
step 00024/05000 (0.48%) | loss: 8.165230 | lrm: 0.62 | dt: 2065.42ms | tok/sec: 7,932 | bf16_mfu: 11.70 | epoch: 1 pq: 0 rg: 2 | total time: 0.48m | eta: 171.3m
step 00025/05000 (0.50%) | loss: 8.109956 | lrm: 0.65 | dt: 2049.47ms | tok/sec: 7,994 | bf16_mfu: 11.79 | epoch: 1 pq: 0 rg: 2 | total time: 0.52m | eta: 171.2m
step 00026/05000 (0.52%) | loss: 8.063723 | lrm: 0.68 | dt: 2081.88ms | tok/sec: 7,869 | bf16_mfu: 11.60 | epoch: 1 pq: 0 rg: 2 | total time: 0.55m | eta: 171.3m
step 00027/05000 (0.54%) | loss: 8.003079 | lrm: 0.70 | dt: 2053.58ms | tok/sec: 7,978 | bf16_mfu: 11.76 | epoch: 1 pq: 0 rg: 2 | total time: 0.59m | eta: 171.2m
step 00028/05000 (0.56%) | loss: 7.956742 | lrm: 0.72 | dt: 2075.84ms | tok/sec: 7,892 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 2 | total time: 0.62m | eta: 171.2m
step 00029/05000 (0.58%) | loss: 7.906719 | lrm: 0.75 | dt: 2076.61ms | tok/sec: 7,889 | bf16_mfu: 11.63 | epoch: 1 pq: 0 rg: 2 | total time: 0.65m | eta: 171.2m
step 00030/05000 (0.60%) | loss: 7.854176 | lrm: 0.78 | dt: 2056.21ms | tok/sec: 7,968 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 2 | total time: 0.69m | eta: 171.1m
step 00031/05000 (0.62%) | loss: 7.800654 | lrm: 0.80 | dt: 2075.61ms | tok/sec: 7,893 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 2 | total time: 0.72m | eta: 171.1m
step 00032/05000 (0.64%) | loss: 7.749434 | lrm: 0.82 | dt: 2074.44ms | tok/sec: 7,898 | bf16_mfu: 11.65 | epoch: 1 pq: 0 rg: 2 | total time: 0.76m | eta: 171.1m
step 00033/05000 (0.66%) | loss: 7.728799 | lrm: 0.85 | dt: 2062.01ms | tok/sec: 7,945 | bf16_mfu: 11.72 | epoch: 1 pq: 0 rg: 2 | total time: 0.79m | eta: 171.1m
step 00034/05000 (0.68%) | loss: 7.758927 | lrm: 0.88 | dt: 2075.79ms | tok/sec: 7,892 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 2 | total time: 0.83m | eta: 171.1m
step 00035/05000 (0.70%) | loss: 7.736159 | lrm: 0.90 | dt: 2059.13ms | tok/sec: 7,956 | bf16_mfu: 11.73 | epoch: 1 pq: 0 rg: 2 | total time: 0.86m | eta: 171.0m
step 00036/05000 (0.72%) | loss: 7.720699 | lrm: 0.93 | dt: 2049.46ms | tok/sec: 7,994 | bf16_mfu: 11.79 | epoch: 1 pq: 0 rg: 2 | total time: 0.90m | eta: 170.9m
step 00037/05000 (0.74%) | loss: 7.696859 | lrm: 0.95 | dt: 2086.51ms | tok/sec: 7,852 | bf16_mfu: 11.58 | epoch: 1 pq: 0 rg: 3 | total time: 0.93m | eta: 170.9m
step 00038/05000 (0.76%) | loss: 7.648587 | lrm: 0.97 | dt: 2052.79ms | tok/sec: 7,981 | bf16_mfu: 11.77 | epoch: 1 pq: 0 rg: 3 | total time: 0.96m | eta: 170.9m
step 00039/05000 (0.78%) | loss: 7.605276 | lrm: 1.00 | dt: 2055.73ms | tok/sec: 7,969 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 3 | total time: 1.00m | eta: 170.8m
step 00040/05000 (0.80%) | loss: 7.566034 | lrm: 1.00 | dt: 2079.72ms | tok/sec: 7,877 | bf16_mfu: 11.62 | epoch: 1 pq: 0 rg: 3 | total time: 1.03m | eta: 170.8m
step 00041/05000 (0.82%) | loss: 7.528033 | lrm: 1.00 | dt: 2058.14ms | tok/sec: 7,960 | bf16_mfu: 11.74 | epoch: 1 pq: 0 rg: 3 | total time: 1.07m | eta: 170.8m
step 00042/05000 (0.84%) | loss: 7.513371 | lrm: 1.00 | dt: 2057.28ms | tok/sec: 7,963 | bf16_mfu: 11.74 | epoch: 1 pq: 0 rg: 3 | total time: 1.10m | eta: 170.7m
step 00043/05000 (0.86%) | loss: 7.473093 | lrm: 1.00 | dt: 2076.95ms | tok/sec: 7,888 | bf16_mfu: 11.63 | epoch: 1 pq: 0 rg: 3 | total time: 1.14m | eta: 170.7m
step 00044/05000 (0.88%) | loss: 7.437801 | lrm: 1.00 | dt: 2053.46ms | tok/sec: 7,978 | bf16_mfu: 11.77 | epoch: 1 pq: 0 rg: 3 | total time: 1.17m | eta: 170.6m
step 00045/05000 (0.90%) | loss: 7.403961 | lrm: 1.00 | dt: 2084.71ms | tok/sec: 7,859 | bf16_mfu: 11.59 | epoch: 1 pq: 0 rg: 3 | total time: 1.21m | eta: 170.6m
step 00046/05000 (0.92%) | loss: 7.372060 | lrm: 1.00 | dt: 2062.32ms | tok/sec: 7,944 | bf16_mfu: 11.71 | epoch: 1 pq: 0 rg: 3 | total time: 1.24m | eta: 170.6m
step 00047/05000 (0.94%) | loss: 7.340916 | lrm: 1.00 | dt: 2054.89ms | tok/sec: 7,973 | bf16_mfu: 11.76 | epoch: 1 pq: 0 rg: 3 | total time: 1.27m | eta: 170.5m
step 00048/05000 (0.96%) | loss: 7.297674 | lrm: 1.00 | dt: 2074.17ms | tok/sec: 7,899 | bf16_mfu: 11.65 | epoch: 1 pq: 0 rg: 3 | total time: 1.31m | eta: 170.5m
step 00049/05000 (0.98%) | loss: 7.289561 | lrm: 1.00 | dt: 2058.25ms | tok/sec: 7,960 | bf16_mfu: 11.74 | epoch: 1 pq: 0 rg: 3 | total time: 1.34m | eta: 170.5m
step 00050/05000 (1.00%) | loss: 7.299583 | lrm: 1.00 | dt: 2061.46ms | tok/sec: 7,947 | bf16_mfu: 11.72 | epoch: 1 pq: 0 rg: 3 | total time: 1.38m | eta: 170.4m
step 00051/05000 (1.02%) | loss: 7.293160 | lrm: 1.00 | dt: 2084.44ms | tok/sec: 7,860 | bf16_mfu: 11.59 | epoch: 1 pq: 0 rg: 3 | total time: 1.41m | eta: 170.4m
step 00052/05000 (1.04%) | loss: 7.271561 | lrm: 1.00 | dt: 2054.50ms | tok/sec: 7,974 | bf16_mfu: 11.76 | epoch: 1 pq: 0 rg: 3 | total time: 1.45m | eta: 170.4m
step 00053/05000 (1.06%) | loss: 7.266301 | lrm: 1.00 | dt: 2057.99ms | tok/sec: 7,961 | bf16_mfu: 11.74 | epoch: 1 pq: 0 rg: 3 | total time: 1.48m | eta: 170.3m
step 00054/05000 (1.08%) | loss: 7.229230 | lrm: 1.00 | dt: 2077.62ms | tok/sec: 7,885 | bf16_mfu: 11.63 | epoch: 1 pq: 0 rg: 3 | total time: 1.52m | eta: 170.3m
step 00055/05000 (1.10%) | loss: 7.209833 | lrm: 1.00 | dt: 2058.23ms | tok/sec: 7,960 | bf16_mfu: 11.74 | epoch: 1 pq: 0 rg: 3 | total time: 1.55m | eta: 170.3m
step 00056/05000 (1.12%) | loss: 7.197286 | lrm: 1.00 | dt: 2055.87ms | tok/sec: 7,969 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 3 | total time: 1.58m | eta: 170.2m
step 00057/05000 (1.14%) | loss: 7.163629 | lrm: 1.00 | dt: 2083.37ms | tok/sec: 7,864 | bf16_mfu: 11.60 | epoch: 1 pq: 0 rg: 3 | total time: 1.62m | eta: 170.2m
step 00058/05000 (1.16%) | loss: 7.135170 | lrm: 1.00 | dt: 2060.20ms | tok/sec: 7,952 | bf16_mfu: 11.73 | epoch: 1 pq: 0 rg: 3 | total time: 1.65m | eta: 170.2m
step 00059/05000 (1.18%) | loss: 7.106473 | lrm: 1.00 | dt: 2058.66ms | tok/sec: 7,958 | bf16_mfu: 11.74 | epoch: 1 pq: 0 rg: 3 | total time: 1.69m | eta: 170.1m
step 00060/05000 (1.20%) | loss: 7.074997 | lrm: 1.00 | dt: 2087.15ms | tok/sec: 7,849 | bf16_mfu: 11.58 | epoch: 1 pq: 0 rg: 4 | total time: 1.72m | eta: 170.1m
step 00061/05000 (1.22%) | loss: 7.055065 | lrm: 1.00 | dt: 2064.87ms | tok/sec: 7,934 | bf16_mfu: 11.70 | epoch: 1 pq: 0 rg: 4 | total time: 1.76m | eta: 170.1m
step 00062/05000 (1.24%) | loss: 7.037053 | lrm: 1.00 | dt: 2054.11ms | tok/sec: 7,976 | bf16_mfu: 11.76 | epoch: 1 pq: 0 rg: 4 | total time: 1.79m | eta: 170.0m
step 00063/05000 (1.26%) | loss: 6.994868 | lrm: 1.00 | dt: 2083.59ms | tok/sec: 7,863 | bf16_mfu: 11.59 | epoch: 1 pq: 0 rg: 4 | total time: 1.83m | eta: 170.0m
step 00064/05000 (1.28%) | loss: 6.984246 | lrm: 1.00 | dt: 2054.16ms | tok/sec: 7,976 | bf16_mfu: 11.76 | epoch: 1 pq: 0 rg: 4 | total time: 1.86m | eta: 170.0m
step 00065/05000 (1.30%) | loss: 6.971847 | lrm: 1.00 | dt: 2061.55ms | tok/sec: 7,947 | bf16_mfu: 11.72 | epoch: 1 pq: 0 rg: 4 | total time: 1.89m | eta: 169.9m
step 00066/05000 (1.32%) | loss: 6.953176 | lrm: 1.00 | dt: 2072.65ms | tok/sec: 7,904 | bf16_mfu: 11.66 | epoch: 1 pq: 0 rg: 4 | total time: 1.93m | eta: 169.9m
step 00067/05000 (1.34%) | loss: 6.930238 | lrm: 1.00 | dt: 2056.58ms | tok/sec: 7,966 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 4 | total time: 1.96m | eta: 169.8m
step 00068/05000 (1.36%) | loss: 6.925384 | lrm: 1.00 | dt: 2053.28ms | tok/sec: 7,979 | bf16_mfu: 11.77 | epoch: 1 pq: 0 rg: 4 | total time: 2.00m | eta: 169.8m
step 00069/05000 (1.38%) | loss: 6.912203 | lrm: 1.00 | dt: 2072.09ms | tok/sec: 7,906 | bf16_mfu: 11.66 | epoch: 1 pq: 0 rg: 4 | total time: 2.03m | eta: 169.8m
step 00070/05000 (1.40%) | loss: 6.919049 | lrm: 1.00 | dt: 2055.33ms | tok/sec: 7,971 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 4 | total time: 2.07m | eta: 169.7m
step 00071/05000 (1.42%) | loss: 6.939910 | lrm: 1.00 | dt: 2051.73ms | tok/sec: 7,985 | bf16_mfu: 11.78 | epoch: 1 pq: 0 rg: 4 | total time: 2.10m | eta: 169.7m
step 00072/05000 (1.44%) | loss: 6.971927 | lrm: 1.00 | dt: 2080.07ms | tok/sec: 7,876 | bf16_mfu: 11.61 | epoch: 1 pq: 0 rg: 4 | total time: 2.13m | eta: 169.7m
step 00073/05000 (1.46%) | loss: 6.957128 | lrm: 1.00 | dt: 2052.67ms | tok/sec: 7,981 | bf16_mfu: 11.77 | epoch: 1 pq: 0 rg: 4 | total time: 2.17m | eta: 169.6m
step 00074/05000 (1.48%) | loss: 6.932785 | lrm: 1.00 | dt: 2055.81ms | tok/sec: 7,969 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 4 | total time: 2.20m | eta: 169.6m
step 00075/05000 (1.50%) | loss: 6.916195 | lrm: 1.00 | dt: 2075.85ms | tok/sec: 7,892 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 4 | total time: 2.24m | eta: 169.5m
step 00076/05000 (1.52%) | loss: 6.914593 | lrm: 1.00 | dt: 2055.74ms | tok/sec: 7,969 | bf16_mfu: 11.75 | epoch: 1 pq: 0 rg: 4 | total time: 2.27m | eta: 169.5m
step 00077/05000 (1.54%) | loss: 6.895743 | lrm: 1.00 | dt: 2052.23ms | tok/sec: 7,983 | bf16_mfu: 11.77 | epoch: 1 pq: 0 rg: 4 | total time: 2.31m | eta: 169.4m
step 00078/05000 (1.56%) | loss: 6.885815 | lrm: 1.00 | dt: 2082.53ms | tok/sec: 7,867 | bf16_mfu: 11.60 | epoch: 1 pq: 0 rg: 4 | total time: 2.34m | eta: 169.4m
step 00079/05000 (1.58%) | loss: 6.875355 | lrm: 1.00 | dt: 2057.25ms | tok/sec: 7,964 | bf16_mfu: 11.74 | epoch: 1 pq: 0 rg: 4 | total time: 2.37m | eta: 169.4m
step 00080/05000 (1.60%) | loss: 6.861540 | lrm: 1.00 | dt: 2054.55ms | tok/sec: 7,974 | bf16_mfu: 11.76 | epoch: 1 pq: 0 rg: 4 | total time: 2.41m | eta: 169.3m
step 00081/05000 (1.62%) | loss: 6.855731 | lrm: 1.00 | dt: 2075.73ms | tok/sec: 7,893 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 4 | total time: 2.44m | eta: 169.3m
step 00082/05000 (1.64%) | loss: 6.850904 | lrm: 1.00 | dt: 2059.95ms | tok/sec: 7,953 | bf16_mfu: 11.73 | epoch: 1 pq: 0 rg: 4 | total time: 2.48m | eta: 169.3m
step 00083/05000 (1.66%) | loss: 6.831453 | lrm: 1.00 | dt: 2083.20ms | tok/sec: 7,864 | bf16_mfu: 11.60 | epoch: 1 pq: 0 rg: 5 | total time: 2.51m | eta: 169.3m
step 00084/05000 (1.68%) | loss: 6.807134 | lrm: 1.00 | dt: 2065.76ms | tok/sec: 7,931 | bf16_mfu: 11.70 | epoch: 1 pq: 0 rg: 5 | total time: 2.55m | eta: 169.2m
step 00085/05000 (1.70%) | loss: 6.791183 | lrm: 1.00 | dt: 2075.24ms | tok/sec: 7,894 | bf16_mfu: 11.64 | epoch: 1 pq: 0 rg: 5 | total time: 2.58m | eta: 169.2m
step 00086/05000 (1.72%) | loss: 6.772487 | lrm: 1.00 | dt: 2080.20ms | tok/sec: 7,876 | bf16_mfu: 11.61 | epoch: 1 pq: 0 rg: 5 | total time: 2.62m | eta: 169.2m
step 00087/05000 (1.74%) | loss: 6.763875 | lrm: 1.00 | dt: 2060.23ms | tok/sec: 7,952 | bf16_mfu: 11.73 | epoch: 1 pq: 0 rg: 5 | total time: 2.65m | eta: 169.1m
step 00088/05000 (1.76%) | loss: 6.746738 | lrm: 1.00 | dt: 2081.98ms | tok/sec: 7,869 | bf16_mfu: 11.60 | epoch: 1 pq: 0 rg: 5 | total time: 2.69m | eta: 169.1m
step 00089/05000 (1.78%) | loss: 6.726195 | lrm: 1.00 | dt: 2072.84ms | tok/sec: 7,904 | bf16_mfu: 11.66 | epoch: 1 pq: 0 rg: 5 | total time: 2.72m | eta: 169.1m

```
