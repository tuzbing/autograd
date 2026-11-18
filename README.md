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
[ARENA-TRAIN-ALLOC] bytes=427819264 buffer_id=37776828032
Step 00000 | Validation bpb: 3.207952
step 00000/05000 (0.00%) | loss: 10.427742 | lrm: 0.03 | dt: 2008.23ms | tok/sec: 8,158 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/05000 (0.02%) | loss: 10.385231 | lrm: 0.05 | dt: 2015.89ms | tok/sec: 8,127 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/05000 (0.04%) | loss: 10.302640 | lrm: 0.07 | dt: 1990.22ms | tok/sec: 8,232 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00003/05000 (0.06%) | loss: 10.151242 | lrm: 0.10 | dt: 2006.37ms | tok/sec: 8,165 | bf16_mfu: 12.04 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00004/05000 (0.08%) | loss: 9.893878 | lrm: 0.12 | dt: 1987.91ms | tok/sec: 8,241 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00005/05000 (0.10%) | loss: 9.579190 | lrm: 0.15 | dt: 2012.79ms | tok/sec: 8,139 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00006/05000 (0.12%) | loss: 9.363239 | lrm: 0.17 | dt: 2014.68ms | tok/sec: 8,132 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00007/05000 (0.14%) | loss: 9.244887 | lrm: 0.20 | dt: 1995.14ms | tok/sec: 8,211 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00008/05000 (0.16%) | loss: 9.098032 | lrm: 0.23 | dt: 2002.47ms | tok/sec: 8,181 | bf16_mfu: 12.06 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00009/05000 (0.18%) | loss: 9.060223 | lrm: 0.25 | dt: 1991.78ms | tok/sec: 8,225 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00010/05000 (0.20%) | loss: 8.964049 | lrm: 0.28 | dt: 2008.09ms | tok/sec: 8,158 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00011/05000 (0.22%) | loss: 8.906732 | lrm: 0.30 | dt: 1993.11ms | tok/sec: 8,220 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 1 | total time: 0.03m | eta: 165.7m
step 00012/05000 (0.24%) | loss: 8.828269 | lrm: 0.33 | dt: 2011.70ms | tok/sec: 8,144 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 1 | total time: 0.07m | eta: 166.5m
step 00013/05000 (0.26%) | loss: 8.755028 | lrm: 0.35 | dt: 1991.27ms | tok/sec: 8,227 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 1 | total time: 0.10m | eta: 166.1m
step 00014/05000 (0.28%) | loss: 8.688152 | lrm: 0.38 | dt: 2037.45ms | tok/sec: 8,041 | bf16_mfu: 11.86 | epoch: 1 pq: 0 rg: 2 | total time: 0.13m | eta: 166.9m
step 00015/05000 (0.30%) | loss: 8.614207 | lrm: 0.40 | dt: 1994.98ms | tok/sec: 8,212 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 2 | total time: 0.17m | eta: 166.6m
step 00016/05000 (0.32%) | loss: 8.530891 | lrm: 0.42 | dt: 1993.76ms | tok/sec: 8,217 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 2 | total time: 0.20m | eta: 166.4m
step 00017/05000 (0.34%) | loss: 8.467439 | lrm: 0.45 | dt: 2009.01ms | tok/sec: 8,155 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 2 | total time: 0.23m | eta: 166.5m
step 00018/05000 (0.36%) | loss: 8.421443 | lrm: 0.47 | dt: 1994.57ms | tok/sec: 8,214 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 2 | total time: 0.27m | eta: 166.3m
step 00019/05000 (0.38%) | loss: 8.378546 | lrm: 0.50 | dt: 1988.12ms | tok/sec: 8,240 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 2 | total time: 0.30m | eta: 166.2m
step 00020/05000 (0.40%) | loss: 8.342298 | lrm: 0.53 | dt: 2007.31ms | tok/sec: 8,162 | bf16_mfu: 12.04 | epoch: 1 pq: 0 rg: 2 | total time: 0.33m | eta: 166.2m
step 00021/05000 (0.42%) | loss: 8.309158 | lrm: 0.55 | dt: 1992.58ms | tok/sec: 8,222 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 2 | total time: 0.37m | eta: 166.1m
step 00022/05000 (0.44%) | loss: 8.268324 | lrm: 0.57 | dt: 1992.29ms | tok/sec: 8,223 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 2 | total time: 0.40m | eta: 166.0m
step 00023/05000 (0.46%) | loss: 8.214371 | lrm: 0.60 | dt: 2008.64ms | tok/sec: 8,156 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 2 | total time: 0.43m | eta: 166.0m
step 00024/05000 (0.48%) | loss: 8.165193 | lrm: 0.62 | dt: 1987.33ms | tok/sec: 8,244 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 2 | total time: 0.47m | eta: 165.9m
step 00025/05000 (0.50%) | loss: 8.109915 | lrm: 0.65 | dt: 1988.16ms | tok/sec: 8,240 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 2 | total time: 0.50m | eta: 165.8m
step 00026/05000 (0.52%) | loss: 8.063681 | lrm: 0.68 | dt: 2013.67ms | tok/sec: 8,136 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.53m | eta: 165.8m
step 00027/05000 (0.54%) | loss: 8.003046 | lrm: 0.70 | dt: 1991.52ms | tok/sec: 8,226 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 2 | total time: 0.57m | eta: 165.7m
step 00028/05000 (0.56%) | loss: 7.956708 | lrm: 0.72 | dt: 2012.38ms | tok/sec: 8,141 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 2 | total time: 0.60m | eta: 165.8m
step 00029/05000 (0.58%) | loss: 7.906694 | lrm: 0.75 | dt: 1990.61ms | tok/sec: 8,230 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 2 | total time: 0.63m | eta: 165.7m
step 00030/05000 (0.60%) | loss: 7.854149 | lrm: 0.78 | dt: 1991.91ms | tok/sec: 8,225 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 2 | total time: 0.67m | eta: 165.6m
step 00031/05000 (0.62%) | loss: 7.800650 | lrm: 0.80 | dt: 2013.95ms | tok/sec: 8,135 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.70m | eta: 165.7m
step 00032/05000 (0.64%) | loss: 7.749618 | lrm: 0.82 | dt: 1987.66ms | tok/sec: 8,242 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 2 | total time: 0.73m | eta: 165.6m
step 00033/05000 (0.66%) | loss: 7.730542 | lrm: 0.85 | dt: 1982.32ms | tok/sec: 8,265 | bf16_mfu: 12.19 | epoch: 1 pq: 0 rg: 2 | total time: 0.77m | eta: 165.5m
step 00034/05000 (0.68%) | loss: 7.759579 | lrm: 0.88 | dt: 2006.15ms | tok/sec: 8,166 | bf16_mfu: 12.04 | epoch: 1 pq: 0 rg: 2 | total time: 0.80m | eta: 165.5m
step 00035/05000 (0.70%) | loss: 7.735326 | lrm: 0.90 | dt: 1992.87ms | tok/sec: 8,221 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 2 | total time: 0.83m | eta: 165.4m
step 00036/05000 (0.72%) | loss: 7.719885 | lrm: 0.93 | dt: 1986.74ms | tok/sec: 8,246 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 2 | total time: 0.87m | eta: 165.3m
step 00037/05000 (0.74%) | loss: 7.696292 | lrm: 0.95 | dt: 2009.77ms | tok/sec: 8,152 | bf16_mfu: 12.02 | epoch: 1 pq: 0 rg: 3 | total time: 0.90m | eta: 165.3m
step 00038/05000 (0.76%) | loss: 7.648122 | lrm: 0.97 | dt: 1991.94ms | tok/sec: 8,225 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 3 | total time: 0.93m | eta: 165.3m
step 00039/05000 (0.78%) | loss: 7.604983 | lrm: 1.00 | dt: 1991.78ms | tok/sec: 8,225 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 3 | total time: 0.97m | eta: 165.2m
step 00040/05000 (0.80%) | loss: 7.565965 | lrm: 1.00 | dt: 2012.79ms | tok/sec: 8,139 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 3 | total time: 1.00m | eta: 165.2m
step 00041/05000 (0.82%) | loss: 7.526966 | lrm: 1.00 | dt: 1989.47ms | tok/sec: 8,235 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.03m | eta: 165.2m
step 00042/05000 (0.84%) | loss: 7.510432 | lrm: 1.00 | dt: 1987.88ms | tok/sec: 8,241 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 3 | total time: 1.07m | eta: 165.1m
step 00043/05000 (0.86%) | loss: 7.469440 | lrm: 1.00 | dt: 2014.42ms | tok/sec: 8,133 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 3 | total time: 1.10m | eta: 165.1m
step 00044/05000 (0.88%) | loss: 7.434224 | lrm: 1.00 | dt: 2001.00ms | tok/sec: 8,187 | bf16_mfu: 12.07 | epoch: 1 pq: 0 rg: 3 | total time: 1.13m | eta: 165.1m
step 00045/05000 (0.90%) | loss: 7.400415 | lrm: 1.00 | dt: 2022.10ms | tok/sec: 8,102 | bf16_mfu: 11.95 | epoch: 1 pq: 0 rg: 3 | total time: 1.17m | eta: 165.1m
step 00046/05000 (0.92%) | loss: 7.368519 | lrm: 1.00 | dt: 1999.90ms | tok/sec: 8,192 | bf16_mfu: 12.08 | epoch: 1 pq: 0 rg: 3 | total time: 1.20m | eta: 165.1m
step 00047/05000 (0.94%) | loss: 7.337160 | lrm: 1.00 | dt: 1991.31ms | tok/sec: 8,227 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 3 | total time: 1.23m | eta: 165.0m
step 00048/05000 (0.96%) | loss: 7.294709 | lrm: 1.00 | dt: 2008.60ms | tok/sec: 8,156 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 3 | total time: 1.27m | eta: 165.0m
step 00049/05000 (0.98%) | loss: 7.290203 | lrm: 1.00 | dt: 1989.91ms | tok/sec: 8,233 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.30m | eta: 165.0m
step 00050/05000 (1.00%) | loss: 7.304641 | lrm: 1.00 | dt: 1987.68ms | tok/sec: 8,242 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 3 | total time: 1.33m | eta: 164.9m
step 00051/05000 (1.02%) | loss: 7.299046 | lrm: 1.00 | dt: 2013.44ms | tok/sec: 8,137 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 3 | total time: 1.37m | eta: 164.9m
step 00052/05000 (1.04%) | loss: 7.279532 | lrm: 1.00 | dt: 1990.39ms | tok/sec: 8,231 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.40m | eta: 164.9m
step 00053/05000 (1.06%) | loss: 7.276299 | lrm: 1.00 | dt: 1988.44ms | tok/sec: 8,239 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 3 | total time: 1.43m | eta: 164.8m
step 00054/05000 (1.08%) | loss: 7.239638 | lrm: 1.00 | dt: 2015.98ms | tok/sec: 8,127 | bf16_mfu: 11.98 | epoch: 1 pq: 0 rg: 3 | total time: 1.47m | eta: 164.8m
step 00055/05000 (1.10%) | loss: 7.218638 | lrm: 1.00 | dt: 1987.48ms | tok/sec: 8,243 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 3 | total time: 1.50m | eta: 164.7m
step 00056/05000 (1.12%) | loss: 7.204387 | lrm: 1.00 | dt: 1989.74ms | tok/sec: 8,234 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.53m | eta: 164.7m
step 00057/05000 (1.14%) | loss: 7.168903 | lrm: 1.00 | dt: 2013.64ms | tok/sec: 8,136 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 3 | total time: 1.57m | eta: 164.7m
step 00058/05000 (1.16%) | loss: 7.138362 | lrm: 1.00 | dt: 1994.48ms | tok/sec: 8,214 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 3 | total time: 1.60m | eta: 164.7m
step 00059/05000 (1.18%) | loss: 7.109035 | lrm: 1.00 | dt: 1990.60ms | tok/sec: 8,230 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 3 | total time: 1.63m | eta: 164.6m
step 00060/05000 (1.20%) | loss: 7.081584 | lrm: 1.00 | dt: 2012.59ms | tok/sec: 8,140 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 4 | total time: 1.67m | eta: 164.6m
step 00061/05000 (1.22%) | loss: 7.072671 | lrm: 1.00 | dt: 1993.33ms | tok/sec: 8,219 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 4 | total time: 1.70m | eta: 164.6m
step 00062/05000 (1.24%) | loss: 7.070149 | lrm: 1.00 | dt: 1989.89ms | tok/sec: 8,233 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 1.73m | eta: 164.5m
step 00063/05000 (1.26%) | loss: 7.051643 | lrm: 1.00 | dt: 2018.07ms | tok/sec: 8,118 | bf16_mfu: 11.97 | epoch: 1 pq: 0 rg: 4 | total time: 1.77m | eta: 164.5m
step 00064/05000 (1.28%) | loss: 7.044447 | lrm: 1.00 | dt: 1993.08ms | tok/sec: 8,220 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 4 | total time: 1.80m | eta: 164.5m
step 00065/05000 (1.30%) | loss: 7.032834 | lrm: 1.00 | dt: 1985.04ms | tok/sec: 8,253 | bf16_mfu: 12.17 | epoch: 1 pq: 0 rg: 4 | total time: 1.83m | eta: 164.4m
step 00066/05000 (1.32%) | loss: 7.014993 | lrm: 1.00 | dt: 2008.20ms | tok/sec: 8,158 | bf16_mfu: 12.03 | epoch: 1 pq: 0 rg: 4 | total time: 1.87m | eta: 164.4m
step 00067/05000 (1.34%) | loss: 6.992309 | lrm: 1.00 | dt: 1991.19ms | tok/sec: 8,228 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 4 | total time: 1.90m | eta: 164.3m
step 00068/05000 (1.36%) | loss: 6.985724 | lrm: 1.00 | dt: 1990.51ms | tok/sec: 8,231 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 1.93m | eta: 164.3m
step 00069/05000 (1.38%) | loss: 6.967371 | lrm: 1.00 | dt: 2011.13ms | tok/sec: 8,146 | bf16_mfu: 12.01 | epoch: 1 pq: 0 rg: 4 | total time: 1.97m | eta: 164.3m
step 00070/05000 (1.40%) | loss: 6.959697 | lrm: 1.00 | dt: 1990.60ms | tok/sec: 8,230 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 2.00m | eta: 164.2m
step 00071/05000 (1.42%) | loss: 6.952044 | lrm: 1.00 | dt: 1991.09ms | tok/sec: 8,228 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 4 | total time: 2.03m | eta: 164.2m
step 00072/05000 (1.44%) | loss: 6.944100 | lrm: 1.00 | dt: 2019.73ms | tok/sec: 8,111 | bf16_mfu: 11.96 | epoch: 1 pq: 0 rg: 4 | total time: 2.07m | eta: 164.2m
step 00073/05000 (1.46%) | loss: 6.919393 | lrm: 1.00 | dt: 1986.94ms | tok/sec: 8,245 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 4 | total time: 2.10m | eta: 164.1m
step 00074/05000 (1.48%) | loss: 6.891868 | lrm: 1.00 | dt: 1990.14ms | tok/sec: 8,232 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 2.13m | eta: 164.1m
step 00075/05000 (1.50%) | loss: 6.873556 | lrm: 1.00 | dt: 2025.45ms | tok/sec: 8,089 | bf16_mfu: 11.93 | epoch: 1 pq: 0 rg: 4 | total time: 2.17m | eta: 164.1m
step 00076/05000 (1.52%) | loss: 6.868666 | lrm: 1.00 | dt: 1990.33ms | tok/sec: 8,231 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 4 | total time: 2.20m | eta: 164.0m
step 00077/05000 (1.54%) | loss: 6.844133 | lrm: 1.00 | dt: 1987.18ms | tok/sec: 8,244 | bf16_mfu: 12.16 | epoch: 1 pq: 0 rg: 4 | total time: 2.23m | eta: 164.0m
step 00078/05000 (1.56%) | loss: 6.833941 | lrm: 1.00 | dt: 2014.32ms | tok/sec: 8,133 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 4 | total time: 2.27m | eta: 164.0m
step 00079/05000 (1.58%) | loss: 6.824215 | lrm: 1.00 | dt: 1996.94ms | tok/sec: 8,204 | bf16_mfu: 12.10 | epoch: 1 pq: 0 rg: 4 | total time: 2.30m | eta: 163.9m
step 00080/05000 (1.60%) | loss: 6.811880 | lrm: 1.00 | dt: 1994.19ms | tok/sec: 8,215 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 4 | total time: 2.33m | eta: 163.9m
step 00081/05000 (1.62%) | loss: 6.809145 | lrm: 1.00 | dt: 2014.62ms | tok/sec: 8,132 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 4 | total time: 2.37m | eta: 163.9m
step 00082/05000 (1.64%) | loss: 6.804279 | lrm: 1.00 | dt: 1992.81ms | tok/sec: 8,221 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 4 | total time: 2.40m | eta: 163.9m
step 00083/05000 (1.66%) | loss: 6.785130 | lrm: 1.00 | dt: 2019.51ms | tok/sec: 8,112 | bf16_mfu: 11.96 | epoch: 1 pq: 0 rg: 5 | total time: 2.43m | eta: 163.8m
step 00084/05000 (1.68%) | loss: 6.760949 | lrm: 1.00 | dt: 1994.22ms | tok/sec: 8,215 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 5 | total time: 2.47m | eta: 163.8m
step 00085/05000 (1.70%) | loss: 6.747167 | lrm: 1.00 | dt: 1990.25ms | tok/sec: 8,232 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 5 | total time: 2.50m | eta: 163.8m
step 00086/05000 (1.72%) | loss: 6.724353 | lrm: 1.00 | dt: 2021.60ms | tok/sec: 8,104 | bf16_mfu: 11.95 | epoch: 1 pq: 0 rg: 5 | total time: 2.53m | eta: 163.8m
step 00087/05000 (1.74%) | loss: 6.711683 | lrm: 1.00 | dt: 1993.69ms | tok/sec: 8,217 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 5 | total time: 2.57m | eta: 163.7m
step 00088/05000 (1.76%) | loss: 6.693410 | lrm: 1.00 | dt: 1988.60ms | tok/sec: 8,238 | bf16_mfu: 12.15 | epoch: 1 pq: 0 rg: 5 | total time: 2.60m | eta: 163.7m
step 00089/05000 (1.78%) | loss: 6.677098 | lrm: 1.00 | dt: 2009.94ms | tok/sec: 8,151 | bf16_mfu: 12.02 | epoch: 1 pq: 0 rg: 5 | total time: 2.63m | eta: 163.6m
step 00090/05000 (1.80%) | loss: 6.683031 | lrm: 1.00 | dt: 1994.88ms | tok/sec: 8,213 | bf16_mfu: 12.11 | epoch: 1 pq: 0 rg: 5 | total time: 2.67m | eta: 163.6m
step 00091/05000 (1.82%) | loss: 6.680419 | lrm: 1.00 | dt: 1998.38ms | tok/sec: 8,198 | bf16_mfu: 12.09 | epoch: 1 pq: 0 rg: 5 | total time: 2.70m | eta: 163.6m
step 00092/05000 (1.84%) | loss: 6.669632 | lrm: 1.00 | dt: 2014.31ms | tok/sec: 8,133 | bf16_mfu: 11.99 | epoch: 1 pq: 0 rg: 5 | total time: 2.73m | eta: 163.6m
step 00093/05000 (1.86%) | loss: 6.658788 | lrm: 1.00 | dt: 1993.34ms | tok/sec: 8,219 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 5 | total time: 2.77m | eta: 163.5m
step 00094/05000 (1.88%) | loss: 6.648844 | lrm: 1.00 | dt: 1991.07ms | tok/sec: 8,228 | bf16_mfu: 12.13 | epoch: 1 pq: 0 rg: 5 | total time: 2.80m | eta: 163.5m
step 00095/05000 (1.90%) | loss: 6.649586 | lrm: 1.00 | dt: 2012.94ms | tok/sec: 8,139 | bf16_mfu: 12.00 | epoch: 1 pq: 0 rg: 5 | total time: 2.83m | eta: 163.5m
step 00096/05000 (1.92%) | loss: 6.637742 | lrm: 1.00 | dt: 1992.70ms | tok/sec: 8,222 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 5 | total time: 2.87m | eta: 163.4m
step 00097/05000 (1.94%) | loss: 6.633759 | lrm: 1.00 | dt: 1989.33ms | tok/sec: 8,235 | bf16_mfu: 12.14 | epoch: 1 pq: 0 rg: 5 | total time: 2.90m | eta: 163.4m
step 00098/05000 (1.96%) | loss: 6.633233 | lrm: 1.00 | dt: 2019.82ms | tok/sec: 8,111 | bf16_mfu: 11.96 | epoch: 1 pq: 0 rg: 5 | total time: 2.93m | eta: 163.4m
step 00099/05000 (1.98%) | loss: 6.614957 | lrm: 1.00 | dt: 1992.87ms | tok/sec: 8,221 | bf16_mfu: 12.12 | epoch: 1 pq: 0 rg: 5 | total time: 2.97m | eta: 163.3m

```
