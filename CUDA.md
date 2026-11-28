# CUDA 执行指南

## 1. 环境准备

安装 Python 编译依赖：

```bash
python3 -m pip install pybind11
```

## 2. 编译

执行编译脚本：

```bash
bash test.sh
```

## 3. 分词器训练

安装分词器依赖：

```bash
python3 -m pip install rustbpe
```

**训练 BPE Tokenizer：**

```bash
PYTHONPATH=./nanochat python -c '
import glob, pyarrow.parquet as pq
from nanochat.tokenizer import RustBPETokenizer

def text_iter(n=0):
    for f in sorted(glob.glob("./dataset/p19/**/train*.parquet", recursive=True)):
        for b in pq.ParquetFile(f).iter_batches(batch_size=16, columns=["text"]):
            for doc in b.column(0).to_pylist():
                if doc:
                    s = doc[:10000]
                    n += len(s)
                    yield s
                    if n >= 2_000_000_000:
                        return

RustBPETokenizer.train_from_iterator(text_iter(), 32768).save("./tokenizer")
'
```

**参数说明：**

- 数据集目录：`./dataset/p19`
- 输入文件：`train*.parquet`
- 文本字段：`text`
- 单条文本最大长度：`10000`
- 训练文本目标规模：`2,000,000,000` 字符
- Tokenizer 词表大小：`32768`
- 输出目录：`./tokenizer`

## 4. CUDA Kernel 调试

执行 CUDA Kernel 测试：

```bash
python3 debug/check_cuda_kernel.py
```

## 5. GPT-2 模型训练

执行训练命令：

```bash
python -B train_gpt2.py \
  --device-type cuda \
  --use-fp16-gemm 0 \
  --data-dir ../dataset/p19 \
  --tokenizer-dir ../tokenizer \
  --depth 12 \
  --aspect-ratio 64 \
  --head-dim 64 \
  --max-seq-len 1024 \
  --device-batch-size 8 \
  --total-batch-size 32768 \
  --optimizer-state mini \
  --grad-dtype bf16 \
  --matrix-lr 0.0024 \
  --warmup-steps 200 \
  --max-norm 1.0 \
  --eval-every 200 \
  --eval-tokens 524288 \
  --num-iterations 10000 \
  --save-every 1000 \
  --model-out '../checkpoint/gpt2_d12/model_{step:06d}.gguf'
```

### 5.1 模型配置

| 参数 | 配置 | 说明 |
|---|---:|---|
| `--depth` | 12 | Transformer 层数 |
| `--aspect-ratio` | 64 | 模型宽度与深度比例参数 |
| `--head-dim` | 64 | Attention Head 维度 |
| `--max-seq-len` | 1024 | 最大序列长度 |
| `--device-batch-size` | 8 | 单设备 Batch Size |
| `--total-batch-size` | 32768 | 总 Batch Size |

### 5.2 优化器与精度配置

| 参数 | 配置 |
|---|---|
| `--optimizer-state` | `mini` |
| `--grad-dtype` | `bf16` |
| `--use-fp16-gemm` | `0` |
| `--matrix-lr` | `0.0024` |
| `--max-norm` | `1.0` |

### 5.3 训练与评估配置

| 参数 | 配置 |
|---|---:|
| `--warmup-steps` | 200 |
| `--eval-every` | 200 |
| `--eval-tokens` | 524288 |
| `--num-iterations` | 10000 |
| `--save-every` | 1000 |

### 5.4 模型输出

输出目录：

```text
../checkpoint/gpt2_d12/
```

模型命名格式：

```text
model_{step:06d}.gguf
```

每隔 1000 个训练迭代保存一次模型。