
export NANOCHAT_BASE_DIR="$HOME/Documents/code/autograd/sft_data"
export HF_HOME="$NANOCHAT_BASE_DIR/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

/Users/a58/Documents/code/nanochat/.venv/bin/python -B - <<'PY'
import os
import shutil
import urllib.request
from pathlib import Path

base = Path(os.environ["NANOCHAT_BASE_DIR"])
tokenizer_src = Path.home() / ".cache/nanochat/tokenizer/tokenizer.pkl"
tokenizer_dst = base / "tokenizer/tokenizer.pkl"

if not tokenizer_src.is_file():
    raise FileNotFoundError(f"缺少预训练 tokenizer: {tokenizer_src}")

tokenizer_dst.parent.mkdir(parents=True, exist_ok=True)
if tokenizer_dst.exists():
    if tokenizer_dst.read_bytes() != tokenizer_src.read_bytes():
        raise RuntimeError("目标目录已有不同 tokenizer，请检查")
else:
    shutil.copy2(tokenizer_src, tokenizer_dst)

urls = {
    "identity_conversations.jsonl":
        "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl",
    "words_alpha.txt":
        "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt",
}
for name, url in urls.items():
    target = base / name
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary.replace(target)
    print("已准备:", target)

from datasets import load_dataset

specs = [
    ("HuggingFaceTB/smol-smoltalk", None, ("train", "test")),
    ("cais/mmlu", "all", ("auxiliary_train", "test")),
    ("openai/gsm8k", "main", ("train", "test")),
]
for repo, config, splits in specs:
    for split in splits:
        args = (repo,) if config is None else (repo, config)
        dataset = load_dataset(*args, split=split)
        print(repo, config, split, len(dataset))

print("SFT 数据准备完成:", base)
PY