import os
import time
from tokenizer import Tokenizer
from dataloader import get_dataloader_batches

B, T = 16, 2048
tokenizer = Tokenizer()
data_dir = os.path.expanduser("~/.cache/nanochat/base_data_climbmix")

print(f"Checking dataloader at {data_dir}...")
t0 = time.time()
try:
    loader = get_dataloader_batches(data_dir, tokenizer, B, T)
    x, y = next(loader)
    print(f"Got first batch in {time.time() - t0:.2f}s. Shape: {x.shape}")
except StopIteration:
    print("Dataloader is empty.")
except Exception as e:
    print(f"Error: {e}")
