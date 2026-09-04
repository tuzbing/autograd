import os
import pyarrow.parquet as pq
import numpy as np

def get_dataloader_batches(parquet_dir, tokenizer, B, T):
    parquet_files = sorted([
        os.path.join(parquet_dir, f) 
        for f in os.listdir(parquet_dir) 
        if f.endswith('.parquet')
    ])
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
        
    bos_token = tokenizer.bos_id
    chunk_size = B * T + 1 
    buffer = []
    for filepath in parquet_files:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            
            for text in texts:
                tokens = [bos_token] + tokenizer.encode(text)
                buffer.extend(tokens)
                
                while len(buffer) >= chunk_size:
                    chunk = buffer[:chunk_size]
                    buffer = buffer[chunk_size - 1:]
                    
                    chunk_np = np.array(chunk, dtype=np.int32)
                    
                    x_np = chunk_np[:-1].reshape(B, T)
                    y_np = chunk_np[1:].reshape(B, T)
                    
                    yield x_np, y_np