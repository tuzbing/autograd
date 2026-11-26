"""
Streaming multi-format text dataset loader for autograd pretraining.
Supports:
  1. Hugging Face split-prefixed Parquet (e.g. ~/Documents/datasets/p19/data/train-*.parquet)
  2. Unsplit Parquet shards (e.g. ~/.cache/nanochat/base_data_climbmix/shard_*.parquet)
  3. Local .jsonl / .ndjson and .txt files
  4. Weighted multi-dataset mixing via "path1:0.3,path2:0.7"
"""

import json
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

TEXT_COLUMN_CANDIDATES = ("text", "content", "markdown", "body", "code")
SUPPORTED_EXTS = (".parquet", ".jsonl", ".ndjson", ".txt")


def _discover_files(root_path):
    root = Path(root_path).expanduser().resolve()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and not p.name.endswith((".tmp", ".part"))
        and not p.name.startswith(".")
    )
    if not files:
        raise RuntimeError(f"No supported dataset files ({SUPPORTED_EXTS}) found under {root}")
    return files


def resolve_split_files(root_path, split):
    """
    Prefer explicit filename split prefixes (train / validation|val|dev / test).
    Fallback to nanochat convention (all-but-last = train, last = val) for unsplit shards.
    """
    files = _discover_files(root_path)

    def match_prefix(prefixes):
        matched = []
        for p in files:
            stem = p.stem.lower()
            # Match both "train-00000..." and "pg19-train-00000..."
            if any(
                stem == pref
                or stem.startswith((pref + "-", pref + "_", pref + "."))
                or f"-{pref}-" in stem
                or f"_{pref}_" in stem
                for pref in prefixes
            ):
                matched.append(p)
        return matched

    train_files = match_prefix(("train",))
    val_files = match_prefix(("validation", "val", "valid", "dev"))
    test_files = match_prefix(("test",))

    has_explicit_splits = bool(train_files or val_files or test_files)
    if has_explicit_splits:
        if split == "train":
            if not train_files:
                raise RuntimeError(f"No train split files found in {root_path}")
            return train_files
        if split == "val":
            chosen = val_files or test_files
            if not chosen:
                raise RuntimeError(f"No validation/test split files found in {root_path}")
            return chosen
        if split == "test":
            if not test_files:
                raise RuntimeError(f"No test split files found in {root_path}")
            return test_files
        raise ValueError(f"Invalid split: {split!r}")

    # Fallback for unsplit shard directories like base_data_climbmix
    if len(files) < 2:
        return files
    if split == "train":
        return files[:-1]
    if split in ("val", "test"):
        return files[-1:]
    raise ValueError(f"Invalid split: {split!r}")


def _split_long_text(text, max_chars=32768):
    """Split book-length strings on paragraph/line boundaries to bound per-call RAM."""
    n = len(text)
    if n <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            cut = text.rfind("\n\n", start + max_chars // 2, end)
            if cut == -1:
                cut = text.rfind("\n", start + max_chars // 2, end)
            if cut == -1:
                cut = text.rfind(" ", start + max_chars // 2, end)
            if cut != -1 and cut > start:
                end = cut + 1
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _pick_parquet_text_col(schema_names, preferred_col=None):
    if preferred_col and preferred_col in schema_names:
        return preferred_col
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in schema_names:
            return candidate
    raise KeyError(
        f"Could not find text column in Parquet schema {schema_names}; "
        f"pass --text-column explicitly."
    )


def iter_document_pieces(
    data_dir,
    split,
    text_column=None,
    parquet_batch_size=16,
    max_chars_per_piece=32768,
    max_epochs=None,
    start_state=None,
):
    """
    Yields batches of (text_piece, is_doc_start) tuples plus progress state.
    Uses pyarrow iter_batches() so giant PG-19 row groups never blow up RAM.
    """
    files = resolve_split_files(data_dir, split)
    num_files = len(files)
    epoch = int(start_state["epoch"]) if start_state else 1
    start_pq = int(start_state["pq_idx"]) if start_state else 0
    start_rg = int(start_state["rg_idx"]) if start_state else 0

    def calc_progress(ep, f_idx, f_frac):
        ep_prog = (f_idx + f_frac) / num_files
        return ((ep - 1) + ep_prog) / max_epochs if max_epochs else ep_prog

    while max_epochs is None or epoch <= max_epochs:
        for file_idx, filepath in enumerate(files):
            if start_state and epoch == start_state["epoch"] and file_idx < start_pq:
                continue
            suffix = filepath.suffix.lower()
            batch_idx = 0

            if suffix == ".parquet":
                pf = pq.ParquetFile(filepath)
                total_rows = max(1, pf.metadata.num_rows)
                rows_done = 0
                col = _pick_parquet_text_col(pf.schema.names, text_column)
                for record_batch in pf.iter_batches(batch_size=parquet_batch_size, columns=[col],):
                    rows_done += record_batch.num_rows
                    if (start_state and epoch == start_state["epoch"] and file_idx == start_pq and batch_idx <= start_rg):
                        batch_idx += 1
                        continue 
                    pieces = []
                    for text in record_batch.column(0).to_pylist():
                        if not text:
                            continue
                        sub_texts = _split_long_text(text, max_chars=max_chars_per_piece)
                        for idx, sub in enumerate(sub_texts):
                            pieces.append((sub, idx == 0))
                    if pieces:
                        yield pieces, {"epoch": epoch, "pq_idx": file_idx, "rg_idx": batch_idx, 
                                       "progress": calc_progress(epoch, file_idx, rows_done / total_rows)}
                    batch_idx += 1

            elif suffix in (".jsonl", ".ndjson"):
                pieces = []
                with filepath.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        col = text_column if (text_column and text_column in obj) else next(
                            (c for c in TEXT_COLUMN_CANDIDATES if c in obj), None
                        )
                        if col is None or not obj[col]:
                            continue
                        sub_texts = _split_long_text(str(obj[col]), max_chars=max_chars_per_piece)
                        for idx, sub in enumerate(sub_texts):
                            pieces.append((sub, idx == 0))
                        if len(pieces) >= parquet_batch_size:
                            yield pieces, {"epoch": epoch, "pq_idx": file_idx, "rg_idx": batch_idx}
                            pieces = []
                            batch_idx += 1
                if pieces:
                    yield pieces, {"epoch": epoch, "pq_idx": file_idx, "rg_idx": batch_idx}

            elif suffix == ".txt":
                text = filepath.read_text(encoding="utf-8", errors="ignore")
                sub_texts = _split_long_text(text, max_chars=max_chars_per_piece)
                pieces = [(sub, idx == 0) for idx, sub in enumerate(sub_texts)]
                for start in range(0, len(pieces), parquet_batch_size):
                    yield pieces[start:start + parquet_batch_size], {
                        "epoch": epoch,
                        "pq_idx": file_idx,
                        "rg_idx": batch_idx,
                    }
                    batch_idx += 1

        epoch += 1


def parse_data_specs(spec_str):
    """
    Parse "path1:weight1,path2:weight2" or single "path1".
    """
    specs = []
    for raw in spec_str.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            path_part, weight_part = item.rsplit(":", 1)
            try:
                weight = float(weight_part)
                specs.append((path_part.strip(), weight))
                continue
            except ValueError:
                pass
        specs.append((item, 1.0))
    if not specs:
        raise ValueError("Empty --data-dir specification")
    total_w = sum(w for _, w in specs)
    return [(p, w / total_w) for p, w in specs]


def universal_bestfit_dataloader(
    data_spec,
    tokenizer,
    batch_size,
    sequence_len,
    split,
    text_column=None,
    tokenizer_threads=4,
    parquet_batch_size=16,
    buffer_size=256,
    seed=42,
    max_epochs=None,
    start_state=None,
):
    """
    Lossless long-document chunking + BOS-aligned best-fit packing:
      - Short docs (<= sequence_len + 1): identical to nanochat best-fit packing.
      - Long docs / books (PG-19): full-row windows are emitted losslessly without
        dropping 99% of the book, and leftover tails enter doc_buffer for best-fit packing.
    """
    row_capacity = sequence_len + 1
    parsed_specs = parse_data_specs(data_spec)
    sources = [
        iter_document_pieces(
            data_dir=path,
            split=split,
            text_column=text_column,
            parquet_batch_size=parquet_batch_size,
            max_epochs=max_epochs,
            start_state=start_state if len(parsed_specs) == 1 else None,
        )
        for path, _ in parsed_specs
    ]
    weights = np.array([w for _, w in parsed_specs], dtype=np.float64)
    rng = np.random.default_rng(seed if split == "train" else seed + 1)

    full_row_queue = []
    doc_buffer = []
    eof = False
    state = {"epoch": 1, "pq_idx": 0, "rg_idx": 0}
    row_buffer = np.empty((batch_size, row_capacity), dtype=np.int32)

    def refill():
        nonlocal state, eof
        if eof:
            return
        src_idx = 0 if len(sources) == 1 else int(rng.choice(len(sources), p=weights))
        try:
            pieces, state = next(sources[src_idx])
        except StopIteration:
            eof = True
            return
        texts = [t for t, _ in pieces]
        is_starts = [s for _, s in pieces]
        encoded_list = tokenizer.encode_batch(texts, num_threads=tokenizer_threads)

        for tokens, is_start in zip(encoded_list, is_starts):
            if not tokens:
                continue
            seq = ([tokenizer.bos_id] + tokens) if is_start else tokens
            # Slice full row_capacity windows losslessly for long texts (e.g. PG-19 books)
            pos = 0
            n = len(seq)
            while n - pos >= row_capacity:
                full_row_queue.append(seq[pos:pos + row_capacity])
                pos += row_capacity
            if pos < n:
                doc_buffer.append(seq[pos:])

    def pop_full_row():
        if split != "train":
            return full_row_queue.pop(0)
        idx = int(rng.integers(len(full_row_queue)))
        row = full_row_queue[idx]
        full_row_queue[idx] = full_row_queue[-1]
        full_row_queue.pop()
        return row

    while True:
        for row_idx in range(batch_size):
            while not eof and len(full_row_queue) < buffer_size and len(doc_buffer) < buffer_size:
                refill()
            # Fast path: complete contiguous windows from long books
            if full_row_queue and (eof or len(doc_buffer) < buffer_size):
                row_buffer[row_idx, :] = pop_full_row()
                continue

            position = 0
            while position < row_capacity:
                while not eof and len(doc_buffer) < buffer_size and not full_row_queue:
                    refill()

                if position == 0 and full_row_queue:
                    row_buffer[row_idx, :] = pop_full_row()
                    position = row_capacity
                    break

                if not doc_buffer and full_row_queue:
                    doc_buffer.append(pop_full_row())

                if not doc_buffer:
                    return

                remaining = row_capacity - position
                best_idx = -1
                best_len = 0
                for idx, tokens in enumerate(doc_buffer):
                    tlen = len(tokens)
                    if best_len < tlen <= remaining:
                        best_idx = idx
                        best_len = tlen

                if best_idx >= 0:
                    tokens = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, position:position + best_len] = tokens
                    position += best_len
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    tokens = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, position:position + remaining] = tokens[:remaining]
                    tail = tokens[remaining:]
                    # Preserve non-trivial remainder instead of discarding it
                    if len(tail) >= 32:
                        doc_buffer.append(tail)
                    position += remaining

        x = row_buffer[:, :-1].copy()
        y = row_buffer[:, 1:].copy()
        yield x, y, state.copy()
