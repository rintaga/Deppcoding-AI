"""Persiapan & pemuatan dataset token (hemat RAM lewat memmap)."""
from __future__ import annotations

import json
from array import array
from pathlib import Path

import numpy as np
import torch

from .tokenizer import ByteBPETokenizer, EOS_ID


def _dtype_for(vocab_size: int):
    return ("H", np.uint16) if vocab_size <= 65535 else ("I", np.uint32)


def tokenize_files(inputs, tokenizer: ByteBPETokenizer, out_path, chunk_bytes: int = 1 << 22) -> int:
    """Tokenisasi file teks (UTF-8) -> file biner token + meta. Mengembalikan jumlah token."""
    code, np_dtype = _dtype_for(tokenizer.vocab_size)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "wb") as out:
        for src in inputs:
            buf, size = [], 0
            with open(src, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    buf.append(line)
                    size += len(line)
                    if size >= chunk_bytes:
                        a = array(code, tokenizer.encode("".join(buf)))
                        a.tofile(out)
                        total += len(a)
                        buf, size = [], 0
            tail = tokenizer.encode("".join(buf)) + [EOS_ID]  # EOS di akhir tiap file
            a = array(code, tail)
            a.tofile(out)
            total += len(a)
    out_path.with_suffix(".meta.json").write_text(
        json.dumps({"dtype": np.dtype(np_dtype).name, "tokens": total, "vocab_size": tokenizer.vocab_size}),
        encoding="utf-8")
    return total


def load_token_array(path, tokenizer_path=None) -> np.ndarray:
    """Muat .bin (memmap) atau .txt (ditokenisasi di memori; hanya untuk data kecil)."""
    path = Path(path)
    if path.suffix == ".bin":
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return np.memmap(path, dtype=np.dtype(meta["dtype"]), mode="r")
    if tokenizer_path is None:
        raise ValueError("tokenizer_path wajib untuk data .txt")
    tok = ByteBPETokenizer.load(tokenizer_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return np.asarray(tok.encode(text) + [EOS_ID], dtype=np.int64)


class BlockSampler:
    """Mengambil potongan acak (seq_len+1 token) dari array token; split train/val berurutan."""

    def __init__(self, tokens: np.ndarray, seq_len: int, val_fraction: float = 0.05):
        self.seq_len = seq_len
        n = len(tokens)
        if n < seq_len + 2:
            raise ValueError(f"Data terlalu pendek: {n} token, butuh minimal {seq_len + 2}. "
                             "Tambah data atau kecilkan --seq-len.")
        split = int(n * (1 - val_fraction))
        if val_fraction > 0 and (n - split) >= seq_len + 2 and split >= seq_len + 2:
            self.train, self.val = tokens[:split], tokens[split:]
        else:
            self.train, self.val = tokens, tokens  # data kecil: val = train (hanya untuk uji teknis)
        self.val_is_train = self.val is self.train

    def get_batch(self, split: str, batch_size: int, generator: torch.Generator):
        arr = self.train if split == "train" else self.val
        hi = len(arr) - self.seq_len - 1
        starts = torch.randint(0, hi + 1, (batch_size,), generator=generator).tolist()
        xs, ys = [], []
        for s in starts:
            chunk = torch.from_numpy(np.asarray(arr[s:s + self.seq_len + 1], dtype=np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs), torch.stack(ys)
