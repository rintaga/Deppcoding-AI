"""Konfigurasi model + rumus hitung parameter (tanpa perlu PyTorch)."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class ModelConfig:
    model_name: str = "custom"
    vocab_size: int = 32768
    hidden_size: int = 2048
    num_layers: int = 32
    num_heads: int = 16
    num_kv_heads: int = 4
    intermediate_size: int = 5632
    context_length: int = 2048
    window_size: int = 0  # 0 = attention penuh (causal); >0 = sliding window
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size harus habis dibagi num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads harus habis dibagi num_kv_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("head_dim harus genap (dibutuhkan RoPE)")
        if self.vocab_size < 259:
            raise ValueError("vocab_size minimal 259 (256 byte + 3 token khusus)")
        if self.window_size < 0 or self.context_length < 1:
            raise ValueError("window_size/context_length tidak valid")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def load(cls, path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def count_parameters_formula(cfg: ModelConfig) -> int:
    """Hitung parameter secara analitik (parameter yang di-tie dihitung sekali)."""
    d, hd = cfg.hidden_size, cfg.head_dim
    attn = d * cfg.num_heads * hd * 2 + 2 * d * cfg.num_kv_heads * hd  # q,o + k,v
    mlp = 3 * d * cfg.intermediate_size
    per_layer = attn + mlp + 2 * d  # + dua RMSNorm
    embed = cfg.vocab_size * d
    total = embed + cfg.num_layers * per_layer + d  # + norm akhir
    if not cfg.tie_embeddings:
        total += embed
    return total
