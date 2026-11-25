"""Decoder-only Transformer: RMSNorm, RoPE, SwiGLU, GQA, sliding-window attention,
gradient checkpointing. Semua parameter diinisialisasi acak (tanpa bobot pretrained)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


def build_rope(head_dim: int, max_len: int, theta: float, device=None):
    half = head_dim // 2
    freq = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    angles = torch.arange(max_len, device=device).float()[:, None] * freq[None, :]
    return angles.cos(), angles.sin()  # masing-masing [max_len, half]


def apply_rope(x, cos, sin):
    """x: [B, H, T, D]; cos/sin: [T, D/2]."""
    half = x.shape[-1] // 2
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def _expand_kv(x, groups: int):
    if groups == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, groups, t, d).reshape(b, h * groups, t, d)


def sliding_window_attention(q, k, v, window: int, dropout_p: float = 0.0):
    """Causal attention terbatas: token i melihat token j dengan i-window < j <= i.

    Dikerjakan per blok berukuran `window` (tiap blok melihat dirinya + blok
    sebelumnya) sehingga biaya O(T * 2W), bukan O(T^2). q,k,v: [B, H, T, D].
    """
    b, h, t, d = q.shape
    w = window
    n = -(-t // w)
    pad = n * w - t
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    qb = q.view(b, h, n, w, d)
    kb = k.view(b, h, n, w, d)
    vb = v.view(b, h, n, w, d)
    zeros = torch.zeros_like(kb[:, :, :1])
    kk = torch.cat([torch.cat([zeros, kb[:, :, :-1]], dim=2), kb], dim=3)  # [B,H,N,2W,D]
    vv = torch.cat([torch.cat([zeros, vb[:, :, :-1]], dim=2), vb], dim=3)
    blk = torch.arange(n, device=q.device)[:, None, None]
    qi = torch.arange(w, device=q.device)[None, :, None]
    kj = torch.arange(2 * w, device=q.device)[None, None, :]
    abs_q = blk * w + qi
    abs_k = (blk - 1) * w + kj
    mask = (abs_k >= 0) & (abs_k <= abs_q) & (abs_k > abs_q - w)  # [N,W,2W]
    out = F.scaled_dot_product_attention(qb, kk, vv, attn_mask=mask, dropout_p=dropout_p)
    return out.reshape(b, h, n * w, d)[:, :, :t]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.window = cfg.window_size
        self.dropout = cfg.dropout
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        groups = self.heads // self.kv_heads
        k, v = _expand_kv(k, groups), _expand_kv(v, groups)
        p = self.dropout if self.training else 0.0
        if self.window and self.window < t:
            y = sliding_window_attention(q, k, v, self.window, p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, self.heads * self.head_dim))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size)
        self.ffn_norm = RMSNorm(cfg.hidden_size)
        self.attn = Attention(cfg)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.ffn_norm(x))


class DecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig | dict):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = ModelConfig.from_dict(cfg)
        self.cfg = cfg
        self.gradient_checkpointing = False
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = build_rope(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * cfg.num_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, std=scale)
            nn.init.normal_(layer.mlp.down.weight, std=scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        t = input_ids.size(1)
        if t > self.cfg.context_length:
            raise ValueError(f"Panjang urutan {t} > context_length {self.cfg.context_length}")
        x = self.embed(input_ids)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}


def load_config(path) -> ModelConfig:
    return ModelConfig.load(path)


def count_parameters(model: nn.Module) -> int:
    """Jumlah parameter unik (weight tying dihitung sekali)."""
    return sum(p.numel() for p in model.parameters())


def count_parameters_meta(cfg: ModelConfig) -> int:
    """Bangun model di device 'meta' (tanpa memori) dan hitung parameter sebenarnya."""
    with torch.device("meta"):
        model = DecoderLM(cfg)
    return count_parameters(model)
