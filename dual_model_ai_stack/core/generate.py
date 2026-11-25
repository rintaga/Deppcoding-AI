"""Muat checkpoint untuk inferensi + sampling teks (tanpa KV-cache: sederhana tapi lambat)."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_model

from .config import ModelConfig
from .model import DecoderLM
from .tokenizer import ByteBPETokenizer, EOS_ID


def load_for_inference(ckpt_dir, device="cpu", dtype=None, tokenizer_path=None):
    ckpt_dir = Path(ckpt_dir)
    cfg = ModelConfig.load(ckpt_dir / "config.json")
    model = DecoderLM(cfg)
    load_model(model, str(ckpt_dir / "model.safetensors"))
    model.to(device=device, dtype=dtype) if dtype else model.to(device)
    model.eval()
    tok_file = Path(tokenizer_path) if tokenizer_path else ckpt_dir / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(f"tokenizer.json tidak ada di {ckpt_dir}")
    return model, ByteBPETokenizer.load(tok_file), cfg


def sample_next(logits, temperature=0.8, top_k=40, top_p=0.95, generator=None):
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits.float() / temperature
    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp[(cum - sp) > top_p] = 0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens=64, temperature=0.8, top_k=40, top_p=0.95,
             seed=None, on_token=None) -> str:
    device = next(model.parameters()).device
    ids = tok.encode(prompt, add_bos=True)
    gen = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    ctx = model.cfg.context_length
    out_ids = []
    for _ in range(max_new_tokens):
        window = torch.tensor([ids[-ctx:]], dtype=torch.long, device=device)
        logits = model(window)["logits"][0, -1].cpu()
        nxt = sample_next(logits, temperature, top_k, top_p, gen)
        if nxt == EOS_ID:
            break
        ids.append(nxt)
        out_ids.append(nxt)
        if on_token:
            on_token(nxt)
    return tok.decode(out_ids)
