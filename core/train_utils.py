"""Training: gradient accumulation, mixed precision, gradient checkpointing,
checkpoint atomik + resume penuh (model, optimizer, scaler, step, RNG)."""
from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from .config import ModelConfig, count_parameters_formula
from .data import BlockSampler, load_token_array
from .model import DecoderLM, count_parameters


# ---------------- checkpoint ----------------
def step_dir(root, step: int) -> Path:
    return Path(root) / f"step_{step:08d}"


def save_checkpoint(root, model, optimizer, scaler, step, cfg: ModelConfig, tokenizer_path=None,
                    extra=None, keep_last: int = 3) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final, tmp = step_dir(root, step), root / f".tmp_step_{step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    save_model(model, str(tmp / "model.safetensors"))
    cfg.save(tmp / "config.json")
    if tokenizer_path and Path(tokenizer_path).exists():
        shutil.copy(tokenizer_path, tmp / "tokenizer.json")
    torch.save({
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "step": step,
        "rng_cpu": torch.get_rng_state(),
        "extra": extra or {},
    }, tmp / "trainer_state.pt")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    (root / "latest.txt").write_text(final.name, encoding="utf-8")
    done = sorted(p for p in root.glob("step_*") if p.is_dir())
    for old in done[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return final


def find_latest_checkpoint(root):
    root = Path(root)
    ptr = root / "latest.txt"
    if ptr.exists():
        cand = root / ptr.read_text(encoding="utf-8").strip()
        if (cand / "model.safetensors").exists():
            return cand
    done = sorted(p for p in root.glob("step_*") if (p / "model.safetensors").exists())
    return done[-1] if done else None


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu") -> int:
    path = Path(path)
    load_model(model, str(path / "model.safetensors"))
    state_file = path / "trainer_state.pt"
    step = 0
    if state_file.exists():
        # File ini dibuat oleh proyek ini sendiri; jangan memuat trainer_state.pt dari sumber tak tepercaya.
        state = torch.load(state_file, map_location=device, weights_only=False)
        step = int(state.get("step", 0))
        if optimizer is not None and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
    return step


# ---------------- utilitas ----------------
def lr_at(step, max_lr, min_lr, warmup, total):
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def estimate_training_memory_gb(n_params: int, dtype: str = "bf16") -> dict:
    """Perkiraan KASAR (tanpa aktivasi): bobot fp32 + grad fp32 + AdamW (2x fp32)."""
    gb = 1024 ** 3
    return {
        "bobot+grad+adam (fp32 master)": n_params * 16 / gb,
        "hanya bobot (bf16/fp16)": n_params * 2 / gb,
        "hanya bobot (fp32)": n_params * 4 / gb,
    }


def pick_device(name: str = "auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(model, lr, weight_decay, betas=(0.9, 0.95)):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


@torch.no_grad()
def evaluate(model, sampler, device, batch_size, batches, amp_ctx, seed=1234):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        x, y = sampler.get_batch("val", batch_size, g)
        with amp_ctx():
            losses.append(model(x.to(device), y.to(device))["loss"].item())
    model.train()
    return sum(losses) / len(losses)


# ---------------- loop utama ----------------
def run_training(cfg: ModelConfig, data_path, tokenizer_path, save_dir, *, steps=100, batch_size=1,
                 grad_accum=1, seq_len=None, lr=3e-4, min_lr_ratio=0.1, warmup=10, weight_decay=0.1,
                 grad_clip=1.0, save_every=100, eval_every=0, eval_batches=5, log_every=10,
                 resume=None, dtype="auto", grad_checkpoint=False, device="auto", seed=1337,
                 keep_last=3, hf_repo=None, force_large=False, log=print) -> dict:
    dev = pick_device(device)
    seq_len = seq_len or min(cfg.context_length, 256)
    if seq_len > cfg.context_length:
        raise ValueError("seq_len melebihi context_length model")
    n_params = count_parameters_formula(cfg)
    log(f"Model {cfg.model_name}: {n_params:,} parameter ({n_params / 1e9:.3f}B) | device={dev}")
    for k, v in estimate_training_memory_gb(n_params).items():
        log(f"  perkiraan memori {k}: {v:.1f} GB")
    if n_params > 300_000_000 and dev.type != "cuda" and not force_large:
        raise SystemExit("Model >300M parameter tanpa GPU CUDA hampir pasti gagal/sangat lambat. "
                         "Gunakan configs/tiny.json di HP, atau tambahkan --force-large jika yakin.")

    use_cuda = dev.type == "cuda"
    if dtype == "auto":
        dtype = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if use_cuda else "fp32"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[dtype]

    def amp_ctx():
        if amp_dtype is None:
            return torch.autocast(device_type=dev.type, enabled=False)
        return torch.autocast(device_type=dev.type, dtype=amp_dtype)

    torch.manual_seed(seed)
    model = DecoderLM(cfg).to(dev)
    assert count_parameters(model) == n_params, "rumus parameter tidak cocok dengan model"
    model.gradient_checkpointing = grad_checkpoint
    model.train()
    optimizer = make_optimizer(model, lr, weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and dtype == "fp16"))

    tokens = load_token_array(data_path, tokenizer_path)
    sampler = BlockSampler(tokens, seq_len)
    if sampler.val_is_train:
        log("PERINGATAN: data terlalu kecil untuk split validasi; val = train (hanya uji teknis).")

    start = 0
    if resume:
        ckpt = find_latest_checkpoint(save_dir) if resume == "auto" else Path(resume)
        if ckpt:
            start = load_checkpoint(ckpt, model, optimizer, scaler, device=dev)
            log(f"Resume dari {ckpt} (step {start})")
        else:
            log("Resume 'auto': belum ada checkpoint, mulai dari awal.")

    min_lr = lr * min_lr_ratio
    last_loss, t0, final_ckpt = float("nan"), time.time(), None
    for step in range(start, steps):
        cur_lr = lr_at(step, lr, min_lr, warmup, steps)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        optimizer.zero_grad(set_to_none=True)
        gen = torch.Generator().manual_seed(seed * 1_000_003 + step)  # deterministik per step -> resume konsisten
        accum_loss = 0.0
        for _ in range(grad_accum):
            x, y = sampler.get_batch("train", batch_size, gen)
            with amp_ctx():
                loss = model(x.to(dev), y.to(dev))["loss"] / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss tidak finite di step {step + 1}; turunkan --lr atau cek data.")
            scaler.scale(loss).backward()
            accum_loss += loss.item()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        last_loss = accum_loss
        s = step + 1
        if s % log_every == 0 or s == steps:
            tps = (batch_size * grad_accum * seq_len) / max(1e-9, (time.time() - t0) / (s - start))
            log(f"step {s}/{steps} loss {last_loss:.4f} lr {cur_lr:.2e} gnorm {float(gn):.2f} ~{tps:,.0f} tok/s")
        if eval_every and s % eval_every == 0:
            log(f"  eval loss: {evaluate(model, sampler, dev, batch_size, eval_batches, amp_ctx):.4f}")
        if save_every and s % save_every == 0 and s != steps:
            save_checkpoint(save_dir, model, optimizer, scaler, s, cfg, tokenizer_path, keep_last=keep_last)
    final_ckpt = save_checkpoint(save_dir, model, optimizer, scaler, max(steps, start), cfg, tokenizer_path,
                                 keep_last=keep_last)
    log(f"Checkpoint akhir: {final_ckpt}")
    if hf_repo:
        from hf_sync.hub import upload_checkpoint
        upload_checkpoint(final_ckpt, hf_repo)
        log(f"Diunggah ke Hugging Face: {hf_repo}")
    return {"step": max(steps, start), "loss": last_loss, "checkpoint": str(final_ckpt), "params": n_params}
