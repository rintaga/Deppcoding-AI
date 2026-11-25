"""Entry point training. Contoh: python train.py --config configs/tiny.json --data data/train.txt"""
import argparse
from pathlib import Path

from core.config import ModelConfig
from core.tokenizer import ByteBPETokenizer
from core.train_utils import run_training


def main():
    p = argparse.ArgumentParser(description="Latih model dari inisialisasi acak")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True, help=".txt (kecil) atau .bin dari scripts/prepare_data.py")
    p.add_argument("--tokenizer", default="outputs/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=None, help="override vocab_size model")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--save-dir", default="checkpoints/run")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--resume", default=None, help="'auto' atau path folder checkpoint")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--hf-repo", default=None, help="mis. username/nama-repo (butuh env HF_TOKEN)")
    p.add_argument("--force-large", action="store_true")
    a = p.parse_args()

    cfg = ModelConfig.load(a.config)
    if a.vocab_size:
        cfg.vocab_size = a.vocab_size
    if not Path(a.tokenizer).exists():
        if not a.data.endswith(".txt"):
            raise SystemExit(f"Tokenizer {a.tokenizer} tidak ada. Jalankan scripts/prepare_data.py dulu.")
        print("Tokenizer belum ada; melatih BPE dari file teks...")
        tok = ByteBPETokenizer.train(Path(a.data).read_text(encoding="utf-8"), cfg.vocab_size)
        tok.save(a.tokenizer)
    tok = ByteBPETokenizer.load(a.tokenizer)
    if tok.vocab_size > cfg.vocab_size:
        raise SystemExit(f"Tokenizer ({tok.vocab_size}) lebih besar dari vocab_size model ({cfg.vocab_size}).")
    run_training(cfg, a.data, a.tokenizer, a.save_dir, steps=a.steps, batch_size=a.batch_size,
                 grad_accum=a.grad_accum, seq_len=a.seq_len, lr=a.lr, warmup=a.warmup,
                 weight_decay=a.weight_decay, save_every=a.save_every, eval_every=a.eval_every,
                 log_every=a.log_every, keep_last=a.keep_last, resume=a.resume, dtype=a.dtype,
                 grad_checkpoint=a.grad_checkpoint, device=a.device, seed=a.seed,
                 hf_repo=a.hf_repo, force_large=a.force_large)


if __name__ == "__main__":
    main()
