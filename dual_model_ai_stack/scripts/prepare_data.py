"""Ubah file .txt menjadi dataset token (.bin) + latih tokenizer bila belum ada."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.data import tokenize_files  # noqa: E402
from core.tokenizer import ByteBPETokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="file .txt UTF-8 atau folder berisi .txt")
    ap.add_argument("--tokenizer", default="outputs/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--out", default="data/train.bin")
    ap.add_argument("--tokenizer-sample-mb", type=int, default=20,
                    help="batas data untuk melatih tokenizer (agar cepat)")
    args = ap.parse_args()

    files = []
    for item in args.inputs:
        p = Path(item)
        files += sorted(p.rglob("*.txt")) if p.is_dir() else [p]
    if not files:
        sys.exit("Tidak ada file .txt ditemukan.")

    if not Path(args.tokenizer).exists():
        limit = args.tokenizer_sample_mb * 1_000_000
        text, size = [], 0
        for f in files:
            t = f.read_text(encoding="utf-8", errors="replace")[: max(0, limit - size)]
            text.append(t)
            size += len(t)
            if size >= limit:
                break
        tok = ByteBPETokenizer.train("\n".join(text), args.vocab_size)
        tok.save(args.tokenizer)
        print(f"Tokenizer dilatih: {tok.vocab_size} token -> {args.tokenizer}")
    else:
        tok = ByteBPETokenizer.load(args.tokenizer)
        print(f"Memakai tokenizer yang ada ({tok.vocab_size} token)")
    n = tokenize_files(files, tok, args.out)
    print(f"Selesai: {n:,} token -> {args.out}")


if __name__ == "__main__":
    main()
