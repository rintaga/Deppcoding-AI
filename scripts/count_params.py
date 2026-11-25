"""Verifikasi jumlah parameter: rumus vs model sungguhan (device meta, tanpa RAM besar)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import ModelConfig, count_parameters_formula  # noqa: E402
from core.model import count_parameters_meta  # noqa: E402

TARGETS = {"1.5b": 1.5e9, "3b": 3.0e9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=["configs/tiny.json", "configs/1.5b.json", "configs/3b.json"])
    ap.add_argument("--check", action="store_true", help="gagal jika menyimpang >5%% dari target / rumus != aktual")
    args = ap.parse_args()
    ok = True
    for path in args.configs:
        cfg = ModelConfig.load(path)
        formula, actual = count_parameters_formula(cfg), count_parameters_meta(cfg)
        stem = Path(path).stem
        line = f"{stem:6s} rumus={formula:>14,} aktual={actual:>14,} ({actual / 1e9:.4f}B)"
        if formula != actual:
            ok = False
            line += "  <-- RUMUS != AKTUAL"
        if stem in TARGETS:
            dev = abs(actual - TARGETS[stem]) / TARGETS[stem]
            line += f"  selisih dari target {dev * 100:.1f}%"
            if dev > 0.05:
                ok = False
                line += "  <-- DI LUAR 5%"
        print(line)
    if args.check and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
