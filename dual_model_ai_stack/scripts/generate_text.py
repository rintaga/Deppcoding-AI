"""Coba model: python scripts/generate_text.py checkpoints/tiny "halo dunia" """
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.generate import generate, load_for_inference  # noqa: E402
from core.train_utils import find_latest_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="folder step_XXXX ATAU folder induk (dipakai checkpoint terbaru)")
    ap.add_argument("prompt")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    path = Path(a.checkpoint)
    if not (path / "model.safetensors").exists():
        path = find_latest_checkpoint(path)
        if path is None:
            sys.exit("Checkpoint tidak ditemukan.")
    model, tok, _ = load_for_inference(path)
    print(generate(model, tok, a.prompt, a.max_new_tokens, a.temperature, seed=a.seed))


if __name__ == "__main__":
    main()
