"""Integrasi Hugging Face Hub. Token HANYA dibaca dari environment (HF_TOKEN), tidak pernah dicetak/disimpan."""
from __future__ import annotations

import argparse
import os
import tempfile
import shutil
from pathlib import Path

WEIGHT_FILES = ["model.safetensors", "config.json", "tokenizer.json"]


def get_token(required: bool = True):
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if required and not tok:
        raise RuntimeError("HF_TOKEN belum di-set. Jalankan: export HF_TOKEN=... (jangan tulis token di kode/commit).")
    return tok


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=get_token())


def model_card(cfg: dict, params: int, trained_steps: int | None = None) -> str:
    status = (f"Dilatih {trained_steps} step (jumlah kecil bukan berarti model berguna)."
              if trained_steps else "Checkpoint belum/tanpa informasi training.")
    return (f"---\nlibrary_name: custom\ntags: [from-scratch, decoder-only]\n---\n"
            f"# {cfg.get('model_name', 'model')}\n\n"
            f"- Parameter (diverifikasi dari model nyata): {params:,}\n"
            f"- Arsitektur: Transformer decoder-only (RMSNorm, RoPE, SwiGLU, GQA, sliding-window opsional)\n"
            f"- Inisialisasi: acak (bukan fine-tune model lain)\n- {status}\n\n"
            f"Memuat butuh kode proyek (`core/`), bukan `transformers`.\n")


def upload_checkpoint(local_dir, repo_id: str, private: bool = True, include_optimizer: bool = False,
                      path_in_repo: str | None = None, commit_message: str = "Upload checkpoint"):
    """Unggah bobot + config + tokenizer (+ optimizer state bila diminta, ukurannya besar)."""
    local = Path(local_dir)
    files = [f for f in WEIGHT_FILES if (local / f).exists()]
    if "model.safetensors" not in files:
        raise FileNotFoundError(f"model.safetensors tidak ada di {local}")
    if include_optimizer and (local / "trainer_state.pt").exists():
        files.append("trainer_state.pt")
    api = _api()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            shutil.copy(local / f, Path(tmp) / f)
        from core.config import ModelConfig, count_parameters_formula
        cfg = ModelConfig.load(local / "config.json")
        (Path(tmp) / "README.md").write_text(model_card(cfg.to_dict(), count_parameters_formula(cfg)), encoding="utf-8")
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=tmp,
                          path_in_repo=path_in_repo, commit_message=commit_message)
    return files


def download_checkpoint(repo_id: str, local_dir, revision: str | None = None):
    """Unduh (bisa dilanjutkan jika terputus: file yang sudah lengkap dilewati)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(local_dir),
                             revision=revision, token=get_token(required=False))


def upload_dataset(local_path, repo_id: str, private: bool = True, path_in_repo: str | None = None):
    api = _api()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    p = Path(local_path)
    if p.is_dir():
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(p), path_in_repo=path_in_repo)
    else:
        api.upload_file(repo_id=repo_id, repo_type="dataset", path_or_fileobj=str(p),
                        path_in_repo=path_in_repo or p.name)


def download_dataset(repo_id: str, local_dir, revision: str | None = None):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir),
                             revision=revision, token=get_token(required=False))


def main():
    ap = argparse.ArgumentParser(description="Sinkronisasi Hugging Face Hub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("upload-model", "upload-dataset"):
        s = sub.add_parser(name)
        s.add_argument("path"); s.add_argument("repo")
        s.add_argument("--public", action="store_true")
        if name == "upload-model":
            s.add_argument("--with-optimizer", action="store_true")
    for name in ("download-model", "download-dataset"):
        s = sub.add_parser(name)
        s.add_argument("repo"); s.add_argument("dest"); s.add_argument("--revision")
    sub.add_parser("whoami")
    a = ap.parse_args()
    if a.cmd == "upload-model":
        print("Terunggah:", upload_checkpoint(a.path, a.repo, not a.public, a.with_optimizer))
    elif a.cmd == "upload-dataset":
        upload_dataset(a.path, a.repo, not a.public); print("Dataset terunggah")
    elif a.cmd == "download-model":
        print("Tersimpan di:", download_checkpoint(a.repo, a.dest, a.revision))
    elif a.cmd == "download-dataset":
        print("Tersimpan di:", download_dataset(a.repo, a.dest, a.revision))
    elif a.cmd == "whoami":
        print("Login sebagai:", _api().whoami()["name"])  # hanya nama, bukan token


if __name__ == "__main__":
    main()
