# Dual Model AI Stack (1.5B + 3B, dari nol)

Ekosistem kecil dan modular untuk melatih dan menjalankan **dua model bahasa dari inisialisasi acak**:

| Model | Config | Parameter (diverifikasi) | d_model | Layer | Head / KV | FFN |
|---|---|---|---|---|---|---|
| tiny (uji) | `configs/tiny.json` | 459.392 | 128 | 2 | 4 / 2 | 384 |
| **A ≈ 1.5B** | `configs/1.5b.json` | **1.510.082.560** | 2048 | 32 | 16 / 4 | 5632 |
| **B ≈ 3B** | `configs/3b.json` | **3.025.318.400** | 2560 | 34 | 20 / 4 | 9216 |

Jumlah parameter dihitung dari model sungguhan (device `meta`) dan dicocokkan dengan rumus analitik:
`python scripts/count_params.py --check`. Tidak ada bobot pretrained dan tidak ada API AI eksternal.

## Isi proyek

```
core/        config, model (RMSNorm, RoPE, SwiGLU, GQA, sliding-window, grad checkpointing),
             tokenizer BPE byte-level, data, training/checkpoint/resume, generate
memory/      memori eksternal: SQLite (riwayat, dokumen, catatan) + pencarian BM25 + kuota disk
agent/       agen Linux: workspace terbatas, allowlist perintah, persetujuan pengguna, diagnosis error
server/      server HTTP (stdlib): antrean, worker, sesi, pilih model 1.5B/3B/tiny, pilih checkpoint
hf_sync/     upload/download model & dataset ke Hugging Face (token hanya dari env HF_TOKEN)
scripts/     count_params, prepare_data, generate_text
tests/       63 tes otomatis
.github/     CI (lint + tes + cek parameter) dan workflow smoke-train
.devcontainer/  konfigurasi GitHub Codespaces
docs/PANDUAN.md  panduan langkah demi langkah (Indonesia, untuk Termux)
```

## Mulai cepat (Linux / Codespaces)

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/count_params.py --check
python train.py --config configs/tiny.json --data data/train.txt --vocab-size 512 --steps 50
```

Panduan lengkap untuk HP Android (Termux): **docs/PANDUAN.md**.

## Batasan yang harus diketahui (jujur)

**Hardware.** Perkiraan memori training (bobot fp32 + gradien + AdamW = 16 byte/param, *belum termasuk aktivasi*):
1.5B ≈ 22,5 GiB, 3B ≈ 45,1 GiB. Praktisnya 1.5B butuh GPU ≥ 40 GB, 3B butuh GPU ≥ 80 GB (dengan `--grad-checkpoint`).
Hanya bobot untuk inferensi: 1.5B ≈ 2,8 GiB (bf16), 3B ≈ 5,6 GiB (bf16).
Skala data: aturan praktis ~20 token per parameter → ~30 miliar token (1.5B) dan ~60 miliar (3B). Dengan asumsi
6·N·D FLOPs dan GPU A100 ±120 TFLOPs efektif, itu sekitar 26 hari (1.5B) dan 105 hari (3B) pada **satu** GPU
(angka kasar, tergantung efisiensi). Kode training saat ini **satu GPU saja** (belum DDP/FSDP).

**Ponsel / Termux.** Realistisnya hanya model `tiny` untuk belajar dan uji alur. Model >300M parameter tanpa GPU
CUDA ditolak oleh `train.py` kecuali `--force-large`. Model tiny menghasilkan teks acak-acakan kecuali dilatih dengan data
sangat banyak; model besar pun tidak otomatis pintar.

**GitHub Actions / Codespaces.** Runner standar tidak punya GPU dan dibatasi waktu, RAM, dan kuota. Di proyek ini
Actions hanya untuk lint, tes, dan smoke-train model tiny. Codespaces dipakai untuk development dan tes CPU.
Kuota gratis berubah-ubah: cek halaman billing akun GitHub Anda.

**Konteks & memori.** Konteks aktif model terbatas (`context_length`, 2048 pada config besar). Memori eksternal
menyimpan lebih banyak dan mengambil potongan relevan ke prompt; ia tumbuh sampai batas kuota disk (`memory_max_mb`),
bukan tak terbatas. Jumlah token juga terbatas oleh vocab (32.768 pada config besar).

**Atensi.** `window_size > 0` memakai sliding-window per blok (biaya ~O(T·2W), diverifikasi sama dengan implementasi
naif lewat tes). `window_size = 0` = atensi kausal penuh. Config 1.5B/3B memakai window 1024 dari context 2048.

**Inferensi.** Generate belum memakai KV-cache (tiap token menghitung ulang konteks), jadi lambat untuk model besar.

**Agen.** Allowlist + workspace terbatas + persetujuan pengguna adalah *guardrail*, bukan sandbox keamanan penuh.
Kode Python yang disetujui tetap berjalan dengan hak akses Anda. Untuk kode tak tepercaya, gunakan container/Codespaces.
Agen bisa dipakai manual lewat perintah `/`; loop otomatis oleh model baru berguna setelah model Anda terlatih baik.

**Tokenizer.** BPE murni Python: nyaman untuk puluhan MB. Untuk korpus besar, latih tokenizer pada sampel
(`--tokenizer-sample-mb`); proses tokenisasi data besar akan lama.

## Status pengujian (apa yang benar-benar dijalankan)

Dijalankan di Linux x86 CPU (Python 3.12, PyTorch 2.15): 63 tes lulus; training tiny, resume, generate,
server HTTP (curl) dan REPL agen dijalankan nyata; forward 1.5B dan 3B diuji pada device `meta`.
**Belum diuji:** Android/Termux, GPU/CUDA, training 1.5B/3B sungguhan, eksekusi workflow di GitHub,
Codespaces, dan upload/download Hugging Face sungguhan (tes memakai API palsu). Workflow YAML hanya divalidasi sintaksnya.
