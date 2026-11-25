# Panduan Lengkap (Termux, tanpa PC)

Semua perintah bisa disalin langsung ke Termux. Tanda `#` = penjelasan (boleh ikut disalin, diabaikan shell).
Ganti `USERNAME`, `REPO`, dsb. dengan milik Anda.

> **Jujur dulu:** di HP Anda realistisnya hanya melatih model **tiny** dan menjalankan agen/memori/server.
> Training 1.5B/3B butuh GPU besar (lihat Langkah 10–11). Alur kerja yang masuk akal: tulis & uji di HP,
> jalankan tes di Codespaces, simpan ke Hugging Face.

---
## 1. Ekstrak ZIP di Termux
```bash
termux-setup-storage          # minta izin Termux membaca penyimpanan HP (tekan "Izinkan")
cd ~                          # pindah ke folder rumah Termux
cp ~/storage/downloads/dual_model_ai_stack.zip .   # salin ZIP dari folder Download HP ke Termux
pkg install -y unzip          # pasang alat pembuka ZIP
unzip dual_model_ai_stack.zip # ekstrak; akan muncul folder dual_model_ai_stack
cd dual_model_ai_stack        # masuk ke folder proyek
ls                            # lihat isinya (harus ada train.py, core/, agent/, ...)
```

## 2. Pasang dependensi
```bash
pkg update -y && pkg upgrade -y          # perbarui daftar & paket Termux
pkg install -y python git nano curl python-numpy   # Python, Git, editor nano, curl, numpy
python -m venv --system-site-packages .venv        # buat lingkungan Python terisolasi (tetap bisa pakai numpy Termux)
source .venv/bin/activate                # aktifkan (ulangi tiap membuka Termux baru, dari folder proyek)
```
Bagian yang **tidak butuh PyTorch** (tokenizer, memori, agen) sudah bisa jalan di sini. Untuk training & server butuh PyTorch:
```bash
pkg install -y tur-repo                  # repositori tambahan komunitas Termux
pkg install -y python-torch              # PyTorch CPU (ketersediaan tergantung perangkat/arsitektur Anda)
pip install safetensors huggingface_hub pytest pyflakes   # paket lain
```
Jika `python-torch` tidak ada/gagal, atau `pip install safetensors` gagal karena butuh kompilasi (Rust):
**pakai GitHub Codespaces (Langkah 5)** untuk semua bagian yang memakai PyTorch. Itu jalur yang paling dapat diandalkan dari HP.
Cek: `python -c "import torch; print(torch.__version__)"`

## 3. Membuka dan mengedit proyek
```bash
nano README.md        # buka editor. Simpan: Ctrl+O lalu Enter. Keluar: Ctrl+X
ls -R | head -50      # lihat struktur folder
cat configs/tiny.json # tampilkan isi file di layar
```
Di keyboard Termux, tombol Ctrl ada di baris tambahan atas (atau geser dari kiri-layar untuk sesi). Alternatif editor
berbasis kode: buka Codespaces (Langkah 5), editornya VS Code di browser.

## 4. Konfigurasi GitHub
1. Di browser HP, buat akun di github.com, lalu **New repository** (nama mis. `dual-ai`, pilih *Private*). Jangan centang README.
2. Buat token: GitHub → Settings → Developer settings → Personal access tokens → *Fine-grained* → pilih repo itu →
   izin **Contents: Read and write** → Generate. **Salin tokennya** (hanya tampil sekali). Jangan dibagikan.
```bash
git config --global user.name "Nama Anda"        # nama yang tampil di commit
git config --global user.email "email@anda.com"  # email akun GitHub
git init -b main                                 # mulai repositori git di folder ini
git add .                                        # tandai semua file untuk disimpan
git commit -m "Versi awal"                       # simpan snapshot
git remote add origin https://github.com/USERNAME/REPO.git   # hubungkan ke repo GitHub
git push -u origin main                          # unggah. Username = username GitHub, Password = TOKEN tadi
```
`.gitignore` sudah mencegah `.env`, checkpoint, dan data ikut terunggah. **Jangan pernah** menulis token di file kode.

## 5. Memakai GitHub Codespaces (dari browser HP)
1. Buka github.com/codespaces di browser (aktifkan "Situs desktop" di menu browser agar lebih nyaman) → **New codespace** → pilih repo Anda.
2. Tunggu proses build. Konfigurasi di `.devcontainer/` akan memasang PyTorch CPU + dependensi otomatis.
3. Buka terminal di dalamnya, lalu uji:
```bash
python -m pytest -q                      # jalankan 63 tes otomatis
python scripts/count_params.py --check   # verifikasi parameter 1.5B dan 3B
```
4. **Hentikan codespace** setelah selesai (menu Codespaces → Stop) agar kuota gratis tidak habis. Kuota bergantung paket akun Anda.
5. Rahasia (token) disimpan di GitHub → Settings → Codespaces → Secrets (mis. `HF_TOKEN`); tersedia sebagai variabel environment.

## 6. Mengatur GitHub Actions
Workflow sudah ada di `.github/workflows/`. Begitu Anda `git push`, tab **Actions** di repo menjalankan `CI`
(lint + tes + cek parameter). Untuk smoke-train tiny: Actions → *Train smoke* → **Run workflow**.
Untuk upload otomatis ke Hugging Face: repo → Settings → Secrets and variables → Actions → **New repository secret**,
nama `HF_TOKEN`, isi token HF. Actions runner standar **tidak punya GPU** — jangan dipakai melatih 1.5B/3B.

## 7. Menghubungkan Hugging Face Hub
1. Daftar di huggingface.co → Settings → Access Tokens → **New token** dengan izin **Write**. Salin.
```bash
read -s -p "Tempel token HF lalu Enter: " HF_TOKEN; export HF_TOKEN; echo   # input tersembunyi, hanya berlaku di sesi ini
python -m hf_sync.hub whoami             # uji: harus menampilkan nama akun Anda (token tidak dicetak)
```
Opsional agar tidak mengetik ulang: simpan di `.env` (sudah di-.gitignore) lalu `set -a; source .env; set +a`.
Repo model dibuat otomatis (privat) saat upload pertama.

## 8. Menyiapkan dataset latihan
Dataset = file teks UTF-8 (`.txt`). Simpan di `data/raw/`.
```bash
mkdir -p data/raw
nano data/raw/contoh.txt          # tempel/tulis teks Anda (makin banyak makin baik)
python scripts/prepare_data.py data/raw --vocab-size 4096 --out data/train.bin
```
Perintah itu: (a) melatih tokenizer BPE → `outputs/tokenizer.json`, (b) mengubah teks jadi token → `data/train.bin`.
Dataset dari Hugging Face: `python -m hf_sync.hub download-dataset USERNAME/DATASET data/raw`.
Mengunggah dataset: `python -m hf_sync.hub upload-dataset data/raw USERNAME/DATASET`.
Catatan: vocab tokenizer harus ≤ `vocab_size` model. Config tiny: 512 → pakai `--vocab-size 512` untuk tiny.

## 9. Melatih model tiny (uji alur)
```bash
python scripts/prepare_data.py data/raw --vocab-size 512 --out data/train.bin
python train.py --config configs/tiny.json --data data/train.bin --vocab-size 512 \
  --seq-len 64 --batch-size 8 --steps 300 --lr 3e-3 --save-every 100 --eval-every 100 \
  --save-dir checkpoints/tiny
python scripts/generate_text.py checkpoints/tiny "halo dunia" --max-new-tokens 40
```
Artinya: `--steps` jumlah langkah, `--save-every` simpan checkpoint tiap N langkah, `--eval-every` hitung loss validasi.
Loss (angka di layar) harus menurun. Hasil teks akan masih kasar — wajar untuk model sekecil itu.
Tidak ada data banyak? Kalau data < ~2×`--seq-len` token akan muncul error "Data terlalu pendek": tambah data atau kecilkan `--seq-len`.

## 10. Melatih model 1.5B (saat hardware memadai)
Syarat: GPU CUDA ≥ 40 GB (mis. A100 40/80GB), data besar (target ~30 miliar token), **bukan** HP/Codespaces biasa.
Kode ini **belum diuji di GPU** dan hanya satu GPU.
```bash
python scripts/prepare_data.py data/raw --vocab-size 32768 --out data/train.bin
python train.py --config configs/1.5b.json --data data/train.bin --seq-len 1024 --batch-size 1 \
  --grad-accum 32 --grad-checkpoint --lr 3e-4 --warmup 2000 --steps 100000 \
  --save-every 500 --eval-every 500 --resume auto --save-dir checkpoints/1.5b --hf-repo USERNAME/model-1.5b
```
`--grad-accum 32` = menumpuk 32 batch kecil jadi satu langkah (hemat memori). `--grad-checkpoint` = hitung ulang aktivasi
(hemat memori, sedikit lebih lambat). `--hf-repo` mengunggah checkpoint akhir (bobot+config+tokenizer) ke Hugging Face.
Penolakan otomatis jika tanpa GPU: tambahkan `--force-large` hanya jika Anda tahu risikonya.

## 11. Mengonfigurasi model 3B
Sama seperti Langkah 10, ganti config: `--config configs/3b.json --save-dir checkpoints/3b`. Butuh GPU ≥ 80 GB
(estimasi 45 GiB hanya untuk bobot+gradien+optimizer, belum aktivasi). Ubah ukuran model dengan mengedit
`configs/*.json`, lalu **selalu** verifikasi: `python scripts/count_params.py configs/namafile.json`.
Aturan: `hidden_size` habis dibagi `num_heads`; `num_heads` habis dibagi `num_kv_heads`; `head_dim` genap.

## 12. Menyimpan & memulihkan checkpoint
Checkpoint tersimpan di `checkpoints/NAMA/step_XXXXXXXX/` (bobot `model.safetensors`, `config.json`, `tokenizer.json`,
`trainer_state.pt` = optimizer+langkah). Penulisan atomik, `--keep-last 3` menghapus yang lama.
```bash
python train.py ... --resume auto                       # lanjut dari checkpoint terbaru di --save-dir
python train.py ... --resume checkpoints/tiny/step_00000100   # lanjut dari checkpoint tertentu
python -m hf_sync.hub upload-model checkpoints/tiny/step_00000300 USERNAME/tiny    # simpan ke HF (tanpa optimizer)
python -m hf_sync.hub upload-model checkpoints/tiny/step_00000300 USERNAME/tiny --with-optimizer  # +optimizer (besar)
python -m hf_sync.hub download-model USERNAME/tiny checkpoints/tiny/step_hf        # pulihkan dari HF
```
Tips resume: pakai `--steps` **yang sama** dengan run awal agar jadwal learning-rate tetap mulus (resume dengan
optimizer yang diunggah → persis sama; tes membuktikan hasil resume identik bit-demi-bit di CPU).

## 13. Menjalankan agen Linux AI
```bash
mkdir -p workspace
python -m agent.agent --workspace workspace              # tanpa model: pakai perintah manual
python -m agent.agent --workspace workspace --model-dir checkpoints/tiny/step_00000300   # + /ask ke model
```
Di prompt `agen>` ketik `/help`. Contoh: `/inspect`, `/write halo.py` (ketik kode, akhiri `EOF`), `/run python halo.py`,
`/check halo.py`, `/diagnose <pesan error>`, `/plan perbaiki bug import`, `/note ...`, `/memory kata`.
Aturan keamanan: hanya file di `workspace/`; hanya perintah allowlist (ls, cat, head, tail, wc, pwd, echo, grep, python
skrip/`-m pytest`, git baca-saja); tanpa shell; **setiap aksi tulis/jalankan meminta Anda mengetik `y`** (termasuk
yang diminta model). Log di `logs/agent.jsonl`. Ini guardrail, bukan sandbox penuh.

## 14. Menjalankan & memakai server AI
```bash
export SERVER_API_KEY="$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"   # kunci acak; catat dengan echo $SERVER_API_KEY
python -m server.app --config configs/server.json       # mendengar di 127.0.0.1:8000 (hanya HP ini)
```
Buka sesi Termux kedua, lalu:
```bash
K=$SERVER_API_KEY
curl -s localhost:8000/health                                                     # status & antrean
curl -s -H "Authorization: Bearer $K" localhost:8000/v1/models                    # model & checkpoint
curl -s -X POST localhost:8000/v1/generate -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"model":"tiny","prompt":"halo","max_new_tokens":30,"session_id":"s1","use_memory":true}'
curl -s -H "Authorization: Bearer $K" localhost:8000/v1/sessions/s1/history        # riwayat sesi
curl -s -X POST localhost:8000/v1/memory/documents -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"source":"catatan1","text":"Termux adalah terminal Android."}'              # tambah dokumen ke memori
curl -s -H "Authorization: Bearer $K" "localhost:8000/v1/memory/search?q=termux"   # cari di memori
```
Pilih model dengan `"model":"1.5b"` atau `"3b"` (butuh checkpoint di `checkpoints/1.5b` / `checkpoints/3b`; kalau
belum ada, server menjawab 409). Ganti checkpoint: `POST /v1/checkpoints/select {"model":"tiny","path":"step_00000100"}`.
Setelan di `configs/server.json`: `workers` (konkurensi), `max_queue` (antrean penuh → HTTP 429),
`max_loaded_models` (berapa model di RAM sekaligus), `memory_max_mb`. Request lama: tambahkan `"async":true` lalu
cek `GET /v1/jobs/ID`. Server hanya mendengar di localhost; jangan buka ke jaringan tanpa `SERVER_API_KEY`.

## 15. Diagnosis error umum
| Pesan | Arti & solusi |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch belum ada. Langkah 2, atau pakai Codespaces. |
| `Killed` / `MemoryError` | RAM habis (Android mematikan proses). Pakai config tiny, kecilkan `--batch-size`/`--seq-len`. |
| `CUDA out of memory` | Kecilkan batch/seq-len, naikkan `--grad-accum`, tambahkan `--grad-checkpoint`. |
| `Model >300M parameter tanpa GPU...` | Perlindungan. Pakai tiny di HP, atau GPU. |
| `Data terlalu pendek` | Data kurang dari `seq_len+2` token. Tambah data / kecilkan `--seq-len`. |
| `Tokenizer (...) lebih besar dari vocab_size` | Latih ulang tokenizer dengan `--vocab-size` ≤ vocab model, atau ubah `--vocab-size` train. |
| `Loss tidak finite` | Turunkan `--lr`, cek data (file kosong/rusak). |
| `HF_TOKEN belum di-set` / `401`/`403` | Ulangi Langkah 7; token harus izin Write dan nama repo benar. |
| `No space left on device` | Hapus checkpoint lama, `pip cache purge`. |
| `externally-managed-environment` (pip) | Aktifkan venv: `source .venv/bin/activate`. |
| `Permission denied` saat `git push` | Pakai token sebagai password, bukan password akun. |
| Server `409 Belum ada checkpoint` | Latih dulu atau unduh dari HF ke `checkpoints/NAMA/`. |
| Server `429` | Antrean penuh; coba lagi / naikkan `max_queue`. |
| Agen `SecurityError ... allowlist` | Perintah/path memang diblokir. Lihat `/help`. |

Lihat log: `tail -n 50 logs/server.log`, `tail -n 20 logs/agent.jsonl`. Jalankan `python -m pytest -q -x` untuk
mencari bagian yang rusak, dan `/diagnose <pesan error>` di agen untuk saran otomatis.
