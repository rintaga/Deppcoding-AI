"""Alat agen Linux yang aman: workspace terbatas, allowlist perintah, tanpa shell.

PENTING: ini guardrail praktis, BUKAN sandbox keamanan. Kode Python yang dijalankan
masih bisa melakukan banyak hal di dalam hak akses pengguna. Untuk kode yang tidak
dipercaya, jalankan di container/Codespaces terpisah.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None


class SecurityError(Exception):
    pass


MAX_OUT = 20_000
SHELL_META = re.compile(r"[;&|`$<>\n\r]")
SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TERM")
DENY_MODULES = {"subprocess", "socket", "ctypes", "shutil", "multiprocessing", "pty", "http", "urllib", "requests"}
DENY_OS_ATTRS = {"system", "popen", "remove", "unlink", "rmdir", "removedirs", "kill", "fork", "killpg",
                 "execv", "execve", "execl", "execlp", "execvp", "spawnl", "spawnv", "chmod", "chown", "rename"}
DENY_NAMES = {"eval", "exec", "compile", "__import__"}


def check_python_source(code: str) -> None:
    """Pemeriksaan AST sederhana (penjaga dari kecelakaan, mudah dilewati orang yang berniat jahat)."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in DENY_MODULES:
                    raise SecurityError(f"import '{a.name}' tidak diizinkan di run_python")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in DENY_MODULES:
                raise SecurityError(f"import dari '{node.module}' tidak diizinkan di run_python")
        elif isinstance(node, ast.Attribute) and node.attr in DENY_OS_ATTRS:
            raise SecurityError(f"pemakaian '.{node.attr}' tidak diizinkan di run_python")
        elif isinstance(node, ast.Name) and node.id in DENY_NAMES:
            raise SecurityError(f"pemakaian '{node.id}' tidak diizinkan di run_python")


class Workspace:
    def __init__(self, root, max_file_bytes: int = 1_000_000, cmd_timeout: int = 30, mem_limit_mb: int = 1024,
                 log_path=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes, self.cmd_timeout, self.mem_limit_mb = max_file_bytes, cmd_timeout, mem_limit_mb
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- util ----------
    def log(self, event: str, **data):
        if not self.log_path:
            return
        rec = {"ts": round(time.time(), 3), "event": event, **{k: (str(v)[:500]) for k, v in data.items()}}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def resolve(self, rel) -> Path:
        p = (self.root / str(rel)).resolve()  # resolve() mengikuti symlink -> dicek setelahnya
        if p != self.root and self.root not in p.parents:
            raise SecurityError(f"Path di luar workspace ditolak: {rel}")
        return p

    # ---------- baca ----------
    def list_dir(self, rel: str = ".") -> list:
        p = self.resolve(rel)
        return sorted(f"{c.name}/" if c.is_dir() else c.name for c in p.iterdir())

    def read_file(self, rel: str, max_bytes: int | None = None) -> str:
        p = self.resolve(rel)
        limit = min(max_bytes or self.max_file_bytes, self.max_file_bytes)
        data = p.read_bytes()[:limit + 1]
        text = data[:limit].decode("utf-8", errors="replace")
        return text + ("\n...[dipotong]" if len(data) > limit else "")

    def search_text(self, pattern: str, rel: str = ".", max_results: int = 50) -> list:
        rx = re.compile(pattern)
        out = []
        for f in sorted(self.resolve(rel).rglob("*")):
            if not f.is_file() or any(x.startswith(".") for x in f.relative_to(self.root).parts):
                continue
            if f.stat().st_size > self.max_file_bytes:
                continue
            try:
                for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        out.append(f"{f.relative_to(self.root)}:{n}: {line.strip()[:200]}")
                        if len(out) >= max_results:
                            return out
            except (UnicodeDecodeError, OSError):
                continue
        return out

    def inspect_project(self, rel: str = ".") -> dict:
        base = self.resolve(rel)
        exts, files, py_errors, todo = {}, 0, [], 0
        for f in base.rglob("*"):
            parts = f.relative_to(base).parts
            if not f.is_file() or any(x.startswith(".") or x in ("__pycache__", "node_modules", "venv") for x in parts):
                continue
            files += 1
            exts[f.suffix or "(tanpa ekstensi)"] = exts.get(f.suffix or "(tanpa ekstensi)", 0) + 1
            if f.suffix == ".py":
                try:
                    src = f.read_text(encoding="utf-8")
                    ast.parse(src)
                    todo += src.count("TODO")
                except SyntaxError as e:
                    py_errors.append(f"{f.relative_to(base)}:{e.lineno}: {e.msg}")
                except UnicodeDecodeError:
                    py_errors.append(f"{f.relative_to(base)}: bukan UTF-8")
        top = sorted(c.name + ("/" if c.is_dir() else "") for c in base.iterdir() if not c.name.startswith("."))
        return {"files": files, "by_extension": exts, "top_level": top, "python_syntax_errors": py_errors, "todo_count": todo}

    # ---------- tulis ----------
    def write_file(self, rel: str, content: str) -> str:
        if len(content.encode("utf-8")) > self.max_file_bytes:
            raise SecurityError("Konten melebihi batas ukuran file")
        p = self.resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Ditulis: {p.relative_to(self.root)} ({len(content)} karakter)"

    def mkdir(self, rel: str) -> str:
        self.resolve(rel).mkdir(parents=True, exist_ok=True)
        return f"Folder dibuat: {rel}"

    def move(self, src: str, dst: str) -> str:
        s, d = self.resolve(src), self.resolve(dst)
        if d.exists():
            raise SecurityError("Tujuan sudah ada; tidak menimpa")
        d.parent.mkdir(parents=True, exist_ok=True)
        s.rename(d)
        return f"Dipindah: {src} -> {dst}"

    # ---------- eksekusi ----------
    def _limits(self):
        if resource is None:
            return None
        mem = self.mem_limit_mb * 1024 * 1024
        cpu = self.cmd_timeout + 5

        def apply():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024,) * 2)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except (ValueError, OSError):
                pass
            os.setsid()
        return apply

    def _run(self, argv: list, timeout: int | None = None) -> dict:
        env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
        env.update(HOME=str(self.root), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        t = timeout or self.cmd_timeout
        try:
            r = subprocess.run(argv, cwd=self.root, env=env, capture_output=True, text=True, timeout=t,
                               stdin=subprocess.DEVNULL, preexec_fn=self._limits())
            res = {"ok": r.returncode == 0, "returncode": r.returncode, "stdout": r.stdout[-MAX_OUT:],
                   "stderr": r.stderr[-MAX_OUT:], "timed_out": False}
        except subprocess.TimeoutExpired as e:
            res = {"ok": False, "returncode": None, "stdout": (e.stdout or "")[-MAX_OUT:] if isinstance(e.stdout, str) else "",
                   "stderr": f"Timeout setelah {t} detik", "timed_out": True}
        return res

    def run_python(self, code: str, timeout: int | None = None) -> dict:
        check_python_source(code)
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=self.root, delete=False, encoding="utf-8") as f:
            f.write(code)
            name = f.name
        try:
            return self._run([sys.executable, "-I", name], timeout)
        finally:
            Path(name).unlink(missing_ok=True)

    # allowlist perintah -> flag yang diizinkan (None = tanpa flag)
    ALLOWED = {
        "ls": {"-l", "-a", "-la", "-lh", "-1", "-R", "-h"},
        "cat": {"-n"}, "head": {"-n"}, "tail": {"-n"}, "wc": {"-l", "-w", "-c"},
        "pwd": set(), "echo": set(), "grep": {"-n", "-i", "-r", "-R", "-E", "-l", "-c", "-v", "-w"},
        "python": None, "python3": None, "git": None,
    }
    GIT_SUB = {"status", "diff", "log", "show", "branch"}
    GIT_BAD = ("--output", "--ext-diff", "--exec", "--textconv", "-c", "-C", "--git-dir", "--work-tree", "--upload-pack")
    PY_MODULES = {"pytest", "py_compile", "compileall", "unittest"}

    def _check_path_arg(self, a: str):
        if a.startswith("~") or a.startswith("/") or ".." in Path(a).parts:
            self.resolve(a)  # melempar SecurityError bila di luar workspace

    def validate_command(self, command: str) -> list:
        if SHELL_META.search(command):
            raise SecurityError("Karakter shell (; & | ` $ < >) tidak diizinkan; tidak ada shell di sini")
        argv = shlex.split(command)
        if not argv:
            raise SecurityError("Perintah kosong")
        name = argv[0]
        if name not in self.ALLOWED:
            raise SecurityError(f"Perintah '{name}' tidak ada di allowlist: {sorted(self.ALLOWED)}")
        args = argv[1:]
        if name in ("python", "python3"):
            if not args:
                raise SecurityError("python tanpa argumen (REPL) tidak diizinkan")
            if args[0] == "-m":
                if len(args) < 2 or args[1] not in self.PY_MODULES:
                    raise SecurityError(f"Modul python yang boleh: {sorted(self.PY_MODULES)}")
            elif args[0].startswith("-") or not args[0].endswith(".py"):
                raise SecurityError("Hanya 'python skrip.py [arg]' atau 'python -m <modul-diizinkan>'")
            else:
                self.resolve(args[0])
            argv = [sys.executable, "-I"] + args if args[0] != "-m" else [sys.executable] + args
            for a in args[1:]:
                self._check_path_arg(a)
            return argv
        if name == "git":
            if not args or args[0] not in self.GIT_SUB:
                raise SecurityError(f"git hanya boleh: {sorted(self.GIT_SUB)}")
            if any(a == b or a.startswith(b + "=") for a in args for b in self.GIT_BAD):
                raise SecurityError("Argumen git berbahaya ditolak")
            for a in args[1:]:
                if not a.startswith("-"):
                    self._check_path_arg(a)
            return argv
        flags = self.ALLOWED[name]
        for a in args:
            if a.startswith("-") and not re.fullmatch(r"-\d+", a):
                if a not in flags:
                    raise SecurityError(f"Flag '{a}' tidak diizinkan untuk {name}")
            else:
                self._check_path_arg(a)
        return argv

    def run_command(self, command: str, timeout: int | None = None) -> dict:
        return self._run(self.validate_command(command), timeout)


# ---------------- deteksi error ----------------
_RULES = [
    (r"ModuleNotFoundError: No module named '?([\w.]+)'?", "modul_tidak_ada",
     "Modul '{0}' belum terpasang. Coba: pip install {0} (atau cek nama paketnya; di Termux beberapa paket butuh `pkg install`)."),
    (r"(SyntaxError|IndentationError|TabError): (.+)", "sintaks",
     "Kesalahan penulisan kode ({0}: {1}). Cek baris yang ditunjuk: kurung/kutip tak tertutup atau indentasi tidak konsisten."),
    (r"CUDA out of memory", "oom_gpu",
     "Memori GPU habis. Kecilkan --batch-size/--seq-len, naikkan --grad-accum, aktifkan --grad-checkpoint, atau pakai config lebih kecil."),
    (r"(MemoryError|Killed\s*$|Cannot allocate memory)", "oom_ram",
     "RAM habis (di Android proses sering langsung 'Killed'). Pakai config tiny, kecilkan batch/seq-len."),
    (r"No space left on device", "disk_penuh", "Penyimpanan penuh. Hapus checkpoint lama (--keep-last) atau cache pip."),
    (r"ImportError: cannot import name '([\w]+)'", "import_nama",
     "Nama '{0}' tidak ada di modul itu: salah ketik, atau versi paket berbeda."),
    (r"FileNotFoundError: .*'([^']+)'", "file_tidak_ada", "File/folder '{0}' tidak ditemukan. Cek path dan folder kerja (pwd)."),
    (r"PermissionError: .*'([^']+)'", "izin", "Tidak punya izin untuk '{0}'."),
    (r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)", "bentuk_tensor",
     "Ukuran tensor tidak cocok ({0} vs {1}). Periksa dimensi (num_heads, head_dim, seq_len) di config/kode."),
    (r"(401|403) Client Error|Invalid (user )?token|Repository Not Found", "hf_auth",
     "Akses Hugging Face ditolak: cek HF_TOKEN (izin write), nama repo, dan apakah repo sudah ada/milik Anda."),
    (r"externally-managed-environment", "pip_env",
     "pip menolak install global. Pakai virtualenv: python -m venv .venv && source .venv/bin/activate."),
    (r"AssertionError", "assert", "Ada assert yang gagal; baca baris yang ditunjuk traceback."),
]


def diagnose(text: str) -> dict:
    """Analisis teks error/traceback dengan aturan (bukan model) -> jenis, lokasi, saran."""
    loc = re.findall(r'File "([^"]+)", line (\d+)', text)
    location = f"{loc[-1][0]}:{loc[-1][1]}" if loc else None
    for pattern, kind, hint in _RULES:
        m = re.search(pattern, text, re.M)
        if m:
            return {"kind": kind, "location": location, "hint": hint.format(*m.groups())}
    return {"kind": "tidak_dikenal", "location": location,
            "hint": "Pola error belum dikenali. Baca baris terakhir traceback dan cari lokasi file:baris-nya."}


def check_python_file(ws: Workspace, rel: str) -> dict:
    """Cek sintaks tanpa menjalankan/menulis apa pun."""
    p = ws.resolve(rel)
    src = p.read_text(encoding="utf-8", errors="replace")
    try:
        ast.parse(src, filename=str(rel))
        return {"ok": True, "message": "Sintaks valid"}
    except SyntaxError as e:
        lines, n = src.splitlines(), e.lineno or 1
        ctx = [f"{i + 1:4d}| {lines[i]}" for i in range(max(0, n - 3), min(len(lines), n + 2))]
        return {"ok": False, "message": f"{type(e).__name__}: {e.msg} (baris {n})", "context": ctx,
                **diagnose(f"{type(e).__name__}: {e.msg}")}
