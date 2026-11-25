"""Agen Linux modular. Semua aksi (tulis/jalankan) WAJIB persetujuan pengguna,
termasuk aksi yang diminta oleh model. Model dari-nol Anda mungkin belum bisa
memanggil alat dengan benar; REPL tetap bisa dipakai manual lewat perintah /."""
from __future__ import annotations

import argparse
import json
import re
import time
from typing import Callable

from .tools import SecurityError, Workspace, check_python_file, diagnose

READ_TOOLS = {"list_dir", "read_file", "search_text", "inspect_project", "check_python", "diagnose"}
ACTION_TOOLS = {"write_file", "mkdir", "move", "run_python", "run_command"}
ARG_SPEC = {  # nama alat -> {argumen: tipe}
    "list_dir": {"rel": str}, "read_file": {"rel": str}, "search_text": {"pattern": str, "rel": str},
    "inspect_project": {"rel": str}, "check_python": {"rel": str}, "diagnose": {"text": str},
    "write_file": {"rel": str, "content": str}, "mkdir": {"rel": str}, "move": {"src": str, "dst": str},
    "run_python": {"code": str}, "run_command": {"command": str},
}
TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.S)

SYSTEM_PROMPT = (
    "Kamu agen Linux. Untuk memakai alat, tulis <tool>{\"name\": \"...\", \"args\": {...}}</tool>. "
    "Alat: " + ", ".join(sorted(ARG_SPEC)) + ". Jawab singkat.")


def parse_tool_call(text: str):
    """Ambil panggilan alat pertama dari output model; validasi nama & tipe argumen."""
    m = TOOL_RE.search(text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
        name, args = call["name"], call.get("args", {})
    except (json.JSONDecodeError, KeyError, TypeError):
        raise ValueError("Format <tool> tidak valid (harus JSON dengan 'name' dan 'args')")
    if name not in ARG_SPEC:
        raise ValueError(f"Alat '{name}' tidak dikenal")
    spec = ARG_SPEC[name]
    if not isinstance(args, dict) or set(args) - set(spec):
        raise ValueError(f"Argumen untuk {name} harus subset dari {sorted(spec)}")
    for k, v in args.items():
        if not isinstance(v, spec[k]):
            raise ValueError(f"Argumen '{k}' harus bertipe {spec[k].__name__}")
    return name, args


def make_plan(task: str) -> list:
    """Perencana berbasis aturan (deterministik, tidak memakai model)."""
    t = task.lower()
    steps = ["Periksa struktur proyek (/inspect) dan baca file yang relevan (/read, /grep)."]
    if any(k in t for k in ("bug", "error", "gagal", "fix", "perbaiki", "debug")):
        steps += ["Jalankan tes/skrip dan simpan pesan error lengkap.", "Analisis error (/diagnose) dan temukan baris penyebab.",
                  "Usulkan perbaikan minimal; terapkan (butuh persetujuan).", "Jalankan ulang tes untuk memastikan."]
    elif any(k in t for k in ("buat", "tambah", "create", "implement", "tulis")):
        steps += ["Tentukan nama file dan antarmuka fungsi.", "Tulis kode (butuh persetujuan untuk menyimpan).",
                  "Tulis tes kecil untuk fitur itu.", "Jalankan tes dan perbaiki kegagalan."]
    else:
        steps += ["Rumuskan tujuan menjadi langkah yang bisa diuji.", "Kerjakan satu langkah, verifikasi, lanjut."]
    return steps + ["Ringkas perubahan dan catat hal yang belum selesai."]


class Agent:
    def __init__(self, workspace: Workspace, confirm: Callable[[str], bool], backend: Callable[[str], str] | None = None,
                 memory=None, session_id: str = "agent", max_steps: int = 5):
        self.ws, self.confirm, self.backend = workspace, confirm, backend
        self.memory, self.session_id, self.max_steps = memory, session_id, max_steps

    # ---------- eksekusi alat ----------
    def execute(self, name: str, args: dict, source: str = "user"):
        """Satu-satunya pintu eksekusi. Kembalikan (ok, hasil)."""
        ws = self.ws
        try:
            if name not in ARG_SPEC:
                raise SecurityError(f"Alat tidak dikenal: {name}")
            if name == "run_command":
                ws.validate_command(args["command"])  # tolak perintah terlarang SEBELUM meminta persetujuan
            if name == "run_python":
                from .tools import check_python_source
                check_python_source(args["code"])
            if name in ACTION_TOOLS:
                desc = f"[{source}] {name} {json.dumps(args, ensure_ascii=False)[:300]}"
                approved = bool(self.confirm(desc))
                ws.log("approval", tool=name, args=args, approved=approved, source=source)
                if not approved:
                    return False, "Ditolak oleh pengguna"
            result = {
                "list_dir": lambda: ws.list_dir(args.get("rel", ".")),
                "read_file": lambda: ws.read_file(args["rel"]),
                "search_text": lambda: ws.search_text(args["pattern"], args.get("rel", ".")),
                "inspect_project": lambda: ws.inspect_project(args.get("rel", ".")),
                "check_python": lambda: check_python_file(ws, args["rel"]),
                "diagnose": lambda: diagnose(args["text"]),
                "write_file": lambda: ws.write_file(args["rel"], args["content"]),
                "mkdir": lambda: ws.mkdir(args["rel"]),
                "move": lambda: ws.move(args["src"], args["dst"]),
                "run_python": lambda: ws.run_python(args["code"]),
                "run_command": lambda: ws.run_command(args["command"]),
            }[name]()
            ws.log("tool", tool=name, args=args, ok=True, source=source)
            return True, result
        except (SecurityError, KeyError, OSError, SyntaxError, ValueError) as e:
            ws.log("tool", tool=name, args=args, ok=False, error=repr(e), source=source)
            return False, f"{type(e).__name__}: {e}"

    # ---------- loop berbasis model ----------
    def ask(self, user_text: str, token_budget: int = 1024) -> str:
        if self.backend is None:
            return "Model belum dimuat (pakai --model-dir). Gunakan perintah / untuk alat manual."
        if self.memory:
            prompt = self.memory.build_prompt(self.session_id, user_text, token_budget, system=SYSTEM_PROMPT)
            self.memory.add_message(self.session_id, "user", user_text)
        else:
            prompt = f"{SYSTEM_PROMPT}\nUser: {user_text}\nAssistant:"
        reply = ""
        for _ in range(self.max_steps):
            reply = self.backend(prompt)
            try:
                call = parse_tool_call(reply)
            except ValueError as e:
                reply = f"{reply}\n[agen] {e}"
                break
            if call is None:
                break
            ok, result = self.execute(call[0], call[1], source="model")
            obs = json.dumps(result, ensure_ascii=False, default=str)[:1500]
            prompt += f" {reply}\n[hasil {'ok' if ok else 'gagal'}] {obs}\nAssistant:"
        if self.memory:
            self.memory.add_message(self.session_id, "assistant", reply)
        return reply

    def generate_code(self, description: str, target: str | None = None) -> str:
        code = self.ask(f"Tulis kode Python untuk: {description}")
        if target:
            ok, res = self.execute("write_file", {"rel": target, "content": code}, source="codegen")
            return str(res)
        return code


HELP = """Perintah:
  /ls [path]          daftar isi          /read <file>       baca file
  /grep <regex>       cari teks           /inspect           ringkasan proyek + cek sintaks
  /check <file.py>    cek sintaks         /diagnose <teks>   analisis pesan error
  /write <file>       tulis file (isi diketik, akhiri dengan baris 'EOF')   [butuh persetujuan]
  /run <perintah>     perintah allowlist  [butuh persetujuan]  mis. /run python -m pytest -q
  /py <kode>          jalankan kode Python satu baris [butuh persetujuan]
  /plan <tugas>       rencana langkah     /note <teks>       simpan catatan memori
  /memory <kata>      cari di memori      /ask <teks>        tanya model (jika dimuat)
  /help  /quit"""


def repl(agent: Agent, memory=None, inp=input, out=print):
    out("Agen siap. Ketik /help. Workspace: " + str(agent.ws.root))
    while True:
        try:
            line = inp("agen> ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            return
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        if cmd in ("/quit", "/exit"):
            return
        if cmd == "/help":
            out(HELP); continue
        if cmd == "/plan":
            out("\n".join(f"{i}. {s}" for i, s in enumerate(make_plan(rest), 1))); continue
        if cmd == "/note" and memory:
            out(f"Catatan disimpan: {memory.add_note('catatan-' + str(int(time.time())), rest)}"); continue
        if cmd == "/memory" and memory:
            for r in memory.search(rest) or [{"source": "-", "text": "(tidak ada hasil)"}]:
                out(f"- {r['source']}: {r['text'][:200]}")
            continue
        if cmd == "/ask":
            out(agent.ask(rest)); continue
        if cmd == "/write":
            out(f"Ketik isi {rest}, akhiri dengan baris EOF:")
            lines = []
            while (l := inp("")) != "EOF":
                lines.append(l)
            name, args = "write_file", {"rel": rest, "content": "\n".join(lines) + "\n"}
        else:
            table = {"/ls": ("list_dir", {"rel": rest or "."}), "/read": ("read_file", {"rel": rest}),
                     "/grep": ("search_text", {"pattern": rest}), "/inspect": ("inspect_project", {"rel": rest or "."}),
                     "/check": ("check_python", {"rel": rest}), "/diagnose": ("diagnose", {"text": rest}),
                     "/run": ("run_command", {"command": rest}), "/py": ("run_python", {"code": rest})}
            if cmd not in table:
                out("Perintah tidak dikenal. /help"); continue
            name, args = table[cmd]
        ok, res = agent.execute(name, args)
        out(res if isinstance(res, str) else json.dumps(res, indent=2, ensure_ascii=False, default=str))


def main():
    ap = argparse.ArgumentParser(description="Agen Linux (workspace terbatas, allowlist, persetujuan)")
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--model-dir", default=None, help="folder checkpoint untuk /ask (opsional)")
    ap.add_argument("--memory-dir", default="memory_data")
    ap.add_argument("--log", default="logs/agent.jsonl")
    a = ap.parse_args()

    from memory.store import MemoryStore
    memory = MemoryStore(a.memory_dir)
    backend = None
    if a.model_dir:
        from core.generate import generate, load_for_inference
        model, tok, _ = load_for_inference(a.model_dir)
        backend = lambda p: generate(model, tok, p, max_new_tokens=128)  # noqa: E731

    def confirm(desc: str) -> bool:
        return input(f"IZINKAN? {desc}\n  ketik 'y' untuk setuju: ").strip().lower() == "y"

    agent = Agent(Workspace(a.workspace, log_path=a.log), confirm, backend, memory)
    repl(agent, memory)


if __name__ == "__main__":
    main()
