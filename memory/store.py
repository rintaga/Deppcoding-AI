"""Memori eksternal: riwayat percakapan, catatan/dokumen berbasis file, pencarian BM25.

Penyimpanan persisten di SQLite (stdlib). Ini BUKAN konteks tak terbatas: model tetap
hanya melihat sejumlah token terbatas per permintaan; memori eksternal hanya menyimpan
lebih banyak dan mengambil potongan yang relevan ke dalam prompt, selama kuota disk cukup.
"""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"\w+", re.UNICODE)
TEXT_EXTS = {".txt", ".md", ".py", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh", ".rst", ".csv"}


class MemoryFull(RuntimeError):
    pass


def _terms(text: str):
    return [w.lower() for w in _WORD.findall(text)]


class MemoryStore:
    def __init__(self, root, max_mb: float = 512, chunk_chars: int = 800, chunk_overlap: int = 100):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "notes").mkdir(exist_ok=True)
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.chunk_chars, self.chunk_overlap = chunk_chars, chunk_overlap
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.root / "memory.db", check_same_thread=False)
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, ts REAL);
            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS sources(source TEXT PRIMARY KEY, kind TEXT, hash TEXT, ts REAL);
            CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, source TEXT, idx INTEGER, text TEXT);
            CREATE INDEX IF NOT EXISTS idx_chunk_source ON chunks(source);
        """)
        self._db.commit()
        self._index = None  # dibangun lazily

    # ---------- kuota ----------
    def usage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def _check_quota(self, extra: int):
        if self.usage_bytes() + extra > self.max_bytes:
            raise MemoryFull(f"Kuota memori {self.max_bytes // (1024 * 1024)} MB penuh; hapus data lama.")

    # ---------- percakapan ----------
    def add_message(self, session_id: str, role: str, content: str):
        with self._lock:
            self._check_quota(len(content.encode("utf-8")) + 200)
            self._db.execute("INSERT INTO messages(session_id, role, content, ts) VALUES(?,?,?,?)",
                             (session_id, role, content, time.time()))
            self._db.commit()

    def history(self, session_id: str, limit: int | None = None):
        with self._lock:
            rows = self._db.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
                                    (session_id,)).fetchall()
        rows = [{"role": r, "content": c} for r, c in rows]
        return rows[-limit:] if limit else rows

    def sessions(self):
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT DISTINCT session_id FROM messages ORDER BY session_id")]

    def clear_session(self, session_id: str):
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self._db.commit()

    # ---------- dokumen & catatan ----------
    def _chunk(self, text: str):
        step = max(1, self.chunk_chars - self.chunk_overlap)
        return [text[i:i + self.chunk_chars] for i in range(0, max(1, len(text)), step) if text[i:i + self.chunk_chars].strip()]

    def add_document(self, source: str, text: str, kind: str = "document") -> int:
        """Tambah/perbarui dokumen. Mengembalikan jumlah chunk (0 jika tidak berubah)."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            row = self._db.execute("SELECT hash FROM sources WHERE source=?", (source,)).fetchone()
            if row and row[0] == digest:
                return 0
            self._check_quota(len(text.encode("utf-8")) * 2)
            self._db.execute("DELETE FROM chunks WHERE source=?", (source,))
            chunks = self._chunk(text)
            self._db.executemany("INSERT INTO chunks(source, idx, text) VALUES(?,?,?)",
                                 [(source, i, c) for i, c in enumerate(chunks)])
            self._db.execute("INSERT OR REPLACE INTO sources(source, kind, hash, ts) VALUES(?,?,?,?)",
                             (source, kind, digest, time.time()))
            self._db.commit()
            self._index = None
            return len(chunks)

    def add_note(self, name: str, text: str) -> Path:
        safe = re.sub(r"[^\w.-]+", "_", name).strip("._") or "catatan"
        path = self.root / "notes" / f"{safe}.md"
        self._check_quota(len(text.encode("utf-8")) * 3)
        path.write_text(text, encoding="utf-8")
        self.add_document(f"note:{safe}", text, kind="note")
        return path

    def index_directory(self, directory, exts=None, max_file_kb: int = 512) -> dict:
        exts = set(exts or TEXT_EXTS)
        added = skipped = 0
        for p in sorted(Path(directory).rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts or any(x.startswith(".") for x in p.parts):
                continue
            if p.stat().st_size > max_file_kb * 1024:
                skipped += 1
                continue
            try:
                self.add_document(f"file:{p}", p.read_text(encoding="utf-8"), kind="file")
                added += 1
            except (UnicodeDecodeError, OSError):
                skipped += 1
        return {"indexed": added, "skipped": skipped}

    def remove_source(self, source: str):
        with self._lock:
            self._db.execute("DELETE FROM chunks WHERE source=?", (source,))
            self._db.execute("DELETE FROM sources WHERE source=?", (source,))
            self._db.commit()
            self._index = None

    # ---------- pencarian BM25 ----------
    def _build_index(self):
        rows = self._db.execute("SELECT id, source, text FROM chunks").fetchall()
        docs, df, total = {}, Counter(), 0
        for cid, src, text in rows:
            tf = Counter(_terms(text))
            length = sum(tf.values())
            docs[cid] = (src, text, tf, length)
            total += length
            df.update(tf.keys())
        self._index = {"docs": docs, "df": df, "avg": (total / len(docs)) if docs else 0.0}

    def search(self, query: str, k: int = 4, k1: float = 1.5, b: float = 0.75):
        with self._lock:
            if self._index is None:
                self._build_index()
            idx = self._index
        docs, df, avg, n = idx["docs"], idx["df"], idx["avg"], len(idx["docs"])
        q = _terms(query)
        scored = []
        for cid, (src, text, tf, length) in docs.items():
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / (avg or 1)))
            if s > 0:
                scored.append((s, src, text))
        scored.sort(key=lambda x: -x[0])
        return [{"score": round(s, 4), "source": src, "text": t} for s, src, t in scored[:k]]

    # ---------- penyusun prompt ----------
    def build_prompt(self, session_id: str, user_message: str, token_budget: int = 1024, count_tokens=None,
                     retrieve_k: int = 3, system: str = "") -> str:
        """Susun prompt yang muat dalam `token_budget` token (konteks aktif tetap terbatas)."""
        count = count_tokens or (lambda s: max(1, len(s) // 4))
        retrieved = self.search(user_message, retrieve_k) if retrieve_k else []
        head = (system + "\n" if system else "")
        if retrieved:
            head += "[Konteks relevan]\n" + "\n".join(f"- ({r['source']}) {r['text']}" for r in retrieved) + "\n"
        tail = f"User: {user_message}\nAssistant:"
        while retrieved and count(head + tail) > token_budget * 0.5:  # jatah konteks maks 50%
            retrieved.pop()
            head = (system + "\n" if system else "")
            if retrieved:
                head += "[Konteks relevan]\n" + "\n".join(f"- ({r['source']}) {r['text']}" for r in retrieved) + "\n"
        turns = []
        used = count(head + tail)
        for m in reversed(self.history(session_id)):
            line = f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n"
            if used + count(line) > token_budget:
                break
            turns.append(line)
            used += count(line)
        return head + "".join(reversed(turns)) + tail

    def stats(self) -> dict:
        with self._lock:
            m = self._db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            c = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            s = self._db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        return {"messages": m, "chunks": c, "sources": s, "usage_mb": round(self.usage_bytes() / 1048576, 3),
                "quota_mb": self.max_bytes // 1048576}

    def close(self):
        with self._lock:
            self._db.close()
