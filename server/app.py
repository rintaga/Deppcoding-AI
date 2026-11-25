"""Server HTTP (stdlib saja, ramah Termux). Jalankan: python -m server.app --config configs/server.json"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from memory.store import MemoryFull, MemoryStore

from .engine import Engine, ModelRegistry, ModelUnavailable, QueueFull

MAX_BODY = 1_000_000


class ApiError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _num(body, key, default, typ, lo, hi):
    v = body.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ApiError(400, f"'{key}' harus berupa angka")
    v = typ(v)
    if not lo <= v <= hi:
        raise ApiError(400, f"'{key}' harus di antara {lo} dan {hi}")
    return v


def make_handler(engine: Engine, memory: MemoryStore, cfg: dict, api_key: str | None, logger):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DualAI/1.0"

        def log_message(self, fmt, *args):  # arahkan ke logger, bukan stderr
            logger.info("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, payload: dict):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ApiError(413, "Body terlalu besar")
            if n == 0:
                return {}
            try:
                body = json.loads(self.rfile.read(n))
            except json.JSONDecodeError:
                raise ApiError(400, "Body bukan JSON valid")
            if not isinstance(body, dict):
                raise ApiError(400, "Body harus objek JSON")
            return body

        def _auth(self):
            if api_key and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {api_key}"):
                raise ApiError(401, "API key salah/kurang (header Authorization: Bearer ...)")

        def _dispatch(self, method: str):
            try:
                url = urlparse(self.path)
                path, qs = url.path.rstrip("/") or "/", parse_qs(url.query)
                if path != "/health":
                    self._auth()
                body = self._body() if method in ("POST", "DELETE") else {}
                code, payload = self.route(method, path, qs, body)
                self._send(code, payload)
            except ApiError as e:
                self._send(e.code, {"error": {"code": e.code, "message": e.message}})
            except QueueFull as e:
                self._send(429, {"error": {"code": 429, "message": str(e)}})
            except ModelUnavailable as e:
                self._send(409, {"error": {"code": 409, "message": str(e)}})
            except MemoryFull as e:
                self._send(507, {"error": {"code": 507, "message": str(e)}})
            except Exception:  # noqa: BLE001
                logger.exception("error tak terduga")
                self._send(500, {"error": {"code": 500, "message": "Kesalahan internal server (lihat log)"}})

        do_GET = lambda self: self._dispatch("GET")  # noqa: E731
        do_POST = lambda self: self._dispatch("POST")  # noqa: E731
        do_DELETE = lambda self: self._dispatch("DELETE")  # noqa: E731

        def route(self, method, path, qs, body):
            if (method, path) == ("GET", "/health"):
                return 200, {"status": "ok", **engine.stats()}
            if (method, path) == ("GET", "/v1/models"):
                return 200, {"models": engine.registry.info()}
            if (method, path) == ("GET", "/v1/checkpoints"):
                name = (qs.get("model") or [""])[0]
                if name not in engine.registry.models:
                    raise ApiError(400, f"parameter model harus salah satu dari {engine.registry.names()}")
                return 200, {"model": name, "checkpoints": engine.registry.checkpoints(name)}
            if (method, path) == ("POST", "/v1/checkpoints/select"):
                try:
                    engine.registry.select_checkpoint(str(body.get("model")), str(body.get("path", "")))
                except KeyError:
                    raise ApiError(400, "model tidak dikenal")
                return 200, {"ok": True}
            if (method, path) == ("POST", "/v1/sessions"):
                sid = str(body.get("id") or os.urandom(6).hex())
                if not re.fullmatch(r"[\w.-]{1,64}", sid):
                    raise ApiError(400, "id sesi hanya boleh huruf/angka/._- (maks 64)")
                return 201, {"session_id": sid}
            if (method, path) == ("GET", "/v1/sessions"):
                return 200, {"sessions": memory.sessions()}
            m = re.fullmatch(r"/v1/sessions/([\w.-]{1,64})(/history)?", path)
            if m and method == "GET" and m.group(2):
                return 200, {"session_id": m.group(1), "messages": memory.history(m.group(1))}
            if m and method == "DELETE" and not m.group(2):
                memory.clear_session(m.group(1))
                return 200, {"ok": True}
            if (method, path) == ("POST", "/v1/generate"):
                return self.generate(body)
            m = re.fullmatch(r"/v1/jobs/(\w+)", path)
            if m and method == "GET":
                job = engine.get_job(m.group(1))
                if not job:
                    raise ApiError(404, "job tidak ditemukan")
                return 200, job.to_dict()
            if (method, path) == ("POST", "/v1/memory/documents"):
                src, text = body.get("source"), body.get("text")
                if not isinstance(src, str) or not isinstance(text, str) or not src or not text:
                    raise ApiError(400, "'source' dan 'text' (string) wajib")
                return 201, {"chunks": memory.add_document(src, text)}
            if (method, path) == ("GET", "/v1/memory/search"):
                q = (qs.get("q") or [""])[0]
                k = max(1, min(20, int((qs.get("k") or ["4"])[0])))
                return 200, {"results": memory.search(q, k)}
            if (method, path) == ("GET", "/v1/memory/stats"):
                return 200, memory.stats()
            raise ApiError(404, f"Rute tidak ada: {method} {path}")

        def generate(self, body):
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ApiError(400, "'prompt' (string) wajib")
            model = body.get("model", "tiny")
            if model not in engine.registry.models:
                raise ApiError(400, f"model harus salah satu dari {engine.registry.names()}")
            sid = body.get("session_id")
            if sid is not None and not re.fullmatch(r"[\w.-]{1,64}", str(sid)):
                raise ApiError(400, "session_id tidak valid")
            params = {"model": model, "prompt": prompt, "session_id": sid,
                      "max_new_tokens": _num(body, "max_new_tokens", 64, int, 1, engine.cap),
                      "temperature": _num(body, "temperature", 0.8, float, 0.0, 5.0),
                      "top_k": _num(body, "top_k", 40, int, 0, 100000),
                      "top_p": _num(body, "top_p", 0.95, float, 0.01, 1.0),
                      "use_memory": bool(body.get("use_memory", False)),
                      "seed": body.get("seed") if isinstance(body.get("seed"), int) else None}
            job = engine.submit(params)
            if body.get("async"):
                return 202, job.to_dict()
            if not job.event.wait(cfg.get("request_timeout_s", 300)):
                return 202, {**job.to_dict(), "note": "Belum selesai; cek GET /v1/jobs/{id}"}
            if job.status == "error":
                raise ApiError(500, job.error or "gagal")
            return 200, job.to_dict()

    return Handler


def build_server(cfg: dict, host=None, port=None):
    log_file = cfg.get("log_file", "logs/server.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dualai.server")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"))
        logger.addHandler(logging.StreamHandler())
    memory = MemoryStore(cfg.get("memory_dir", "memory_data"), cfg.get("memory_max_mb", 512))
    registry = ModelRegistry(cfg["models"], cfg.get("max_loaded_models", 1), cfg.get("device", "cpu"))
    engine = Engine(registry, memory, cfg.get("workers", 1), cfg.get("max_queue", 16), logger=logger)
    api_key = os.environ.get("SERVER_API_KEY") or None
    handler = make_handler(engine, memory, cfg, api_key, logger)
    httpd = ThreadingHTTPServer((host or cfg.get("host", "127.0.0.1"), cfg.get("port", 8000) if port is None else port), handler)
    httpd.daemon_threads = True
    httpd.engine, httpd.memory = engine, memory
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/server.json")
    ap.add_argument("--host"); ap.add_argument("--port", type=int)
    ap.add_argument("--workers", type=int); ap.add_argument("--device")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.workers:
        cfg["workers"] = a.workers
    if a.device:
        cfg["device"] = a.device
    httpd = build_server(cfg, a.host, a.port)
    host, port = httpd.server_address[:2]
    if host not in ("127.0.0.1", "localhost") and not os.environ.get("SERVER_API_KEY"):
        print("PERINGATAN: server terbuka ke jaringan tanpa SERVER_API_KEY. Set SERVER_API_KEY dulu.")
    print(f"Server di http://{host}:{port}  (Ctrl+C untuk berhenti)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.engine.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
