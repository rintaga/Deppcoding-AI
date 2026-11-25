"""Inti server: registry model (1.5B / 3B / tiny), antrean permintaan, worker, sesi."""
from __future__ import annotations

import queue
import threading
import time
import uuid
from pathlib import Path

from core.train_utils import find_latest_checkpoint


class QueueFull(Exception):
    pass


class ModelUnavailable(Exception):
    pass


class _Loaded:
    def __init__(self, model, tok, cfg, path):
        self.model, self.tok, self.cfg, self.path = model, tok, cfg, path
        self.lock = threading.Lock()  # satu generate per model pada satu waktu
        self.last_used = time.time()


class ModelRegistry:
    def __init__(self, models: dict, max_loaded: int = 1, device: str = "cpu", dtype=None):
        self.models, self.max_loaded, self.device, self.dtype = models, max_loaded, device, dtype
        self._loaded: dict[str, _Loaded] = {}
        self._selected: dict[str, str] = {}
        self._lock = threading.Lock()

    def names(self):
        return list(self.models)

    def checkpoints(self, name: str):
        d = Path(self.models[name]["checkpoint_dir"])
        return sorted(p.name for p in d.glob("step_*") if (p / "model.safetensors").exists()) if d.exists() else []

    def select_checkpoint(self, name: str, path: str):
        if name not in self.models:
            raise KeyError(name)
        root = Path(self.models[name]["checkpoint_dir"]).resolve()
        target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if root not in target.parents or not (target / "model.safetensors").exists():
            raise ModelUnavailable("Checkpoint harus berada di dalam checkpoint_dir model dan berisi model.safetensors")
        with self._lock:
            self._selected[name] = str(target)
            self._loaded.pop(name, None)

    def info(self):
        out = []
        for n, m in self.models.items():
            out.append({"name": n, "config": m["config"], "loaded": n in self._loaded,
                        "checkpoints": self.checkpoints(n), "selected": self._selected.get(n)})
        return out

    def ensure_available(self, name: str):
        """Cek cepat (tanpa memuat model): nama dikenal & ada checkpoint."""
        if name not in self.models:
            raise ModelUnavailable(f"Model '{name}' tidak dikenal. Pilihan: {self.names()}")
        if not (self._selected.get(name) or find_latest_checkpoint(self.models[name]["checkpoint_dir"])):
            raise ModelUnavailable(f"Belum ada checkpoint untuk '{name}' di {self.models[name]['checkpoint_dir']}. "
                                   "Latih dulu atau unduh dari Hugging Face.")

    def get(self, name: str) -> _Loaded:
        if name not in self.models:
            raise ModelUnavailable(f"Model '{name}' tidak dikenal. Pilihan: {self.names()}")
        with self._lock:
            if name in self._loaded:
                self._loaded[name].last_used = time.time()
                return self._loaded[name]
            ckpt = self._selected.get(name) or find_latest_checkpoint(self.models[name]["checkpoint_dir"])
            if not ckpt:
                raise ModelUnavailable(f"Belum ada checkpoint untuk '{name}' di {self.models[name]['checkpoint_dir']}. "
                                       "Latih dulu atau unduh dari Hugging Face.")
            while len(self._loaded) >= self.max_loaded:  # buang yang paling lama tidak dipakai
                idle = [k for k, v in self._loaded.items() if not v.lock.locked()]
                if not idle:
                    raise ModelUnavailable("Semua slot model sedang dipakai; coba lagi sebentar.")
                del self._loaded[min(idle, key=lambda k: self._loaded[k].last_used)]
            from core.generate import load_for_inference
            model, tok, cfg = load_for_inference(ckpt, device=self.device, dtype=self.dtype)
            self._loaded[name] = _Loaded(model, tok, cfg, str(ckpt))
            return self._loaded[name]


class Job:
    def __init__(self, params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.params, self.status, self.result, self.error = params, "queued", None, None
        self.created, self.event = time.time(), threading.Event()

    def to_dict(self):
        return {"id": self.id, "status": self.status, "result": self.result, "error": self.error,
                "age_s": round(time.time() - self.created, 2)}


class Engine:
    def __init__(self, registry: ModelRegistry, memory, workers: int = 1, max_queue: int = 16,
                 max_new_tokens_cap: int = 512, logger=None):
        self.registry, self.memory, self.cap, self.log = registry, memory, max_new_tokens_cap, logger
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._stop = threading.Event()
        self.threads = [threading.Thread(target=self._worker, daemon=True, name=f"worker-{i}") for i in range(max(1, workers))]
        for t in self.threads:
            t.start()

    def submit(self, params: dict) -> Job:
        if "model" in params:
            self.registry.ensure_available(params["model"])
        job = Job(params)
        try:
            self.q.put_nowait(job)
        except queue.Full:
            raise QueueFull("Antrean penuh; coba lagi nanti")
        with self._jobs_lock:
            self.jobs[job.id] = job
            if len(self.jobs) > 1000:  # jaga memori: buang job selesai paling lama
                for k in [k for k, j in sorted(self.jobs.items(), key=lambda kv: kv[1].created) if j.status in ("done", "error")][:200]:
                    self.jobs.pop(k, None)
        return job

    def _worker(self):
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            job.status = "running"
            try:
                job.result = self._run(job.params)
                job.status = "done"
            except Exception as e:  # noqa: BLE001 - dilaporkan ke klien, server tetap hidup
                job.status, job.error = "error", f"{type(e).__name__}: {e}"
                if self.log:
                    self.log.error("job %s gagal: %s", job.id, job.error)
            finally:
                job.event.set()

    def _run(self, p: dict) -> dict:
        from core.generate import generate
        m = self.registry.get(p["model"])
        n_new = min(int(p["max_new_tokens"]), self.cap, m.cfg.context_length - 1)
        sid, prompt = p.get("session_id"), p["prompt"]
        if p.get("use_memory") and sid:
            budget = max(16, m.cfg.context_length - n_new)
            prompt = self.memory.build_prompt(sid, p["prompt"], budget, count_tokens=lambda s: len(m.tok.encode(s)))
        t0 = time.time()
        with m.lock:
            text = generate(m.model, m.tok, prompt, n_new, p["temperature"], p["top_k"], p["top_p"], p.get("seed"))
        if sid:
            self.memory.add_message(sid, "user", p["prompt"])
            self.memory.add_message(sid, "assistant", text)
        return {"model": p["model"], "checkpoint": m.path, "text": text, "seconds": round(time.time() - t0, 3)}

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)

    def stats(self):
        return {"queue_size": self.q.qsize(), "queue_max": self.q.maxsize, "workers": len(self.threads),
                "loaded_models": [k for k in self.registry._loaded], "jobs_tracked": len(self.jobs)}

    def shutdown(self):
        self._stop.set()
