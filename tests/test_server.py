import json
import threading
import time
import urllib.error
import urllib.request

import pytest
from server.app import build_server
from server.engine import Engine, ModelRegistry, QueueFull
from memory.store import MemoryStore


def call(base, method, path, body=None, key=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def srv(tmp_path, tiny_ckpt, monkeypatch):
    monkeypatch.delenv("SERVER_API_KEY", raising=False)
    cfg = {"port": 0, "workers": 1, "max_queue": 4, "memory_dir": str(tmp_path / "mem"), "log_file": str(tmp_path / "s.log"),
           "models": {"tiny": {"config": "configs/tiny.json", "checkpoint_dir": str(tiny_ckpt[0])},
                      "3b": {"config": "configs/3b.json", "checkpoint_dir": str(tmp_path / "none")}}}
    httpd = build_server(cfg)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", httpd
    httpd.shutdown(); httpd.engine.shutdown(); httpd.server_close(); httpd.memory.close()


def test_health_models_generate_sessions_memory(srv):
    base, _ = srv
    assert call(base, "GET", "/health")[1]["status"] == "ok"
    models = call(base, "GET", "/v1/models")[1]["models"]
    assert {m["name"] for m in models} == {"tiny", "3b"}
    code, sess = call(base, "POST", "/v1/sessions", {"id": "s1"})
    assert code == 201
    code, r = call(base, "POST", "/v1/generate", {"model": "tiny", "prompt": "halo", "max_new_tokens": 5, "session_id": "s1", "temperature": 0})
    assert code == 200 and r["status"] == "done" and isinstance(r["result"]["text"], str)
    hist = call(base, "GET", "/v1/sessions/s1/history")[1]["messages"]
    assert [m["role"] for m in hist] == ["user", "assistant"]
    call(base, "POST", "/v1/memory/documents", {"source": "d", "text": "termux android linux"})
    assert call(base, "GET", "/v1/memory/search?q=termux")[1]["results"][0]["source"] == "d"
    code, r = call(base, "POST", "/v1/generate", {"model": "tiny", "prompt": "halo", "max_new_tokens": 3, "async": True})
    assert code == 202
    for _ in range(100):
        j = call(base, "GET", f"/v1/jobs/{r['id']}")[1]
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "done"


def test_errors(srv):
    base, _ = srv
    assert call(base, "POST", "/v1/generate", {"model": "tiny"})[0] == 400
    assert call(base, "POST", "/v1/generate", {"model": "xx", "prompt": "a"})[0] == 400
    assert call(base, "POST", "/v1/generate", {"model": "tiny", "prompt": "a", "temperature": "panas"})[0] == 400
    assert call(base, "POST", "/v1/generate", {"model": "tiny", "prompt": "a", "max_new_tokens": 99999})[0] == 400
    code, r = call(base, "POST", "/v1/generate", {"model": "3b", "prompt": "a"})
    assert code == 409 and "checkpoint" in r["error"]["message"].lower()
    assert call(base, "GET", "/nope")[0] == 404
    assert call(base, "POST", "/v1/sessions", {"id": "../x"})[0] == 400


def test_checkpoint_select_blocks_escape(srv):
    base, _ = srv
    names = call(base, "GET", "/v1/checkpoints?model=tiny")[1]["checkpoints"]
    assert call(base, "POST", "/v1/checkpoints/select", {"model": "tiny", "path": names[0]})[0] == 200
    assert call(base, "POST", "/v1/checkpoints/select", {"model": "tiny", "path": "/etc"})[0] == 409
    assert call(base, "POST", "/v1/checkpoints/select", {"model": "tiny", "path": "../.."})[0] == 409


def test_api_key_enforced(tmp_path, tiny_ckpt, monkeypatch):
    monkeypatch.setenv("SERVER_API_KEY", "rahasia")
    cfg = {"port": 0, "memory_dir": str(tmp_path / "m"), "log_file": str(tmp_path / "l.log"),
           "models": {"tiny": {"config": "configs/tiny.json", "checkpoint_dir": str(tiny_ckpt[0])}}}
    httpd = build_server(cfg)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert call(base, "GET", "/v1/models")[0] == 401
        assert call(base, "GET", "/v1/models", key="salah")[0] == 401
        assert call(base, "GET", "/v1/models", key="rahasia")[0] == 200
        assert call(base, "GET", "/health")[0] == 200
    finally:
        httpd.shutdown(); httpd.engine.shutdown(); httpd.server_close(); httpd.memory.close()


def test_queue_limit(tmp_path):
    mem = MemoryStore(tmp_path)
    eng = Engine(ModelRegistry({}), mem, workers=1, max_queue=1)
    gate = threading.Event()
    eng._run = lambda p: gate.wait(5) or {}
    eng.submit({"x": 1}); time.sleep(0.3)  # worker mengambil job 1
    eng.submit({"x": 2})                    # mengisi antrean
    with pytest.raises(QueueFull):
        eng.submit({"x": 3})
    gate.set(); eng.shutdown()


def test_concurrent_requests_all_served(srv):
    base, _ = srv
    out = []
    def go(i):
        out.append(call(base, "POST", "/v1/generate", {"model": "tiny", "prompt": f"tes {i}", "max_new_tokens": 3, "temperature": 0})[0])
    ts = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sorted(out) == [200] * 4
