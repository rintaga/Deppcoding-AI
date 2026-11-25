import pytest
from hf_sync import hub


def test_token_required_and_not_hardcoded(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        hub.get_token()
    import pathlib, re
    for f in pathlib.Path(".").rglob("*.py"):
        assert not re.search(r"hf_[A-Za-z0-9]{30,}", f.read_text(encoding="utf-8")), f


def test_upload_sends_only_weights_by_default(tmp_path, tiny_ckpt, monkeypatch):
    sent = {}

    class FakeApi:
        def create_repo(self, *a, **k): sent["repo"] = (a, k)
        def upload_folder(self, **k):
            import os
            sent["files"] = sorted(os.listdir(k["folder_path"]))

    monkeypatch.setattr(hub, "_api", lambda: FakeApi())
    hub.upload_checkpoint(tiny_ckpt[1], "user/tiny")
    assert "trainer_state.pt" not in sent["files"] and "model.safetensors" in sent["files"] and "README.md" in sent["files"]
    assert sent["repo"][1]["private"] is True
    sent.clear()
    hub.upload_checkpoint(tiny_ckpt[1], "user/tiny", include_optimizer=True)
    assert "trainer_state.pt" in sent["files"]
