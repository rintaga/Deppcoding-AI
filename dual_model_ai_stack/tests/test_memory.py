import pytest
from memory.store import MemoryFull, MemoryStore


def test_history_search_persist(tmp_path):
    m = MemoryStore(tmp_path)
    m.add_message("s", "user", "halo")
    m.add_message("s", "assistant", "hai")
    m.add_document("doc1", "Termux adalah terminal Android. Python berjalan di Termux.")
    m.add_document("doc2", "Resep nasi goreng dengan telur dan kecap.")
    m.close()
    m2 = MemoryStore(tmp_path)  # persisten setelah ditutup
    assert [x["content"] for x in m2.history("s")] == ["halo", "hai"]
    assert m2.search("termux android")[0]["source"] == "doc1"
    assert m2.search("kata yang tidak ada") == []
    assert m2.add_document("doc1", "Termux adalah terminal Android. Python berjalan di Termux.") == 0  # tak berubah


def test_prompt_respects_budget(tmp_path):
    m = MemoryStore(tmp_path)
    for i in range(50):
        m.add_message("s", "user", f"pesan panjang nomor {i} " * 5)
    p = m.build_prompt("s", "tanya", token_budget=100)
    assert len(p) // 4 <= 110 and p.endswith("User: tanya\nAssistant:")


def test_quota_enforced(tmp_path):
    m = MemoryStore(tmp_path, max_mb=0.01)
    with pytest.raises(MemoryFull):
        m.add_document("big", "x " * 50_000)


def test_index_directory_and_notes(tmp_path):
    src = tmp_path / "proj"; src.mkdir()
    (src / "a.py").write_text("def hitung_gaji(): pass")
    (src / "b.bin").write_bytes(b"\x00\x01")
    m = MemoryStore(tmp_path / "mem")
    assert m.index_directory(src)["indexed"] == 1
    m.add_note("ide", "ingat: pakai gradient checkpointing")
    assert m.search("hitung_gaji")[0]["source"].startswith("file:")
    assert m.search("gradient")[0]["source"] == "note:ide"
