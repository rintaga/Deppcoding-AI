import json
from core.tokenizer import ByteBPETokenizer, BOS_ID, EOS_ID, PAD_ID


def test_roundtrip_unicode_and_whitespace(corpus):
    tok = ByteBPETokenizer.load(corpus[1])
    for s in ["halo dunia", "  spasi   ganda\n\nbaris", "é 日本語 😀", "", "a"]:
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress_and_ids_in_range(corpus):
    tok = ByteBPETokenizer.load(corpus[1])
    s = "halo dunia ini tes model bahasa"
    ids = tok.encode(s)
    assert len(ids) < len(s.encode())
    assert max(ids) < tok.vocab_size and not {PAD_ID, BOS_ID, EOS_ID} & set(ids)


def test_special_tokens_and_save_load(tmp_path, corpus):
    tok = ByteBPETokenizer.load(corpus[1])
    ids = tok.encode("halo", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID and tok.decode(ids) == "halo"
    tok.save(tmp_path / "t.json")
    assert ByteBPETokenizer.load(tmp_path / "t.json").encode("halo dunia") == tok.encode("halo dunia")


def test_loads_legacy_format(tmp_path):
    legacy = {"vocab_bytes": [], "merges": [[104, 97]], "special_tokens": {}}
    (tmp_path / "old.json").write_text(json.dumps(legacy))
    tok = ByteBPETokenizer.load(tmp_path / "old.json")
    assert tok.encode("ha") == [259]


def test_training_is_deterministic():
    text = "abab abab cdcd cdcd abcd " * 50
    assert ByteBPETokenizer.train(text, 300).merges == ByteBPETokenizer.train(text, 300).merges
