import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ModelConfig  # noqa: E402
from core.tokenizer import ByteBPETokenizer  # noqa: E402
from core.train_utils import run_training  # noqa: E402

TINY = dict(model_name="tiny-test", vocab_size=512, hidden_size=64, num_layers=2, num_heads=4,
            num_kv_heads=2, intermediate_size=128, context_length=64, window_size=32)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    rnd = random.Random(0)
    words = "halo dunia ini tes model bahasa kecil belajar python linux termux github".split()
    f = d / "train.txt"
    f.write_text("\n".join(" ".join(rnd.choice(words) for _ in range(12)) for _ in range(600)), encoding="utf-8")
    tok = ByteBPETokenizer.train(f.read_text(encoding="utf-8"), 400)
    tok.save(d / "tokenizer.json")
    return f, d / "tokenizer.json"


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory, corpus):
    f, tokp = corpus
    out = tmp_path_factory.mktemp("ckpt")
    res = run_training(ModelConfig(**TINY), f, tokp, out, steps=4, batch_size=2, seq_len=32,
                       save_every=0, log_every=100, log=lambda *_: None)
    return out, Path(res["checkpoint"])
