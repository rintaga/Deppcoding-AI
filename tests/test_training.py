import torch
from core.config import ModelConfig
from core.data import BlockSampler, load_token_array, tokenize_files
from core.tokenizer import ByteBPETokenizer
from core.train_utils import run_training
from tests.conftest import TINY

QUIET = dict(log=lambda *_: None, log_every=1000)


def test_loss_decreases(tmp_path, corpus):
    f, tokp = corpus
    r1 = run_training(ModelConfig(**TINY), f, tokp, tmp_path / "a", steps=1, batch_size=4, seq_len=32, warmup=1, save_every=0, lr=3e-3, **QUIET)
    r2 = run_training(ModelConfig(**TINY), f, tokp, tmp_path / "b", steps=60, batch_size=4, seq_len=32, warmup=5, save_every=0, lr=3e-3, **QUIET)
    assert r2["loss"] < r1["loss"] - 0.5


def test_resume_is_exact_when_lr_schedule_matches(tmp_path, corpus):
    f, tokp = corpus
    kw = dict(batch_size=2, seq_len=32, warmup=2, lr=1e-3, save_every=4, **QUIET)
    cfg = ModelConfig(**TINY)
    run_training(cfg, f, tokp, tmp_path / "full", steps=8, keep_last=10, **kw)
    # salin checkpoint step 4 dari run penuh lalu resume -> harus identik di step 8
    import shutil
    (tmp_path / "re").mkdir()
    shutil.copytree(tmp_path / "full" / "step_00000004", tmp_path / "re" / "step_00000004")
    run_training(cfg, f, tokp, tmp_path / "re", steps=8, resume=str(tmp_path / "re" / "step_00000004"), **kw)
    from safetensors.torch import load_file
    a = load_file(str(tmp_path / "full" / "step_00000008" / "model.safetensors"))
    b = load_file(str(tmp_path / "re" / "step_00000008" / "model.safetensors"))
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_keep_last_prunes(tmp_path, corpus):
    f, tokp = corpus
    run_training(ModelConfig(**TINY), f, tokp, tmp_path, steps=6, batch_size=2, seq_len=32, save_every=1, keep_last=2, **QUIET)
    assert len(list(tmp_path.glob("step_*"))) == 2


def test_grad_checkpoint_and_accum_run(tmp_path, corpus):
    f, tokp = corpus
    r = run_training(ModelConfig(**TINY), f, tokp, tmp_path, steps=3, batch_size=2, grad_accum=3, seq_len=32,
                     grad_checkpoint=True, save_every=0, **QUIET)
    assert torch.isfinite(torch.tensor(r["loss"]))


def test_bin_dataset_pipeline(tmp_path, corpus):
    f, tokp = corpus
    tok = ByteBPETokenizer.load(tokp)
    n = tokenize_files([f], tok, tmp_path / "d.bin")
    arr = load_token_array(tmp_path / "d.bin")
    assert len(arr) == n
    s = BlockSampler(arr, 16)
    x, y = s.get_batch("train", 3, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (3, 16) and torch.equal(x[:, 1:], y[:, :-1])


def test_large_model_without_gpu_is_refused(tmp_path, corpus):
    import pytest
    f, tokp = corpus
    with pytest.raises(SystemExit):
        run_training(ModelConfig.load("configs/1.5b.json"), f, tokp, tmp_path, steps=1, device="cpu", **QUIET)
