import copy
import pytest
import torch
import torch.nn.functional as F
from core.config import ModelConfig
from core.model import DecoderLM, count_parameters, sliding_window_attention
from tests.conftest import TINY


def naive_window(q, k, v, w):
    t = q.shape[2]
    i = torch.arange(t)[:, None]
    j = torch.arange(t)[None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i) & (j > i - w))


@pytest.mark.parametrize("t,w", [(16, 4), (17, 4), (10, 16), (33, 8), (7, 7), (64, 1)])
def test_sliding_window_matches_naive(t, w):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, t, 8) for _ in range(3))
    got = sliding_window_attention(q, k, v, w) if w < t else naive_window(q, k, v, w)
    assert torch.allclose(got, naive_window(q, k, v, w), atol=1e-5)


def test_forward_shapes_loss_and_tying():
    m = DecoderLM(ModelConfig(**TINY))
    x = torch.randint(0, 512, (2, 20))
    out = m(x, x)
    assert out["logits"].shape == (2, 20, 512) and torch.isfinite(out["loss"])
    assert m.lm_head.weight is m.embed.weight
    assert count_parameters(m) > 0


def test_causality_future_tokens_do_not_change_past():
    m = DecoderLM(ModelConfig(**TINY)).eval()
    x = torch.randint(0, 512, (1, 40))
    y = x.clone()
    y[0, 30:] = (y[0, 30:] + 7) % 512
    a, b = m(x)["logits"], m(y)["logits"]
    assert torch.allclose(a[:, :30], b[:, :30], atol=1e-5)
    assert not torch.allclose(a[:, 35], b[:, 35], atol=1e-5)


def test_window_limits_receptive_field():
    cfg = ModelConfig(**{**TINY, "num_layers": 1, "window_size": 4})
    m = DecoderLM(cfg).eval()
    x = torch.randint(0, 512, (1, 30))
    y = x.clone()
    y[0, 0] = (y[0, 0] + 1) % 512  # token jauh di luar window posisi 20 (1 layer)
    assert torch.allclose(m(x)["logits"][:, 20], m(y)["logits"][:, 20], atol=1e-6)


def test_gradient_checkpointing_gives_same_gradients():
    torch.manual_seed(0)
    a = DecoderLM(ModelConfig(**TINY))
    b = copy.deepcopy(a)
    b.gradient_checkpointing = True
    a.train(); b.train()
    x = torch.randint(0, 512, (2, 40))
    a(x, x)["loss"].backward()
    b(x, x)["loss"].backward()
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-5), n


def test_too_long_sequence_rejected():
    m = DecoderLM(ModelConfig(**TINY))
    with pytest.raises(ValueError):
        m(torch.randint(0, 512, (1, 65)))
