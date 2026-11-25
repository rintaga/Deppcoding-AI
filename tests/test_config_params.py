import pytest
from core.config import ModelConfig, count_parameters_formula
from core.model import count_parameters_meta


@pytest.mark.parametrize("name,target", [("tiny", None), ("1.5b", 1.5e9), ("3b", 3.0e9)])
def test_formula_matches_real_model_and_target(name, target):
    cfg = ModelConfig.load(f"configs/{name}.json")
    actual = count_parameters_meta(cfg)
    assert count_parameters_formula(cfg) == actual
    if target:
        assert abs(actual - target) / target < 0.05


def test_invalid_configs():
    with pytest.raises(ValueError):
        ModelConfig(hidden_size=100, num_heads=3)
    with pytest.raises(ValueError):
        ModelConfig(num_heads=16, num_kv_heads=5)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=100)


@pytest.mark.parametrize("name", ["1.5b", "3b"])
def test_real_configs_forward_on_meta_device(name):
    import torch
    from core.model import DecoderLM
    cfg = ModelConfig.load(f"configs/{name}.json")
    with torch.device("meta"):
        m = DecoderLM(cfg)
        x = torch.zeros(1, cfg.context_length, dtype=torch.long)
        out = m(x, x)
    assert out["logits"].shape == (1, cfg.context_length, cfg.vocab_size)
