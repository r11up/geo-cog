"""Shape and wiring checks for the deep models on random tensors (no training)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbeeg.deep import DualTower, build  # noqa: E402


def test_all_models_produce_one_output_per_target():
    x = torch.randn(4, 65, 400)
    for name in ("eegnet", "conformer"):
        assert build(name, 65, 400, 3)(x).shape == (4, 3)
    out = build("dualtower", 65, 400, 3)(x, torch.randn(4, 7))
    assert out.shape == (4, 3)


def test_dualtower_attention_is_between_time_tokens_and_scene_categories():
    m = DualTower(65, 400, 7, 3).eval()
    _, a_es, a_se = m(torch.randn(2, 65, 400), torch.randn(2, 7), return_attention=True)
    assert a_es.shape == (2, 38, 7)        # 38 EEG time tokens over 7 scene categories
    assert a_se.shape == (2, 7, 38)
    assert torch.allclose(a_es.sum(-1), torch.ones(2, 38), atol=1e-5)


def test_positional_embedding_is_a_trained_parameter():
    m = DualTower(65, 400, 7, 3)
    assert any(n.endswith("eeg.pos") for n, _ in m.named_parameters())
