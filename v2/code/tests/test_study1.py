"""Shape and wiring checks for the re-implemented original architecture (random inputs)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbeeg.contrast import EnvSampler  # noqa: E402
from urbeeg.study1 import Study1Model, grid_adjacency  # noqa: E402

GRID = pd.DataFrame([(r, c) for r in range(10) for c in range(10)], columns=["row", "col"])


def test_adjacency_is_symmetric_normalised_queen():
    A = grid_adjacency(GRID)
    assert np.allclose(A, A.T)
    assert (A[0] > 0).sum() == 4                  # corner: itself + 3 neighbours
    assert (A[55] > 0).sum() == 9                 # interior: itself + 8 neighbours


def test_all_fusion_modes_run():
    A, F = grid_adjacency(GRID), np.random.default_rng(0).normal(size=(100, 4))
    x, v = torch.randn(8, 30, 10), torch.randn(8, 4)
    for mode in ("bi", "uni", "eeg_only", "uni_res", "concat", "tok_bi", "tok_uni"):
        out, align = Study1Model(A, F, mode=mode)(x, v)
        assert out.shape == (8, 5) and align.ndim == 0


def test_token_paths_pool_to_the_single_vector_outputs():
    """The token-level modes align the same pooled embeddings as the single-vector modes."""
    A, F = grid_adjacency(GRID), np.random.default_rng(0).normal(size=(100, 4))
    m = Study1Model(A, F, mode="tok_bi", n_extra=6).eval()
    x, v, t = torch.randn(4, 30, 10), torch.randn(4, 4), torch.randn(4, 6)
    with torch.no_grad():
        assert torch.allclose(m.eeg(x, t, tokens=True).mean(1), m.eeg(x, t), atol=1e-5)
        assert torch.allclose(m.env(v, tokens=True).mean(1), m.env(v), atol=1e-5)
        assert m.eeg(x, t, tokens=True).shape == (4, 31, 256) and m.env(v, tokens=True).shape == (4, 100, 256)


def test_single_key_attention_is_a_linear_map_of_the_value():
    """With one key the attention weight is 1, so 'uni' passes the classifier only a function of v_env."""
    A, F = grid_adjacency(GRID), np.random.default_rng(0).normal(size=(100, 4))
    m = Study1Model(A, F, mode="uni").eval()
    v = torch.randn(3, 4)
    with torch.no_grad():
        out1, _ = m(torch.randn(3, 30, 10), v)
        out2, _ = m(torch.randn(3, 30, 10), v)
    assert torch.allclose(out1, out2, atol=1e-5)    # EEG input does not change the prediction


def test_original_ablation_samplers_break_the_label_link_in_training():
    E = np.random.default_rng(1).normal(size=(100, 4))
    y = np.repeat(np.arange(5), 3000)
    for kind in ("unconditional", "shuffled_labels"):
        v = EnvSampler(E, kind, h=0.3, seed=2).train(y)
        means = np.array([v[y == k].mean(0) for k in range(5)])
        assert np.abs(means - means.mean(0)).max() < 0.1
