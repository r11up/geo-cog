from pathlib import Path

import numpy as np

from data.alignment import PECM


def test_unconditional_env_independent_of_labels():
    """env vectors from sample_unconditional() must not depend on label order."""
    rng = np.random.default_rng(0)
    node_feats = rng.normal(size=(100, 6))

    labels_a = np.array([0, 1, 2, 3, 4] * 20)
    labels_b = np.random.default_rng(1).permutation(labels_a)

    pecm_a = PECM(node_feats, random_state=42)
    pecm_b = PECM(node_feats, random_state=42)
    env_a = pecm_a.sample_unconditional(len(labels_a))
    env_b = pecm_b.sample_unconditional(len(labels_b))

    np.testing.assert_allclose(env_a, env_b)


def test_multimodal_cv_test_split_uses_unconditional_sampling():
    """Guards against a future regression reintroducing pecm.sample(y_te)."""
    src = (Path(__file__).parents[1] / "training" / "cross_validation.py").read_text()
    assert "env_te   = _sample_env_without_labels(pecm, len(y_te))" in src
