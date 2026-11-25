"""Checks of the contrast-analysis sampler and folds on synthetic grids (no EEG needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbeeg.contrast import EnvSampler, class_subsets, subject_folds  # noqa: E402

rng = np.random.default_rng(0)
E = rng.normal(size=(100, 4))
y = np.repeat(np.arange(5), 4000)


def test_prior_is_label_conditioned_in_training_only():
    s = EnvSampler(E, "prior", h=0.3, seed=1)
    tr, te = s.train(y), s.test(y)
    tr_means = np.array([tr[y == k].mean(0) for k in range(5)])
    te_means = np.array([te[y == k].mean(0) for k in range(5)])
    assert np.abs(tr_means[1] - tr_means[3]).max() > 0.5       # sadness vs happiness differ
    assert np.abs(te_means - te_means.mean(0)).max() < 0.08    # test: no class differences


def test_paired_classes_share_subsets_and_reversal_swaps_them():
    sub, rev = class_subsets(E), class_subsets(E, reversed_=True)
    assert np.array_equal(sub[1], sub[2]) and np.array_equal(sub[3], sub[4])
    assert np.array_equal(rev[1], sub[3]) and np.array_equal(rev[3], sub[1])


def test_matched_random_copies_class_moments_without_grid_values():
    p, m = EnvSampler(E, "prior", h=0.3, seed=2), EnvSampler(E, "matched_random", h=0.3, seed=2)
    a, b = p.train(y), m.train(y)
    for k in range(5):
        assert np.abs(a[y == k].mean(0) - b[y == k].mean(0)).max() < 0.1
    assert not np.isin(np.round(b, 6), np.round(E, 6)).all(axis=1).any()


def test_noise_and_removed_inputs_carry_no_label_information():
    n = EnvSampler(E, "noise", seed=3).test(y)
    assert np.abs(np.array([n[y == k].mean(0) for k in range(5)])).max() < 0.08
    assert np.all(EnvSampler(E, "prior_removed_test", seed=3).test(y) == 0)


def test_subject_folds_are_disjoint_and_cover_everyone():
    subj = np.repeat(np.arange(1, 43), 10)
    folds = subject_folds(subj, n_folds=5, n_rep=3, seed=0)
    for rep in range(3):
        tests = np.concatenate([te for r, f, tr, va, te in folds if r == rep])
        assert sorted(tests) == list(range(1, 43))
    for _, _, tr, va, te in folds:
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
