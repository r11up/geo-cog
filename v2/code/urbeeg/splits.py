"""Crossed person x scene cross-validation.

In each test cell, the test set is (held-out participants) x (held-out scenes) and
the training set is (remaining participants) x (remaining scenes). Pairs that mix a
held-out participant with a training scene (or the reverse) are used by neither,
so no person and no scene is ever seen in both training and test.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cell:
    rep: int
    p_fold: int
    s_fold: int
    train_participants: tuple
    test_participants: tuple
    train_scenes: tuple
    test_scenes: tuple


def _split_even(items, n_folds, rng):
    items = list(items)
    rng.shuffle(items)
    return [tuple(sorted(items[i::n_folds])) for i in range(n_folds)]


def scene_folds(scenes, strata, n_folds, rng):
    """Assign scenes to folds so each fold gets a similar mix of each stratum."""
    folds = [[] for _ in range(n_folds)]
    offset = 0
    for s in sorted(set(strata.values())):
        group = [sc for sc in scenes if strata[sc] == s]
        rng.shuffle(group)
        for i, sc in enumerate(group):
            folds[(i + offset) % n_folds].append(sc)
        offset += len(group)
    return [tuple(sorted(f)) for f in folds]


def crossed_cells(participants, scenes, scene_strata=None, n_p=5, n_s=4, n_rep=3, seed=0):
    participants, scenes = sorted(participants), sorted(scenes)
    strata = scene_strata or {sc: 0 for sc in scenes}
    cells = []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed + rep)
        p_folds = _split_even(participants, n_p, rng)
        s_folds = scene_folds(scenes, strata, n_s, rng)
        for i, tp in enumerate(p_folds):
            for j, ts in enumerate(s_folds):
                cells.append(Cell(rep, i, j,
                                  tuple(p for p in participants if p not in tp), tp,
                                  tuple(s for s in scenes if s not in ts), ts))
    return cells


def inner_cells(cell: Cell, k: int = 3, seed: int = 0):
    """k inner splits of the training block for hyperparameter selection. Split i holds out
    training-participant fold i x training-scene fold i (so every validation set is new in
    both people and scenes) and trains on the other folds x other folds."""
    rng = np.random.default_rng(seed + 1000 * cell.rep + 10 * cell.p_fold + cell.s_fold)
    pf = _split_even(cell.train_participants, k, rng)
    sf = _split_even(cell.train_scenes, k, rng)
    out = []
    for i in range(k):
        out.append((tuple(p for p in cell.train_participants if p not in pf[i]), pf[i],
                    tuple(s for s in cell.train_scenes if s not in sf[i]), sf[i]))
    return out


def inner_validation(cell: Cell, frac: float = 0.2, seed: int = 0):
    """Hold out a share of the training participants AND training scenes for early
    stopping / hyperparameter choice, so validation is also person- and scene-novel."""
    rng = np.random.default_rng(seed + 1000 * cell.rep + 10 * cell.p_fold + cell.s_fold)
    tp, ts = list(cell.train_participants), list(cell.train_scenes)
    rng.shuffle(tp)
    rng.shuffle(ts)
    n_vp, n_vs = max(1, round(frac * len(tp))), max(1, round(frac * len(ts)))
    return (tuple(sorted(tp[n_vp:])), tuple(sorted(tp[:n_vp])),
            tuple(sorted(ts[n_vs:])), tuple(sorted(ts[:n_vs])))
