"""Run models over the crossed person x scene cells and collect out-of-fold predictions.

Targets
-------
raw        each training participant's ratings z-scored with that person's ratings of the
           training scenes (training data only). Test ratings are stored unscaled; the
           metrics used (per-person correlation) do not depend on their scale.
deviation  a person's z-rating minus the scene's consensus. Consensus = mean z-rating of
           the scene by the training participants (excluding the person themself for
           training rows). Every person, test people included, is z-scored with their
           ratings of the training scenes only, so no test rating defines its own target.
"""
import numpy as np
import pandas as pd

from .splits import Cell, inner_cells


def _z_stats(ratings: pd.DataFrame, participants, rows_mask):
    """Per-person mean/SD of each scale over the rows selected by rows_mask."""
    sub = ratings[rows_mask]
    g = sub.groupby(participants[rows_mask])
    return g.mean(), g.std(ddof=0).replace(0, 1.0)


def cell_targets(data, cell: Cell, targets, mode="raw"):
    part, stim = data["participant"], data["stimulus"]
    R = data["ratings"][targets]
    in_tr_s = np.isin(stim, cell.train_scenes)
    mu, sd = _z_stats(R, part, in_tr_s)          # every person, over training scenes only
    Z = (R - mu.loc[part].to_numpy()) / sd.loc[part].to_numpy()
    Z = Z.to_numpy()
    if mode == "raw":
        return Z
    in_tr_p = np.isin(part, cell.train_participants)
    df = pd.DataFrame(Z, columns=targets).assign(s=stim)
    tr = df[in_tr_p]
    S = tr.groupby("s")[targets].sum().loc[stim].to_numpy()      # NaNs skipped
    N = tr.groupby("s")[targets].count().loc[stim].to_numpy()
    own = in_tr_p[:, None] & ~np.isnan(Z)                        # leave self out
    consensus = (S - np.where(own, Z, 0.0)) / (N - own)
    return Z - consensus


def run_cells(data, cells, make_models, targets, mode="raw", log=print):
    part, stim = data["participant"], data["stimulus"]
    rows_out = []
    for ci, cell in enumerate(cells):
        Y = cell_targets(data, cell, targets, mode)
        train = np.where(np.isin(part, cell.train_participants) & np.isin(stim, cell.train_scenes))[0]
        test = np.where(np.isin(part, cell.test_participants) & np.isin(stim, cell.test_scenes))[0]
        inner = [(np.where(np.isin(part, tp) & np.isin(stim, ts))[0],
                  np.where(np.isin(part, vp) & np.isin(stim, vs))[0])
                 for tp, vp, ts, vs in inner_cells(cell)]
        assert not set(part[test]) & set(part[train]) and not set(stim[test]) & set(stim[train])
        Y_fit = np.full_like(Y, np.nan)
        # models see only training targets; a missing rating becomes the person's mean (0)
        Y_fit[train] = np.nan_to_num(Y[train], nan=0.0)
        seed = 1000 * cell.rep + 10 * cell.p_fold + cell.s_fold
        truth = Y[test] if mode == "deviation" else data["ratings"][targets].to_numpy()[test]
        for m in make_models(seed):
            m.fit(data, Y_fit, inner, train)
            pred = m.predict(data, test)
            for k, t in enumerate(targets):
                rows_out.append(pd.DataFrame({
                    "rep": cell.rep, "p_fold": cell.p_fold, "s_fold": cell.s_fold,
                    "model": m.name, "target": t, "mode": mode,
                    "participant": part[test], "stimulus": stim[test],
                    "y_true": truth[:, k], "y_pred": pred[:, k]}))
        log(f"cell {ci + 1}/{len(cells)} (rep {cell.rep}, P{cell.p_fold}, S{cell.s_fold}) "
            f"train {len(train)} test {len(test)}")
    return pd.concat(rows_out, ignore_index=True)
