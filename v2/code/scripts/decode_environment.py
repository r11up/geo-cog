"""Decoding the viewed environment from EEG (Urban Appraisal, co-registered).

For each person x scene pair, EEG features averaged over the scene's presentations predict the
scene's environmental composition, for people and scenes held out together (crossed 5 x 4 folds,
3 repetitions). Targets: high vs low greenery (stimulus code HT/LT, AUC), continuous greenery share
(Pearson r), high vs low built form (HB/LB), and controls: shuffled scene labels, low-level image
statistics as targets, and greenery residualised on low-level statistics (fitted on training scenes).

    ~/.venvs/urbeeg/bin/python -I scripts/decode_environment.py --jobs 6
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                      # one BLAS thread per worker process
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import pearsonr, wilcoxon  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from urbeeg.config import RESULTS, SEED  # noqa: E402
from urbeeg.dataset import processed_participants  # noqa: E402
from urbeeg.splits import crossed_cells  # noqa: E402
from run_linear import build_data  # noqa: E402

LOW = ["Brightness", "Saturation", "HSVHue", "Entropy", "StraightEdgeDensity", "NonStraightEdgeDensity",
       "FractalD", "depth_metric_mean"]
TASKS = {  # name: (kind, target column or recipe)
    "greenery_code": ("binary", "code_HT"),
    "built_code": ("binary", "code_HB"),
    "greenery_share": ("continuous", "greenery"),
    "greenery_shuffled": ("binary_shuffled", "code_HT"),
    "greenery_resid_lowlevel": ("continuous_resid", "greenery"),
    "brightness": ("continuous", "Brightness"),
    "saturation": ("continuous", "Saturation"),
}


def tangent_per_person(cov, people):
    """Band-wise tangent-space vectors, re-centred at each person's own mean (no labels used)."""
    from pyriemann.tangentspace import TangentSpace
    out = np.zeros((len(cov), cov.shape[1] * cov.shape[2] * (cov.shape[2] + 1) // 2), np.float32)
    for p in np.unique(people):
        m = people == p
        out[m] = np.hstack([TangentSpace(metric="riemann").fit_transform(cov[m, b]) for b in range(cov.shape[1])])
    return out


def zscore_per_person(X, people):
    X = X.astype(np.float64).copy()
    for p in np.unique(people):
        m = people == p
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
    return X


def run_cell(cell, X, data, task, seed):
    kind, col = TASKS[task]
    part, stim, scene = data["participant"], data["stimulus"], data["scene"]
    tr = np.isin(part, cell.train_participants) & np.isin(stim, cell.train_scenes)
    te = np.isin(part, cell.test_participants) & np.isin(stim, cell.test_scenes)
    y_scene = scene[col].astype(float)
    if kind == "continuous_resid":               # residual of greenery on low-level stats, training scenes
        lr = LinearRegression().fit(scene.loc[list(cell.train_scenes), LOW], y_scene.loc[list(cell.train_scenes)])
        y_scene = y_scene - pd.Series(lr.predict(scene[LOW]), index=scene.index)
    y = y_scene.reindex(stim).to_numpy()
    ytr = y[tr].copy()
    if kind == "binary_shuffled":                # break the scene-label link within training scenes
        rng = np.random.default_rng(seed)
        perm = dict(zip(cell.train_scenes, rng.permutation(y_scene.loc[list(cell.train_scenes)].to_numpy())))
        ytr = np.array([perm[s] for s in stim[tr]])
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    if kind.startswith("binary"):
        m = LogisticRegression(C=0.01, max_iter=3000).fit(Xtr, ytr.astype(int))
        score = m.decision_function(Xte)
    else:
        m = Ridge(alpha=1e4, solver="lsqr").fit(Xtr, (ytr - ytr.mean()) / (ytr.std() + 1e-12))
        score = m.predict(Xte)
    return pd.DataFrame({"rep": cell.rep, "participant": part[te], "stimulus": stim[te], "y": y[te],
                         "score": score})


def per_person(df, kind):
    rows = []
    for (rep, p), g in df.groupby(["rep", "participant"]):
        if kind.startswith("binary"):
            yb = g.y.to_numpy() > 0.5
            v = roc_auc_score(yb, g.score) if 0 < yb.sum() < len(yb) else np.nan
            b = balanced_accuracy_score(yb, g.score.to_numpy() > 0)
        else:
            v, b = pearsonr(g.score, g.y)[0], np.nan
        rows.append({"rep": rep, "participant": p, "value": v, "bacc": b})
    return pd.DataFrame(rows).groupby("participant")[["value", "bacc"]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--features", nargs="+", default=["erp_windows", "covariance", "erp_plus_cov"])
    args = ap.parse_args()
    t0 = time.time()
    subs = processed_participants()
    data = build_data(subs)
    data["participant"] = np.asarray(data["participant"]).astype(str)
    data["stimulus"] = np.asarray(data["stimulus"]).astype(str)
    people = data["participant"]
    feats = {"erp_windows": zscore_per_person(data["erp_win"], people),
             "covariance": zscore_per_person(tangent_per_person(data["cov"], people), people)}
    feats["erp_plus_cov"] = np.hstack([feats["erp_windows"], feats["covariance"]])
    scenes = sorted(data["scene"].index)
    strata = data["scene"][["code_HT", "code_HB"]].astype(str).agg("".join, axis=1).to_dict()
    cells = crossed_cells(subs, scenes, strata, n_p=5, n_s=4, n_rep=3, seed=SEED)
    print(f"{len(subs)} participants, {len(people)} pairs, features ready in {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(SEED)
    rows, oof = [], []
    for fname, X in feats.items():
        if fname not in args.features:
            continue
        for task, (kind, _) in TASKS.items():
            parts = Parallel(n_jobs=args.jobs)(delayed(run_cell)(c, X, data, task, SEED + i) for i, c in enumerate(cells))
            df = pd.concat(parts, ignore_index=True)
            pp = per_person(df, kind)
            v = pp["value"].dropna().to_numpy()
            boot = rng.choice(v, size=(2000, len(v))).mean(1)
            null = 0.5 if kind.startswith("binary") else 0.0
            rows.append({"features": fname, "task": task, "metric": "AUC" if kind.startswith("binary") else "r",
                         "mean": v.mean(), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
                         "n": len(v), "p_vs_chance": wilcoxon(v - null).pvalue,
                         "balanced_acc": pp["bacc"].mean()})
            oof.append(df.assign(features=fname, task=task))
            print(f"{fname:13s} {task:24s} {rows[-1]['metric']} = {v.mean():.3f} "
                  f"[{rows[-1]['lo']:.3f}, {rows[-1]['hi']:.3f}]", flush=True)
    out = RESULTS / f"decode_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.concat(oof).to_csv(out / "predictions.csv.gz", index=False)
    print("saved", out, f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
