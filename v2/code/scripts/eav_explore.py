"""EAV, EEG-only pipeline exploration on the 30 development subjects (lockbox excluded).

The analysis plan was fixed before any run. Every variant run is appended to
results/tracker.tsv and never deleted; results/tracker.md is the leaderboard.

    ~/.venvs/urbeeg/bin/python -I scripts/eav_explore.py --jobs 6
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.contrast import env_matrix, macro_f1, subject_folds  # noqa: E402
from urbeeg.study1 import grid_adjacency, train_eval  # noqa: E402

torch.set_num_threads(2)
TRACK, BOARD = RESULTS / "tracker.tsv", RESULTS / "tracker.md"
GRID = [(0.3, 1e-4), (0.3, 1e-2), (0.5, 1e-4), (0.5, 1e-2)]          # dropout x weight decay


def load_dev():
    z = np.load(DERIVED / "eav" / "eav_features_v2.npz")
    lock = [int(s) for s in (DERIVED / "eav" / "lockbox_subjects.txt").read_text().split()]
    keep = ~np.isin(z["subject"], lock)
    return {k: z[k][keep] for k in z.files}, lock


def feats(D, kind):
    flat = D["X"].reshape(len(D["X"]), -1)
    return {"depsd": flat, "tangent": D["T"], "both": np.hstack([flat, D["T"]])}[kind]


def fit_variant(name, D, E, A, tr, va, te, seed):
    """Return test probabilities (n_te, 5) and a dict of choices made on validation data."""
    y = D["y"]
    if name.startswith("tower"):
        T = D["T"] if "both" in name else None
        grid = GRID if name.endswith("tuned") else [(0.3, 1e-4)]
        best = None
        for drop, wd in grid:
            _, proba, f_val = train_eval(D["X"], y, tr, va, te, "noise", E, A, 0.75, 1.0, seed,
                                         mode="eeg_only", T=T, drop=drop, wd=wd, return_proba=True)
            if best is None or f_val > best[0] + 1e-9:
                best = (f_val, proba, {"dropout": drop, "weight_decay": wd})
        return best[1], best[2]
    if name.startswith("logreg"):
        from sklearn.linear_model import LogisticRegression
        F = feats(D, name.split("_")[1])
        best = None
        for C in (0.001, 0.01, 0.1):
            m = LogisticRegression(C=C, max_iter=2000).fit(F[tr], y[tr])
            f_val = macro_f1(y[va], m.predict(F[va]))
            if best is None or f_val > best[0] + 1e-9:
                best = (f_val, m, {"C": C})
        return best[1].predict_proba(F[te]), best[2]
    if name.startswith("hgb"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        F = feats(D, name.split("_")[1])
        m = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, early_stopping=True,
                                           random_state=seed).fit(F[tr], y[tr])
        return m.predict_proba(F[te]), {"n_iter": int(m.n_iter_)}
    raise ValueError(name)


def score(pred: pd.DataFrame) -> pd.DataFrame:
    """Per-subject epoch accuracy, trial accuracy (mean probability over a trial's epochs), F1."""
    P = np.stack(pred["proba"].to_numpy())
    pred = pred.assign(pred=P.argmax(1))
    rows = []
    for s, g in pred.groupby("subject"):
        Pg = np.stack(g["proba"].to_numpy())
        tr_ids = g["trial"].to_numpy()
        trials = np.unique(tr_ids)
        tp = np.array([Pg[tr_ids == t].mean(0).argmax() for t in trials])
        ty = np.array([g["y"].to_numpy()[tr_ids == t][0] for t in trials])
        rows.append({"subject": s, "acc_epoch": (g.y == g.pred).mean(), "acc_trial": (tp == ty).mean(),
                     "f1_epoch": macro_f1(g.y.to_numpy(), g.pred.to_numpy())})
    return pd.DataFrame(rows)


def boot_ci(v, rng):
    b = rng.choice(v, size=(2000, len(v))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def track(row):
    new = not TRACK.exists()
    pd.DataFrame([row]).to_csv(TRACK, sep="\t", index=False, header=new, mode="a")
    t = pd.read_csv(TRACK, sep="\t")
    dev = t[t.phase == "dev"].sort_values("acc_trial", ascending=False)
    lines = ["# EAV pipeline tracker", "",
             f"Updated {datetime.now():%d %b %Y %H:%M}. Development subjects only (lockbox excluded); "
             "every run is kept. Chance = 20%.", "",
             "| # | variant | trial acc (%) [95% CI] | epoch acc (%) | macro-F1 | choices | when |",
             "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(dev.itertuples(), 1):
        lines.append(f"| {i} | {r.variant} | {100 * r.acc_trial:.1f} [{100 * r.acc_trial_lo:.1f}, "
                     f"{100 * r.acc_trial_hi:.1f}] | {100 * r.acc_epoch:.1f} | {r.f1_epoch:.3f} | "
                     f"{r.choices} | {r.when} |")
    other = t[t.phase != "dev"]
    if len(other):
        lines += ["", "## Final evaluations", "", "| phase | variant | trial acc (%) | epoch acc (%) | when |",
                  "|---|---|---|---|---|"]
        for r in other.itertuples():
            lines.append(f"| {r.phase} | {r.variant} | {100 * r.acc_trial:.1f} | {100 * r.acc_epoch:.1f} | {r.when} |")
    BOARD.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--variants", nargs="+", default=[
        "tower_depsd", "logreg_depsd", "logreg_tangent", "logreg_both", "hgb_both",
        "tower_depsd_tuned", "tower_both_tuned"])
    args = ap.parse_args()
    D, lock = load_dev()
    grid = pd.read_csv(DERIVED / "eav" / "astana_grid.csv")
    E, A = env_matrix(grid), grid_adjacency(grid)
    folds = subject_folds(D["subject"], n_folds=5, n_rep=1, seed=SEED)
    out_dir = RESULTS / "explore"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dev subjects: {len(np.unique(D['subject']))}, epochs: {len(D['y'])}, lockbox: {lock}", flush=True)

    def job(name, fold):
        rep, f, tr_s, va_s, te_s = fold
        tr, va, te = (np.where(np.isin(D["subject"], s))[0] for s in (tr_s, va_s, te_s))
        proba, choice = fit_variant(name, D, E, A, tr, va, te, SEED + f)
        return pd.DataFrame({"fold": f, "subject": D["subject"][te], "trial": D["trial"][te], "y": D["y"][te],
                             "proba": list(proba), "choice": json.dumps(choice)})

    rng = np.random.default_rng(SEED)
    for name in args.variants:            # one variant at a time so the tracker fills as we go
        t0 = time.time()
        parts = Parallel(n_jobs=args.jobs)(delayed(job)(name, fd) for fd in folds)
        pred = pd.concat(parts, ignore_index=True)
        sc = score(pred)
        lo, hi = boot_ci(sc.acc_trial.to_numpy(), rng)
        choices = "; ".join(sorted(set(pred.choice)))
        track({"when": f"{datetime.now():%d %b %H:%M}", "phase": "dev", "variant": name,
               "n_subjects": len(sc), "acc_epoch": sc.acc_epoch.mean(), "acc_trial": sc.acc_trial.mean(),
               "acc_trial_lo": lo, "acc_trial_hi": hi, "f1_epoch": sc.f1_epoch.mean(),
               "choices": choices if len(choices) < 120 else f"{len(set(pred.choice))} fold-wise settings",
               "seconds": round(time.time() - t0)})
        pred.assign(proba=[json.dumps(np.round(p, 4).tolist()) for p in pred.proba]).to_csv(
            out_dir / f"{name}.csv.gz", index=False)
        print(f"{name}: trial {100 * sc.acc_trial.mean():.1f}%, epoch {100 * sc.acc_epoch.mean():.1f}% "
              f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
