"""Run the EAV contrast analysis (all conditions + threshold/bandwidth sensitivity).

    ~/.venvs/urbeeg/bin/python -I scripts/run_contrast.py --jobs 3
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.contrast import (env_matrix, fit_predict, per_subject_scores,  # noqa: E402
                             select_bandwidth, subject_folds)

MAIN = ["eeg_only", "prior", "noise", "matched_random", "prior_removed_test", "prior_reversed",
        "prior_test_labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    z = np.load(DERIVED / "eav" / "eav_features.npz")
    X, y, subj = z["X"].reshape(len(z["y"]), -1), z["y"], z["subject"]
    E = env_matrix(pd.read_csv(DERIVED / "eav" / "astana_grid.csv"))
    h = select_bandwidth(E, seed=SEED)
    print(f"{len(np.unique(subj))} subjects, {len(y)} epochs; selected KDE bandwidth h = {h:.3f}")

    conds = [(c, 0.75, h) for c in MAIN]
    conds += [("prior", q, hh) for q in (0.67, 0.75, 0.90) for hh in (h / 2, h, 2 * h)
              if not (q == 0.75 and hh == h)]
    folds = subject_folds(subj, n_rep=1 if args.smoke else args.reps, seed=SEED)
    if args.smoke:
        folds, conds = folds[:1], conds[:3]

    def run(cond, q, hh, fold):
        rep, f, tr_s, va_s, te_s = fold
        tr, va, te = (np.where(np.isin(subj, s))[0] for s in (tr_s, va_s, te_s))
        seed = SEED + 100 * rep + f
        pred = fit_predict(X, y, tr, va, te, cond, E, q, hh, seed)
        name = cond if (q == 0.75 and hh == h) else f"prior_q{q:.2f}_h{hh / h:.1f}x"
        return pd.DataFrame({"condition": name, "rep": rep, "fold": f, "subject": subj[te],
                             "y": y[te], "pred": pred})

    jobs = [(c, q, hh, fd) for c, q, hh in conds for fd in folds]
    preds = pd.concat(Parallel(n_jobs=args.jobs, verbose=5)(delayed(run)(*j) for j in jobs),
                      ignore_index=True)
    scores = per_subject_scores(preds)
    rng = np.random.default_rng(SEED)
    summ = []
    for c, g in scores.groupby("condition"):
        row = {"condition": c, "n_subjects": len(g)}
        for m in ("acc", "f1"):
            v = g[m].to_numpy()
            boot = rng.choice(v, size=(2000, len(v))).mean(1)
            row.update({m: v.mean(), f"{m}_lo": np.percentile(boot, 2.5), f"{m}_hi": np.percentile(boot, 97.5)})
        summ.append(row)
    summ = pd.DataFrame(summ).sort_values("condition")
    piv = scores.pivot(index="subject", columns="condition", values="f1")
    comps = []
    for a, b in [("prior", "eeg_only"), ("prior", "noise"), ("prior", "matched_random"),
                 ("prior", "prior_reversed"), ("prior", "prior_removed_test"),
                 ("noise", "eeg_only"), ("matched_random", "eeg_only"), ("prior_test_labels", "prior")]:
        if a in piv and b in piv:
            d = (piv[a] - piv[b]).to_numpy()
            boot = rng.choice(d, size=(2000, len(d))).mean(1)
            comps.append({"a": a, "b": b, "mean_diff_f1": d.mean(), "lo": np.percentile(boot, 2.5),
                          "hi": np.percentile(boot, 97.5),
                          "wilcoxon_p": wilcoxon(d).pvalue if np.any(d != 0) else np.nan})
    if args.smoke:      # checks only, no accuracy (partial runs must not steer decisions)
        print(f"smoke: {len(preds)} predictions, conditions {sorted(preds.condition.unique())}, "
              f"classes predicted {sorted(np.unique(preds.pred))}")
        return
    out = RESULTS / f"contrast_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    preds.to_csv(out / "predictions.csv.gz", index=False)
    scores.to_csv(out / "per_subject.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(comps).to_csv(out / "comparisons.csv", index=False)
    json.dump({"bandwidth": h, "reps": args.reps, "seed": SEED, "conditions": [c[0] for c in conds]},
              open(out / "config.json", "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
