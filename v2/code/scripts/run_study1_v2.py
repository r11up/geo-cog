"""Study 1, final run (analysis plan fixed before the run).

EEG branch: DE/PSD + re-centred tangent-space covariance token, hyperparameters chosen per fold on
validation data. Proposed model: environmental vectors drawn WITHOUT the label in training and at
test (label-free pairing). The original label-conditioned PECM and its controls are ablations.

Protocols
  lockbox    train on the 30 development participants (4 for early stopping), test on the 12 lockbox
             participants; 3 seeds
  cv         subject-level 5-fold x 3 on all 42
  benchmark  published EAV protocol: per participant, first 70% of trials train, last 30% test; 3 seeds

    ~/.venvs/urbeeg/bin/python -I scripts/run_study1_v2.py --protocol lockbox --jobs 6
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from urbeeg import tracker  # noqa: E402
from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.contrast import env_matrix, macro_f1, select_bandwidth, subject_folds  # noqa: E402
from urbeeg.study1 import grid_adjacency, train_eval  # noqa: E402
from run_study1 import coarsen  # noqa: E402
from eav_benchmark import split as benchmark_split  # noqa: E402

torch.set_num_threads(2)
GRID = [(0.3, 1e-4), (0.3, 1e-2), (0.5, 1e-4), (0.5, 1e-2)]
# name: (environment sampling, fusion mode, grid)
FUSION = {
    "proposed": ("unconditional", "bi", "1km"),
    "proposed_uni": ("unconditional", "uni", "1km"),
    "proposed_noise": ("noise", "bi", "1km"),
    "pecm": ("prior", "bi", "1km"),
    "pecm_matched_random": ("matched_random", "bi", "1km"),
    "pecm_reversed": ("prior_reversed", "bi", "1km"),
    "pecm_shuffled_labels": ("shuffled_labels", "bi", "1km"),
    "pecm_removed_at_test": ("prior_removed_test", "bi", "1km"),
    "pecm_diag_test_labels": ("prior_test_labels", "bi", "1km"),
    "pecm_2km": ("prior", "bi", "2km"),
    "pecm_5km": ("prior", "bi", "5km"),
}
BENCH_FUSION = ["proposed", "pecm"]


def fold_sets(protocol, D, lock):
    subj = D["subject"]
    out = []
    if protocol == "cv":
        for rep, f, tr_s, va_s, te_s in subject_folds(subj, n_rep=3, seed=SEED):
            out.append((rep, f, *(np.where(np.isin(subj, s))[0] for s in (tr_s, va_s, te_s))))
    elif protocol == "lockbox":
        dev = np.setdiff1d(np.unique(subj), lock)
        for rep in range(3):
            va_s = np.random.default_rng(SEED + rep).choice(dev, 4, replace=False)
            tr_s = np.setdiff1d(dev, va_s)
            out.append((rep, 0, *(np.where(np.isin(subj, s))[0] for s in (tr_s, va_s, lock))))
    elif protocol == "benchmark":
        tr, va, te = benchmark_split(D, np.ones(len(subj), bool))
        out = [(rep, 0, tr, va, te) for rep in range(3)]
    return out


def logreg(D, tr, va, te):
    from sklearn.linear_model import LogisticRegression
    F = np.hstack([D["X"].reshape(len(D["X"]), -1), D["T"]])
    best = None
    for C in (1e-4, 1e-3, 1e-2, 1e-1):
        m = LogisticRegression(C=C, max_iter=3000).fit(F[tr], D["y"][tr])
        f = macro_f1(D["y"][va], m.predict(F[va]))
        if best is None or f > best[0] + 1e-9:
            best = (f, m)
    return best[1].predict_proba(F[te])


def per_subject(pred):
    rows = []
    for (c, s), g in pred.groupby(["condition", "subject"]):
        P = np.stack(g.p.to_numpy())
        accs, taccs, f1s = [], [], []
        for _, gr in g.groupby("rep"):
            Pr = np.stack(gr.p.to_numpy()); yr = gr.y.to_numpy(); tr_ = gr.trial.to_numpy()
            accs.append((Pr.argmax(1) == yr).mean()); f1s.append(macro_f1(yr, Pr.argmax(1)))
            ts = np.unique(tr_)
            taccs.append(np.mean([Pr[tr_ == t].mean(0).argmax() == yr[tr_ == t][0] for t in ts]))
        rows.append({"condition": c, "subject": s, "acc_epoch": np.mean(accs), "acc_trial": np.mean(taccs),
                     "f1": np.mean(f1s)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["lockbox", "cv", "benchmark"], required=True)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    z = np.load(DERIVED / "eav" / "eav_features_v2.npz")
    D = {k: z[k] for k in z.files}
    lock = np.array([int(s) for s in (DERIVED / "eav" / "lockbox_subjects.txt").read_text().split()])
    grid = pd.read_csv(DERIVED / "eav" / "astana_grid.csv")
    grids = {"1km": grid, "2km": coarsen(grid, 2), "5km": coarsen(grid, 5)}
    env = {k: (env_matrix(g), grid_adjacency(g)) for k, g in grids.items()}
    bw = {k: select_bandwidth(E, seed=SEED) for k, (E, _) in env.items()}
    folds = fold_sets(args.protocol, D, lock)
    fusion = BENCH_FUSION if args.protocol == "benchmark" else list(FUSION)
    print(f"{args.protocol}: {len(folds)} folds, conditions: {fusion}", flush=True)

    def tune(fold, cfg):
        rep, f, tr, va, te = fold
        E, A = env["1km"]
        _, P, fv = train_eval(D["X"], D["y"], tr, va, te, "noise", E, A, 0.75, bw["1km"], SEED + 100 * rep + f,
                              mode="eeg_only", T=D["T"], drop=cfg[0], wd=cfg[1], return_proba=True)
        return fv, P

    t0 = time.time()
    tuned = Parallel(n_jobs=args.jobs)(delayed(tune)(fd, cfg) for fd in folds for cfg in GRID)
    best = {}
    for i, fd in enumerate(folds):
        res = tuned[i * len(GRID):(i + 1) * len(GRID)]
        k = int(np.argmax([r[0] for r in res]))
        best[i] = (GRID[k], res[k][1])
    print(f"tuning done ({time.time() - t0:.0f} s): {[best[i][0] for i in best]}", flush=True)

    def run(i, name):
        rep, f, tr, va, te = folds[i]
        cond, mode, g = FUSION[name]
        E, A = env[g]
        drop, wd = best[i][0]
        _, P, _ = train_eval(D["X"], D["y"], tr, va, te, cond, E, A, 0.75, bw[g], SEED + 100 * rep + f, mode=mode,
                             T=D["T"], drop=drop, wd=wd, return_proba=True)
        return i, name, P

    out = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run)(i, n) for i in range(len(folds)) for n in fusion)
    lr = Parallel(n_jobs=args.jobs)(delayed(logreg)(D, fd[2], fd[3], fd[4]) for fd in folds)
    frames = []

    def frame(i, name, P):
        rep, f, tr, va, te = folds[i]
        return pd.DataFrame({"condition": name, "rep": rep, "fold": f, "subject": D["subject"][te],
                             "trial": D["trial"][te], "y": D["y"][te], "p": list(P)})
    for i in range(len(folds)):
        frames += [frame(i, "eeg_tower", best[i][1]), frame(i, "eeg_logreg", lr[i])]
    frames += [frame(i, n, P) for i, n, P in out]
    pred = pd.concat(frames, ignore_index=True)
    sc = per_subject(pred)
    rng = np.random.default_rng(SEED)
    summ = []
    for c, g in sc.groupby("condition"):
        row = {"condition": c, "n_subjects": len(g)}
        for m in ("acc_trial", "acc_epoch", "f1"):
            lo, hi = tracker.boot_ci(g[m].to_numpy(), rng)
            row.update({m: g[m].mean(), f"{m}_lo": lo, f"{m}_hi": hi})
        summ.append(row)
    summ = pd.DataFrame(summ)
    comps = []
    for a, b in [("proposed", "eeg_tower"), ("proposed", "eeg_logreg"), ("eeg_logreg", "eeg_tower"),
                 ("proposed", "proposed_noise"), ("proposed", "proposed_uni"), ("proposed", "pecm"),
                 ("pecm", "eeg_tower"), ("pecm", "pecm_matched_random"), ("pecm", "pecm_reversed"),
                 ("pecm", "pecm_shuffled_labels"), ("pecm", "pecm_removed_at_test"),
                 ("pecm_diag_test_labels", "pecm"), ("pecm_2km", "pecm"), ("pecm_5km", "pecm")]:
        for m in ("acc_trial", "acc_epoch"):
            piv = sc.pivot(index="subject", columns="condition", values=m)
            if a in piv and b in piv:
                d = (piv[a] - piv[b]).to_numpy()
                lo, hi = tracker.boot_ci(d, rng)
                comps.append({"metric": m, "a": a, "b": b, "mean_diff": d.mean(), "lo": lo, "hi": hi,
                              "wilcoxon_p": wilcoxon(d).pvalue if np.any(d != 0) else np.nan})
    res = RESULTS / f"study1v2_{args.protocol}_{datetime.now():%Y%m%d_%H%M}"
    res.mkdir(parents=True)
    pred.assign(p=[json.dumps(np.round(x, 4).tolist()) for x in pred.p]).to_csv(res / "predictions.csv.gz", index=False)
    sc.to_csv(res / "per_subject.csv", index=False)
    summ.to_csv(res / "summary.csv", index=False)
    pd.DataFrame(comps).to_csv(res / "comparisons.csv", index=False)
    json.dump({"protocol": args.protocol, "lockbox": lock.tolist(), "tuned": {i: best[i][0] for i in best},
               "bandwidth": bw}, open(res / "config.json", "w"), indent=1)
    for r in summ.itertuples():
        tracker.log({"when": f"{datetime.now():%d %b %H:%M}", "phase": f"final-{args.protocol}",
                     "variant": r.condition, "n_subjects": r.n_subjects, "acc_epoch": r.acc_epoch,
                     "acc_trial": r.acc_trial, "acc_trial_lo": r.acc_trial_lo, "acc_trial_hi": r.acc_trial_hi,
                     "f1_epoch": r.f1, "choices": "", "seconds": round(time.time() - t0)})
    print("saved", res, flush=True)


if __name__ == "__main__":
    main()
