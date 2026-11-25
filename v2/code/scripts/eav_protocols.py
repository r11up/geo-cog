"""EAV, EEG only, under the evaluation protocols of published work, with the pipeline selected on the
development participants (DE/PSD + re-centred tangent-space covariances, logistic regression).

  p2  Gong et al. 2026 (LEL): per participant, trials assigned at random 60/10/30 (stratified by class),
      all windows of a trial in one set; 20 s trials, prediction = mean of the four 5 s segment
      probabilities; 5 random assignments. Models: pooled (one model on every participant's training
      trials) and per participant.
  p3  Lee, Suresh Chandra & Hansen 2026: leave-one-subject-out, 2 s epochs, balanced accuracy and
      weighted F1 per held-out participant; 4 training participants for validation.

    ~/.venvs/urbeeg/bin/python -I scripts/eav_protocols.py --protocol p2 --jobs 6
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
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

from urbeeg import tracker  # noqa: E402
from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.contrast import macro_f1  # noqa: E402

CS = (1e-4, 1e-3, 1e-2, 1e-1)


def load(name):
    z = np.load(DERIVED / "eav" / name)
    D = {k: z[k] for k in z.files}
    D["F"] = np.hstack([D["X"].reshape(len(D["X"]), -1), D["T"]]).astype(np.float32)
    return D


def fit_lr(F, y, tr, va, cs=CS):
    best = None
    for C in cs:
        m = LogisticRegression(C=C, max_iter=3000).fit(F[tr], y[tr])
        f = macro_f1(y[va], m.predict(F[va]))
        if best is None or f > best[0] + 1e-9:
            best = (f, m, C)
    return best[1], best[2]


def p2_split(D, seed):
    """Per participant and class: 12 training, 2 validation, 6 test trials (of 20)."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for s in np.unique(D["subject"]):
        idx = np.where(D["subject"] == s)[0]
        trials = np.unique(D["trial"][idx])
        lab = {t: D["y"][idx][D["trial"][idx] == t][0] for t in trials}
        for c in np.unique(list(lab.values())):
            tc = rng.permutation([t for t in trials if lab[t] == c])
            n = len(tc)
            n_tr, n_va = int(round(0.6 * n)), int(round(0.1 * n))
            for t, dst in [(t, tr) for t in tc[:n_tr]] + [(t, va) for t in tc[n_tr:n_tr + n_va]] + \
                          [(t, te) for t in tc[n_tr + n_va:]]:
                dst.extend(idx[D["trial"][idx] == t])
    return np.array(tr), np.array(va), np.array(te)


def trial_scores(D, te, P):
    df = pd.DataFrame({"subject": D["subject"][te], "trial": D["trial"][te], "y": D["y"][te]})
    df["p"] = list(P)
    rows = []
    for s, g in df.groupby("subject"):
        ts = np.unique(g.trial)
        Pg = np.stack(g.p.to_numpy())
        tp = np.array([Pg[g.trial.to_numpy() == t].mean(0).argmax() for t in ts])
        ty = np.array([g.y.to_numpy()[g.trial.to_numpy() == t][0] for t in ts])
        rows.append({"subject": s, "acc_trial": (tp == ty).mean(), "f1_trial": f1_score(ty, tp, average="macro"),
                     "acc_segment": (Pg.argmax(1) == g.y.to_numpy()).mean()})
    return pd.DataFrame(rows)


def run_p2(args):
    D = load("eav_features_v2.npz")
    out = []
    for rep in range(5):
        tr, va, te = p2_split(D, SEED + rep)
        m, C = fit_lr(D["F"], D["y"], tr, va)                       # pooled
        sc = trial_scores(D, te, m.predict_proba(D["F"][te])).assign(model="logreg_pooled", rep=rep)

        def one(s):
            mask = lambda idx: idx[D["subject"][idx] == s]
            ms, _ = fit_lr(D["F"], D["y"], mask(tr), mask(va))
            return mask(te), ms.predict_proba(D["F"][mask(te)])
        parts = Parallel(n_jobs=args.jobs)(delayed(one)(s) for s in np.unique(D["subject"]))
        te_s = np.concatenate([p[0] for p in parts]); P_s = np.concatenate([p[1] for p in parts])
        sc2 = trial_scores(D, te_s, P_s).assign(model="logreg_per_participant", rep=rep)
        out += [sc, sc2]
        print(f"rep {rep}: pooled {100 * sc.acc_trial.mean():.1f}% (C={C}), per participant "
              f"{100 * sc2.acc_trial.mean():.1f}%", flush=True)
    return pd.concat(out)


def p4_split(D, seed):
    """Per participant: random 70/30 split of 5 s segments; 10% of the training segments for validation."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for s in np.unique(D["subject"]):
        idx = rng.permutation(np.where(D["subject"] == s)[0])
        n_te = int(round(0.3 * len(idx))); n_va = int(round(0.1 * (len(idx) - n_te)))
        te.extend(idx[:n_te]); va.extend(idx[n_te:n_te + n_va]); tr.extend(idx[n_te + n_va:])
    return np.array(tr), np.array(va), np.array(te)


def run_p4(args):
    D = load("eav_features_v2.npz")
    out = []
    for rep in range(5):
        tr, va, te = p4_split(D, SEED + rep)
        m, C = fit_lr(D["F"], D["y"], tr, va)
        sc = trial_scores(D, te, m.predict_proba(D["F"][te])).assign(model="logreg_pooled", rep=rep)

        def one(s):
            mask = lambda idx: idx[D["subject"][idx] == s]
            ms, _ = fit_lr(D["F"], D["y"], mask(tr), mask(va))
            return mask(te), ms.predict_proba(D["F"][mask(te)])
        parts = Parallel(n_jobs=args.jobs)(delayed(one)(s) for s in np.unique(D["subject"]))
        sc2 = trial_scores(D, np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        out += [sc, sc2.assign(model="logreg_per_participant", rep=rep)]
        print(f"rep {rep}: pooled {100 * sc.acc_segment.mean():.1f}%, per participant {100 * sc2.acc_segment.mean():.1f}%",
              flush=True)
    return pd.concat(out)


def run_p3(args):
    D = load("eav_features_v2_2s.npz")
    subs = np.unique(D["subject"])

    def fold(s):
        rng = np.random.default_rng(SEED + int(s))
        others = np.setdiff1d(subs, [s])
        val = rng.choice(others, 4, replace=False)
        tr = np.where(np.isin(D["subject"], np.setdiff1d(others, val)))[0]
        va = np.where(np.isin(D["subject"], val))[0]
        te = np.where(D["subject"] == s)[0]
        m, C = fit_lr(D["F"], D["y"], tr, va, cs=(1e-4, 1e-3, 1e-2))
        pred = m.predict(D["F"][te])
        y = D["y"][te]
        return {"subject": s, "bal_acc": balanced_accuracy_score(y, pred), "f1_weighted": f1_score(y, pred, average="weighted"),
                "C": C, "model": "logreg_loso", "rep": 0}
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(fold)(s) for s in subs)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["p2", "p3", "p4"], required=True)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    res = {"p2": run_p2, "p3": run_p3, "p4": run_p4}[args.protocol](args)
    out = RESULTS / f"eav_{args.protocol}_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    res.to_csv(out / "per_subject.csv", index=False)
    lock = [int(s) for s in (DERIVED / "eav" / "lockbox_subjects.txt").read_text().split()]
    rng = np.random.default_rng(SEED)
    metric = {"p2": "acc_trial", "p3": "bal_acc", "p4": "acc_segment"}[args.protocol]
    summ = []
    for model, g in res.groupby("model"):
        per = g.groupby("subject")[metric].mean()
        lo, hi = tracker.boot_ci(per.to_numpy(), rng)
        rep_means = g.groupby("rep")[metric].mean()
        row = {"protocol": args.protocol, "model": model, "metric": metric, "mean": per.mean(), "lo": lo, "hi": hi,
               "sd_over_subjects": per.std(), "sd_over_reps": rep_means.std(), "lockbox_mean": per[per.index.isin(lock)].mean()}
        if args.protocol in ("p2", "p4"):
            row["f1_macro"] = g.groupby("subject")["f1_trial"].mean().mean()
            row["acc_segment"] = g.groupby("subject")["acc_segment"].mean().mean()
        else:
            row["f1_weighted"] = g.groupby("subject")["f1_weighted"].mean().mean()
        summ.append(row)
        tracker.log({"when": f"{datetime.now():%d %b %H:%M}", "phase": f"published-{args.protocol}", "variant": model,
                     "n_subjects": len(per), "acc_epoch": row.get("acc_segment", per.mean()), "acc_trial": per.mean(),
                     "acc_trial_lo": lo, "acc_trial_hi": hi, "f1_epoch": row.get("f1_macro", row.get("f1_weighted")),
                     "choices": "", "seconds": round(time.time() - t0)})
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(summ).round(4).to_string(index=False))
    print("saved", out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
