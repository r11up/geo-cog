"""EAV, EEG only: further methods under the published protocols P1 and P3 (declared 26 Nov 2026, 11:38,
amended 11:39). Selection stays on validation data across all rounds; every candidate is reported.

  p1  cbramod_long       CBraMod fine-tuned for 50 epochs: 3-epoch warm-up, cosine decay from 1e-4,
                         label smoothing 0.1
      cbramod_long_lr    mean of its probabilities and those of pooled LR (training windows every 1 s)
      perpart_lr         LR per participant (training windows every 1 s)
      pooled_perpart_lr  mean of pooled and per-participant LR
      selftrain_lr       pooled LR refitted with the more confident half of each participant's unlabeled
                         evaluation windows and their predicted labels (transductive)
      Selection: first 50 training trials vs the next 20 (the next 20 act as the unlabeled block).
  p3  selftrain          the same self-training on the held-out participant (protocol's 2 s windows)
      weighted           training participants weighted by exp(-d / median d), d the mean Riemannian distance
                         between band-wise mean covariances (no labels)

    ~/.venvs/urbeeg/bin/python -I scripts/eav_round5.py --protocol p1
"""
import argparse
import copy
import json
import math
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
from pyriemann.geometry.distance import distance_riemann  # noqa: E402
from pyriemann.geometry.mean import mean_riemann  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

import eav_cbramod as cb  # noqa: E402
import eav_newmethods as nm  # noqa: E402
from urbeeg import tracker  # noqa: E402
from urbeeg.config import RESULTS, SEED  # noqa: E402
from urbeeg.contrast import macro_f1  # noqa: E402


# ------------------------------------------------------------------ logistic regression helpers
def lr_select(F, y, tr, va, cs=nm.CS, w=None):
    """LR with C chosen by macro-F1 on `va` (warm start along the C path)."""
    m = LogisticRegression(max_iter=3000, warm_start=True)
    best = None
    for C in cs:
        m.set_params(C=C).fit(F[tr], y[tr], sample_weight=w)
        f = macro_f1(y[va], m.predict(F[va]))
        if best is None or f > best[0] + 1e-9:
            best = (f, C, m.coef_.copy(), m.intercept_.copy())
    m.coef_, m.intercept_ = best[2], best[3]
    return m, best[1]


def confident_half(P, groups):
    """Indices (into P) of the more confident half of the rows within each group, with predicted labels."""
    keep = []
    conf = P.max(1)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        keep.append(idx[conf[idx] >= np.median(conf[idx])])
    keep = np.concatenate(keep)
    return keep, P[keep].argmax(1)


def self_train(F, y, tr, unl, P_unl, groups, C, w=None, w_unl=None):
    """Refit LR (fixed C) on the labelled windows plus the confident half of the unlabeled ones."""
    k, lab = confident_half(P_unl, groups)
    X_idx = np.concatenate([tr, unl[k]])
    labels = np.concatenate([y[tr], lab])
    sw = None if w is None else np.concatenate([w, w_unl[k] if w_unl is not None else np.ones(len(k))])
    m = LogisticRegression(C=C, max_iter=3000).fit(F[X_idx], labels, sample_weight=sw)
    return m.predict_proba(F[unl])


# ------------------------------------------------------------------ CBraMod with the longer recipe
def finetune_long(X, Y, train_trials, val_keys, seed, epochs=50, warmup=3, lr=1e-4, smoothing=0.1, win=500):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = cb.new_model(2 * win).to(cb.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = math.ceil(len(train_trials) * 4 / 64)
    total, warm = epochs * steps, warmup * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm))))
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
    yv = np.array([Y[s, t] for s, t, _ in val_keys])
    best = (-1.0, None, -1)
    for ep in range(epochs):
        model.train()
        pairs = np.repeat(np.asarray(train_trials), 4, axis=0)
        starts = rng.integers(0, nm.TRIAL - win + 1, len(pairs))
        order = rng.permutation(len(pairs))
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            keys = [(pairs[j, 0], pairs[j, 1], starts[j]) for j in idx]
            xb = torch.as_tensor(cb.windows_of(X, keys, win), device=cb.DEV)
            yb = torch.as_tensor(np.array([Y[s, t] for s, t, _ in keys]), device=cb.DEV)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            sched.step()
        f1 = macro_f1(yv, cb.predict(model, X, val_keys, win).argmax(1))
        print(f"    epoch {ep + 1}: validation macro-F1 {f1:.3f}", flush=True)
        if f1 > best[0] + 1e-9:
            best = (f1, copy.deepcopy(model.state_dict()), ep + 1)
    model.load_state_dict(best[1])
    return model, best[2]


# ------------------------------------------------------------------ P1
def run_p1(args):
    X, Y = cb.load_trials()
    W, meta = cb.load_windows_base(500)
    sub, tpos, y, std = meta["subject"], meta["tpos"], meta["y"], meta["std"]
    last = {}
    for limit in (50, 70):
        v = np.zeros(len(y), bool)
        for s in np.unique(sub):
            for k in np.unique(y):
                ts = tpos[(sub == s) & (y == k) & (tpos < limit)]
                if len(ts):
                    v |= (sub == s) & (tpos == ts.max())
        last[limit] = v
    splits = {"selection": dict(tr=(tpos < 50) & ~last[50], es=std & last[50], ev=std & (tpos >= 50) & (tpos < 70)),
              "test": dict(tr=(tpos < 70) & ~last[70], es=std & last[70], ev=std & (tpos >= 70))}
    F = nm.features(W, meta, nm.BANDS5, sub)
    print(f"data ready on {cb.DEV}", flush=True)
    results, scores = {}, {}
    for split, d in splits.items():
        tr, es, ev = np.where(d["tr"])[0], np.where(d["es"])[0], np.where(d["ev"])[0]
        va_c = ev if split == "selection" else es                     # as in rounds 4 and 4b
        t = time.time()
        m_pool, C = lr_select(F, y, tr, va_c)
        P_pool = m_pool.predict_proba(F[ev])
        P_pp = np.zeros_like(P_pool)
        for s in np.unique(sub):
            tr_s, va_s, ev_s = tr[sub[tr] == s], va_c[sub[va_c] == s], np.where(sub[ev] == s)[0]
            m_s, _ = lr_select(F, y, tr_s, va_s)
            P_s = m_s.predict_proba(F[ev[ev_s]])
            full = np.zeros((len(ev_s), P_pool.shape[1]))                # classes absent in a participant's training
            full[:, np.searchsorted(m_pool.classes_, m_s.classes_)] = P_s
            P_pp[ev_s] = full
        P_st = self_train(F, y, tr, ev, P_pool, sub[ev], C)
        print(f"  {split}: LR candidates ({time.time() - t:.0f} s)", flush=True)
        t = time.time()
        trials = sorted({(s - 1, tp) for s, tp in zip(sub[tr], tpos[tr])})
        model, best_ep = finetune_long(X, Y, trials, cb.keys_from(meta, es), SEED + (10 if split == "selection" else 11))
        P_cl = cb.predict(model, X, cb.keys_from(meta, ev), 500)
        print(f"  {split}: CBraMod long, best epoch {best_ep} ({time.time() - t:.0f} s)", flush=True)
        del model
        torch.mps.empty_cache() if cb.DEV == "mps" else None
        cands = {"cbramod_long": P_cl, "cbramod_long_lr": (P_cl + P_pool) / 2, "perpart_lr": P_pp,
                 "pooled_perpart_lr": (P_pool + P_pp) / 2, "selftrain_lr": P_st}
        for name, P in cands.items():
            sc = nm.per_subject_epoch_scores(meta, ev, P, 500)
            if split == "selection":
                scores[name] = sc.acc_epoch.mean()
            else:
                results[name] = sc.assign(model=name, rep=0)
        print("  " + ", ".join(f"{k} {100 * (scores[k] if split == 'selection' else results[k].acc_epoch.mean()):.1f}"
                               for k in cands), flush=True)
    return results, scores


# ------------------------------------------------------------------ P3
def subject_distances(W, meta):
    """Mean over bands of the Riemannian distance between participants' mean covariances (standard windows)."""
    subs = np.unique(meta["subject"])
    D = np.zeros((len(subs), len(subs)))
    for b in nm.BANDS5:
        M = [mean_riemann(W["cov"][b][(meta["subject"] == s) & meta["std"]].astype(np.float64)) for s in subs]
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                D[i, j] = D[j, i] = D[i, j] + distance_riemann(M[i], M[j]) / len(nm.BANDS5)
    return subs, D


def _p3_fold(F, y, subk, s, others_tr, vals, D, subs):
    pos = {v: i for i, v in enumerate(subs)}
    tr = np.where(np.isin(subk, others_tr))[0]
    va = np.where(np.isin(subk, vals))[0]
    te = np.where(subk == s)[0]
    m, C = lr_select(F, y, tr, va, cs=nm.CS[:5])
    out = {}

    def weights(targets):
        d = np.array([np.mean([D[pos[j], pos[t]] for t in targets]) for j in others_tr])
        w = np.exp(-d / np.median(d))
        w = w / w.mean()
        return w[np.searchsorted(others_tr, subk[tr])]
    for split, idx, targets in (("val", va, vals), ("test", te, [s])):
        P0 = m.predict_proba(F[idx])
        P_st = self_train(F, y, tr, idx, P0, np.zeros(len(idx)), C)
        w = weights(targets)
        P_w = LogisticRegression(C=C, max_iter=3000).fit(F[tr], y[tr], sample_weight=w).predict_proba(F[idx])
        out[split] = {"selftrain": P_st, "weighted": P_w}
    rows = [{"subject": s, "bal_acc": balanced_accuracy_score(y[te], out["test"][k].argmax(1)),
             "f1_weighted": f1_score(y[te], out["test"][k].argmax(1), average="weighted"), "C": C, "model": k, "rep": 0}
            for k in out["test"]]
    vscores = {k: balanced_accuracy_score(y[va], out["val"][k].argmax(1)) for k in out["val"]}
    return rows, vscores


def run_p3(args):
    W, meta = cb.load_windows_base(200)
    sub, y, std = meta["subject"], meta["y"], meta["std"]
    keep = np.where(std)[0]
    F = nm.features(W, meta, nm.BANDS5, sub)[keep]
    subs, D = subject_distances(W, meta)
    subk, yk = sub[keep], y[keep]
    del W                                                               # free the covariances before the folds run
    folds = []
    for s in subs:
        rng = np.random.default_rng(SEED + int(s))                     # same validation participants as before
        others = np.setdiff1d(subs, [s])
        vals = rng.choice(others, 4, replace=False)
        folds.append((s, np.setdiff1d(others, vals), vals))
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(_p3_fold)(F, yk, subk, s, o, v, D, subs) for s, o, v in folds)
    rows = pd.DataFrame([r for fold_rows, _ in res for r in fold_rows])
    results = {k: g.reset_index(drop=True) for k, g in rows.groupby("model")}
    scores = {k: float(np.mean([v[k] for _, v in res])) for k in results}
    return results, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["p1", "p3"], required=True)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    results, scores = (run_p1 if args.protocol == "p1" else run_p3)(args)
    out = RESULTS / f"eav_round5_{args.protocol}_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    metric = "acc_epoch" if args.protocol == "p1" else "bal_acc"
    rng = np.random.default_rng(SEED)
    summ = []
    for name, res in results.items():
        per = res.groupby("subject")[metric].mean()
        lo, hi = tracker.boot_ci(per.to_numpy(), rng)
        row = {"protocol": args.protocol, "candidate": name, "metric": metric, "validation": scores[name],
               "test_mean": per.mean(), "test_lo": lo, "test_hi": hi}
        for extra in ("acc_trial", "f1_epoch", "f1_weighted"):
            if extra in res:
                row[extra] = res.groupby("subject")[extra].mean().mean()
        summ.append(row)
        tracker.log({"when": f"{datetime.now():%d %b %H:%M}", "phase": f"round5-{args.protocol}", "variant": name,
                     "n_subjects": len(per), "acc_epoch": row.get("acc_epoch", per.mean()), "acc_trial": per.mean(),
                     "acc_trial_lo": lo, "acc_trial_hi": hi, "f1_epoch": row.get("f1_epoch", row.get("f1_weighted")),
                     "choices": f"validation {scores[name]:.4f}", "seconds": round(time.time() - t0)})
        print(f"{name}: validation {100 * scores[name]:.1f}, test {metric} {100 * per.mean():.1f} "
              f"[{100 * lo:.1f}, {100 * hi:.1f}]", flush=True)
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    pd.concat([r.assign(model=k) for k, r in results.items()]).to_csv(out / "per_subject.csv", index=False)
    json.dump({"protocol": args.protocol, "device": cb.DEV, "seconds": round(time.time() - t0)}, open(out / "meta.json", "w"))
    print("saved", out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
