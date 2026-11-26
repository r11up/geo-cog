"""EAV, EEG only: new methods under the evaluation protocols of published work.

Candidates, declared before running (26 Nov 2026, 09:13):
  base   DE and log power (5 bands) with band-wise OAS covariances in the tangent space at the
         participant's Riemannian mean, z-scored within participant; logistic regression
  align  P1 only: training and test trials of each participant re-centred and standardised separately,
         each at its own mean and without labels, against slow drift over the recording
  fb7    seven frequency bands for both feature types
  crops  training windows cut every 1 s within each training trial; test windows unchanged
  ens    soft vote of logistic regression and shrinkage LDA
  all    every component that applies to the protocol
Every band is filtered over the whole 20 s trial before windows are cut. The method reported for a
protocol is the candidate with the best validation score; every candidate is logged.

  p1  per participant, trials in time order, first 70% train, last 30% test; 5 s windows; one pooled model.
      Selection: first 50 training trials vs the next 20 (same block-wise processing).
  p2  per participant, trials at random 60/10/30 within class; 20 s trial = mean of its four 5 s windows;
      five random assignments; one model per participant. Selection: validation trials.
  p3  leave one subject out; 2 s windows; balanced accuracy; four training participants for validation.
      Selection: validation participants.

    ~/.venvs/urbeeg/bin/python -I scripts/eav_newmethods.py --protocol p1 --jobs 3
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
from joblib import Parallel, delayed  # noqa: E402
from pyriemann.geometry.mean import mean_riemann  # noqa: E402
from pyriemann.geometry.tangentspace import tangent_space  # noqa: E402
from scipy.signal import butter, sosfiltfilt, welch  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

from urbeeg import tracker  # noqa: E402
from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.contrast import macro_f1  # noqa: E402

FS, TRIAL, N_SUB, N_TRIAL = 100, 2000, 42, 100
BANDS5 = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
BANDS7 = [(1, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]
ALLB = sorted(set(BANDS5) | set(BANDS7))
CS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2)
CANDIDATES = {"p1": ["base", "align", "fb7", "crops", "ens", "all"],
              "p2": ["base", "fb7", "crops", "ens", "all"],
              "p3": ["base", "fb7", "crops", "ens", "all"],
              "p5": ["base", "fb7", "ens"]}


def parts_of(name, protocol):
    if name == "all":
        return {"align": protocol == "p1", "fb7": True, "crops": True, "ens": True}
    return {k: name == k for k in ("align", "fb7", "crops", "ens")}


# ------------------------------------------------------------------ window features
def oas(X):
    """OAS-shrunk covariance of each window (n, channels, samples), as sklearn.covariance.oas."""
    X = X - X.mean(-1, keepdims=True)
    n, p = X.shape[-1], X.shape[1]
    S = X @ X.transpose(0, 2, 1) / n
    mu = np.trace(S, axis1=1, axis2=2) / p
    alpha = (S ** 2).mean(axis=(1, 2))
    den = (n + 1) * (alpha - mu ** 2 / p)
    shrink = np.where(den > 0, np.minimum((alpha + mu ** 2) / np.where(den > 0, den, 1.0), 1.0), 1.0)
    S = (1 - shrink)[:, None, None] * S
    S[:, np.arange(p), np.arange(p)] += (shrink * mu)[:, None]
    return S


def participant_windows(ep, win):
    """ep: (400, 30, 500), the four consecutive 5 s epochs of each trial in time order (CAR, 1-45 Hz,
    100 Hz). Windows of `win` samples start every 1 s within each 20 s trial."""
    x = np.asarray(ep, dtype=np.float64).reshape(N_TRIAL, 4, 30, 500).transpose(0, 2, 1, 3).reshape(N_TRIAL, 30, TRIAL)
    starts = np.arange(0, TRIAL - win + 1, FS)

    def cut(a):
        return np.stack([a[..., s:s + win] for s in starts], axis=1).reshape(-1, a.shape[1], win)
    nper = 200 if win >= 400 else 100
    f, p = welch(cut(x), fs=FS, nperseg=nper, noverlap=nper // 2, axis=-1)
    cov, de, lp = {}, {}, {}
    for b in ALLB:
        xb = cut(sosfiltfilt(butter(4, b, btype="bandpass", fs=FS, output="sos"), x, axis=-1))
        cov[b] = oas(xb).astype(np.float32)
        de[b] = (0.5 * np.log(2 * np.pi * np.e * xb.var(axis=-1) + 1e-20)).astype(np.float32)
        lp[b] = np.log(p[..., (f >= b[0]) & (f < b[1])].mean(axis=-1) + 1e-20).astype(np.float32)
    return cov, de, lp, np.repeat(np.arange(N_TRIAL), len(starts)), np.tile(starts, N_TRIAL)


def load_windows(win, jobs):
    cache = DERIVED / "eav" / f"eav_windows_{win // FS}s.npz"
    if cache.exists():
        z = np.load(cache)
        W = {"cov": {b: z[f"cov_{b[0]}_{b[1]}"] for b in ALLB}, "de": {b: z[f"de_{b[0]}_{b[1]}"] for b in ALLB},
             "lp": {b: z[f"lp_{b[0]}_{b[1]}"] for b in ALLB}}
        meta = {k: z[k] for k in ("subject", "tpos", "start")}
    else:
        raw = np.load(DERIVED / "eav" / "eav_raw.npy", mmap_mode="r")
        res = Parallel(n_jobs=jobs, verbose=5)(
            delayed(participant_windows)(raw[s * 400:(s + 1) * 400], win) for s in range(N_SUB))
        W = {k: {b: np.concatenate([r[i][b] for r in res]) for b in ALLB} for i, k in enumerate(("cov", "de", "lp"))}
        meta = {"subject": np.concatenate([np.full(len(r[3]), s + 1) for s, r in enumerate(res)]),
                "tpos": np.concatenate([r[3] for r in res]), "start": np.concatenate([r[4] for r in res])}
        np.savez(cache, **{f"{k}_{b[0]}_{b[1]}": W[k][b] for k in ("cov", "de", "lp") for b in ALLB}, **meta)
    z = np.load(DERIVED / "eav" / "eav_features_v2.npz")                 # labels, same participant/trial order
    ytrial = z["y"].reshape(N_SUB, N_TRIAL, 4)[:, :, 0]
    meta["y"] = ytrial[meta["subject"] - 1, meta["tpos"]]
    meta["std"] = meta["start"] % win == 0
    return W, meta


def features(W, meta, bands, block):
    """Tangent vectors at the Riemannian mean of each block's standard windows, with DE and log power;
    every column z-scored over the block's standard windows. Blocks never mix participants; no labels."""
    out = None
    for blk in np.unique(block):
        w = block == blk
        ref = (meta["std"] & w)[w]
        parts = [W["de"][b][w] for b in bands] + [W["lp"][b][w] for b in bands]
        for b in bands:
            C = W["cov"][b][w].astype(np.float64)
            parts.append(tangent_space(C, mean_riemann(C[ref]), metric="riemann"))
        F = np.hstack(parts)
        mu, sd = F[ref].mean(0), F[ref].std(0) + 1e-8
        if out is None:
            out = np.empty((len(block), F.shape[1]), np.float32)
        out[w] = (F - mu) / sd
    return out


# ------------------------------------------------------------------ model
def fit_predict(F, y, tr, va, targets, ens, cs=CS):
    """Logistic regression with C chosen by macro-F1 on `va` (warm start along the C path), optionally
    averaged with shrinkage LDA. Returns probabilities for `va` and for each index array in `targets`."""
    m = LogisticRegression(max_iter=3000, warm_start=True)
    best = None
    for C in cs:
        m.set_params(C=C).fit(F[tr], y[tr])
        f = macro_f1(y[va], m.predict(F[va]))
        if best is None or f > best[0] + 1e-9:
            best = (f, C, m.coef_.copy(), m.intercept_.copy())
    m.coef_, m.intercept_ = best[2], best[3]
    models = [m]
    if ens:
        models.append(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F[tr], y[tr]))
    prob = lambda idx: np.mean([mm.predict_proba(F[idx]) for mm in models], axis=0)
    return prob(va), [prob(t) for t in targets], best[1]


def per_subject_epoch_scores(meta, idx, P, win):
    df = pd.DataFrame({"subject": meta["subject"][idx], "tpos": meta["tpos"][idx], "y": meta["y"][idx]})
    df["p"] = list(P)
    rows = []
    for s, g in df.groupby("subject"):
        Pg = np.stack(g.p.to_numpy())
        tp = np.array([Pg[g.tpos.to_numpy() == t].mean(0).argmax() for t in np.unique(g.tpos)])
        ty = np.array([g.y.to_numpy()[g.tpos.to_numpy() == t][0] for t in np.unique(g.tpos)])
        rows.append({"subject": s, "acc_epoch": (Pg.argmax(1) == g.y.to_numpy()).mean(), "acc_trial": (tp == ty).mean(),
                     "f1_epoch": macro_f1(g.y.to_numpy(), Pg.argmax(1)), "f1_trial": f1_score(ty, tp, average="macro"),
                     "bal_acc": balanced_accuracy_score(g.y.to_numpy(), Pg.argmax(1)),
                     "f1_weighted": f1_score(g.y.to_numpy(), Pg.argmax(1), average="weighted")})
    return pd.DataFrame(rows)


def _p2_one(F, y, tr, va, te, ens):
    Pva, (Pte,), _ = fit_predict(F, y, tr, va, [te], ens)
    return va, Pva, te, Pte


def _p3_fold(F, y, s, tr, va, te, ens, name):
    Pva, (Pte,), C = fit_predict(F, y, tr, va, [te], ens, cs=CS[:5])
    pred = Pte.argmax(1)
    return ({"subject": s, "bal_acc": balanced_accuracy_score(y[te], pred),
             "f1_weighted": f1_score(y[te], pred, average="weighted"), "C": C, "model": name, "rep": 0},
            balanced_accuracy_score(y[va], Pva.argmax(1)))


# ------------------------------------------------------------------ protocols
def run_p1(W, meta, name, win):
    c = parts_of(name, "p1")
    bands = BANDS7 if c["fb7"] else BANDS5
    sub, tpos, y, std = meta["subject"], meta["tpos"], meta["y"], meta["std"]
    use = std | c["crops"]
    # selection run: first 50 training trials vs the next 20, test trials unused
    block = sub * 10 + ((tpos >= 50).astype(int) if c["align"] else 0)
    block = np.where(tpos >= 70, sub * 10 + 2, block) if c["align"] else block
    F = features(W, meta, bands, block)
    itr = np.where(use & (tpos < 50))[0]
    iva = np.where(std & (tpos >= 50) & (tpos < 70))[0]
    Pva, _, _ = fit_predict(F, y, itr, iva, [], c["ens"])
    sel = per_subject_epoch_scores(meta, iva, Pva, win)
    # test run: published split; last training trial of each class for C
    block = sub * 10 + ((tpos >= 70).astype(int) if c["align"] else 0)
    F = features(W, meta, bands, block)
    val = np.zeros(len(y), bool)
    for s in np.unique(sub):
        for k in np.unique(y):
            ts = tpos[(sub == s) & (y == k) & (tpos < 70)]
            val |= (sub == s) & (tpos == ts.max())
    tr = np.where(use & (tpos < 70) & ~val)[0]
    va = np.where(std & val)[0]
    te = np.where(std & (tpos >= 70))[0]
    _, (Pte,), C = fit_predict(F, y, tr, va, [te], c["ens"])
    sc = per_subject_epoch_scores(meta, te, Pte, win)
    return sc.assign(model=name, rep=0, C=C), sel.acc_epoch.mean()


def run_p2(W, meta, name, win, jobs):
    from eav_protocols import p2_split
    c = parts_of(name, "p2")
    bands = BANDS7 if c["fb7"] else BANDS5
    F = features(W, meta, bands, meta["subject"])
    z = np.load(DERIVED / "eav" / "eav_features_v2.npz")
    D = {k: z[k] for k in ("y", "subject", "trial")}
    sub, tpos, y, std = meta["subject"], meta["tpos"], meta["y"], meta["std"]
    use = std | c["crops"]
    out, sel = [], []
    for rep in range(5):
        tr0, va0, te0 = p2_split(D, SEED + rep)                     # same assignment as the earlier P2 run
        key = lambda idx: set(zip(D["subject"][idx], (idx % 400) // 4))
        trk, vak, tek = key(tr0), key(va0), key(te0)
        pairs = list(zip(sub, tpos))
        in_tr = np.array([p in trk for p in pairs]); in_va = np.array([p in vak for p in pairs])
        in_te = np.array([p in tek for p in pairs])

        jobs_in = [(np.where((sub == s) & use & in_tr)[0], np.where((sub == s) & std & in_va)[0],
                    np.where((sub == s) & std & in_te)[0]) for s in np.unique(sub)]
        parts = Parallel(n_jobs=jobs)(delayed(_p2_one)(F, y, tr, va, te, c["ens"]) for tr, va, te in jobs_in)
        va = np.concatenate([p[0] for p in parts]); Pva = np.concatenate([p[1] for p in parts])
        te = np.concatenate([p[2] for p in parts]); Pte = np.concatenate([p[3] for p in parts])
        out.append(per_subject_epoch_scores(meta, te, Pte, win).assign(model=name, rep=rep))
        sel.append(per_subject_epoch_scores(meta, va, Pva, win).acc_trial.mean())
        print(f"  {name} rep {rep}: test trial acc {100 * out[-1].acc_trial.mean():.1f}%", flush=True)
    return pd.concat(out), float(np.mean(sel))


def p5_split(meta, seed):
    """Per participant and class: half of the 5 s segments for testing; 30% of the rest for validation."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for s in np.unique(meta["subject"]):
        for k in np.unique(meta["y"]):
            idx = rng.permutation(np.where(meta["std"] & (meta["subject"] == s) & (meta["y"] == k))[0])
            n_te = len(idx) // 2
            rest = idx[n_te:]
            n_va = int(round(0.3 * len(rest)))
            te.extend(idx[:n_te]); va.extend(rest[:n_va]); tr.extend(rest[n_va:])
    return np.array(tr), np.array(va), np.array(te)


def run_p5(W, meta, name, win, jobs):
    """Malallah & Iqbal 2025: per participant, random 50/50 split of the 5 s segments, eight assignments."""
    c = parts_of(name, "p5")
    bands = BANDS7 if c["fb7"] else BANDS5
    F = features(W, meta, bands, meta["subject"])
    sub, y = meta["subject"], meta["y"]
    out, sel = [], []
    for rep in range(8):
        tr0, va0, te0 = p5_split(meta, SEED + 100 + rep)
        jobs_in = [(tr0[sub[tr0] == s], va0[sub[va0] == s], te0[sub[te0] == s]) for s in np.unique(sub)]
        parts = Parallel(n_jobs=jobs)(delayed(_p2_one)(F, y, tr, va, te, c["ens"]) for tr, va, te in jobs_in)
        va = np.concatenate([p[0] for p in parts]); Pva = np.concatenate([p[1] for p in parts])
        te = np.concatenate([p[2] for p in parts]); Pte = np.concatenate([p[3] for p in parts])
        out.append(per_subject_epoch_scores(meta, te, Pte, win).assign(model=name, rep=rep))
        sel.append(per_subject_epoch_scores(meta, va, Pva, win).acc_epoch.mean())
        print(f"  {name} rep {rep}: test segment acc {100 * out[-1].acc_epoch.mean():.1f}%", flush=True)
    return pd.concat(out), float(np.mean(sel))


def run_p3(W, meta, name, win, jobs):
    c = parts_of(name, "p3")
    bands = BANDS7 if c["fb7"] else BANDS5
    F = features(W, meta, bands, meta["subject"])
    sub, y, std = meta["subject"], meta["y"], meta["std"]
    use = std | c["crops"]
    subs = np.unique(sub)

    folds = []
    for s in subs:
        rng = np.random.default_rng(SEED + int(s))                  # same validation participants as before
        others = np.setdiff1d(subs, [s])
        vals = rng.choice(others, 4, replace=False)
        folds.append((s, np.where(use & np.isin(sub, np.setdiff1d(others, vals)))[0],
                      np.where(std & np.isin(sub, vals))[0], np.where(std & (sub == s))[0]))
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(_p3_fold)(F, y, s, tr, va, te, c["ens"], name)
                                           for s, tr, va, te in folds)
    return pd.DataFrame([r[0] for r in res]), float(np.mean([r[1] for r in res]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["p1", "p2", "p3", "p5"], required=True)
    ap.add_argument("--candidates", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--resume", default=None, help="results folder of a stopped run; its finished candidates are kept")
    args = ap.parse_args()
    win = 200 if args.protocol == "p3" else 500
    t0 = time.time()
    W, meta = load_windows(win, args.jobs)
    print(f"windows ready: {len(meta['y'])} ({time.time() - t0:.0f} s)", flush=True)
    if args.resume:
        out = Path(args.resume)
        summ = pd.read_csv(out / "summary.csv").drop(columns="selected", errors="ignore").to_dict("records")
        allres = [pd.read_csv(out / "per_subject.csv")]
    else:
        out = RESULTS / f"eav_new_{args.protocol}_{datetime.now():%Y%m%d_%H%M}"
        out.mkdir(parents=True)
        summ, allres = [], []
    finished = {r["candidate"] for r in summ}
    metric = {"p1": "acc_epoch", "p2": "acc_trial", "p3": "bal_acc", "p5": "acc_epoch"}[args.protocol]
    rng = np.random.default_rng(SEED)
    for name in args.candidates or CANDIDATES[args.protocol]:
        if name in finished:
            print(f"{name}: kept from {out.name}", flush=True)
            continue
        t1 = time.time()
        if args.protocol == "p1":
            res, vscore = run_p1(W, meta, name, win)
        elif args.protocol == "p2":
            res, vscore = run_p2(W, meta, name, win, args.jobs)
        elif args.protocol == "p5":
            res, vscore = run_p5(W, meta, name, win, args.jobs)
        else:
            res, vscore = run_p3(W, meta, name, win, args.jobs)
        allres.append(res)
        per = res.groupby("subject")[metric].mean()
        lo, hi = tracker.boot_ci(per.to_numpy(), rng)
        row = {"protocol": args.protocol, "candidate": name, "metric": metric, "validation": vscore,
               "test_mean": per.mean(), "test_lo": lo, "test_hi": hi, "seconds": round(time.time() - t1)}
        for extra in ("acc_epoch", "acc_trial", "f1_epoch", "f1_trial", "f1_weighted"):
            if extra in res and extra != metric:
                row[extra] = res.groupby("subject")[extra].mean().mean()
        summ.append(row)
        tracker.log({"when": f"{datetime.now():%d %b %H:%M}", "phase": f"new-{args.protocol}", "variant": name,
                     "n_subjects": len(per), "acc_epoch": row.get("acc_epoch", per.mean()), "acc_trial": per.mean(),
                     "acc_trial_lo": lo, "acc_trial_hi": hi, "f1_epoch": row.get("f1_epoch", row.get("f1_weighted")),
                     "choices": f"validation {vscore:.4f}", "seconds": row["seconds"]})
        pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
        pd.concat(allres).to_csv(out / "per_subject.csv", index=False)
        print(f"{name}: validation {100 * vscore:.1f}, test {metric} {100 * per.mean():.1f} "
              f"[{100 * lo:.1f}, {100 * hi:.1f}] ({row['seconds']} s)", flush=True)
    S = pd.DataFrame(summ)
    S["selected"] = S.validation == S.validation.max()
    S.to_csv(out / "summary.csv", index=False)
    json.dump({"protocol": args.protocol, "selected": S.loc[S.selected, "candidate"].tolist(),
               "seconds": round(time.time() - t0)}, open(out / "meta.json", "w"))
    print(S.round(4).to_string(index=False))
    print("saved", out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
