"""EAV, EEG only: the EEG foundation model CBraMod (pretrained on TUEG) under the published protocols P1 and
P3 (declared 26 Nov 2026, 09:35). Input: the preprocessed signal (CAR, 1-45 Hz, unit variance per
participant) resampled to 200 Hz and multiplied by 0.1.

  p1  cbramod_ft      fine-tuned end to end, one model for all participants, four random 5 s windows per
                      training trial and epoch, the epoch with the best validation macro-F1 kept
      cbramod_ft_lr   mean of its class probabilities and those of the selected logistic regression
      cbramod_frozen  logistic regression on frozen CBraMod features with the base features
      Selection: first 50 training trials vs the next 20, every candidate re-trained for that split.
  p3  cbramod_frozen_only, base_cbramod: frozen features of each 2 s window, leave one subject out.

    ~/.venvs/urbeeg/bin/python -I scripts/eav_cbramod.py --protocol p1
"""
import argparse
import copy
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
from scipy.signal import resample_poly  # noqa: E402

import eav_newmethods as nm  # noqa: E402
from urbeeg import tracker  # noqa: E402
from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.contrast import macro_f1  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
FS2, SCALE = 200, 0.1


def load_trials():
    """(42, 100, 30, 4000) float16 trials at 200 Hz, and (42, 100) labels."""
    raw = np.load(DERIVED / "eav" / "eav_raw.npy", mmap_mode="r")
    out = np.empty((nm.N_SUB, nm.N_TRIAL, 30, 2 * nm.TRIAL), np.float16)
    for s in range(nm.N_SUB):
        x = np.asarray(raw[s * 400:(s + 1) * 400], np.float32).reshape(nm.N_TRIAL, 4, 30, 500)
        x = x.transpose(0, 2, 1, 3).reshape(nm.N_TRIAL, 30, nm.TRIAL)
        out[s] = (resample_poly(x, 2, 1, axis=-1) * SCALE).astype(np.float16)
    z = np.load(DERIVED / "eav" / "eav_features_v2.npz")
    return out, z["y"].reshape(nm.N_SUB, nm.N_TRIAL, 4)[:, :, 0]


def load_windows_base(win):
    """Round-4 window features restricted to the five base bands (less memory)."""
    z = np.load(DERIVED / "eav" / f"eav_windows_{win // nm.FS}s.npz")
    W = {k: {b: z[f"{k}_{b[0]}_{b[1]}"] for b in nm.BANDS5} for k in ("cov", "de", "lp")}
    meta = {k: z[k] for k in ("subject", "tpos", "start")}
    y = np.load(DERIVED / "eav" / "eav_features_v2.npz")["y"].reshape(nm.N_SUB, nm.N_TRIAL, 4)[:, :, 0]
    meta["y"] = y[meta["subject"] - 1, meta["tpos"]]
    meta["std"] = meta["start"] % win == 0
    return W, meta


def new_model(n_times, n_outputs=5):
    from braindecode.models import CBraMod
    return CBraMod.from_pretrained("braindecode/cbramod-pretrained", n_outputs=n_outputs, n_chans=30,
                                   n_times=n_times, sfreq=FS2)


def windows_of(X, keys, win):
    """keys: (subject index 0-41, trial position, start at 100 Hz) -> float32 array (n, 30, win at 200 Hz)."""
    return np.stack([X[s, t, :, 2 * st:2 * st + 2 * win] for s, t, st in keys]).astype(np.float32)


def predict(model, X, keys, win, bs=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(keys), bs):
            xb = torch.as_tensor(windows_of(X, keys[i:i + bs], win), device=DEV)
            out.append(torch.softmax(model(xb), dim=1).float().cpu().numpy())
    return np.concatenate(out)


def frozen_features(X, keys, win, bs=128):
    model = new_model(2 * win).to(DEV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(keys), bs):
            f = model(torch.as_tensor(windows_of(X, keys[i:i + bs], win), device=DEV), return_features=True)
            f = f["features"] if isinstance(f, dict) else f
            out.append(f.float().mean(1).flatten(1).cpu().numpy())          # mean over channels
    return np.concatenate(out)


def finetune(X, Y, train_trials, val_keys, seed, epochs=20, win=500):
    """train_trials: (subject index, trial position) pairs; val_keys: standard windows for model selection."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = new_model(2 * win).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
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
            xb = torch.as_tensor(windows_of(X, keys, win), device=DEV)
            yb = torch.as_tensor(np.array([Y[s, t] for s, t, _ in keys]), device=DEV)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
        f1 = macro_f1(yv, predict(model, X, val_keys, win).argmax(1))
        print(f"    epoch {ep + 1}: validation macro-F1 {f1:.3f}", flush=True)
        if f1 > best[0] + 1e-9:
            best = (f1, copy.deepcopy(model.state_dict()), ep + 1)
    model.load_state_dict(best[1])
    return model, best[2]


def keys_from(meta, idx):
    return list(zip(meta["subject"][idx] - 1, meta["tpos"][idx], meta["start"][idx]))


def run_p1(args):
    X, Y = load_trials()
    W, meta = load_windows_base(500)
    print(f"data ready on {DEV}", flush=True)
    sub, tpos, y, std = meta["subject"], meta["tpos"], meta["y"], meta["std"]
    # splits as in round 4: inner (selection) and published (test)
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
    Fb = nm.features(W, meta, nm.BANDS5, sub)                               # base features, participant blocks
    allkeys = keys_from(meta, np.arange(len(y)))
    t = time.time()
    Fc = frozen_features(X, allkeys, 500)
    for s in np.unique(sub):                                                # z-score within participant
        m = sub == s
        ref = m & std
        Fc[m] = (Fc[m] - Fc[ref].mean(0)) / (Fc[ref].std(0) + 1e-8)
    Fbc = np.hstack([Fb, Fc]).astype(np.float32)
    print(f"frozen features {Fc.shape} ({time.time() - t:.0f} s)", flush=True)
    results, scores = {}, {}
    for split, d in splits.items():
        tr_idx = np.where(d["tr"])[0]                                        # every 1 s window of training trials
        es_idx, ev_idx = np.where(d["es"])[0], np.where(d["ev"])[0]
        Pcv = {}
        # selected round-4 method: logistic regression, training windows every 1 s
        va_for_c = ev_idx if split == "selection" else es_idx
        _, (P_lr,), _ = nm.fit_predict(Fb, y, tr_idx, va_for_c, [ev_idx], False)
        _, (P_fr,), _ = nm.fit_predict(Fbc, y, tr_idx, va_for_c, [ev_idx], False)
        trials = sorted({(s - 1, tp) for s, tp in zip(sub[tr_idx], tpos[tr_idx])})
        t = time.time()
        model, best_ep = finetune(X, Y, trials, keys_from(meta, es_idx), SEED + (0 if split == "selection" else 1))
        P_ft = predict(model, X, keys_from(meta, ev_idx), 500)
        print(f"  {split}: fine-tuned, best epoch {best_ep} ({time.time() - t:.0f} s)", flush=True)
        Pcv = {"cbramod_ft": P_ft, "cbramod_ft_lr": (P_ft + P_lr) / 2, "cbramod_frozen": P_fr, "lr_crops": P_lr}
        for name, P in Pcv.items():
            sc = nm.per_subject_epoch_scores(meta, ev_idx, P, 500)
            if split == "selection":
                scores[name] = sc.acc_epoch.mean()
            else:
                results[name] = sc.assign(model=name, rep=0)
        del model
        torch.mps.empty_cache() if DEV == "mps" else None
    return results, scores


def run_p3(args):
    X, Y = load_trials()
    W, meta = load_windows_base(200)
    sub, y, std = meta["subject"], meta["y"], meta["std"]
    keep = np.where(std)[0]                                                 # the protocol's 2 s windows
    t = time.time()
    Fc = frozen_features(X, keys_from(meta, keep), 200)
    for s in np.unique(sub[keep]):
        m = sub[keep] == s
        Fc[m] = (Fc[m] - Fc[m].mean(0)) / (Fc[m].std(0) + 1e-8)
    print(f"frozen features {Fc.shape} ({time.time() - t:.0f} s)", flush=True)
    Fb = nm.features(W, meta, nm.BANDS5, sub)[keep]
    sets = {"cbramod_frozen_only": Fc.astype(np.float32), "base_cbramod": np.hstack([Fb, Fc]).astype(np.float32)}
    subk, yk = sub[keep], y[keep]
    subs = np.unique(subk)
    results, scores = {}, {}
    for name, F in sets.items():
        folds = []
        for s in subs:
            rng = np.random.default_rng(SEED + int(s))                     # same validation participants as before
            others = np.setdiff1d(subs, [s])
            vals = rng.choice(others, 4, replace=False)
            folds.append((s, np.where(np.isin(subk, np.setdiff1d(others, vals)))[0],
                          np.where(np.isin(subk, vals))[0], np.where(subk == s)[0]))
        res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(nm._p3_fold)(F, yk, s, tr, va, te, False, name)
                                                    for s, tr, va, te in folds)
        results[name] = pd.DataFrame([r[0] for r in res])
        scores[name] = float(np.mean([r[1] for r in res]))
        print(f"{name}: validation {100 * scores[name]:.1f}, test {100 * results[name].bal_acc.mean():.1f}", flush=True)
    return results, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["p1", "p3"], required=True)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    results, scores = (run_p1 if args.protocol == "p1" else run_p3)(args)
    out = RESULTS / f"eav_cbramod_{args.protocol}_{datetime.now():%Y%m%d_%H%M}"
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
        tracker.log({"when": f"{datetime.now():%d %b %H:%M}", "phase": f"new-{args.protocol}-cbramod", "variant": name,
                     "n_subjects": len(per), "acc_epoch": row.get("acc_epoch", per.mean()), "acc_trial": per.mean(),
                     "acc_trial_lo": lo, "acc_trial_hi": hi, "f1_epoch": row.get("f1_epoch", row.get("f1_weighted")),
                     "choices": f"validation {scores[name]:.4f}", "seconds": round(time.time() - t0)})
        print(f"{name}: validation {100 * scores[name]:.1f}, test {metric} {100 * per.mean():.1f} "
              f"[{100 * lo:.1f}, {100 * hi:.1f}]", flush=True)
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    pd.concat([r.assign(model=k) for k, r in results.items()]).to_csv(out / "per_subject.csv", index=False)
    json.dump({"protocol": args.protocol, "device": DEV, "seconds": round(time.time() - t0)}, open(out / "meta.json", "w"))
    print("saved", out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
