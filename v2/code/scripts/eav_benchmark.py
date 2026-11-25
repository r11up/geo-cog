"""EAV, EEG only, the published protocol (Lee et al. 2024): per participant, speaking trials in
chronological order, first 70% for training and last 30% for testing; 5 s segments.
Protocol fixed before any run. Validation for early stopping / C: the last
training trial of each class. Development participants only unless --final.

    ~/.venvs/urbeeg/bin/python -I scripts/eav_benchmark.py --variants logreg_both_pooled ...
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

from urbeeg import tracker  # noqa: E402
from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.contrast import env_matrix, macro_f1  # noqa: E402
from urbeeg.study1 import grid_adjacency, train_eval  # noqa: E402

torch.set_num_threads(2)


def load(final=False):
    z = np.load(DERIVED / "eav" / "eav_features_v2.npz")
    D = {k: z[k] for k in z.files}
    lock = [int(s) for s in (DERIVED / "eav" / "lockbox_subjects.txt").read_text().split()]
    raw_path = DERIVED / "eav" / "eav_raw.npy"
    D["raw"] = np.load(raw_path, mmap_mode="r") if raw_path.exists() else None
    keep = np.ones(len(D["y"]), bool) if final else ~np.isin(D["subject"], lock)
    return D, keep, lock


def split(D, keep):
    """Per participant: trials sorted by time; first 70 train, last 30 test; last training trial
    of each class -> validation."""
    tr, va, te = [], [], []
    for s in np.unique(D["subject"][keep]):
        idx = np.where((D["subject"] == s) & keep)[0]
        trials = np.unique(D["trial"][idx])                       # chronological
        cut = int(round(0.7 * len(trials)))
        train_t, test_t = trials[:cut], trials[cut:]
        lab = {t: D["y"][idx][D["trial"][idx] == t][0] for t in train_t}
        val_t = [max(t for t in train_t if lab[t] == c) for c in np.unique(list(lab.values()))]
        for t in trials:
            rows = idx[D["trial"][idx] == t]
            (te if t in test_t else va if t in val_t else tr).extend(rows)
    return np.array(tr), np.array(va), np.array(te)


def feats(D, kind):
    flat = D["X"].reshape(len(D["X"]), -1)
    return {"depsd": flat, "tangent": D["T"], "both": np.hstack([flat, D["T"]])}[kind]


def fit_logreg(F, y, tr, va, te):
    from sklearn.linear_model import LogisticRegression
    best = None
    for C in (1e-4, 1e-3, 1e-2, 1e-1):
        m = LogisticRegression(C=C, max_iter=3000).fit(F[tr], y[tr])
        f = macro_f1(y[va], m.predict(F[va]))
        if best is None or f > best[0] + 1e-9:
            best = (f, m, C)
    return best[1].predict_proba(F[te]), {"C": best[2]}


def fit_raw(name, D, tr, va, te, seed, epochs=80, patience=10):
    from braindecode.models import EEGConformer, EEGNet, ShallowFBCSPNet
    torch.manual_seed(seed)
    n_ch, n_t = D["raw"].shape[1:]
    published = name == "eegnetpub"   # Lee et al. 2024: F1=8, kernel 300, D=8, F2=16; Adam, batch 32, 100 epochs
    if published:
        epochs, patience = 100, 10 ** 9
    net = {"eegnetpub": lambda: EEGNet(n_chans=n_ch, n_outputs=5, n_times=n_t, F1=8, D=8, F2=16, kernel_length=300),
           "eegnet": lambda: EEGNet(n_chans=n_ch, n_outputs=5, n_times=n_t),
           "shallow": lambda: ShallowFBCSPNet(n_chans=n_ch, n_outputs=5, n_times=n_t, final_conv_length="auto"),
           "conformer": lambda: EEGConformer(n_chans=n_ch, n_outputs=5, n_times=n_t, final_fc_length="auto")}[name]()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = net.to(dev)
    opt = (torch.optim.Adam(net.parameters(), lr=1e-3) if published
           else torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3))
    bs = 32 if published else 64
    X = lambda idx: torch.as_tensor(np.asarray(D["raw"][np.sort(idx)], dtype=np.float32)).to(dev)
    ytr = torch.as_tensor(D["y"][np.sort(tr)]).to(dev)
    Xtr, Xva, Xte = X(tr), X(va), X(te)
    g = torch.Generator().manual_seed(seed)
    best, state, bad = -1, None, 0
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(Xtr), generator=g).split(bs):
            b = b.to(dev)
            loss = torch.nn.functional.cross_entropy(net(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            f = macro_f1(D["y"][np.sort(va)], net(Xva).argmax(1).cpu().numpy())
        if published:                                     # no early stopping: keep the final model
            state = {k: v.clone() for k, v in net.state_dict().items()}
        elif f > best + 1e-4:
            best, bad, state = f, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        P = torch.softmax(net(Xte), 1).cpu().numpy()
    return P[np.argsort(np.argsort(te))], {}                # back to the order of `te`


def fit(name, D, E, A, tr, va, te, seed):
    name = name.replace("_persubject", "").replace("_pooled", "")
    if name.endswith("_raw"):
        return fit_raw(name[:-4], D, tr, va, te, seed)
    if name.startswith("logreg"):
        return fit_logreg(feats(D, name.split("_")[1]), D["y"], tr, va, te)
    if name.startswith("hgb"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        F = feats(D, name.split("_")[1])
        m = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=seed).fit(F[tr], D["y"][tr])
        return m.predict_proba(F[te]), {"n_iter": int(m.n_iter_)}
    if name.startswith("tower"):
        T = D["T"] if "both" in name else None
        best = None
        for drop, wd in [(0.3, 1e-4), (0.3, 1e-2), (0.5, 1e-4), (0.5, 1e-2)]:
            _, P, f = train_eval(D["X"], D["y"], tr, va, te, "noise", E, A, 0.75, 1.0, seed, mode="eeg_only",
                                 T=T, drop=drop, wd=wd, return_proba=True)
            if best is None or f > best[0] + 1e-9:
                best = (f, P, {"dropout": drop, "weight_decay": wd})
        return best[1], best[2]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--final", action="store_true", help="all 42 participants (after selection)")
    args = ap.parse_args()
    D, keep, lock = load(args.final)
    tr, va, te = split(D, keep)
    grid = pd.read_csv(DERIVED / "eav" / "astana_grid.csv")
    E, A = env_matrix(grid), grid_adjacency(grid)
    rng = np.random.default_rng(SEED)
    phase = "final-benchmark" if args.final else "dev-benchmark"
    out_dir = RESULTS / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    subs = np.unique(D["subject"][keep])
    print(f"{phase}: {len(subs)} participants; train {len(tr)}, val {len(va)}, test {len(te)} epochs", flush=True)
    for name in args.variants:
        t0 = time.time()
        if name.endswith("persubject"):           # one model per participant, as in the original paper
            def one(s):
                m = lambda idx: idx[D["subject"][idx] == s]
                P, c = fit(name, D, E, A, m(tr), m(va), m(te), SEED + int(s))
                return m(te), P, json.dumps(c)
            parts = Parallel(n_jobs=args.jobs)(delayed(one)(s) for s in subs)
            te_idx = np.concatenate([p[0] for p in parts]); P = np.concatenate([p[1] for p in parts])
            choices = f"{len(set(p[2] for p in parts))} participant-wise settings"
        else:
            torch.set_num_threads(6)                 # one pooled model: give it the cores
            P, c = fit(name, D, E, A, tr, va, te, SEED)
            torch.set_num_threads(2)
            te_idx, choices = te, json.dumps(c)
        pred = pd.DataFrame({"subject": D["subject"][te_idx], "trial": D["trial"][te_idx], "y": D["y"][te_idx],
                             "pred": P.argmax(1)})
        pred["p"] = list(P)
        rows = []
        for s, g in pred.groupby("subject"):
            Pg = np.stack(g.p.to_numpy())
            tp = [Pg[g.trial.to_numpy() == t].mean(0).argmax() for t in np.unique(g.trial)]
            ty = [g.y.to_numpy()[g.trial.to_numpy() == t][0] for t in np.unique(g.trial)]
            rows.append({"acc_epoch": (g.y == g.pred).mean(), "acc_trial": np.mean(np.array(tp) == np.array(ty)),
                         "f1": macro_f1(g.y.to_numpy(), g.pred.to_numpy())})
        sc = pd.DataFrame(rows)
        lo, hi = tracker.boot_ci(sc.acc_trial.to_numpy(), rng)
        tracker.log({"when": f"{datetime.now():%d %b %H:%M}", "phase": phase, "variant": name,
                     "n_subjects": len(sc), "acc_epoch": sc.acc_epoch.mean(), "acc_trial": sc.acc_trial.mean(),
                     "acc_trial_lo": lo, "acc_trial_hi": hi, "f1_epoch": sc.f1.mean(), "choices": choices,
                     "seconds": round(time.time() - t0)})
        pred.drop(columns="p").assign(p=[json.dumps(np.round(x, 4).tolist()) for x in P]).to_csv(
            out_dir / f"{phase}_{name}.csv.gz", index=False)
        print(f"{name}: epoch {100 * sc.acc_epoch.mean():.1f}%, trial {100 * sc.acc_trial.mean():.1f}% "
              f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
