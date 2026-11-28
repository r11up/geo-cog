"""EAV, round 7 (declared 28 Nov 2026, 23:10): does a random split inflate a convolutional network as much as
the reference covariance pipeline? One model per participant, all 42 participants:

  eegnet  EEGNet with the dataset paper's recipe (F1 8, D 8, F2 16, kernel 300 samples, Adam 1e-3, batch 32,
          100 epochs, no early stopping, final model kept)
  logreg  DE/LP + re-centred tangent-space covariances, logistic regression, C chosen by validation macro-F1

Both are trained on the same training set of each split, made by the existing split functions:
  P1  eav_benchmark.split: chronological 70/30 within participant (one split)
  P2  eav_protocols.p2_split: random trials 60/10/30, five assignments
  P4  eav_protocols.p4_split: random 70/30 segments, 10% of training for validation, five assignments

Primary: per participant, split effect d = segment accuracy under P4 minus under P1, for each model, and the paired
difference d_eegnet - d_logreg with a 95% bootstrap interval over participants. The decision rule is in
ANALYSIS-PLAN.md (28 Nov 2026, 23:10).

On Kaggle, the private datasets eav-epochs-private and eav-study1-private and the code dataset are attached; the
output holds result tables only. Locally the derived/ folder is used.

    python -I scripts/eav_eegnet_splits.py [--quick]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK_DERIVED = "/kaggle/working/derived"
if ON_KAGGLE:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "braindecode", "pyriemann"], check=False)

    def find(marker):
        for depth in range(1, 6):
            hits = glob.glob("/kaggle/input/" + "*/" * depth + marker)
            if hits:
                return hits[0][: -len(marker)].rstrip("/")
        raise SystemExit(f"{marker} not found")

    raw_dir, feat_dir, code_root = find("eav_raw.npy"), find("eav_features_v2.npz"), find("urbeeg/config.py")
    os.makedirs(f"{WORK_DERIVED}/eav", exist_ok=True)
    for src in (f"{raw_dir}/eav_raw.npy", f"{raw_dir}/eav_labels.npz", f"{feat_dir}/eav_features_v2.npz",
                f"{feat_dir}/lockbox_subjects.txt"):
        dst = f"{WORK_DERIVED}/eav/{os.path.basename(src)}"
        if not os.path.exists(dst):
            os.symlink(src, dst)
    os.environ["URBEEG_DERIVED"] = WORK_DERIVED
    os.environ["URBEEG_RESULTS"] = "/kaggle/working/results"
    sys.path[:0] = [code_root, code_root + "/scripts"]
else:
    CODE = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(CODE), str(CODE / "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import eav_benchmark  # noqa: E402
import eav_protocols  # noqa: E402
from urbeeg import tracker  # noqa: E402
from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.contrast import macro_f1  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MARGIN = 5.0          # points; rule (i)


def load():
    z = np.load(DERIVED / "eav" / "eav_features_v2.npz")
    D = {k: z[k] for k in z.files}
    D["F"] = np.hstack([D["X"].reshape(len(D["X"]), -1), D["T"]]).astype(np.float32)
    D["raw"] = np.load(DERIVED / "eav" / "eav_raw.npy", mmap_mode="r")
    assert len(D["raw"]) == len(D["y"]), "raw epochs and features differ in length"
    lab = DERIVED / "eav" / "eav_labels.npz"
    if lab.exists():                                   # the raw epochs must be in the order of the features
        L = np.load(lab)
        for k in ("y", "subject", "trial"):
            assert np.array_equal(L[k], D[k]), f"order of {k} differs between raw epochs and features"
    return D


def make_eegnet(n_ch, n_t):
    from braindecode import models
    cls = getattr(models, "EEGNet", None) or getattr(models, "EEGNetv4")
    return cls(n_chans=n_ch, n_outputs=5, n_times=n_t, F1=8, D=8, F2=16, kernel_length=300)


def fit_eegnet(D, tr, te, seed, epochs):
    torch.manual_seed(seed)
    n_ch, n_t = D["raw"].shape[1:]
    net = make_eegnet(n_ch, n_t).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtr = torch.as_tensor(np.asarray(D["raw"][tr], dtype=np.float32), device=DEV)
    ytr = torch.as_tensor(D["y"][tr], device=DEV)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(Xtr), generator=g).split(32):
            if len(b) < 2:                             # batch norm needs two samples
                continue
            b = b.to(DEV)
            loss = torch.nn.functional.cross_entropy(net(Xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        Xte = torch.as_tensor(np.asarray(D["raw"][te], dtype=np.float32), device=DEV)
        return torch.softmax(net(Xte), 1).cpu().numpy()


def fit_logreg(D, tr, va, te):
    m, C = eav_protocols.fit_lr(D["F"], D["y"], tr, va)
    return m.predict_proba(D["F"][te]), C


def score(D, te, P):
    y, trial = D["y"][te], D["trial"][te]
    ts = np.unique(trial)
    tp = np.array([P[trial == t].mean(0).argmax() for t in ts])
    ty = np.array([y[trial == t][0] for t in ts])
    return {"acc_segment": float((P.argmax(1) == y).mean()), "acc_trial": float((tp == ty).mean()),
            "f1_segment": float(macro_f1(y, P.argmax(1)))}


def splits(D, protocol, n_rep):
    if protocol == "p1":
        return [eav_benchmark.split(D, np.ones(len(D["y"]), bool))]
    fn = {"p2": eav_protocols.p2_split, "p4": eav_protocols.p4_split}[protocol]
    return [fn(D, SEED + rep) for rep in range(n_rep)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 participants, 2 epochs, 1 assignment")
    args = ap.parse_args()
    t0 = time.time()
    D = load()
    subjects = np.unique(D["subject"])
    epochs, n_rep = (2, 1) if args.quick else (100, 5)
    if args.quick:
        subjects = subjects[:2]
    rows = []
    for protocol in ("p1", "p2", "p4"):
        for rep, (tr, va, te) in enumerate(splits(D, protocol, n_rep)):
            sub = lambda idx, s: idx[D["subject"][idx] == s]

            def lr_one(s):
                P, C = fit_logreg(D, sub(tr, s), sub(va, s), sub(te, s))
                return s, P, C
            lr = {s: (P, C) for s, P, C in Parallel(n_jobs=4)(delayed(lr_one)(s) for s in subjects)}
            for s in subjects:
                te_s = sub(te, s)
                P_e = fit_eegnet(D, sub(tr, s), te_s, SEED + 1000 * rep + int(s), epochs)
                base = {"protocol": protocol, "rep": rep, "subject": int(s), "n_train": len(sub(tr, s)),
                        "n_test": len(te_s)}
                rows.append(base | {"model": "eegnet"} | score(D, te_s, P_e))
                rows.append(base | {"model": "logreg", "C": lr[s][1]} | score(D, te_s, lr[s][0]))
            g = pd.DataFrame(rows).query("protocol == @protocol and rep == @rep").groupby("model")
            print(f"{protocol} rep {rep}: " + ", ".join(f"{m} seg {100 * d.acc_segment.mean():.1f} trial "
                  f"{100 * d.acc_trial.mean():.1f}" for m, d in g) + f" ({time.time() - t0:.0f} s)", flush=True)
    res = pd.DataFrame(rows)
    out = RESULTS / f"eav_eegnet_splits_{datetime.now():%Y%m%d_%H%M}{'_quick' if args.quick else ''}"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "per_subject.csv", index=False)

    rng = np.random.default_rng(SEED)
    per = res.groupby(["model", "protocol", "subject"])[["acc_segment", "acc_trial"]].mean().reset_index()
    summ = []
    for (model, protocol), g in per.groupby(["model", "protocol"]):
        metric = "acc_trial" if protocol == "p2" else "acc_segment"
        v = 100 * g[metric].to_numpy()
        lo, hi = tracker.boot_ci(v, rng)
        summ.append({"model": model, "protocol": protocol, "metric": metric, "mean": v.mean(), "lo": lo, "hi": hi,
                     "acc_segment": 100 * g.acc_segment.mean(), "acc_trial": 100 * g.acc_trial.mean(),
                     "n_subjects": len(g)})
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)

    seg = per.pivot_table(index="subject", columns=["model", "protocol"], values="acc_segment") * 100
    d = {m: (seg[(m, "p4")] - seg[(m, "p1")]).to_numpy() for m in ("eegnet", "logreg")}
    diff = d["eegnet"] - d["logreg"]
    stats = {m: {"mean": float(d[m].mean()), "ci": [float(x) for x in tracker.boot_ci(d[m], rng)]} for m in d}
    lo, hi = (float(x) for x in tracker.boot_ci(diff, rng))
    if hi < 0:
        verdict = "ii: logreg gains more from the split; P2 not reported as a gain"
    elif lo > 0:
        verdict = "iii: EEGNet gains more from the split; P2 reported as a method gain"
    elif lo >= -MARGIN and hi <= MARGIN:
        verdict = "i: alike within +/-5 points; P2 reported as a method gain with both split effects"
    else:
        verdict = "iv: inconclusive; P2 reported without a gain claim, with both split effects"
    stats["difference_eegnet_minus_logreg"] = {"mean": float(diff.mean()), "ci": [lo, hi]}
    stats["verdict"] = verdict
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    md5 = {f: hashlib.md5(Path(sys.modules[f].__file__).read_bytes()).hexdigest() for f in ("eav_benchmark", "eav_protocols")}
    import braindecode
    (out / "meta.json").write_text(json.dumps({"declared": "28 Nov 2026 23:10", "device": DEV, "epochs": epochs,
                                               "n_rep": n_rep, "n_subjects": int(len(subjects)), "quick": args.quick,
                                               "braindecode": braindecode.__version__, "torch": torch.__version__,
                                               "split_code_md5": md5, "seconds": round(time.time() - t0)}, indent=2))
    print(pd.DataFrame(summ).round(2).to_string(index=False))
    print(json.dumps(stats, indent=2))
    print("saved", out, f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if ON_KAGGLE:                                  # no EEG data or links in the output
            shutil.rmtree(WORK_DERIVED, ignore_errors=True)
