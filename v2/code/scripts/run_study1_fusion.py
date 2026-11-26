"""Study 1, round 6: fusion-operator controls and baselines on the cross-validation folds of the final run.

Same folds, seeds and per-fold dropout / weight decay as the saved subject-level cross-validation
(study1v2_cv_*, read from its config.json, not re-tuned); same training as every Study 1 network.

Fusion operators (urbeeg.study1): 'uni_res' single-direction cross-attention with a residual path;
'concat' concatenation without attention; 'tok_bi' / 'tok_uni' token-level cross-attention between the
EEG-tower tokens and the grid-node tokens, in both directions or EEG-query only.
Baselines with class probabilities (trial accuracy): Random Forest on the logistic-regression features,
bidirectional LSTM over the channel feature vectors with the covariance vector as a further step, and
the environment-only MLP trained on the prior. Runs on the GPUs of the machine if there are any (task k on
GPU k mod n), otherwise on the CPU.

    python -I scripts/run_study1_fusion.py --jobs 4
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
from urbeeg.contrast import EnvSampler, env_matrix, macro_f1, select_bandwidth  # noqa: E402
from urbeeg.study1 import Study1Model, grid_adjacency, train_eval  # noqa: E402
from run_study1_v2 import fold_sets, per_subject  # noqa: E402

torch.set_num_threads(2)
# name: (environment sampling, fusion mode)
FUSION = {
    "proposed_uni_res": ("unconditional", "uni_res"),
    "proposed_concat": ("unconditional", "concat"),
    "proposed_tok_bi": ("unconditional", "tok_bi"),
    "proposed_tok_uni": ("unconditional", "tok_uni"),
    "proposed_tok_bi_noise": ("noise", "tok_bi"),
    "pecm_concat": ("prior", "concat"),
    "pecm_tok_bi": ("prior", "tok_bi"),
    "pecm_matched_random_concat": ("matched_random", "concat"),
    "pecm_matched_random_tok_bi": ("matched_random", "tok_bi"),
}
BASELINES = ["baseline_rf", "baseline_lstm", "baseline_env_only"]
COMPARISONS = [
    ("proposed_uni_res", "eeg_tower"), ("proposed_concat", "eeg_tower"), ("proposed_tok_bi", "eeg_tower"),
    ("proposed_tok_uni", "eeg_tower"), ("proposed", "proposed_uni_res"), ("proposed", "proposed_concat"),
    ("proposed_tok_bi", "proposed_tok_uni"), ("proposed_tok_bi", "proposed_tok_bi_noise"),
    ("proposed_tok_bi", "proposed"), ("proposed_uni_res", "proposed_uni"),
    ("pecm_concat", "eeg_tower"), ("pecm_tok_bi", "eeg_tower"),
    ("pecm_concat", "pecm_matched_random_concat"), ("pecm_tok_bi", "pecm_matched_random_tok_bi"),
    ("pecm", "pecm_concat"), ("pecm", "pecm_tok_bi"),
    ("eeg_tower", "baseline_rf"), ("eeg_tower", "baseline_lstm"), ("eeg_logreg", "baseline_rf"),
    ("eeg_logreg", "baseline_lstm"),
]


class LSTMNet(torch.nn.Module):
    """Two-layer bidirectional LSTM over the 30 channel feature vectors; the covariance vector is
    linearly embedded to the same width and appended as a 31st step."""

    def __init__(self, n_extra, drop):
        super().__init__()
        self.extra = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(n_extra, 10))
        self.lstm = torch.nn.LSTM(10, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.head = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(256, 5))

    def forward(self, x, t):
        h = torch.cat([x, self.extra(t)[:, None]], 1)
        return self.head(self.lstm(h)[0].mean(1))


def baseline(kind, D, tr, va, te, E, h, seed, drop, wd, device=None):
    """Class probabilities for the test epochs."""
    if kind == "baseline_rf":
        from sklearn.ensemble import RandomForestClassifier
        F = np.hstack([D["X"].reshape(len(D["X"]), -1), D["T"]])
        m = RandomForestClassifier(200, max_depth=20, n_jobs=2, random_state=seed).fit(F[tr], D["y"][tr])
        return m.predict_proba(F[te])
    if kind == "baseline_env_only":
        from sklearn.neural_network import MLPClassifier
        s = EnvSampler(E, "prior", h=h, seed=seed)
        m = MLPClassifier((128, 64), max_iter=200, random_state=seed).fit(s.train(D["y"][tr]), D["y"][tr])
        return m.predict_proba(s.test(D["y"][te]))
    if kind == "baseline_lstm":
        dev = torch.device(device) if device else torch.device("cpu")
        torch.manual_seed(seed)
        net = LSTMNet(D["T"].shape[1], drop).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        X, T, y = D["X"], D["T"], D["y"]
        Xtr, Ttr, Ytr = t(X[tr]), t(T[tr]), torch.as_tensor(y[tr], device=dev)
        g = torch.Generator().manual_seed(seed)
        fwd = lambda idx: torch.cat([net(t(X[idx[s:s + 512]]), t(T[idx[s:s + 512]])) for s in range(0, len(idx), 512)])
        best, state, bad = -1.0, None, 0
        for _ in range(40):
            net.train()
            for b in torch.randperm(len(tr), generator=g).split(128):
                b = b.to(dev)
                loss = torch.nn.functional.cross_entropy(net(Xtr[b], Ttr[b]), Ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                f = macro_f1(y[va], fwd(va).argmax(1).cpu().numpy())
            if f > best + 1e-4:
                best, bad, state = f, 0, {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 6:
                    break
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            return torch.softmax(fwd(te), 1).cpu().numpy()
    raise ValueError(kind)


def n_params(D, E, A):
    out = {}
    for mode in ("eeg_only", "bi", "uni", "uni_res", "concat", "tok_bi", "tok_uni"):
        m = Study1Model(A, E, mode=mode, n_extra=D["T"].shape[1])
        out[mode] = int(sum(p.numel() for p in m.parameters()))
    out["lstm"] = int(sum(p.numel() for p in LSTMNet(D["T"].shape[1], 0.3).parameters()))
    out["env_only_mlp"] = int(E.shape[1] * 128 + 128 + 128 * 64 + 64 + 64 * 5 + 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="subset of conditions (debugging)")
    ap.add_argument("--folds", type=int, default=None, help="first n folds only (debugging)")
    args = ap.parse_args()
    z = np.load(DERIVED / "eav" / "eav_features_v2.npz")
    D = {k: z[k] for k in z.files}
    lock = np.array([int(s) for s in (DERIVED / "eav" / "lockbox_subjects.txt").read_text().split()])
    grid = pd.read_csv(DERIVED / "eav" / "astana_grid.csv")
    E, A = env_matrix(grid), grid_adjacency(grid)
    ref = sorted(RESULTS.glob("study1v2_cv_*"))[-1]
    cfg = json.load(open(ref / "config.json"))
    bw = cfg["bandwidth"]["1km"]                  # the reference run's bandwidth
    bw_check = select_bandwidth(E, seed=SEED)     # recomputed here, recorded as a check
    tuned = {int(k): tuple(v) for k, v in cfg["tuned"].items()}
    folds = fold_sets("cv", D, lock)
    if args.folds:
        folds = folds[:args.folds]
    names = list(FUSION) + BASELINES
    if args.only:
        names = [n for n in names if n in args.only]
    print(f"reference run {ref.name}; {len(folds)} folds; conditions {names}", flush=True)
    t0 = time.time()

    n_gpu = torch.cuda.device_count()
    print(f"GPUs: {n_gpu}", flush=True)

    def run(k, i, name):
        rep, f, tr, va, te = folds[i]
        drop, wd = tuned[i]
        seed = SEED + 100 * rep + f
        device = f"cuda:{k % n_gpu}" if n_gpu else None
        if name in BASELINES:
            return i, name, baseline(name, D, tr, va, te, E, bw, seed, drop, wd, device=device)
        cond, mode = FUSION[name]
        _, P, _ = train_eval(D["X"], D["y"], tr, va, te, cond, E, A, 0.75, bw, seed, mode=mode,
                             T=D["T"], drop=drop, wd=wd, return_proba=True, device=device)
        return i, name, P

    jobs = [(i, n) for n in names for i in range(len(folds))]
    out = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run)(k, i, n) for k, (i, n) in enumerate(jobs))
    frames = []
    for i, name, P in out:
        rep, f, tr, va, te = folds[i]
        frames.append(pd.DataFrame({"condition": name, "rep": rep, "fold": f, "subject": D["subject"][te],
                                    "trial": D["trial"][te], "y": D["y"][te], "p": list(P)}))
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
    allsc = pd.concat([pd.read_csv(ref / "per_subject.csv"), sc], ignore_index=True)
    comps = []
    for a, b in COMPARISONS:
        for m in ("acc_trial", "acc_epoch"):
            piv = allsc.pivot(index="subject", columns="condition", values=m)
            if a in piv and b in piv:
                d = (piv[a] - piv[b]).dropna().to_numpy()
                lo, hi = tracker.boot_ci(d, rng)
                comps.append({"metric": m, "a": a, "b": b, "mean_diff": d.mean(), "lo": lo, "hi": hi,
                              "wilcoxon_p": wilcoxon(d).pvalue if np.any(d != 0) else np.nan})
    res = RESULTS / f"study1fusion_cv_{datetime.now():%Y%m%d_%H%M}"
    res.mkdir(parents=True)
    pred.assign(p=[json.dumps(np.round(x, 4).tolist()) for x in pred.p]).to_csv(res / "predictions.csv.gz", index=False)
    sc.to_csv(res / "per_subject.csv", index=False)
    summ.to_csv(res / "summary.csv", index=False)
    pd.DataFrame(comps).to_csv(res / "comparisons.csv", index=False)
    json.dump({"reference_run": ref.name, "tuned": {i: tuned[i] for i in range(len(folds))}, "bandwidth": bw,
               "bandwidth_recomputed": bw_check,
               "test_subjects": {i: sorted(np.unique(D["subject"][fd[4]]).tolist()) for i, fd in enumerate(folds)},
               "n_params": n_params(D, E, A), "conditions": names, "n_folds": len(folds), "n_gpu": n_gpu,
               "gpu": torch.cuda.get_device_name(0) if n_gpu else "", "seconds": round(time.time() - t0)},
              open(res / "config.json", "w"), indent=1)
    if not args.only and not args.folds:
        for r in summ.itertuples():
            tracker.log({"when": f"{datetime.now():%d %b %H:%M}", "phase": "final-cv-round6",
                         "variant": r.condition, "n_subjects": r.n_subjects, "acc_epoch": r.acc_epoch,
                         "acc_trial": r.acc_trial, "acc_trial_lo": r.acc_trial_lo, "acc_trial_hi": r.acc_trial_hi,
                         "f1_epoch": r.f1, "choices": "", "seconds": round(time.time() - t0)})
    print(summ[["condition", "acc_trial", "acc_epoch", "f1"]].to_string(index=False), flush=True)
    print("saved", res, flush=True)


if __name__ == "__main__":
    main()
