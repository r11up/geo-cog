"""Study 1: re-run of the original submission's experiments (Tables 2-5) with the original
architecture, plus input controls.

    ~/.venvs/urbeeg/bin/python -I scripts/run_study1.py --jobs 4
    ~/.venvs/urbeeg/bin/python -I scripts/run_study1.py --smoke      # timing only
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
from scipy.stats import wilcoxon  # noqa: E402

from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.contrast import (EnvSampler, env_matrix, macro_f1, per_subject_scores,  # noqa: E402
                             select_bandwidth, subject_folds)
from urbeeg.study1 import grid_adjacency, train_eval  # noqa: E402

torch.set_num_threads(2)


def coarsen(grid: pd.DataFrame, k: int) -> pd.DataFrame:
    """Aggregate the 1 km grid into k x k km cells (mean of member cells)."""
    g = grid.copy()
    g["row"], g["col"] = g["row"] // k, g["col"] // k
    out = g.groupby(["row", "col"])[["ndvi", "ndbi", "building", "roads"]].mean().reset_index()
    return out


def baseline(kind, X, y, tr, va, te, E, h, seed):
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(200, max_depth=20, n_jobs=2, random_state=seed)
        return m.fit(X[tr].reshape(len(tr), -1), y[tr]).predict(X[te].reshape(len(te), -1))
    if kind == "env_only":
        from sklearn.neural_network import MLPClassifier
        s = EnvSampler(E, "prior", h=h, seed=seed)
        m = MLPClassifier((128, 64), max_iter=200, random_state=seed)
        return m.fit(s.train(y[tr]), y[tr]).predict(s.test(y[te]))
    if kind == "lstm":
        torch.manual_seed(seed)
        lstm = torch.nn.LSTM(10, 128, num_layers=2, batch_first=True, bidirectional=True)
        head = torch.nn.Linear(256, 5)
        params = list(lstm.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)
        fwd = lambda x: head(lstm(x)[0].mean(1))
        Xtr, Ytr = t(X[tr]), torch.as_tensor(y[tr])
        best, state, bad = -1, None, 0
        for _ in range(40):
            for b in torch.randperm(len(tr)).split(128):
                loss = torch.nn.functional.cross_entropy(fwd(Xtr[b]), Ytr[b])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                f = macro_f1(y[va], fwd(t(X[va])).argmax(1).numpy())
            if f > best + 1e-4:
                best, bad, state = f, 0, [p.detach().clone() for p in params]
            else:
                bad += 1
                if bad >= 6:
                    break
        with torch.no_grad():
            for p, s_ in zip(params, state):
                p.copy_(s_)
            return fwd(t(X[te])).argmax(1).numpy()
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--singapore-only", action="store_true",
                    help="only the environmental-domain shift (needs derived/eav/singapore_grid.csv)")
    args = ap.parse_args()

    z = np.load(DERIVED / "eav" / "eav_features.npz")
    X, y, subj = z["X"], z["y"], z["subject"]
    grid = pd.read_csv(DERIVED / "eav" / "astana_grid.csv")
    E, A = env_matrix(grid), grid_adjacency(grid)
    sg_path = DERIVED / "eav" / "singapore_grid.csv"
    sg = pd.read_csv(sg_path) if sg_path.exists() else None
    h = select_bandwidth(E, seed=SEED)
    scales = {k: coarsen(grid, k) for k in (2, 5)}
    print(f"{len(np.unique(subj))} subjects, {len(y)} epochs, h = {h:.3f}, Singapore grid: {sg is not None}")

    # (name, environment condition, fusion mode)
    fusion = [("eeg_only", "noise", "eeg_only"), ("proposed_bi", "prior", "bi"),
              ("unidirectional", "prior", "uni"), ("diag_test_labels", "prior_test_labels", "bi"),
              ("ctrl_noise", "noise", "bi"), ("ctrl_matched_random", "matched_random", "bi"),
              ("ctrl_removed_at_test", "prior_removed_test", "bi"),
              ("abl_reversed", "prior_reversed", "bi"), ("abl_random_pairing", "unconditional", "bi"),
              ("abl_shuffled_labels", "shuffled_labels", "bi")]
    folds = subject_folds(subj, n_rep=1 if args.smoke else args.reps, seed=SEED)
    if args.smoke:
        folds, fusion = folds[:1], fusion[:2]
    if args.singapore_only:
        if sg is None:
            sys.exit("no Singapore grid yet")
        fusion = [("proposed_bi", "prior", "bi")]

    def run_fusion(name, cond, mode, fold):
        rep, f, tr_s, va_s, te_s = fold
        tr, va, te = (np.where(np.isin(subj, s))[0] for s in (tr_s, va_s, te_s))
        seed = SEED + 100 * rep + f
        t0 = time.time()
        rows = []
        if name == "proposed_bi":
            pred, model = train_eval(X, y, tr, va, te, cond, E, A, 0.75, h, seed, mode, return_model=True)
            if not args.singapore_only:
                rows.append(("proposed_bi", pred))
            if sg is not None:            # environmental-domain shift: EEG tower frozen
                Es, As = env_matrix(sg), grid_adjacency(sg)
                hs = select_bandwidth(Es, seed=SEED)
                rows.append(("singapore_shift", train_eval(X, y, tr, va, te, "prior", Es, As, 0.75, hs,
                                                           seed, "bi", init_from=model, freeze_eeg=True)))
            for k, gk in ({} if args.singapore_only else scales).items():  # spatial scale (MAUP)
                Ek, Ak = env_matrix(gk), grid_adjacency(gk)
                rows.append((f"scale_{k}km", train_eval(X, y, tr, va, te, "prior", Ek, Ak, 0.75,
                                                        select_bandwidth(Ek, seed=SEED), seed, "bi")))
        else:
            rows.append((name, train_eval(X, y, tr, va, te, cond, E, A, 0.75, h, seed, mode)))
        return [pd.DataFrame({"condition": n, "rep": rep, "fold": f, "subject": subj[te], "y": y[te],
                              "pred": p, "seconds": time.time() - t0}) for n, p in rows]

    def run_baseline(kind, fold):
        rep, f, tr_s, va_s, te_s = fold
        tr, va, te = (np.where(np.isin(subj, s))[0] for s in (tr_s, va_s, te_s))
        pred = baseline(kind, X, y, tr, va, te, E, h, SEED + 100 * rep + f)
        return [pd.DataFrame({"condition": f"baseline_{kind}", "rep": rep, "fold": f,
                              "subject": subj[te], "y": y[te], "pred": pred, "seconds": 0.0})]

    jobs = [delayed(run_fusion)(n, c, m, fd) for n, c, m in fusion for fd in folds]
    if not args.smoke and not args.singapore_only:
        jobs += [delayed(run_baseline)(k, fd) for k in ("rf", "lstm", "env_only") for fd in folds]
    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=5)(jobs)
    preds = pd.concat([d for part in out for d in part], ignore_index=True)
    if args.smoke:
        print(f"smoke: {time.time() - t0:.0f} s; per fit (s):",
              preds.groupby("condition").seconds.first().round(0).to_dict())
        return

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
    piv_acc = scores.pivot(index="subject", columns="condition", values="acc")
    comps = []
    for a, b in [("proposed_bi", "eeg_only"), ("unidirectional", "eeg_only"), ("proposed_bi", "unidirectional"),
                 ("proposed_bi", "ctrl_noise"), ("proposed_bi", "ctrl_matched_random"),
                 ("proposed_bi", "ctrl_removed_at_test"), ("proposed_bi", "abl_reversed"),
                 ("proposed_bi", "abl_random_pairing"), ("proposed_bi", "abl_shuffled_labels"),
                 ("diag_test_labels", "proposed_bi"), ("singapore_shift", "proposed_bi"),
                 ("scale_2km", "proposed_bi"), ("scale_5km", "proposed_bi"), ("ctrl_noise", "eeg_only")]:
        if a in piv_acc and b in piv_acc:
            d = (piv_acc[a] - piv_acc[b]).to_numpy()
            boot = rng.choice(d, size=(2000, len(d))).mean(1)
            comps.append({"a": a, "b": b, "mean_diff_acc": d.mean(), "lo": np.percentile(boot, 2.5),
                          "hi": np.percentile(boot, 97.5),
                          "wilcoxon_p": wilcoxon(d).pvalue if np.any(d != 0) else np.nan})
    res = RESULTS / f"study1{'_singapore' if args.singapore_only else ''}_{datetime.now():%Y%m%d_%H%M}"
    res.mkdir(parents=True)
    preds.to_csv(res / "predictions.csv.gz", index=False)
    scores.to_csv(res / "per_subject.csv", index=False)
    pd.DataFrame(summ).to_csv(res / "summary.csv", index=False)
    pd.DataFrame(comps).to_csv(res / "comparisons.csv", index=False)
    split = preds.groupby(["condition", "rep", "fold"]).apply(
        lambda d: pd.Series({"acc": (d.y == d.pred).mean(), "f1": macro_f1(d.y.to_numpy(), d.pred.to_numpy())}),
        include_groups=False).reset_index()
    split.to_csv(res / "per_split.csv", index=False)
    json.dump({"bandwidth": h, "reps": args.reps, "seed": SEED, "singapore": sg is not None},
              open(res / "config.json", "w"), indent=1)
    print("saved", res)


if __name__ == "__main__":
    main()
