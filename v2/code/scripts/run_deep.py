"""Deep models over the same crossed cells as the linear models (repetition 0 by default).

    ~/.venvs/urbeeg/bin/python -I scripts/run_deep.py --models eegnet conformer dualtower
    ~/.venvs/urbeeg/bin/python -I scripts/run_deep.py --smoke     # timing only, no accuracy

Writes one file per (model, scene input, cell) so an interrupted run resumes where it stopped.
"""
import argparse
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from urbeeg.config import RESULTS, SEED  # noqa: E402
from urbeeg.dataset import PRIMARY_TARGETS, processed_participants  # noqa: E402
from urbeeg.deep import EpochStore, device, train_predict  # noqa: E402
from urbeeg.evaluate import cell_targets  # noqa: E402
from urbeeg.splits import crossed_cells, inner_cells  # noqa: E402

sys.path.insert(0, str(CODE / "scripts"))
from run_linear import N_PARTICIPANTS, build_data  # noqa: E402

VARIANTS = {"eegnet": ["eeg"], "conformer": ["eeg"], "dualtower": ["real", "noise", "shuffled"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(VARIANTS))
    ap.add_argument("--reps", type=int, nargs="+", default=[0])
    ap.add_argument("--modes", nargs="+", default=["raw"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "deep"))
    args = ap.parse_args()

    subs = processed_participants()
    if not args.smoke and len(subs) < N_PARTICIPANTS:
        sys.exit(f"only {len(subs)}/{N_PARTICIPANTS} participants preprocessed; use --smoke")
    data = build_data(subs)
    store = EpochStore(subs)
    print(f"{len(subs)} participants, epochs in memory {store.x.nbytes / 1e9:.1f} GB, device {device()}")
    scenes = sorted(data["scene"].index)
    strata = data["scene"][["code_HT", "code_HB"]].astype(str).agg("".join, axis=1).to_dict()
    cells = [c for c in crossed_cells(subs, scenes, strata, n_p=5, n_s=4, n_rep=max(args.reps) + 1,
                                      seed=SEED) if c.rep in args.reps]
    if args.smoke:
        cells = cells[:1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part, stim = data["participant"], data["stimulus"]
    targets = PRIMARY_TARGETS

    for mode in args.modes:
        for cell in cells:
            Y = np.nan_to_num(cell_targets(data, cell, targets, mode), nan=0.0)
            tp, vp, ts, vs = inner_cells(cell)[0]
            sel_tr = np.where(np.isin(part, tp) & np.isin(stim, ts))[0]
            sel_va = np.where(np.isin(part, vp) & np.isin(stim, vs))[0]
            test = np.where(np.isin(part, cell.test_participants) & np.isin(stim, cell.test_scenes))[0]
            truth = (cell_targets(data, cell, targets, mode)[test] if mode == "deviation"
                     else data["ratings"][targets].to_numpy()[test])
            for name in args.models:
                for variant in VARIANTS[name]:
                    f = out / f"{name}_{variant}_{mode}_r{cell.rep}_p{cell.p_fold}_s{cell.s_fold}.csv.gz"
                    if f.exists() and not args.smoke:
                        continue
                    deadline = float(os.environ.get("URBEEG_DEADLINE", "inf"))
                    if time.time() > deadline:      # leave time for Kaggle to save outputs
                        print("deadline reached; stopping before", f.name, flush=True)
                        return
                    seed = SEED + 1000 * cell.rep + 10 * cell.p_fold + cell.s_fold
                    t = time.time()
                    pred = train_predict(name, store, data, Y, sel_tr, sel_va, test, seed,
                                         scene_input=variant if name == "dualtower" else "real",
                                         max_epochs=2 if args.smoke else 40)
                    label = name if name != "dualtower" else f"dualtower_{variant}"
                    if args.smoke:
                        print(f"smoke {label}: {time.time() - t:.0f} s for 2 epochs, "
                              f"{len(sel_tr)} training pairs, NaN preds {int(np.isnan(pred).sum())}")
                        continue
                    pd.concat([pd.DataFrame({
                        "rep": cell.rep, "p_fold": cell.p_fold, "s_fold": cell.s_fold,
                        "model": label, "target": t_, "mode": mode,
                        "participant": part[test], "stimulus": stim[test],
                        "y_true": truth[:, k], "y_pred": pred[:, k]})
                        for k, t_ in enumerate(targets)]).to_csv(f, index=False)


if __name__ == "__main__":
    main()
