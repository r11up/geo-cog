"""Run the linear models and controls over all crossed person x scene cells.

    ~/.venvs/urbeeg/bin/python -I scripts/run_linear.py                 # full analysis
    ~/.venvs/urbeeg/bin/python -I scripts/run_linear.py --smoke         # pipeline check only

The full analysis refuses to run until all 63 participants are preprocessed, because the
fold assignment depends on the participant set. A --smoke run uses whatever is available,
one repetition and a few cells, and prints only checks (shapes, NaNs, timing), never
accuracy, so that partial data cannot steer any analysis decision.
"""
import argparse
import os
import hashlib
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

from urbeeg.config import RESULTS, SEED  # noqa: E402
from urbeeg.dataset import ALL_TARGETS, load_pairs, processed_participants  # noqa: E402
from urbeeg.evaluate import run_cells  # noqa: E402
from urbeeg.metrics import compare, summarize  # noqa: E402
from urbeeg.models_linear import model_zoo  # noqa: E402
from urbeeg.scene import low_level_features, scene_features  # noqa: E402
from urbeeg.splits import crossed_cells  # noqa: E402

N_PARTICIPANTS = 62          # 63 recorded, sub-91 excluded (config.EXCLUDED)


def code_fingerprint() -> dict:
    return {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            for p in sorted((CODE / "urbeeg").glob("*.py"))}


def build_data(subs):
    data = load_pairs(subs)
    csv = os.environ.get("URBEEG_SCENE_CSV")
    scene = (pd.read_csv(csv, index_col=0) if csv else
             scene_features().join(low_level_features(), how="left"))
    if scene.isna().any().any():
        raise ValueError("missing scene features")
    data["scene"] = scene
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--modes", nargs="+", default=["raw", "deviation"])
    args = ap.parse_args()

    subs = processed_participants()
    if not args.smoke and len(subs) < N_PARTICIPANTS:
        sys.exit(f"only {len(subs)}/{N_PARTICIPANTS} participants preprocessed; use --smoke")
    t0 = time.time()
    data = build_data(subs)
    print(f"{len(subs)} participants, {len(data['participant'])} pairs, features in {time.time() - t0:.0f} s")
    scenes = sorted(data["scene"].index)
    strata = data["scene"][["code_HT", "code_HB"]].astype(str).agg("".join, axis=1).to_dict()
    cells = crossed_cells(subs, scenes, strata, n_p=5, n_s=4,
                          n_rep=1 if args.smoke else args.reps, seed=SEED)
    if args.smoke:
        cells = cells[:2]

    def one(cell, mode):
        return run_cells(data, [cell], model_zoo, ALL_TARGETS, mode=mode, log=lambda *_: None)

    t1 = time.time()
    parts = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one)(c, m) for m in args.modes for c in cells)
    oof = pd.concat(parts, ignore_index=True)
    elapsed = time.time() - t1

    if args.smoke:
        print(f"smoke: {len(cells)} cells x {len(args.modes)} modes in {elapsed:.0f} s "
              f"({elapsed / len(cells) / len(args.modes):.0f} s per cell-mode)")
        print("rows:", len(oof), "| NaN predictions:", int(oof.y_pred.isna().sum()),
              "| models:", oof.model.nunique(), "| targets:", oof.target.nunique())
        return

    out = RESULTS / f"linear_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    oof.to_csv(out / "oof.csv.gz", index=False)
    summary = summarize(oof)
    summary.to_csv(out / "summary.csv", index=False)
    pairs = [("fusion", "scene"), ("fusion", "eeg"), ("scene", "ctrl_scene_noise"),
             ("scene", "ctrl_scene_shuffled"), ("scene", "scene_lowlevel_only"),
             ("scene+lowlevel", "scene_lowlevel_only"), ("eeg", "ctrl_eeg_shuffled"),
             ("fusion", "ctrl_fusion_noise"),
             *[("scene", f"scene_minus_{c}") for c in ("greenery", "buildings", "sky", "ground", "vehicles")]]
    pd.concat([compare(oof, a, b) for a, b in pairs]).to_csv(out / "comparisons.csv", index=False)
    json.dump({"participants": subs, "n_cells": len(cells), "reps": args.reps, "seed": SEED,
               "modes": args.modes, "targets": ALL_TARGETS, "code": code_fingerprint(),
               "seconds": round(elapsed)}, open(out / "config.json", "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
