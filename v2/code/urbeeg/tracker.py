"""Append-only experiment tracker: results/tracker.tsv (every run) and results/tracker.md (leaderboards)."""
from datetime import datetime

import numpy as np
import pandas as pd

from .config import RESULTS

TRACK, BOARD = RESULTS / "tracker.tsv", RESULTS / "tracker.md"
TITLES = {
    "dev": "Unseen people (subject-independent), 30 development participants",
    "dev-benchmark": "EAV published protocol (per participant: first 70% of trials train, last 30% test), "
                     "30 development participants",
}


def boot_ci(v, rng, n=2000):
    b = rng.choice(v, size=(n, len(v))).mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def log(row: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = not TRACK.exists()
    pd.DataFrame([row]).to_csv(TRACK, sep="\t", index=False, header=new, mode="a")
    render()


def render():
    t = pd.read_csv(TRACK, sep="\t")
    lines = ["# EAV pipeline tracker", "",
             f"Updated {datetime.now():%d %b %Y %H:%M}. Every run is kept; selection uses development "
             "participants only, the 12 lockbox participants are untouched until the final run. Chance = 20%.", ""]
    for phase, g in t.groupby("phase", sort=False):
        g = g.sort_values("acc_epoch" if "benchmark" in phase else "acc_trial", ascending=False)
        lines += [f"## {TITLES.get(phase, phase)}", "",
                  "| # | variant | epoch acc (%) | trial acc (%) [95% CI] | macro-F1 | choices | when |",
                  "|---|---|---|---|---|---|---|"]
        for i, r in enumerate(g.itertuples(), 1):
            lines.append(f"| {i} | {r.variant} | {100 * r.acc_epoch:.1f} | {100 * r.acc_trial:.1f} "
                         f"[{100 * r.acc_trial_lo:.1f}, {100 * r.acc_trial_hi:.1f}] | {r.f1_epoch:.3f} | "
                         f"{r.choices} | {r.when} |")
        if "benchmark" in phase:
            lines += ["", "Published, same protocol, all 42 participants: EEGNet 60.0% (Lee et al. 2024), "
                      "EEGformer 53.5% (Lee et al. 2024); Hyper-MML EEG only 71.83% (protocol not stated)."]
        lines.append("")
    BOARD.write_text("\n".join(lines))
