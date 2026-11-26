"""Holm adjustment over the nine primary outcomes of the extension (declared 26 Nov 2026): per-person
correlations of scene composition, EEG and their fusion with valence, arousal and stress (raw ratings),
each tested against zero with a two-sided Wilcoxon signed-rank test over participants.

    ~/.venvs/urbeeg/bin/python -I scripts/holm_primary.py
"""
import glob
import sys
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from urbeeg.config import RESULTS  # noqa: E402
from urbeeg.metrics import per_person_r  # noqa: E402

MODELS, TARGETS = ("scene", "eeg", "fusion"), ("valence", "arousal", "stress")


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    run = Path(sorted(glob.glob(str(RESULTS / "linear_*")))[-1])
    oof = pd.read_csv(run / "oof.csv.gz")
    oof = oof[oof.model.isin(MODELS) & oof.target.isin(TARGETS) & (oof["mode"] == "raw")]
    pp = per_person_r(oof)
    rows = []
    for m in MODELS:
        for t in TARGETS:
            z = pp[(pp.model == m) & (pp.target == t)]["z"].to_numpy()
            rows.append({"model": m, "target": t, "n": len(z), "r": float(np.tanh(z.mean())),
                         "p": float(wilcoxon(z).pvalue)})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p.to_numpy())
    df["significant_holm"] = df.p_holm < 0.05
    out = RESULTS / f"holm_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    df.to_csv(out / "summary.csv", index=False)
    print(f"source: {run.name}")
    print(df.round(4).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
