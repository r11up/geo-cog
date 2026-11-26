"""Equivalence tests (two one-sided tests) for the Study 1 primary outcomes, declared 26 Nov 2026:
(a) proposed model minus EEG tower, (b) PECM minus matched-random input; per-participant trial accuracy
from the subject-level cross-validation (42 participants, mean over the three repetitions).
Smallest effect of interest +/-3 percentage points; equivalent if the 90% interval lies within it.

    ~/.venvs/urbeeg/bin/python -I scripts/equivalence_tests.py
"""
import glob
import sys
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from urbeeg.config import RESULTS, SEED  # noqa: E402

MARGIN = 3.0
PAIRS = [("proposed", "eeg_tower"), ("pecm", "pecm_matched_random")]


def tost(d, margin):
    """Paired TOST on differences d (percentage points)."""
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    p_lower = 1 - stats.t.cdf((m + margin) / se, n - 1)       # H0: mean <= -margin
    p_upper = stats.t.cdf((m - margin) / se, n - 1)           # H0: mean >= +margin
    lo, hi = stats.t.interval(0.90, n - 1, loc=m, scale=se)
    return m, lo, hi, max(p_lower, p_upper)


def main():
    run = Path(sorted(glob.glob(str(RESULTS / "study1v2_cv_*")))[-1])
    ps = pd.read_csv(run / "per_subject.csv")
    acc = ps.groupby(["condition", "subject"]).acc_trial.mean().unstack(0) * 100
    rng = np.random.default_rng(SEED)
    rows = []
    for a, b in PAIRS:
        d = (acc[a] - acc[b]).dropna().to_numpy()
        m, lo, hi, p = tost(d, MARGIN)
        boot = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
        rows.append({"comparison": f"{a}_vs_{b}", "n": len(d), "mean_diff": m, "ci90_lo": lo, "ci90_hi": hi,
                     "p_tost": p, "boot90_lo": np.percentile(boot, 5), "boot90_hi": np.percentile(boot, 95),
                     "margin": MARGIN, "equivalent": bool(lo > -MARGIN and hi < MARGIN)})
    out = RESULTS / f"equivalence_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(f"source: {run.name}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
