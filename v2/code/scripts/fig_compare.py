"""Figure: our results against published results on the same data and splits (the values behind Table 5).
Published value: hollow grey diamond. Ours: filled point with the 95% bootstrap interval over participants
(EAV) or the mean +/- 2 SD over 20 partitions (Urban Appraisal group means). Reads results/table5_forest.csv,
written by make_results_tex.py."""
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from urbeeg.config import PROJECT_ROOT, RESULTS  # noqa: E402

INK, INK2, GRID = "#1F2328", "#52514e", "#e4e3df"
PUB, OURS = "#8C8C8C", "#2A6FDB"
PANELS = {"EAV": ("A  EAV: five affective states from EEG (accuracy, %)", "%"),
          "UA": ("B  Urban Appraisal: group-mean appraisal of 55 scenes (r)", "r")}


def panel(ax, d, title, unit):
    d = d.iloc[::-1].reset_index(drop=True)                    # first row at the top
    y = range(len(d))
    for i, r in d.iterrows():
        ax.plot([r.lo, r.hi], [i, i], color=OURS, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(r.ours, i, "o", ms=6.5, color=OURS, mec="white", mew=1.2, zorder=3)
        ax.plot(r.published, i, "D", ms=7, mfc="none", mec=PUB, mew=1.4, zorder=4)
        txt = f"{r.ours:.1f}" if unit == "%" else f"{r.ours:.3f}"
        ax.text(r.hi, i, "  " + txt, va="center", ha="left", fontsize=7, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d.label, fontsize=7.5, color=INK)
    ax.set_ylim(-0.6, len(d) - 0.4)
    lo = min(d.lo.min(), d.published.min()); hi = max(d.hi.max(), d.published.max())
    pad = (hi - lo) * 0.12
    ax.set_xlim(lo - pad, hi + 2.2 * pad)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=INK2, length=2, labelsize=7)
    ax.set_title(title, loc="left", fontsize=8, color=INK, pad=6)


def main():
    d = pd.read_csv(RESULTS / "table5_forest.csv")
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.6})
    n = {k: int((d.panel == k).sum()) for k in PANELS}
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 0.42 * (n["EAV"] + n["UA"]) + 1.3),
                             gridspec_kw={"height_ratios": [max(n["EAV"], 1), max(n["UA"], 1)]})
    for ax, (key, (title, unit)) in zip(axes, PANELS.items()):
        panel(ax, d[d.panel == key], title, unit)
    h = [plt.Line2D([], [], marker="D", ls="", ms=7, mfc="none", mec=PUB, mew=1.4, label="Published"),
         plt.Line2D([], [], marker="o", ls="-", lw=2, ms=6.5, color=OURS, mec="white", label="Ours (interval)")]
    fig.legend(handles=h, loc="upper right", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ms = Path(os.environ.get("URBEEG_MS", PROJECT_ROOT / "manuscript"))
    (ms / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ms / "figures" / "fig_compare.pdf")
    out = PROJECT_ROOT / "figures" / "data"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_compare.png", dpi=200)
    print("saved", ms / "figures" / "fig_compare.pdf")


if __name__ == "__main__":
    main()
