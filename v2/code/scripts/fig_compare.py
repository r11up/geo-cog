"""Figure: our results against published results on the same data and splits (the values behind Table 5),
drawn at the printed width of the manuscript so that font sizes are final. Each row joins the published
value (hollow diamond) to ours (filled point with the 95% bootstrap interval over participants for EAV, or
over the 55 scenes for the Urban Appraisal group means). The right-hand columns give both values and the gain,
in bold where our interval lies above the published value (EAV) or where the paired difference from our
reproduction of the published model on the same scenes is above zero (group means). EAV rows are ordered by the
published value; panel B shows the split-half noise ceiling. Reads results/table5_forest.csv, written by
make_results_tex.py."""
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from urbeeg.config import PROJECT_ROOT, RESULTS  # noqa: E402

INK, INK2, MUTED, GRID = "#1F2328", "#52514e", "#77766f", "#e4e3df"
PUB, OURS, LINK = "#6f6e69", "#2A6FDB", "#cfcec9"
WIDTH_IN = 372 / 72                                         # text width of the manuscript
PANELS = [("EAV", "A  EAV: five affective states from EEG (accuracy, %)", (50, 90), [50, 60, 70, 80, 90], 1),
          ("UA", "B  Urban Appraisal, Tan & Deng protocol: group means of 55 scenes (r)", (0.5, 1.0),
           [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 3)]
COL1, COL2 = 1.05, 1.47                                     # right-hand columns, in axes fractions


def fmt(v, nd):
    return str(Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def noise_ceiling():
    f = RESULTS / "vision5" / "group_benchmark_v5.csv"
    if not f.exists():
        return None
    V = pd.read_csv(f)
    c = V[V.features == "noise_ceiling_split_half"].mean_r
    return float(c.iloc[0]) if len(c) else None


def panel(fig, ax, d, title, xlim, ticks, nd, ceiling=None):
    n = len(d)
    yax = ax.get_yaxis_transform()                          # x in axes fraction, y in data units
    for y, r in zip(range(n - 1, -1, -1), d.itertuples()):
        higher = bool(r.higher) if "higher" in r._fields else r.lo > r.published
        ax.plot([r.published, r.ours], [y, y], color=LINK, lw=3, solid_capstyle="butt", zorder=1)
        ax.plot([r.lo, r.hi], [y, y], color=OURS, lw=1.8, solid_capstyle="round", zorder=2)
        ax.plot(r.ours, y, "o", ms=5.5, color=OURS, mec="white", mew=0.9, zorder=3)
        ax.plot(r.published, y, "D", ms=5.2, mfc="white", mec=PUB, mew=1.2, zorder=4)
        ax.text(-0.04, y + 0.17, r.name, transform=yax, ha="right", va="center", fontsize=7.5, color=INK)
        ax.text(-0.04, y - 0.21, r.detail, transform=yax, ha="right", va="center", fontsize=6.5, color=MUTED)
        pub_nd = 1 if nd == 1 else 2
        ax.text(COL1, y, f"{fmt(r.published, pub_nd)} → {fmt(r.ours, nd)}", transform=yax, ha="left",
                va="center", fontsize=7, color=INK, clip_on=False)
        gain = f"+{fmt(r.ours - r.published, nd)}" if r.ours >= r.published else fmt(r.ours - r.published, nd)
        ax.text(COL2, y, gain if higher else f"{gain} (tie)", transform=yax, ha="left", va="center",
                fontsize=7, color=INK if higher else MUTED, fontweight="bold" if higher else "normal",
                clip_on=False)
    if ceiling:
        ax.axvline(ceiling, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.text(ceiling - 0.004, n - 0.45, "noise ceiling", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="x", colors=INK2, length=2, labelsize=7, pad=2)
    box = ax.get_position()
    fig.text(0.01, box.y1 + 0.035, title, ha="left", va="bottom", fontsize=8, color=INK, fontweight="bold")
    yh = ax.get_ylim()[1]
    for x, h in [(COL1, "Published → ours"), (COL2, "Gain")]:
        ax.text(x, yh, h, transform=yax, ha="left", va="bottom", fontsize=6.5, color=MUTED, clip_on=False)


def main():
    d = pd.read_csv(RESULTS / "table5_forest.csv")
    if "name" not in d:
        d["name"], d["detail"] = d["label"], ""
    d["detail"] = d["detail"].fillna("")
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    parts = [d[d.panel == "EAV"].sort_values("published"), d[d.panel == "UA"]]
    nA, nB = len(parts[0]), len(parts[1])
    fig = plt.figure(figsize=(WIDTH_IN, 0.31 * (nA + nB) + 1.25))
    gs = fig.add_gridspec(2, 1, height_ratios=[max(nA, 1), max(nB, 1)], hspace=0.62, left=0.34, right=0.70,
                          top=0.86, bottom=0.07)
    ceiling = noise_ceiling()
    for k, ((key, title, xlim, ticks, nd), part) in enumerate(zip(PANELS, parts)):
        ax = fig.add_subplot(gs[k])
        panel(fig, ax, part, title, xlim, ticks, nd, ceiling if key == "UA" else None)
    h = [plt.Line2D([], [], marker="D", ls="", ms=5.2, mfc="white", mec=PUB, mew=1.2, label="Published"),
         plt.Line2D([], [], marker="o", ls="-", lw=1.8, ms=5.5, color=OURS, mec="white", label="Ours, with interval"),
         plt.Line2D([], [], ls="-", lw=3, color=LINK, label="Gain")]
    fig.legend(handles=h, loc="upper left", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.005, 1.0),
               handlelength=1.6, columnspacing=1.4)
    fig.text(0.01, 0.948, "Bold gain: our interval lies above the published value (A); the paired gain over our "
             "reproduction of the published model is above 0 (B)", ha="left", va="top", fontsize=6.3, color=MUTED)
    ms = Path(os.environ.get("URBEEG_MS", PROJECT_ROOT / "manuscript"))
    (ms / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ms / "figures" / "fig_compare.pdf")
    out = PROJECT_ROOT / "figures" / "data"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_compare.png", dpi=300)
    print("saved", ms / "figures" / "fig_compare.pdf")


if __name__ == "__main__":
    main()
