"""Figure: how far our results lie above published results on the same data and splits (the values behind the
comparison table), drawn at the printed width of the manuscript so that font sizes are final.
Each row shows the gain with its 95% bootstrap interval and a line at zero (no gain):
  A, EAV: our accuracy minus the published accuracy, interval over participants;
  B, Urban Appraisal group means: the paired gain in r over our reproduction of the published model on the same
     scenes, interval over scenes (scene bootstrap).
A gain is bold, and its point blue, when the interval lies above zero; otherwise it is a tie (grey).
The right-hand columns give the published and our values (and, in B, our reproduction).
Reads results/table5_forest.csv, written by make_results_tex.py."""
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
WIN, TIE = "#2A6FDB", "#9a9993"
W_IN = 372 / 72                                              # text width of the manuscript (inches)
ROW, AXIS, TITLE, HEAD, GAP = 0.26, 0.40, 0.19, 0.13, 0.08   # vertical layout (inches)
X0, X1 = 1.70, 3.48                                          # plot area (inches from the left edge)
C1, C2 = 3.60, 4.58                                          # right-hand columns
PANELS = [
    dict(key="EAV", title="A  EAV: five affective states from EEG, strictest published splits",
         xlabel="Gain over the published accuracy (percentage points)", xlim=(-5, 10), ticks=[-5, 0, 5, 10],
         nd=1, gain_head="Gain"),
    dict(key="UA", title="B  Urban Appraisal: group means of 55 scenes (Tan & Deng protocol)",
         xlabel="Paired gain in r over our reproduction", xlim=(-0.02, 0.15), ticks=[0, 0.05, 0.10, 0.15],
         nd=3, gain_head="Paired gain"),
]


def fmt(v, nd, sign=False):
    s = str(Decimal(str(v)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
    s = s.replace("-", "−")
    return ("+" + s) if sign and v >= 0 else s


def draw_panel(fig, d, p, top, H):
    n = len(d)
    h = n * ROW
    ax = fig.add_axes([X0 / W_IN, (top - TITLE - HEAD - h) / H, (X1 - X0) / W_IN, h / H])
    yax = ax.get_yaxis_transform()                           # x in axes fraction, y in data units
    fx = lambda inch: (inch - X0) / (X1 - X0)                 # inches from the left edge -> axes fraction
    ax.axvline(0, color=INK2, lw=0.9, zorder=1)
    for y, r in zip(range(n - 1, -1, -1), d.itertuples()):
        win = bool(r.higher)
        col = WIN if win else TIE
        ax.plot([r.gain_lo, r.gain_hi], [y, y], color=col, lw=2.0, solid_capstyle="round", zorder=2)
        ax.plot(r.gain, y, "o", ms=5.6, color=col, mec="white", mew=0.9, zorder=3)
        ax.text(fx(X0 - 0.08), y + 0.17, r.name, transform=yax, ha="right", va="center", fontsize=7.5, color=INK,
                clip_on=False)
        ax.text(fx(X0 - 0.08), y - 0.21, r.detail, transform=yax, ha="right", va="center", fontsize=6.5,
                color=MUTED, clip_on=False)
        pub_nd = 1 if p["nd"] == 1 else 2
        values = f"{fmt(r.published, pub_nd)} → {fmt(r.ours, p['nd'])}"
        if p["key"] == "UA":
            ax.text(fx(C1), y + 0.15, values, transform=yax, ha="left", va="center", fontsize=7, color=INK,
                    clip_on=False)
            ax.text(fx(C1), y - 0.22, f"reproduced {fmt(r.reference, p['nd'])}", transform=yax, ha="left",
                    va="center", fontsize=6.2, color=MUTED, clip_on=False)
        else:
            ax.text(fx(C1), y, values, transform=yax, ha="left", va="center", fontsize=7, color=INK, clip_on=False)
        gain = fmt(r.gain, p["nd"], sign=True)
        ax.text(fx(C2), y, gain if win else f"{gain} (tie)", transform=yax, ha="left", va="center", fontsize=7,
                color=INK if win else MUTED, fontweight="bold" if win else "normal", clip_on=False)
    ax.set_xlim(*p["xlim"])
    ax.set_xticks(p["ticks"])
    ax.set_xticklabels(["0" if t == 0 else fmt(t, 0 if p["nd"] == 1 else 2, sign=t > 0) for t in p["ticks"]])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="x", colors=INK2, length=2, labelsize=7, pad=2)
    ax.set_xlabel(p["xlabel"], fontsize=7, color=INK2, labelpad=3)
    fig.text(0.04 / W_IN, (top - 0.02) / H, p["title"], ha="left", va="top", fontsize=8, color=INK,
             fontweight="bold")
    yh = n - 0.4
    for x, head in [(C1, "Published → ours"), (C2, p["gain_head"])]:
        ax.text(fx(x), yh + 0.12, head, transform=yax, ha="left", va="bottom", fontsize=6.5, color=MUTED,
                clip_on=False)
    return top - TITLE - HEAD - h - AXIS - GAP


def main():
    d = pd.read_csv(RESULTS / "table5_forest.csv")
    d["detail"] = d["detail"].fillna("")
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    parts = [d[d.panel == "EAV"].sort_values("published"), d[d.panel == "UA"]]
    H = sum(TITLE + HEAD + len(x) * ROW + AXIS for x in parts) + GAP + 0.06
    fig = plt.figure(figsize=(W_IN, H))
    top = H - 0.03
    for part, p in zip(parts, PANELS):
        if len(part):
            top = draw_panel(fig, part, p, top, H)
    ms = Path(os.environ.get("URBEEG_MS", PROJECT_ROOT / "manuscript"))
    (ms / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ms / "figures" / "fig_compare.pdf")
    out = PROJECT_ROOT / "figures" / "data"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_compare.png", dpi=300)
    print("saved", ms / "figures" / "fig_compare.pdf", f"({W_IN:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    main()
