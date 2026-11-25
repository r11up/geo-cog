"""Figure: scene composition of the 56 Urban Appraisal stimuli (real data: the dataset
authors' SegFormer-B5 / ADE20K pixel counts). Writes PDF + PNG and the plotted values."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from urbeeg.config import PROJECT_ROOT  # noqa: E402
from urbeeg.scene import scene_features  # noqa: E402

OUT = PROJECT_ROOT / "figures" / "data"
# Stack bottom -> top like a street view; colours are categorical slots 5..1 so every
# adjacent pair is a validated pair (dataviz validator: all checks pass, light mode).
STACK = [("vehicles", "Vehicles", "#e87ba4"), ("ground", "Road and pavement", "#eda100"),
         ("greenery", "Greenery", "#1baf7a"), ("buildings", "Buildings", "#eb6834"),
         ("sky", "Sky", "#2a78d6")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def main():
    f = scene_features()
    f["code"] = f.index.str[:4]
    OUT.mkdir(parents=True, exist_ok=True)
    f[[c for c, _, _ in STACK] + ["people", "other", "code"]].to_csv(OUT / "fig_scene_composition.csv")

    plt.rcParams.update({"font.family": "Helvetica", "font.size": 8, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    codes = ["HTHB", "HTLB", "LTHB", "LTLB"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.06}, facecolor=SURFACE)
    for ax, code in zip(axes, codes):
        g = f[f.code == code].sort_values("greenery", ascending=False)
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        for col, label, colour in STACK:
            v = g[col].to_numpy() * 100
            ax.bar(x, v, bottom=bottom, width=0.78, color=colour, edgecolor=SURFACE,
                   linewidth=0.8, label=label)
            bottom += v
        ax.set_xticks(x, g.index, rotation=90, fontsize=6)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_title(f"{code[:2]} · {code[2:]}  (n = {len(g)})", fontsize=8, color=INK2, pad=4)
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.6, len(g) - 0.4)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Share of image (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.4, fontsize=8)
    fig.text(0.5, -0.06, "Scenes, sorted by greenery within each stimulus design code "
             "(HT = high tree cover)", ha="center", color=INK2, fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_scene_composition.{ext}", dpi=300, bbox_inches="tight",
                    facecolor=SURFACE)
    print("saved", OUT / "fig_scene_composition.pdf")


if __name__ == "__main__":
    main()
