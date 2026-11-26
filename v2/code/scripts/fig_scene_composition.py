"""Figure: the 56 Urban Appraisal scenes. (a) Composition of each image (the dataset authors'
SegFormer-B5 / ADE20K pixel counts, grouped into categories), sorted by greenery within each
stimulus design code. (b) Spearman correlation over the 56 scenes between each composition
share and the scene mean of each of the nine rating scales (62 analysed participants;
descriptive). Drawn at the printed width of the manuscript, so font sizes are final.
Writes PDF + PNG and the plotted values."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from urbeeg.config import PROJECT_ROOT  # noqa: E402
from urbeeg.dataset import processed_participants, ratings_wide  # noqa: E402
from urbeeg.scene import scene_features  # noqa: E402

OUT = PROJECT_ROOT / "figures" / "data"
MS_FIG = PROJECT_ROOT / "manuscript" / "figures" / "fig6_scene_composition.pdf"
# Stack bottom -> top like a street view; colours are categorical slots 5..1 so every
# adjacent pair is a validated pair (dataviz validator: all checks pass, light mode).
STACK = [("vehicles", "Vehicles", "#e87ba4"), ("ground", "Road and pavement", "#eda100"),
         ("greenery", "Greenery", "#1baf7a"), ("buildings", "Buildings", "#eb6834"),
         ("sky", "Sky", "#2a78d6")]
SCALES = ["valence", "arousal", "stress", "dominance", "openness", "safety", "beauty", "hominess",
          "fascination"]
CODES = [("HTHB", "HT · HB"), ("HTLB", "HT · LB"), ("LTHB", "LT · HB"), ("LTLB", "LT · LB")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# Diverging scale: blue <-> red poles, neutral grey midpoint.
DIVERGE = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ee8a84", "#b52b2b"])
WIDTH_IN = 372 / 72                                        # text width of the manuscript


def scene_means():
    subs = processed_participants()
    r = ratings_wide(subs)
    return r.groupby(level="stimulus").mean()[SCALES], len(subs)


def main():
    f = scene_features()
    f["code"] = f.index.str[:4]
    means, n_people = scene_means()
    means = means.reindex(f.index)
    assert means.notna().all().all() and len(f) == 56, "ratings and scenes do not line up"
    rho = np.array([[spearmanr(f[c], means[s]).statistic for s in SCALES] for c, _, _ in STACK[::-1]])

    OUT.mkdir(parents=True, exist_ok=True)
    f[[c for c, _, _ in STACK] + ["people", "other", "code"]].join(means).to_csv(OUT / "fig_scene_composition.csv")
    with open(OUT / "fig_scene_composition_rho.csv", "w") as fh:
        fh.write("category," + ",".join(SCALES) + "\n")
        for (c, _, _), row in zip(STACK[::-1], rho):
            fh.write(c + "," + ",".join(f"{v:.3f}" for v in row) + "\n")

    plt.rcParams.update({"font.family": "Helvetica", "font.size": 7.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(WIDTH_IN, 4.4), facecolor=SURFACE)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.12, 1], hspace=0.47, top=0.915, bottom=0.075,
                             left=0.115, right=0.985)
    top = outer[0].subgridspec(1, 4, wspace=0.07)
    axes = [fig.add_subplot(top[0, i]) for i in range(4)]
    for k, (ax, (code, label)) in enumerate(zip(axes, CODES)):
        g = f[f.code == code].sort_values("greenery", ascending=False)
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        for col, name, colour in STACK:
            v = g[col].to_numpy() * 100
            ax.bar(x, v, bottom=bottom, width=0.8, color=colour, edgecolor=SURFACE, linewidth=0.4,
                   label=name)
            bottom += v
        ax.set_title(f"{label}  (n = {len(g)})", fontsize=7.5, color=INK, pad=3)
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.6, len(g) - 0.4)
        ax.set_xticks([])
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="y", length=0, pad=2)
        if k:
            ax.set_yticklabels([])
    axes[0].set_ylabel("Share of image (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.55, 1.005), handlelength=1.0, handletextpad=0.4, columnspacing=1.0,
               fontsize=7.5)
    pa = axes[0].get_position()
    fig.text((pa.x0 + axes[-1].get_position().x1) / 2, pa.y0 - 0.012, "Scenes, sorted by greenery within "
             "each design code", ha="center", va="top", color=INK2, fontsize=7)

    axh = fig.add_subplot(outer[1])
    im = axh.imshow(rho, cmap=DIVERGE, vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            v = rho[i, j]
            axh.text(j, i, f"{v:.2f}".replace("-", "−"), ha="center", va="center", fontsize=7,
                     color="#ffffff" if abs(v) > 0.55 else INK)
    axh.set_xticks(range(len(SCALES)), [s.capitalize() for s in SCALES], fontsize=7.5, rotation=30,
                   ha="left", rotation_mode="anchor")
    axh.set_yticks(range(len(STACK)), [name for _, name, _ in STACK[::-1]], fontsize=7.5)
    axh.tick_params(length=0, pad=3)
    axh.xaxis.set_ticks_position("top")
    for s in axh.spines.values():
        s.set_visible(False)
    axh.set_xticks(np.arange(-0.5, len(SCALES)), minor=True)
    axh.set_yticks(np.arange(-0.5, len(STACK)), minor=True)
    axh.grid(which="minor", color=SURFACE, linewidth=1.2)
    axh.tick_params(which="minor", length=0)
    axh.axvline(2.5, color=INK, linewidth=1.0)
    cb = fig.colorbar(im, ax=axh, fraction=0.035, pad=0.02, ticks=[-0.8, -0.4, 0, 0.4, 0.8])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7, pad=2)
    cb.ax.set_yticklabels(["−0.8", "−0.4", "0", "0.4", "0.8"])
    cb.set_label("Spearman ρ", fontsize=7, labelpad=2)
    axh.set_xlabel(f"Mean rating of the {n_people} participants (primary scales left of the line)",
                   fontsize=7, color=INK2, labelpad=4)

    fig.text(0.005, 0.995, "a", fontsize=10, fontweight="bold", va="top")
    fig.text(0.005, axh.get_position().y1 + 0.075, "b", fontsize=10, fontweight="bold", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_scene_composition.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02,
                    facecolor=SURFACE)
    if MS_FIG.parent.exists():
        fig.savefig(MS_FIG, bbox_inches="tight", pad_inches=0.02, facecolor=SURFACE)
    print("saved", OUT / "fig_scene_composition.pdf")


if __name__ == "__main__":
    main()
