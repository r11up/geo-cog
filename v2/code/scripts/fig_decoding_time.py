"""Figure: time-resolved decoding of the viewed scene's greenery and built form from EEG, for people and
scenes held out together (AUC per 50 ms window, mean over 62 participants with 95% interval; shaded:
significant clusters, sign-flip permutation test)."""
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from urbeeg.config import PROJECT_ROOT, RESULTS  # noqa: E402

INK, INK2, GRID = "#1F2328", "#52514e", "#e4e3df"
SERIES = {"greenery_code": ("High vs low greenery (HT/LT)", "#1baf7a"),
          "built_code": ("High vs low built form (HB/LB)", "#eb6834")}


def main():
    a = json.load(open(RESULTS / "decode_kaggle" / "partA_time_resolved.json"))
    mid = np.array([(w0 + w1) / 2 for w0, w1 in a["windows_ms"]])
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.axvline(0, color=INK2, lw=0.6)
    for task, (label, col) in SERIES.items():
        m, sd = np.array(a[task]["auc_mean"]), np.array(a[task]["auc_sd"])
        ci = 1.96 * sd / np.sqrt(62)
        ax.fill_between(mid, m - ci, m + ci, color=col, alpha=0.18, lw=0)
        ax.plot(mid, m, color=col, lw=2, label=label)
        for c in a[task]["clusters"]:
            if c["p"] < 0.05:
                w = c["windows"]
                y = 0.455 if task == "greenery_code" else 0.44
                ax.plot([a["windows_ms"][w[0]][0], a["windows_ms"][w[-1]][1]], [y, y], color=col, lw=3,
                        solid_capstyle="butt")
    ax.set_xlim(-100, 1000)
    ax.set_ylim(0.43, 0.72)
    ax.set_xlabel("Time from scene onset (ms)", color=INK)
    ax.set_ylabel("Decoding AUC", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, length=2)
    ax.text(1000, 0.503, "chance", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.text(-95, 0.436, "significant clusters", fontsize=7, color=INK2, va="center")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.tight_layout()
    out = PROJECT_ROOT / "figures" / "data"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_decoding_time.pdf")
    fig.savefig(out / "fig_decoding_time.png", dpi=200)
    print("saved", out / "fig_decoding_time.pdf")


if __name__ == "__main__":
    main()
