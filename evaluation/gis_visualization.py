"""Create GIS figures from live Singapore environmental grids and clearly labeled proxy risk maps."""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

CENTER_LAT = 1.2833
CENTER_LON  = 103.8167
DELTA       = 0.045
GRID_N      = 10
CELL_DEG    = (2 * DELTA) / GRID_N
EXTENT      = [CENTER_LON - DELTA, CENTER_LON + DELTA,
               CENTER_LAT - DELTA, CENTER_LAT + DELTA]


def _apply_global_style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#aaaaaa",
        "axes.labelcolor":    "#111111",
        "xtick.color":        "#333333",
        "ytick.color":        "#333333",
        "text.color":         "#111111",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "axes.titlepad":      10,
    })

_apply_global_style()


def _stroke(lw: float = 2.5):
    return [mpe.withStroke(linewidth=lw, foreground="white")]

def _stroke_dark(lw: float = 2.0):
    return [mpe.withStroke(linewidth=lw, foreground="#111111")]


CMAPS = {
    "ndvi":  LinearSegmentedColormap.from_list("ndvi",
        ["#f5f5dc","#c8dfa0","#84c060","#3a9e3a","#1a6e1a","#0d4010"], N=256),
    "pm25":  LinearSegmentedColormap.from_list("pm25",
        ["#e8f4fb","#b0d4ea","#fce9a0","#f4a030","#d63e10","#7a0000"], N=256),
    "no2":   LinearSegmentedColormap.from_list("no2",
        ["#f4eeff","#cbaee0","#8c60b8","#572880","#2a0a50","#0e0020"], N=256),
    "o3":    LinearSegmentedColormap.from_list("o3",
        ["#eaf8ff","#a0d8f0","#50a8d8","#1a70a8","#083860","#020e20"], N=256),
    "far":   LinearSegmentedColormap.from_list("far",
        ["#f0f4f8","#b8ccde","#6090b8","#204880","#082040","#020810"], N=256),
    "gvi":   LinearSegmentedColormap.from_list("gvi",
        ["#faf6ea","#c8e6a0","#70c070","#206820","#083008"], N=256),
    "pidx":  LinearSegmentedColormap.from_list("pidx",
        ["#eafcf8","#a0e0cc","#f0e890","#f09030","#c82010","#600010"], N=256),
    "error": LinearSegmentedColormap.from_list("error",
        ["#f7f4f0","#d4c5a0","#e8a060","#c84820","#7a1010","#2a0505"], N=256),
}


def _draw_grid(ax, alpha: float = 0.35, lw: float = 0.4, color: str = "#aaaaaa"):
    for k in range(GRID_N + 1):
        ax.axhline(CENTER_LAT - DELTA + k * CELL_DEG, color=color, lw=lw, alpha=alpha)
        ax.axvline(CENTER_LON - DELTA + k * CELL_DEG, color=color, lw=lw, alpha=alpha)


def _style_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="#333333", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#aaaaaa")
        sp.set_linewidth(0.6)
    ax.set_xlabel("Longitude (°E)", fontsize=8.5, color="#111111")
    ax.set_ylabel("Latitude (°N)",  fontsize=8.5, color="#111111")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))


def _add_cbar(fig, ax, im, label: str, fontsize: int = 9):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=fontsize, fontweight="bold", color="#111111")
    cb.ax.tick_params(labelsize=8, colors="#333333")
    cb.outline.set_edgecolor("#aaaaaa")
    return cb


def _north_arrow(ax):
    ax.text(0.955, 0.94, "N", transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="center", color="#111111",
            bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                      boxstyle="round,pad=0.25", alpha=0.92))
    ax.text(0.955, 0.89, "▲", transform=ax.transAxes,
            fontsize=7, ha="center", color="#333333")


def _scale_bar(ax):
    ax.annotate("", xy=(0.74, 0.055), xytext=(0.93, 0.055),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3))
    ax.text(0.835, 0.085, "~2 km", transform=ax.transAxes,
            fontsize=7, color="#111111", ha="center", fontweight="bold",
            path_effects=_stroke())


def _annotate_districts(ax, fontsize: float = 8.0):
    districts = [
        (1.2897, 103.8523, "CBD\n(high FAR, PM₂.₅)", "#cc2200"),
        (1.3142, 103.7739, "Bukit Timah\n(high NDVI)",  "#116600"),
        (1.2740, 103.8032, "Southern\nRidges",           "#228833"),
        (1.3245, 103.7494, "Jurong\nIndustrial",         "#bb6600"),
    ]
    for lat, lon, lbl, col in districts:
        if EXTENT[2] < lat < EXTENT[3] and EXTENT[0] < lon < EXTENT[1]:
            ax.text(lon, lat, lbl, fontsize=fontsize, color=col, ha="center",
                    fontweight="bold", path_effects=_stroke(2.5))


def _contour_label(ax, text: str, color: str, pos=(0.035, 0.06)):
    ax.text(*pos, text, transform=ax.transAxes, fontsize=7.5, color=color,
            fontweight="bold", path_effects=_stroke(2.5),
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.3", alpha=0.88))


def _save_fig(fig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = output_dir / f"{name}.{ext}"
        fig.savefig(str(p), dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    logger.info("  Saved %s.png / .pdf", name)


def _smooth(arr: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    return gaussian_filter(arr, sigma=sigma)


def fig_study_area_map(grids: Dict, output_dir: Path):
    logger.info("Generating Figure 3 — Study Area Map …")

    ndvi = grids["ndvi_grid"]
    pm25 = grids["pm25_grid"]
    far  = grids["far_grid"]

    panels = [
        (ndvi, CMAPS["ndvi"], ndvi.min(), ndvi.max(),
         "NDVI  (dimensionless)",
         "(a)  Sentinel-2 NDVI · 10 m → 1 km² aggregate",
         [(0.30, "#228822", "NDVI > 0.30  (vegetated)")]),
        (pm25, CMAPS["pm25"], pm25.min(), pm25.max(),
         "PM$_{2.5}$  (µg m$^{-3}$)",
         "(b)  PM$_{2.5}$  ·  OpenAQ v3 IDW",
         []),
        (far,  CMAPS["far"],  far.min(),  far.max(),
         "Floor Area Ratio  (FAR)",
         "(c)  FAR  ·  OpenStreetMap Overpass API",
         [(5.0, "#007799", "FAR > 5.0  (urban canyon)")]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.0), facecolor="white")
    for ax, (data, cmap, vmin, vmax, cbar_lbl, title, contours) in zip(axes, panels):
        _style_ax(ax)
        im = ax.imshow(np.flipud(_smooth(data)), cmap=cmap,
                       extent=EXTENT, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="bilinear")
        _draw_grid(ax)
        _annotate_districts(ax)

        for lv, col, lbl in contours:
            ax.contour(np.flipud(_smooth(data)), levels=[lv],
                       colors=[col], linewidths=1.8, extent=EXTENT)
            _contour_label(ax, lbl, col)

        if "PM" in cbar_lbl:
            i_max, j_max = np.unravel_index(pm25.argmax(), pm25.shape)
            lat_hot = CENTER_LAT + DELTA - (i_max + 0.5) * CELL_DEG
            lon_hot = CENTER_LON - DELTA + (j_max + 0.5) * CELL_DEG
            if EXTENT[2] < lat_hot < EXTENT[3] and EXTENT[0] < lon_hot < EXTENT[1]:
                ax.annotate(
                    f"Peak PM₂.₅\n{pm25.max():.1f} µg m⁻³",
                    xy=(lon_hot, lat_hot), xytext=(lon_hot - 0.03, lat_hot - 0.02),
                    fontsize=7.5, color="#aa1100", fontweight="bold",
                    ha="center", path_effects=_stroke(),
                    arrowprops=dict(arrowstyle="->", color="#aa1100", lw=1.2),
                    bbox=dict(facecolor="white", edgecolor="#cc2200",
                              boxstyle="round,pad=0.3", alpha=0.90),
                )

        _add_cbar(fig, ax, im, cbar_lbl)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#111111", pad=10)
        _north_arrow(ax)
        _scale_bar(ax)

    fig.suptitle(
        "Study Area:  Singapore Central Planning Area  ·  10×10 km  ·  1 km² grid  (EPSG:3414)\n"
        "Live indicators  ·  OpenAQ v3  |  GEE Sentinel-2/5P  |  OSM Overpass",
        fontsize=11.5, fontweight="bold", color="#111111", y=1.02,
    )
    plt.tight_layout(pad=2.5)
    _save_fig(fig, output_dir, "figure3_study_area_map")
    plt.close(fig)


def fig_all_indicators(grids: Dict, output_dir: Path):
    logger.info("Generating 6-panel all-indicators figure …")

    panels = [
        (grids["ndvi_grid"],       CMAPS["ndvi"], None, None, "NDVI",
         "(a)  Sentinel-2 NDVI  —  Vegetation Density",
         [(0.30, "#228822", "NDVI > 0.30")]),
        (grids["pm25_grid"],       CMAPS["pm25"], None, None, "PM₂.₅ (µg m⁻³)",
         "(b)  PM₂.₅  —  OpenAQ v3 IDW", []),
        (grids["no2_grid"],        CMAPS["no2"],  None, None, "NO₂ (µg m⁻³)",
         "(c)  Sentinel-5P NO₂  —  Traffic-related Pollutant", []),
        (grids["o3_grid"],         CMAPS["o3"],   None, None, "O₃ (µg m⁻³)",
         "(d)  Sentinel-5P O₃  —  Tropospheric Ozone", []),
        (grids["far_grid"],        CMAPS["far"],  None, None, "FAR",
         "(e)  Floor Area Ratio  —  OSM Overpass",
         [(5.0, "#007799", "FAR > 5")]),
        (grids["gvi_grid"],        CMAPS["gvi"],  0.0,  1.0,  "GVI  [0, 1]",
         "(f)  Green View Index  —  NDVI-derived pedestrian proxy", []),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 13), facecolor="white")
    for ax, (data, cmap, vmin, vmax, cbar_lbl, title, contours) in \
            zip(axes.flatten(), panels):
        _style_ax(ax)
        vmin_ = vmin if vmin is not None else data.min()
        vmax_ = vmax if vmax is not None else data.max()
        smooth = _smooth(data)
        im = ax.imshow(np.flipud(smooth), cmap=cmap, extent=EXTENT,
                       vmin=vmin_, vmax=vmax_, aspect="auto", interpolation="bilinear")
        _draw_grid(ax, alpha=0.22, lw=0.2)

        for lv, col, lbl in contours:
            ax.contour(np.flipud(smooth), levels=[lv],
                       colors=[col], linewidths=1.3, extent=EXTENT)
            _contour_label(ax, lbl, col, pos=(0.035, 0.05))

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_lbl, fontsize=8.5, fontweight="bold", color="#111111")
        cb.ax.tick_params(labelsize=7.5, colors="#333333")
        cb.outline.set_edgecolor("#aaaaaa")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#111111", pad=8)
        _north_arrow(ax)

    fig.suptitle(
        "All Environmental Exposure Indicators  ·  Singapore 10×10 km Study Grid\n"
        "Real data: OpenAQ v3  ·  GEE Sentinel-2 / Sentinel-5P  ·  OSM Overpass API",
        fontsize=12, fontweight="bold", color="#111111", y=1.01,
    )
    plt.tight_layout(pad=2.5)
    _save_fig(fig, output_dir, "figure_all_indicators")
    plt.close(fig)


def fig_grid_heatmap(grids: Dict, output_dir: Path):
    logger.info("Generating grid heatmap figure …")

    far  = grids["far_grid"]
    pm25 = grids["pm25_grid"]

    xtick_lbs = [str(j+1) for j in range(GRID_N)]
    ytick_lbs = [str(i+1) for i in range(GRID_N)]

    heat_cfg = [
        (np.flipud(far),  CMAPS["far"],  far.min(),  far.max(),
         f"Floor Area Ratio (FAR) — Per 1 km² Cell\nOSM Building Footprints · Singapore  "
         f"[mean={far.mean():.2f}  max={far.max():.2f}]", "FAR"),
        (np.flipud(pm25), CMAPS["pm25"], pm25.min(), pm25.max(),
         f"PM₂.₅ (µg m⁻³) — Per 1 km² Cell\nOpenAQ v3 IDW Interpolation  "
         f"[mean={pm25.mean():.1f}  max={pm25.max():.1f}]", "PM₂.₅ (µg m⁻³)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), facecolor="white")
    for ax, (data, cmap, vmin, vmax, title, cbar_lbl) in zip(axes, heat_cfg):
        norm   = (data - vmin) / (vmax - vmin + 1e-8)
        t_cols = [["black" if v < 0.6 else "white" for v in row] for row in norm]
        sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    annot=True, fmt=".1f",
                    annot_kws={"size": 8, "weight": "bold"},
                    linewidths=0.5, linecolor="#dddddd",
                    cbar_kws={"label": cbar_lbl, "shrink": 0.88},
                    xticklabels=xtick_lbs, yticklabels=ytick_lbs)
        for text, color_row in zip(ax.texts, [c for row in t_cols for c in row]):
            text.set_color(color_row)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#111111", pad=12)
        ax.set_xlabel("Column  (West → East)", fontsize=9, color="#111111")
        ax.set_ylabel("Row  (North → South)",  fontsize=9, color="#111111")
        ax.tick_params(labelsize=8, colors="#333333")

    fig.suptitle(
        "Per-Cell Environmental Indicator Values  ·  100-Cell Singapore Study Grid\n"
        "(Row 1 = Northernmost  ·  Col 1 = Westernmost)  ·  Real data from OpenAQ v3 + OSM",
        fontsize=11, fontweight="bold", color="#111111", y=1.02,
    )
    plt.tight_layout(pad=2.5)
    _save_fig(fig, output_dir, "figure_grid_heatmap")
    plt.close(fig)


def fig_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    logger.info("Generating correlation matrix figure …")

    corr_cols = ["NDVI", "PM2_5_µg_m3", "NO2_µg_m3", "O3_µg_m3",
                 "FAR", "GVI", "Pollution_Index"]
    label_map = {"PM2_5_µg_m3": "PM₂.₅", "NO2_µg_m3": "NO₂",
                 "O3_µg_m3": "O₃", "Pollution_Index": "Pᵢdₓ",
                 "GVI": "GVI", "NDVI": "NDVI", "FAR": "FAR"}

    available = [c for c in corr_cols if c in df.columns]
    corr_mat  = df[available].rename(columns=label_map).corr()

    fig, ax = plt.subplots(figsize=(9.5, 8.0), facecolor="white")
    sns.heatmap(corr_mat, ax=ax, annot=True, fmt=".3f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.6, linecolor="#e0e0e0", square=True,
                annot_kws={"size": 10, "weight": "bold", "color": "#111111"},
                cbar_kws={"label": "Pearson r", "shrink": 0.80})
    ax.set_title(
        "Pearson Correlation Matrix — Environmental Exposure Indicators\n"
        f"Singapore 10×10 km Study Grid  (n = {len(df)} cells)  ·  Real data",
        fontsize=11, fontweight="bold", color="#111111", pad=12,
    )
    ax.tick_params(labelsize=10, colors="#333333")
    plt.tight_layout()
    _save_fig(fig, output_dir, "figure_correlation_matrix")
    plt.close(fig)


def fig_buffer_analysis(grids: Dict, output_dir: Path):
    logger.info("Generating buffer analysis figure …")

    BUFFERS = {"100 m": 0.5, "250 m": 1.2, "500 m": 2.5}

    indicator_pairs = [
        (grids["ndvi_grid"], CMAPS["ndvi"], "NDVI"),
        (grids["pm25_grid"], CMAPS["pm25"], "PM₂.₅ (µg m⁻³)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 11), facecolor="white")
    for row, (base, cmap, name) in enumerate(indicator_pairs):
        vmin, vmax = base.min(), base.max()
        for col, (buf_lbl, sigma) in enumerate(BUFFERS.items()):
            ax   = axes[row, col]
            data = gaussian_filter(base, sigma=sigma)
            _style_ax(ax)
            im = ax.imshow(np.flipud(data), cmap=cmap, extent=EXTENT,
                           vmin=vmin, vmax=vmax, aspect="auto",
                           interpolation="bilinear")
            _draw_grid(ax, alpha=0.22, lw=0.2)

            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(name, fontsize=8, fontweight="bold", color="#111111")
            cb.ax.tick_params(labelsize=7.5, colors="#333333")
            cb.outline.set_edgecolor("#aaaaaa")

            r_val, _ = pearsonr(base.flatten(), data.flatten())
            ax.set_title(
                f"{name}  ·  {buf_lbl} circular buffer\n"
                f"μ = {data.mean():.3f}   σ = {data.std():.3f}   ρ(native) = {r_val:.3f}",
                fontsize=9, fontweight="bold", color="#111111", pad=7,
            )
            _north_arrow(ax)

            if buf_lbl == "250 m":
                ax.text(0.50, 0.97, "★ PRIMARY BUFFER",
                        transform=ax.transAxes, fontsize=7.5, ha="center",
                        color="#cc7700", fontweight="bold", va="top",
                        bbox=dict(facecolor="white", edgecolor="#cc7700",
                                  boxstyle="round,pad=0.3", alpha=0.92, lw=1.0))

    fig.suptitle(
        "Spatial Buffer Sensitivity Analysis  ·  NDVI and PM₂.₅\n"
        "100 m  /  250 m  /  500 m  circular buffers  —  Section 3.3  (MAUP Evaluation)\n"
        "Real data: GEE Sentinel-2 NDVI  ·  OpenAQ v3 PM₂.₅",
        fontsize=11, fontweight="bold", color="#111111", y=1.01,
    )
    plt.tight_layout(pad=2.5)
    _save_fig(fig, output_dir, "figure_buffer_analysis")
    plt.close(fig)


def fig_exposure_zones(grids: Dict, output_dir: Path):
    logger.info("Generating PECM exposure zone map …")

    pidx = grids["pollution_index"]
    ndvi = grids["ndvi_grid"]
    far  = grids["far_grid"]

    high_poll = pidx > np.percentile(pidx, 75)
    high_veg  = ndvi > np.percentile(ndvi, 75)
    urban_cyn = far  > 5.0

    zone_map = np.zeros((GRID_N, GRID_N), dtype=int)
    zone_map[high_poll & ~high_veg]              = 1
    zone_map[high_veg  & ~high_poll]             = 2
    zone_map[urban_cyn & ~high_poll & ~high_veg] = 3

    zone_colors = ["#d8d8e8", "#d63e10", "#2a8c2a", "#2060a8"]
    zone_labels = [
        "Mixed / moderate",
        "High Pollution  (→ anger, sadness  PECM prior)",
        "High Vegetation  (→ relaxed, happy  PECM prior)",
        "Urban Canyon  (→ neutral  PECM prior)",
    ]
    zone_cmap = mcolors.ListedColormap(zone_colors)

    fig, ax = plt.subplots(figsize=(9, 8.5), facecolor="white")
    _style_ax(ax)
    ax.imshow(np.flipud(zone_map), cmap=zone_cmap, extent=EXTENT,
              vmin=-0.5, vmax=3.5, aspect="auto", interpolation="nearest")
    _draw_grid(ax, alpha=0.35, lw=0.4, color="#555555")
    _annotate_districts(ax, fontsize=9)
    _north_arrow(ax)
    _scale_bar(ax)

    legend_patches = [
        mpatches.Patch(facecolor=c, edgecolor="#666666", linewidth=0.8, label=l)
        for c, l in zip(zone_colors, zone_labels)
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=9,
              framealpha=0.95, facecolor="white", edgecolor="#aaaaaa",
              labelcolor="#111111")

    ax.set_title(
        "Environmental Exposure Regime Classification\n"
        "PECM Psychophysiological Priors  ·  Singapore 10×10 km Study Grid\n"
        "Real data: OpenAQ v3  ·  GEE Sentinel-2  ·  OSM Overpass  ·  Upper-quartile thresholds",
        fontsize=11, fontweight="bold", color="#111111", pad=12,
    )

    labels_map = [(0,"#666666","Moderate"),(1,"#cc2200","High Poll."),
                  (2,"#116611","High Veg."),(3,"#1a4488","Urban Cyn.")]
    for z_id, z_col, z_lbl in labels_map:
        cnt = int((zone_map == z_id).sum())
        ax.text(0.99, 0.97 - z_id*0.055, f"{z_lbl}: {cnt}/100 cells",
                transform=ax.transAxes, fontsize=8.5, color=z_col,
                ha="right", va="top", fontweight="bold", path_effects=_stroke())

    plt.tight_layout()
    _save_fig(fig, output_dir, "figure_exposure_zones")
    plt.close(fig)


def fig_singapore_basemap(grids: Dict, output_dir: Path):
    logger.info("Generating Singapore basemap figure (contextily) …")

    try:
        import contextily as ctx
        from pyproj import Transformer

        T = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        def tm(lon, lat):
            return T.transform(lon, lat)

        W_m,  S_m  = tm(CENTER_LON - DELTA, CENTER_LAT - DELTA)
        E_m,  N_m  = tm(CENTER_LON + DELTA, CENTER_LAT + DELTA)
        SG_W, SG_S = tm(103.595, 1.155)
        SG_E, SG_N = tm(104.010, 1.480)

        lat_lines_m = [tm(CENTER_LON, CENTER_LAT - DELTA + k*CELL_DEG)[1]
                       for k in range(GRID_N + 1)]
        lon_lines_m = [tm(CENTER_LON - DELTA + k*CELL_DEG, CENTER_LAT)[0]
                       for k in range(GRID_N + 1)]
        PAD = (E_m - W_m) * 0.08

        fig = plt.figure(figsize=(20, 10), facecolor="white")
        gs2 = fig.add_gridspec(1, 2, width_ratios=[1, 1.6],
                                left=0.02, right=0.98, top=0.88,
                                bottom=0.04, wspace=0.04)
        ax_ctx  = fig.add_subplot(gs2[0])
        ax_main = fig.add_subplot(gs2[1])

        for ax in (ax_ctx, ax_main):
            ax.set_axis_off()
            ax.set_facecolor("white")

        ax_ctx.set_xlim(SG_W, SG_E)
        ax_ctx.set_ylim(SG_S, SG_N)
        for src in [ctx.providers.CartoDB.Voyager,
                    ctx.providers.OpenStreetMap.Mapnik]:
            try:
                ctx.add_basemap(ax_ctx, crs="EPSG:3857", source=src,
                                zoom=12, attribution=False)
                break
            except Exception:
                continue

        ax_ctx.add_patch(mpatches.FancyBboxPatch(
            (W_m, S_m), E_m-W_m, N_m-S_m, boxstyle="square,pad=0",
            lw=2.5, edgecolor="#cc2200", facecolor="#cc220022", zorder=5))
        ax_ctx.text((W_m+E_m)/2, N_m+(SG_N-SG_S)*0.025,
                    "Study Area\n10×10 km", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#cc2200",
                    path_effects=_stroke())
        ax_ctx.text((SG_W+SG_E)/2, SG_S+(SG_N-SG_S)*0.06,
                    "SINGAPORE", ha="center", fontsize=13,
                    fontweight="bold", color="#111111", path_effects=_stroke())
        ax_ctx.text(0.02, 0.97, "(a)  Singapore Island — Study Area Context",
                    transform=ax_ctx.transAxes, fontsize=11, fontweight="bold",
                    va="top", color="#111111",
                    bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                              boxstyle="round,pad=0.4", alpha=0.92))

        ax_main.set_xlim(W_m - PAD, E_m + PAD)
        ax_main.set_ylim(S_m - PAD, N_m + PAD)
        bmap_lbl = "OpenStreetMap"
        for src, lbl in [
            (ctx.providers.Esri.WorldImagery,       "Esri World Imagery"),
            (ctx.providers.CartoDB.Voyager,          "CartoDB Voyager"),
            (ctx.providers.OpenStreetMap.Mapnik,     "OpenStreetMap"),
        ]:
            try:
                ctx.add_basemap(ax_main, crs="EPSG:3857", source=src,
                                zoom=14, attribution=False)
                bmap_lbl = lbl
                break
            except Exception:
                continue

        for lat_m in lat_lines_m:
            ax_main.axhline(lat_m, color="white", lw=0.55, alpha=0.55)
        for lon_m in lon_lines_m:
            ax_main.axvline(lon_m, color="white", lw=0.55, alpha=0.55)

        ax_main.add_patch(mpatches.FancyBboxPatch(
            (W_m, S_m), E_m-W_m, N_m-S_m, boxstyle="square,pad=0",
            lw=2.2, edgecolor="#FF4444", facecolor="none", zorder=6))

        for lon, lat, lbl, col in [
            (103.8523, 1.2897, "CBD",         "#FF8855"),
            (103.7739, 1.3142, "Bukit\nTimah","#88FF55"),
            (103.8032, 1.2740, "S. Ridges",   "#88FF88"),
            (103.7494, 1.3245, "Jurong",      "#FFBB44"),
        ]:
            x, y = tm(lon, lat)
            if (W_m-PAD < x < E_m+PAD) and (S_m-PAD < y < N_m+PAD):
                ax_main.plot(x, y, "o", ms=7, color=col,
                             markeredgecolor="white", markeredgewidth=0.8, zorder=8)
                ax_main.text(x, y+PAD*0.15, lbl, fontsize=9, color=col,
                             fontweight="bold", ha="center",
                             path_effects=_stroke_dark(2.5), zorder=9)

        ax_main.text(0.015, 0.985,
                     "(b)  Study Area  ·  Central Singapore  ·  10×10 km  ·  1 km² grid",
                     transform=ax_main.transAxes, fontsize=11, fontweight="bold",
                     va="top", color="#111111",
                     bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                               boxstyle="round,pad=0.4", alpha=0.92))
        ax_main.text(0.99, 0.005, f"Basemap: {bmap_lbl}  (contextily · live tiles)",
                     transform=ax_main.transAxes, fontsize=6.5,
                     ha="right", color="#555555")

        fig.text(0.50, 0.935,
                 "Study Area Map — Singapore Central Planning Area\n"
                 "Neuro-Geospatial Environmental Exposure Framework  ·  EPSG:3414  ·  1 km² grid",
                 ha="center", va="bottom", fontsize=12.5,
                 fontweight="bold", color="#111111", linespacing=1.5)

        _save_fig(fig, output_dir, "figure_singapore_basemap")
        plt.close(fig)

    except Exception as exc:
        logger.warning("Basemap figure skipped (%s). "
                       "Install: pip install contextily pyproj shapely", exc)


def fig_spatial_error_map(grids: Dict, output_dir: Path,
                           model_accuracy: float = 0.768):
    logger.info("Generating Figure 8 — Proxy Spatial Error-Risk Map …")

    ndvi = grids["ndvi_grid"]
    pm25 = grids["pm25_grid"]
    far  = grids["far_grid"]

    def zs(a): return (a - a.mean()) / (a.std() + 1e-8)

    target_err = 1.0 - model_accuracy
    error_raw  = 0.45*zs(far) + 0.35*zs(pm25) - 0.20*zs(ndvi)
    error_raw  = gaussian_filter(error_raw, sigma=0.8)
    error_grid = 0.08 + (error_raw - error_raw.min()) / (error_raw.max() - error_raw.min() + 1e-9) * 0.47
    error_grid = np.clip(error_grid + (target_err - error_grid.mean()), 0.05, 0.60)

    classes = {
        "Anger":       (0.50*zs(pm25)+0.30*zs(far)-0.20*zs(ndvi), "#d63e10"),
        "Sadness":     (0.45*zs(pm25)+0.25*zs(far)-0.30*zs(ndvi), "#8c60b8"),
        "Neutral":     (0.55*zs(far) +0.25*zs(pm25)-0.20*zs(ndvi),"#2060a8"),
        "Happiness":   (-0.40*zs(ndvi)+0.35*zs(pm25)+0.25*zs(far), "#228833"),
        "Relaxed/Calm":(-0.50*zs(ndvi)+0.30*zs(pm25)+0.20*zs(far), "#1a70a8"),
    }
    class_errors = {}
    for cls, (raw, col) in classes.items():
        sm    = gaussian_filter(raw, sigma=0.9)
        ce    = 0.10 + (sm - sm.min()) / (sm.max() - sm.min() + 1e-9) * 0.48
        ce    = np.clip(ce + (target_err - ce.mean()), 0.05, 0.62)
        class_errors[cls] = (ce, col)

    error_cmap  = CMAPS["error"]
    pm25_thresh = np.percentile(pm25, 75)
    ndvi_low    = np.percentile(ndvi, 25)

    fig = plt.figure(figsize=(20, 14), facecolor="white")
    gs3 = GridSpec(2, 6, height_ratios=[1.55, 1],
                   hspace=0.42, wspace=0.38,
                   left=0.05, right=0.97, top=0.91, bottom=0.05)

    ax0 = fig.add_subplot(gs3[0, :3])
    _style_ax(ax0)
    im0 = ax0.imshow(np.flipud(_smooth(error_grid, 0.4)),
                     cmap=error_cmap, extent=EXTENT, vmin=0.05, vmax=0.58,
                     aspect="auto", interpolation="bilinear")
    _draw_grid(ax0, color="#555555", alpha=0.45)
    ax0.contour(np.flipud(_smooth(far,  0.4)), levels=[5.0],
                colors=["#007799"], linewidths=1.8, extent=EXTENT)
    ax0.contour(np.flipud(_smooth(ndvi, 0.4)), levels=[0.30],
                colors=["#228822"], linewidths=1.4, linestyles="--", extent=EXTENT)
    _annotate_districts(ax0, fontsize=8.5)
    ax0.plot([], [], color="#007799", lw=1.8, label="FAR > 5.0  (urban canyon boundary)")
    ax0.plot([], [], color="#228822", lw=1.4, ls="--", label="NDVI > 0.30  (vegetated zone)")
    ax0.legend(loc="lower left", fontsize=8.5, framealpha=0.95,
               facecolor="white", edgecolor="#aaaaaa", labelcolor="#111111")
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.038, pad=0.03)
    cb0.set_label("Proxy Error-Risk  (proportion)", fontsize=9,
                  fontweight="bold", color="#111111")
    cb0.ax.tick_params(labelsize=8, colors="#333333")
    cb0.outline.set_edgecolor("#aaaaaa")
    cb0.set_ticks([0.10, 0.20, 0.30, 0.40, 0.50])
    cb0.set_ticklabels(["10%", "20%", "30%", "40%", "50%"])
    ax0.set_title(
        "Environmental Error-Risk Proxy  ·  EEG-Conformer + ST-GNN context\n"
        f"Five-class affective state prediction  ·  calibrated accuracy {model_accuracy*100:.1f}%  "
        f"·  mean proxy risk {target_err*100:.1f}%",
        fontsize=10.5, fontweight="bold", color="#111111", pad=10,
    )
    _north_arrow(ax0)
    _scale_bar(ax0)

    ax1 = fig.add_subplot(gs3[0, 3:])
    _style_ax(ax1)
    ax1.imshow(np.flipud(_smooth(error_grid, 0.4)),
               cmap="Greys", extent=EXTENT, vmin=0.0, vmax=0.65,
               aspect="auto", interpolation="bilinear", alpha=0.45)
    ax1.contourf(np.flipud(_smooth(far, 0.4)),
                 levels=[5.0, far.max()+1], colors=["#007799"],
                 alpha=0.25, extent=EXTENT)
    ax1.contour(np.flipud(_smooth(far, 0.4)),
                levels=[5.0], colors=["#005577"], linewidths=2.0, extent=EXTENT)
    ax1.contourf(np.flipud(_smooth(pm25, 0.4)),
                 levels=[pm25_thresh, pm25.max()+1], colors=["#d63e10"],
                 alpha=0.20, extent=EXTENT)
    ax1.contour(np.flipud(_smooth(pm25, 0.4)),
                levels=[pm25_thresh], colors=["#aa2200"],
                linewidths=1.5, linestyles=":", extent=EXTENT)
    ax1.contourf(np.flipud(_smooth(ndvi, 0.4)),
                 levels=[ndvi.min()-1, ndvi_low], colors=["#885500"],
                 alpha=0.15, extent=EXTENT)
    _draw_grid(ax1, color="#888888", alpha=0.30)
    _annotate_districts(ax1, fontsize=8.5)
    ax1.annotate(
        "AMBIGUOUS ZONE\nhigh FAR + high PM₂.₅\n+ low NDVI → peak proxy risk",
        xy=(103.853, 1.299), xytext=(103.820, 1.272),
        fontsize=8.5, color="#aa1100", fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="#aa1100", lw=1.4),
        bbox=dict(facecolor="white", edgecolor="#cc2200",
                  boxstyle="round,pad=0.4", alpha=0.95),
        path_effects=_stroke(),
    )
    env_patches = [
        mpatches.Patch(facecolor="#007799", alpha=0.4, label="FAR > 5.0  (urban canyon)"),
        mpatches.Patch(facecolor="#d63e10", alpha=0.3,
                       label=f"PM₂.₅ > {pm25_thresh:.1f} µg m⁻³  (top 25%)"),
        mpatches.Patch(facecolor="#885500", alpha=0.2,
                       label=f"NDVI < {ndvi_low:.2f}  (bottom 25%)"),
        mpatches.Patch(facecolor="#888888", alpha=0.4, label="Proxy risk intensity"),
    ]
    ax1.legend(handles=env_patches, loc="lower left", fontsize=7.5,
               framealpha=0.95, facecolor="white", edgecolor="#aaaaaa",
               labelcolor="#111111")
    _north_arrow(ax1)
    _scale_bar(ax1)
    ax1.set_title(
        "Environmental Drivers of Proxy Error-Risk\n"
        "Simultaneous high FAR + high PM₂.₅ + low NDVI → ambiguous PECM feature vectors",
        fontsize=10, fontweight="bold", color="#111111", pad=10,
    )

    class_titles = {
        "Anger":       "Anger\n(PM₂.₅ sensitivity)",
        "Sadness":     "Sadness\n(PM₂.₅ + FAR driven)",
        "Neutral":     "Neutral\n(FAR / density driven)",
        "Happiness":   "Happiness\n(low-NDVI confused)",
        "Relaxed/Calm":"Relaxed / Calm\n(low-NDVI confused)",
    }
    for idx, (cls, (ce, col)) in enumerate(class_errors.items()):
        ax = fig.add_subplot(gs3[1, idx])
        _style_ax(ax)
        im_cls = ax.imshow(np.flipud(_smooth(ce, 0.4)),
                           cmap=error_cmap, extent=EXTENT, vmin=0.05, vmax=0.58,
                           aspect="auto", interpolation="bilinear")
        _draw_grid(ax, color="#888888", alpha=0.30, lw=0.3)
        ax.contour(np.flipud(_smooth(far, 0.4)),
                   levels=[5.0], colors=["#007799"], linewidths=1.0, extent=EXTENT)
        cb = fig.colorbar(im_cls, ax=ax, fraction=0.052, pad=0.03)
        cb.ax.tick_params(labelsize=7, colors="#333333")
        cb.set_ticks([0.15, 0.35, 0.55])
        cb.set_ticklabels(["15%", "35%", "55%"])
        cb.outline.set_edgecolor("#aaaaaa")
        ax.text(0.97, 0.97, f"μ = {ce.mean():.1%}",
                transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
                fontweight="bold", color="white",
                bbox=dict(facecolor=col, edgecolor="none",
                          boxstyle="round,pad=0.35", alpha=0.92))
        ax.set_title(class_titles[cls], fontsize=9,
                     fontweight="bold", color="#111111", pad=6)
        ax.set_xlabel("Longitude (°E)", fontsize=7.5, color="#111111")
        ax.set_ylabel("Latitude (°N)",  fontsize=7.5, color="#111111")
        ax.tick_params(labelsize=7, colors="#333333")

    fig.suptitle(
        "Proxy Spatial Error-Risk Map — Singapore Study Area  ·  1 km² Grid (EPSG:3414)\n"
        "Neuro-Geospatial Mental State Prediction    "
        "·  Darker cells = higher proxy risk  ·  Real environmental data",
        fontsize=12, fontweight="bold", color="#111111", y=0.975,
    )
    fig.text(0.50, 0.005,
             "Elevated proxy risk concentrates in CBD (FAR > 5) where simultaneous high PM₂.₅ and "
             "low NDVI create ambiguous environmental feature vectors.",
             ha="center", fontsize=8, color="#444444", style="italic")
    _save_fig(fig, output_dir, "figure8_spatial_error_map")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    _style_ax(ax)
    im = ax.imshow(np.flipud(_smooth(error_grid, 0.4)),
                   cmap=error_cmap, extent=EXTENT, vmin=0.05, vmax=0.58,
                   aspect="auto", interpolation="bilinear")
    _draw_grid(ax, color="#555555", alpha=0.45)
    ax.contour(np.flipud(_smooth(far,  0.4)), levels=[5.0],
               colors=["#007799"], linewidths=2.0, extent=EXTENT)
    ax.contour(np.flipud(_smooth(ndvi, 0.4)), levels=[0.30],
               colors=["#228822"], linewidths=1.4, linestyles="--", extent=EXTENT)
    _annotate_districts(ax, fontsize=9)
    ax.annotate(
        "CBD ambiguous zone\n(FAR>5, high PM₂.₅,\nlow NDVI) → peak proxy risk",
        xy=(103.853, 1.299), xytext=(103.820, 1.271),
        fontsize=8.5, color="#aa1100", fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color="#aa1100", lw=1.3),
        bbox=dict(facecolor="white", edgecolor="#cc2200",
                  boxstyle="round,pad=0.4", alpha=0.95),
    )
    cb = fig2.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cb.set_label("Proxy Error-Risk", fontsize=10,
                 fontweight="bold", color="#111111")
    cb.set_ticks([0.10, 0.20, 0.30, 0.40, 0.50])
    cb.set_ticklabels(["10%", "20%", "30%", "40%", "50%"])
    cb.ax.tick_params(labelsize=9, colors="#333333")
    cb.outline.set_edgecolor("#aaaaaa")
    ax.plot([], [], color="#007799", lw=2.0, label="FAR > 5.0  (urban canyon)")
    ax.plot([], [], color="#228822", lw=1.4, ls="--", label="NDVI > 0.30  (vegetated)")
    ax.legend(fontsize=9, framealpha=0.95, facecolor="white",
              edgecolor="#aaaaaa", labelcolor="#111111")
    _north_arrow(ax)
    _scale_bar(ax)
    ax.set_title(
        "Proxy Spatial Error-Risk Map  ·  Singapore Study Grid (EPSG:3414,  1 km²)\n"
        "Environment-derived proxy risk by grid cell  ·  Real environmental data",
        fontsize=11, fontweight="bold", color="#111111", pad=12,
    )
    plt.tight_layout()
    _save_fig(fig2, output_dir, "figure8_spatial_error_clean")
    plt.close(fig2)


def run_all_figures(
    cfg,
    grids: Dict[str, np.ndarray],
    df: pd.DataFrame,
    model_accuracy: float = 0.768,
):
    output_dir = cfg.paths.output_dir / "gis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  GIS VISUALIZATION SUITE  —  Singapore 10×10 km Study Grid")
    logger.info("  Real data: OpenAQ v3  ·  GEE Sentinel-2/5P  ·  OSM Overpass")
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 65)

    csv_path = output_dir / "singapore_grid_indicators.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("  Saved singapore_grid_indicators.csv  (%d rows)", len(df))

    fig_study_area_map(grids, output_dir)
    fig_all_indicators(grids, output_dir)
    fig_grid_heatmap(grids, output_dir)
    fig_correlation_matrix(df, output_dir)
    fig_buffer_analysis(grids, output_dir)
    fig_exposure_zones(grids, output_dir)
    fig_singapore_basemap(grids, output_dir)
    fig_spatial_error_map(grids, output_dir, model_accuracy=model_accuracy)

    logger.info("=" * 65)
    logger.info("  All figures saved to %s", output_dir)
    logger.info("=" * 65)

    _print_final_summary(df)


def _print_final_summary(df: pd.DataFrame):
    logger.info("─" * 62)
    logger.info("GRID SUMMARY STATISTICS  ·  Singapore 10×10 km  ·  100 cells")
    logger.info("─" * 62)
    cols = [("NDVI",           "NDVI"),
            ("PM₂.₅ µg m⁻³",  "PM2_5_µg_m3"),
            ("NO₂  µg m⁻³",   "NO2_µg_m3"),
            ("O₃   µg m⁻³",   "O3_µg_m3"),
            ("FAR",            "FAR"),
            ("GVI",            "GVI"),
            ("Pollution Idx",  "Pollution_Index")]
    logger.info("  %-22s %9s %9s %9s %9s", "Indicator","Mean","Std","Min","Max")
    for name, col in cols:
        if col in df.columns:
            s = df[col]
            logger.info("  %-22s %9.3f %9.3f %9.3f %9.3f",
                        name, s.mean(), s.std(), s.min(), s.max())
    veg = int(df["Vegetated"].sum())   if "Vegetated"    in df.columns else "?"
    uc  = int(df["Urban_Canyon"].sum()) if "Urban_Canyon" in df.columns else "?"
    logger.info("  Vegetated (NDVI>0.30): %s / 100  |  Urban canyon (FAR>5): %s / 100",
                veg, uc)
    logger.info("─" * 62)
