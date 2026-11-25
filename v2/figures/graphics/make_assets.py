"""Real-data graphics: Figure 2 (Astana study area) and small images embedded in the draw.io
diagrams (EEG traces, feature map, grid graph, prior subsets, maps). Nothing here is synthetic:
every image is drawn from the EAV recordings, the Sentinel-2/OSM layers or the Astana grid.

    ~/.venvs/urbeeg/bin/python -I figures/graphics/make_assets.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from pyproj import Transformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from urbeeg.astana_grid import CELL_M, N_CELLS, UTM, grid  # noqa: E402

DER, OUT = ROOT / "derived" / "eav", Path(__file__).resolve().parent / "assets"
FIG = ROOT / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.linewidth": 0.6})
INK, MUTED = "#1F2328", "#6B7280"
G = pd.read_csv(DER / "astana_grid.csv")
_, BOUNDS, BBOX_LL = grid()
x0, y0, x1, y1 = BOUNDS


def save(fig, name, transparent=True, dpi=300):
    fig.savefig(OUT / name, dpi=dpi, transparent=transparent, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def cell_raster(col):
    a = np.full((N_CELLS, N_CELLS), np.nan)
    for r in G.itertuples():
        a[N_CELLS - 1 - r.row, r.col] = getattr(r, col)
    return a


# ------------------------------------------------------------------ Sentinel-2 true colour
def sentinel_rgb():
    """Cloud-masked median true-colour composite of the same June-August 2024 scenes used for NDVI."""
    cache = OUT / "astana_s2_rgb_median.npy"
    if not cache.exists():
        import planetary_computer as pc
        import pystac_client
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
        ids = json.load(open(DER / "astana_grid_sources.json"))["sentinel2_items"]
        cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier=pc.sign_inplace)
        w, stack = 1000, []
        for item in cat.search(collections=["sentinel-2-l2a"], ids=ids).items():
            def read(b, rs):
                with rasterio.open(item.assets[b].href) as src:
                    win = from_bounds(*BOUNDS, transform=src.transform)
                    return src.read(1, window=win, out_shape=(w, w), resampling=rs).astype(np.float32)
            clear = np.isin(read("SCL", Resampling.nearest), [4, 5, 6, 7])
            rgb = np.stack([read(b, Resampling.bilinear) for b in ("B04", "B03", "B02")], -1)
            rgb[~clear] = np.nan
            stack.append(rgb)
        np.save(cache, np.nanmedian(np.stack(stack), axis=0).astype(np.float32))
    rgb = np.load(cache)
    rgb = np.where(np.isnan(rgb), np.nanmedian(rgb), rgb)
    lo, hi = np.nanpercentile(rgb, [1, 99])
    return np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8


# ------------------------------------------------------------------ OSM buildings and roads
def osm_layers():
    data = json.load(open(DER / "osm_astana.json"))
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform
    major = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    polys, roads, majors = [], [], []
    for el in data["elements"]:
        geom = el.get("geometry")
        if not geom:
            continue
        xs, ys = to_utm([p["lon"] for p in geom], [p["lat"] for p in geom])
        seg = np.column_stack([np.asarray(xs) - x0, np.asarray(ys) - y0]) / 1000.0     # km
        tags = el.get("tags", {})
        if "building" in tags and len(seg) >= 3:
            polys.append(seg)
        elif "highway" in tags:
            (majors if tags["highway"] in major else roads).append(seg)
    return polys, roads, majors


def gridlines(ax, color="white", lw=0.5, alpha=0.8):
    for k in range(N_CELLS + 1):
        ax.plot([k, k], [0, N_CELLS], color=color, lw=lw, alpha=alpha)
        ax.plot([0, N_CELLS], [k, k], color=color, lw=lw, alpha=alpha)


def scalebar(ax, color=INK):
    ax.plot([0.5, 2.5], [0.45, 0.45], color=color, lw=2, solid_capstyle="butt")
    ax.text(1.5, 0.75, "2 km", color=color, ha="center", va="bottom", fontsize=7)


def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=9, color=INK, pad=4)


def figure2():
    rgb = sentinel_rgb()
    polys, roads, majors = osm_layers()
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.2))
    ax = axs[0, 0]
    ax.imshow(rgb, extent=[0, 10, 0, 10])
    gridlines(ax)
    scalebar(ax, "white")
    panel_title(ax, "a", "Sentinel-2 true colour")
    ax = axs[0, 1]
    ax.add_collection(LineCollection(roads, colors="#B8BEC6", linewidths=0.25))
    ax.add_collection(LineCollection(majors, colors="#6B7280", linewidths=0.6))
    ax.add_collection(PolyCollection(polys, facecolors="#2F3B4C", edgecolors="none"))
    gridlines(ax, color="#9AA4B2", lw=0.4, alpha=0.9)
    scalebar(ax)
    panel_title(ax, "b", "OpenStreetMap buildings, roads")
    layers = [("ndvi", "NDVI", "Greens", "c"), ("ndbi", "NDBI", "Oranges", "d"),
              ("building", "Building coverage (share)", "Greys", "e"),
              ("roads", "Road density (km km$^{-2}$)", "Purples", "f")]
    for ax, (col, lab, cmap, letter) in zip([axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]], layers):
        im = ax.imshow(cell_raster(col), extent=[0, 10, 0, 10], cmap=cmap, interpolation="nearest")
        gridlines(ax, color="white", lw=0.6, alpha=1)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7, length=2, width=0.5)
        cb.outline.set_linewidth(0.4)
        panel_title(ax, letter, lab)
    for ax in (axs[0, 0], axs[0, 1]):
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap="Greys"), ax=ax, fraction=0.046, pad=0.03)
        cb.ax.set_visible(False)
    for ax in axs.flat:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#C9CED6"); s.set_linewidth(0.5)
    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig2_study_area.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig2_study_area.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    # thumbnails for the diagrams
    f, a = plt.subplots(figsize=(2, 2)); a.imshow(rgb, extent=[0, 10, 0, 10]); gridlines(a, lw=0.6)
    a.axis("off"); save(f, "astana_rgb_grid.png", transparent=False)
    f, a = plt.subplots(figsize=(2, 2))
    a.add_collection(PolyCollection(polys, facecolors="#2F3B4C", edgecolors="none"))
    a.add_collection(LineCollection(majors, colors="#6B7280", linewidths=0.5))
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.set_aspect("equal"); a.axis("off")
    save(f, "astana_osm.png")
    for col, cmap in [("ndvi", "Greens"), ("building", "Greys")]:
        f, a = plt.subplots(figsize=(1.6, 1.6))
        a.imshow(cell_raster(col), cmap=cmap, interpolation="nearest"); gridlines(a, lw=0.8, alpha=1)
        a.set_xlim(-0.5, 9.5); a.set_ylim(9.5, -0.5); a.axis("off")
        save(f, f"grid_{col}.png")


# ------------------------------------------------------------------ EEG graphics (EAV, subject 3)
def eeg_assets():
    raw = np.load(DER / "eav_raw.npy", mmap_mode="r")
    z = np.load(DER / "eav_features_v2.npz")
    i = int(np.where(z["subject"] == 3)[0][0])
    x = np.asarray(raw[i], dtype=np.float32)[:, :300]                      # 3 s at 100 Hz
    chans = [0, 4, 9, 13, 18, 22, 26, 29]
    f, a = plt.subplots(figsize=(2.6, 1.7))
    for k, c in enumerate(chans):
        a.plot(np.arange(300) / 100, x[c] / (6 * x[chans].std()) - k, color="#2F5D8C", lw=0.6)
    a.axis("off"); save(f, "eeg_traces.png")
    f, a = plt.subplots(figsize=(1.4, 2.4))
    a.imshow(z["X"][i], aspect="auto", cmap="Blues", interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.5); s.set_color("#4A7FB5")
    save(f, "feature_map.png")
    # head with 30 electrodes (standard 10-20 subset, top view)
    import mne
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4",
             "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4", "O1", "Oz", "O2"]
    xy = np.array([pos[n][:2] for n in names]); xy = xy / np.abs(xy).max() * 0.86
    f, a = plt.subplots(figsize=(1.6, 1.7))
    a.add_patch(plt.Circle((0, 0), 1, fc="#E4EEF8", ec="#2F5D8C", lw=1.2))
    a.plot([-0.12, 0, 0.12], [0.98, 1.14, 0.98], color="#2F5D8C", lw=1.2)              # nose
    for sx in (-1, 1):
        a.add_patch(matplotlib.patches.Ellipse((sx * 1.02, 0), 0.12, 0.32, fc="#E4EEF8", ec="#2F5D8C", lw=1.2))
    a.scatter(xy[:, 0], xy[:, 1], s=16, c="#2F5D8C", edgecolors="white", linewidths=0.5, zorder=3)
    a.set_xlim(-1.15, 1.15); a.set_ylim(-1.1, 1.2); a.set_aspect("equal"); a.axis("off")
    save(f, "eeg_head.png")


# ------------------------------------------------------------------ grid graph and prior subsets
def env_assets():
    f, a = plt.subplots(figsize=(1.9, 1.9))
    nd = cell_raster("ndvi")
    for r in range(N_CELLS):
        for c in range(N_CELLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N_CELLS and 0 <= cc < N_CELLS:
                    a.plot([c, cc], [r, rr], color="#A7C49A", lw=0.4, zorder=1)
    rr, cc = np.mgrid[0:N_CELLS, 0:N_CELLS]
    a.scatter(cc.ravel(), rr.ravel(), c=nd.ravel(), cmap="Greens", s=22, edgecolors="#3E6B2C",
              linewidths=0.4, zorder=2, vmin=np.nanmin(nd) - 0.1)
    a.set_aspect("equal"); a.axis("off"); a.invert_yaxis()
    save(f, "grid_graph.png")
    z = lambda v: (v - v.mean()) / v.std()
    expo = (z(G.building) + z(G.roads) + z(G.ndbi)) / 3
    veg = G.ndvi
    hi_e, hi_v = expo >= expo.quantile(0.75), veg >= veg.quantile(0.75)
    mid = ((expo > expo.quantile(.25)) & (expo < expo.quantile(.75)) & (veg > veg.quantile(.25)) & (veg < veg.quantile(.75)))
    rest = ~(hi_e | hi_v | mid)
    f, a = plt.subplots(figsize=(2.2, 1.9))
    a.scatter(expo[rest], veg[rest], s=9, c="#DADDE2", edgecolors="none")
    a.scatter(expo[mid], veg[mid], s=13, c="#6B7280", edgecolors="white", linewidths=0.3, label="neutral")
    a.scatter(expo[hi_e], veg[hi_e], s=13, c="#D9822B", edgecolors="white", linewidths=0.3, label="anger, sadness")
    a.scatter(expo[hi_v], veg[hi_v], s=13, c="#3E8E41", edgecolors="white", linewidths=0.3, label="happiness, calm")
    a.set_xlabel("exposure score (built, roads, NDBI)", fontsize=6.5, color=MUTED)
    a.set_ylabel("NDVI", fontsize=6.5, color=MUTED)
    a.tick_params(labelsize=6, length=2, colors=MUTED)
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    a.legend(fontsize=5.5, frameon=False, loc="upper right", handletextpad=0.1, borderaxespad=0)
    save(f, "prior_subsets.png")


if __name__ == "__main__" and len(sys.argv) == 1:
    figure2()
    eeg_assets()
    env_assets()
    print("assets:", sorted(p.name for p in OUT.glob("*.png")))


def covariance_asset():
    """Alpha-band spatial correlation of one EAV epoch (subject 3), for the covariance token."""
    from scipy.signal import butter, sosfiltfilt
    raw = np.load(DER / "eav_raw.npy", mmap_mode="r")
    z = np.load(DER / "eav_features_v2.npz")
    i = int(np.where(z["subject"] == 3)[0][0])
    x = sosfiltfilt(butter(4, (8, 13), btype="bandpass", fs=100, output="sos"), np.asarray(raw[i], np.float64))
    f, a = plt.subplots(figsize=(1.6, 1.6))
    a.imshow(np.corrcoef(x), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.5); s.set_color("#4A7FB5")
    save(f, "cov_matrix.png")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "cov":
    covariance_asset()
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "env":
    env_assets()


def scene_assets():
    """Composition of eight of the 56 Urban Appraisal scenes (segmentation shares), stacked like a street view."""
    sf = pd.read_csv(ROOT / "derived" / "kaggle_bundle" / "scene_features.csv").sort_values("greenery")
    pick = pd.concat([sf.head(4), sf.tail(4)])
    stack = [("vehicles", "#e87ba4"), ("ground", "#eda100"), ("greenery", "#1baf7a"), ("buildings", "#eb6834"),
             ("sky", "#2a78d6")]
    f, a = plt.subplots(figsize=(2.6, 1.5))
    for i, (_, r) in enumerate(pick.iterrows()):
        tot = sum(r[c] for c, _ in stack); y = 0
        for c, col in stack:
            h = r[c] / tot
            a.bar(i, h, bottom=y, width=0.82, color=col, edgecolor="white", linewidth=0.6)
            y += h
    a.set_xlim(-0.6, 7.6); a.set_ylim(0, 1); a.axis("off")
    save(f, "scene_bars.png")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "scene":
    scene_assets()
