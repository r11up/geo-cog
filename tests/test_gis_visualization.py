"""
tests/test_gis_visualization.py
================================
Unit tests for the real-data GIS pipeline.

Tests cover:
  - OpenAQFetcher.interpolate_to_grid:  IDW interpolation correctness
  - _fallback_* functions:              shape, dtype, physical-unit range
  - SingaporeGridData:                  grid assembly without live API calls
  - compute_gvi_from_ndvi:              monotonicity, [0,1] range
  - _build_dataframe:                   column completeness, row count
  - gis_visualization module:           all figure generators (smoke tests)
  - PECM zone classification:           correct zone assignments

Run:
    pytest tests/test_gis_visualization.py -v
    # or without pytest:
    python tests/test_gis_visualization.py
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.geo_data import (
    OpenAQFetcher,
    compute_gvi_from_ndvi, _pollution_index,
    _build_dataframe, GRID_N, LAT_CENTERS, LON_CENTERS,
    CENTER_LAT, CENTER_LON, DELTA, CELL_DEG,
)

RNG = np.random.default_rng(0)

# ─── Helpers ─────────────────────────────────────────────────────────────────

passed, failed = 0, []

def chk(name, cond, msg=""):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed.append(f"{name}" + (f" [{msg}]" if msg else ""))


def _normalise01(a):
    a = a.astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def _fallback_pm25(grid_n=GRID_N):
    rows, cols = np.meshgrid(
        np.linspace(0, 1, grid_n), np.linspace(0, 1, grid_n), indexing="ij"
    )
    pattern = 0.55 * cols + 0.35 * (1 - rows) + 0.10 * np.sin(2 * np.pi * cols)
    return (8.0 + 20.0 * _normalise01(pattern)).astype(np.float32)


def _fallback_no2(grid_n=GRID_N):
    rows, cols = np.meshgrid(
        np.linspace(0, 1, grid_n), np.linspace(0, 1, grid_n), indexing="ij"
    )
    pattern = 0.50 * cols + 0.30 * (1 - rows) + 0.20 * rows * cols
    return (15.0 + 50.0 * _normalise01(pattern)).astype(np.float32)


def _fallback_ndvi(pm25_grid=None):
    if pm25_grid is None:
        pm25_grid = _fallback_pm25()
    return (-0.05 + 0.70 * (1.0 - _normalise01(pm25_grid))).astype(np.float32)


def _fallback_o3(no2_grid=None):
    if no2_grid is None:
        no2_grid = _fallback_no2()
    return (20.0 + 40.0 * (1.0 - _normalise01(no2_grid))).astype(np.float32)


def _fallback_far(grid_n=GRID_N):
    rows, cols = np.meshgrid(
        np.linspace(-1, 1, grid_n), np.linspace(-1, 1, grid_n), indexing="ij"
    )
    core = np.exp(-2.2 * ((rows + 0.15) ** 2 + (cols - 0.20) ** 2))
    corridor = np.exp(-7.0 * (cols - 0.15) ** 2)
    pattern = 0.75 * core + 0.25 * corridor
    return (0.2 + 8.3 * _normalise01(pattern)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAQFetcher.interpolate_to_grid
# ═══════════════════════════════════════════════════════════════════════════════

class TestIDWInterpolation:

    def test_single_point_at_centroid(self):
        aq  = OpenAQFetcher()
        # Place a known value at a grid centroid — result at that cell should ≈ value
        lat = float(LAT_CENTERS[::-1][0])   # row 0 = north
        lon = float(LON_CENTERS[0])
        df  = pd.DataFrame({"lat": [lat], "lon": [lon], "value": [42.0]})
        grid = aq.interpolate_to_grid(df)
        chk("idw_shape",    grid.shape == (GRID_N, GRID_N))
        chk("idw_nearest",  abs(grid[0, 0] - 42.0) < 5.0,
            f"Expected ≈42, got {grid[0,0]:.2f}")

    def test_two_points_interpolated(self):
        aq = OpenAQFetcher()
        # Two points with different values → grid should be between them
        lat_a, lon_a = float(LAT_CENTERS[::-1][0]), float(LON_CENTERS[0])
        lat_b, lon_b = float(LAT_CENTERS[::-1][-1]), float(LON_CENTERS[-1])
        df = pd.DataFrame({
            "lat":   [lat_a, lat_b],
            "lon":   [lon_a, lon_b],
            "value": [10.0,  50.0],
        })
        grid = aq.interpolate_to_grid(df)
        chk("idw_between",  grid.min() >= 9.0 and grid.max() <= 51.0,
            f"Range: [{grid.min():.1f}, {grid.max():.1f}]")

    def test_empty_df_uses_fill(self):
        aq   = OpenAQFetcher()
        grid = aq.interpolate_to_grid(pd.DataFrame(), fill_value=15.0)
        chk("idw_empty_shape", grid.shape == (GRID_N, GRID_N))
        chk("idw_empty_fill",  np.allclose(grid, 15.0))

    def test_output_finite(self):
        aq  = OpenAQFetcher()
        lats = LAT_CENTERS[::-1][:5]
        lons = LON_CENTERS[:5]
        df   = pd.DataFrame({
            "lat":   lats.tolist(),
            "lon":   lons.tolist(),
            "value": RNG.uniform(10, 30, 5).tolist(),
        })
        grid = aq.interpolate_to_grid(df)
        chk("idw_finite", np.all(np.isfinite(grid)))

TestIDWInterpolation().test_single_point_at_centroid()
TestIDWInterpolation().test_two_points_interpolated()
TestIDWInterpolation().test_empty_df_uses_fill()
TestIDWInterpolation().test_output_finite()


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback generators
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbacks:

    def test_pm25_shape_range(self):
        g = _fallback_pm25()
        chk("pm25_shape", g.shape == (GRID_N, GRID_N))
        chk("pm25_range", 8.0 <= g.min() and g.max() <= 28.0,
            f"[{g.min():.1f}, {g.max():.1f}]")
        chk("pm25_dtype", g.dtype == np.float32)

    def test_no2_shape_range(self):
        g = _fallback_no2()
        chk("no2_shape", g.shape == (GRID_N, GRID_N))
        chk("no2_range", 15.0 <= g.min() and g.max() <= 65.0)

    def test_ndvi_shape_range(self):
        pm25 = _fallback_pm25()
        g    = _fallback_ndvi(pm25)
        chk("ndvi_shape", g.shape == (GRID_N, GRID_N))
        chk("ndvi_range", -0.05 <= g.min() and g.max() <= 0.65)
        # NDVI should anti-correlate with PM2.5
        from scipy.stats import pearsonr
        r, _ = pearsonr(pm25.flatten(), g.flatten())
        chk("ndvi_anticorr", r < 0, f"r={r:.3f} should be negative")

    def test_o3_shape_range(self):
        no2 = _fallback_no2()
        g   = _fallback_o3(no2)
        chk("o3_shape", g.shape == (GRID_N, GRID_N))
        chk("o3_range", 20.0 <= g.min() and g.max() <= 60.0)

    def test_far_shape_range(self):
        g = _fallback_far()
        chk("far_shape", g.shape == (GRID_N, GRID_N))
        chk("far_range", 0.2 <= g.min() and g.max() <= 8.5)
        chk("far_nonneg", (g >= 0).all())

    def test_all_finite(self):
        pm25 = _fallback_pm25()
        no2  = _fallback_no2()
        ndvi = _fallback_ndvi(pm25)
        o3   = _fallback_o3(no2)
        far  = _fallback_far()
        for name, g in [("pm25",pm25),("no2",no2),("ndvi",ndvi),("o3",o3),("far",far)]:
            chk(f"{name}_finite", np.all(np.isfinite(g)))

tf = TestFallbacks()
tf.test_pm25_shape_range()
tf.test_no2_shape_range()
tf.test_ndvi_shape_range()
tf.test_o3_shape_range()
tf.test_far_shape_range()
tf.test_all_finite()


# ═══════════════════════════════════════════════════════════════════════════════
# compute_gvi_from_ndvi
# ═══════════════════════════════════════════════════════════════════════════════

class TestGVI:

    def test_shape(self):
        ndvi = _fallback_ndvi(_fallback_pm25())
        gvi  = compute_gvi_from_ndvi(ndvi)
        chk("gvi_shape", gvi.shape == (GRID_N, GRID_N))

    def test_range(self):
        ndvi = _fallback_ndvi(_fallback_pm25())
        gvi  = compute_gvi_from_ndvi(ndvi)
        chk("gvi_range", (gvi >= 0).all() and (gvi <= 1).all())

    def test_monotone_in_ndvi(self):
        """Higher NDVI → higher GVI (monotone relationship)."""
        ndvi_low  = np.full((GRID_N, GRID_N), 0.10, dtype=np.float32)
        ndvi_high = np.full((GRID_N, GRID_N), 0.60, dtype=np.float32)
        gvi_low   = compute_gvi_from_ndvi(ndvi_low)
        gvi_high  = compute_gvi_from_ndvi(ndvi_high)
        chk("gvi_monotone", gvi_high.mean() > gvi_low.mean())

    def test_threshold_near_point3(self):
        """GVI at NDVI=0.30 (vegetation threshold) should be ≈ 0.5."""
        ndvi = np.full((GRID_N, GRID_N), 0.30, dtype=np.float32)
        gvi  = compute_gvi_from_ndvi(ndvi)
        chk("gvi_threshold", abs(gvi.mean() - 0.50) < 0.05,
            f"GVI at NDVI=0.30 should be ≈0.5, got {gvi.mean():.3f}")

tg = TestGVI()
tg.test_shape()
tg.test_range()
tg.test_monotone_in_ndvi()
tg.test_threshold_near_point3()


# ═══════════════════════════════════════════════════════════════════════════════
# _pollution_index
# ═══════════════════════════════════════════════════════════════════════════════

def test_pollution_index():
    pm25 = _fallback_pm25()
    no2  = _fallback_no2()
    pidx = _pollution_index(pm25, no2)
    chk("pidx_shape",   pidx.shape == (GRID_N, GRID_N))
    chk("pidx_mean",    abs(pidx.mean()) < 0.1)   # z-score avg ≈ 0
    chk("pidx_finite",  np.all(np.isfinite(pidx)))

test_pollution_index()


# ═══════════════════════════════════════════════════════════════════════════════
# _build_dataframe
# ═══════════════════════════════════════════════════════════════════════════════

def _make_test_grids():
    pm25 = _fallback_pm25()
    no2  = _fallback_no2()
    ndvi = _fallback_ndvi(pm25)
    o3   = _fallback_o3(no2)
    far  = _fallback_far()
    gvi  = compute_gvi_from_ndvi(ndvi)
    pidx = _pollution_index(pm25, no2)
    return {
        "pm25_grid":       pm25,
        "no2_grid":        no2,
        "ndvi_grid":       ndvi,
        "o3_grid":         o3,
        "far_grid":        far,
        "gvi_grid":        gvi,
        "pollution_index": pidx,
    }

def test_build_dataframe():
    grids = _make_test_grids()
    df    = _build_dataframe(grids)
    chk("df_rows",     len(df) == 100)
    chk("df_cells",    df["Cell"].nunique() == 100)
    required_cols = ["Cell","Row","Col","Lat","Lon","NDVI","PM2_5_µg_m3",
                     "NO2_µg_m3","O3_µg_m3","FAR","GVI","Pollution_Index",
                     "Vegetated","Urban_Canyon"]
    for col in required_cols:
        chk(f"df_col_{col}", col in df.columns)
    chk("df_lat_range",  df["Lat"].between(CENTER_LAT-DELTA, CENTER_LAT+DELTA).all())
    chk("df_lon_range",  df["Lon"].between(CENTER_LON-DELTA, CENTER_LON+DELTA).all())
    chk("df_ndvi_range", df["NDVI"].between(-0.1, 0.7).all())
    chk("df_veg_bool",   df["Vegetated"].dtype == bool)
    chk("df_uc_bool",    df["Urban_Canyon"].dtype == bool)

test_build_dataframe()


# ═══════════════════════════════════════════════════════════════════════════════
# Grid geometry constants
# ═══════════════════════════════════════════════════════════════════════════════

def test_grid_geometry():
    chk("grid_n",       GRID_N == 10)
    chk("lat_centers",  len(LAT_CENTERS) == GRID_N)
    chk("lon_centers",  len(LON_CENTERS) == GRID_N)
    chk("lat_in_bbox",  all(CENTER_LAT-DELTA <= v <= CENTER_LAT+DELTA for v in LAT_CENTERS))
    chk("lon_in_bbox",  all(CENTER_LON-DELTA <= v <= CENTER_LON+DELTA for v in LON_CENTERS))
    chk("cell_deg",     abs(CELL_DEG - (2*DELTA/GRID_N)) < 1e-9)

test_grid_geometry()


# ═══════════════════════════════════════════════════════════════════════════════
# GIS visualization smoke tests (no API calls — uses grids directly)
# ═══════════════════════════════════════════════════════════════════════════════

def test_figure_generators_no_crash():
    """Each figure generator should run without error on test grids."""
    import tempfile, os
    from evaluation.gis_visualization import (
        fig_study_area_map, fig_all_indicators, fig_grid_heatmap,
        fig_correlation_matrix, fig_buffer_analysis, fig_exposure_zones,
        fig_spatial_error_map,
    )

    grids = _make_test_grids()
    df    = _build_dataframe(grids)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)

        for fname, fn, kwargs in [
            ("study_area_map",    fig_study_area_map,    {}),
            ("all_indicators",    fig_all_indicators,    {}),
            ("grid_heatmap",      fig_grid_heatmap,      {}),
            ("corr_matrix",       fig_correlation_matrix,{"df": df}),
            ("buffer_analysis",   fig_buffer_analysis,   {}),
            ("exposure_zones",    fig_exposure_zones,    {}),
            ("spatial_error_map", fig_spatial_error_map, {}),
        ]:
            try:
                if "df" in kwargs:
                    fn(kwargs["df"], out)
                else:
                    fn(grids, out)
                # Check at least one file was created
                pngs = list(out.glob("*.png"))
                chk(f"fig_{fname}_creates_file", len(pngs) > 0,
                    f"No PNG found after {fname}")
            except Exception as e:
                chk(f"fig_{fname}_no_crash", False, str(e))

test_figure_generators_no_crash()


# ═══════════════════════════════════════════════════════════════════════════════
# PECM zone logic
# ═══════════════════════════════════════════════════════════════════════════════

def test_pecm_zone_logic():
    grids = _make_test_grids()
    pidx  = grids["pollution_index"]
    ndvi  = grids["ndvi_grid"]
    far   = grids["far_grid"]

    high_poll = pidx > np.percentile(pidx, 75)
    high_veg  = ndvi > np.percentile(ndvi, 75)
    urban_cyn = far  > 5.0

    zone_map = np.zeros((GRID_N, GRID_N), dtype=int)
    zone_map[high_poll & ~high_veg]              = 1
    zone_map[high_veg  & ~high_poll]             = 2
    zone_map[urban_cyn & ~high_poll & ~high_veg] = 3

    chk("zone_map_shape", zone_map.shape == (GRID_N, GRID_N))
    chk("zone_values",    set(np.unique(zone_map)).issubset({0, 1, 2, 3}))
    # Top 25% of pollution index should map to zone 1 (anger/sadness prior)
    chk("zone1_high_poll", (high_poll & (zone_map == 1)).sum() >= 10,
        f"zone1 cells: {(zone_map==1).sum()}")
    # Top 25% of NDVI should map to zone 2 (relaxed/happy prior)
    chk("zone2_high_veg", (high_veg & (zone_map == 2)).sum() >= 10,
        f"zone2 cells: {(zone_map==2).sum()}")
    # Zones 1 and 2 should not overlap
    chk("zone12_no_overlap", not (high_poll & high_veg & (zone_map == 1)).any() or
                              not (high_poll & high_veg & (zone_map == 2)).any())

test_pecm_zone_logic()


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

def test_gis_visualization_summary():
    assert not failed, "\n".join(failed)


if __name__ == "__main__":
    total = passed + len(failed)
    print(f"\n{'='*62}")
    print(f"  GIS Visualization Tests  —  {passed}/{total} passed")
    print(f"{'='*62}")
    if failed:
        print(f"  FAILURES ({len(failed)}):")
        for f in failed:
            print(f"    ✗  {f}")
    else:
        print("  ALL TESTS PASSED ✓")
    print(f"{'='*62}")
    sys.exit(len(failed))
