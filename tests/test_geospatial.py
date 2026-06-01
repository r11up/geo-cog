"""
test_geospatial.py
Unit tests for geospatial data construction.

Tests mock live API builders so no network calls are made.
Integration tests (live API) are in tests/test_geo_data_integration.py.

Run:
    python -m pytest tests/test_geospatial.py -v
"""

import sys
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.geospatial import (
    _make_grid_centroids,
    _queen_adjacency,
    _grids_to_node_features,
    build_astana_grid,
    build_singapore_grid,
    build_city_grid,
    coarsen_pollution_features,
    FEAT_IDX,
    FEAT_NAMES,
)
from data.alignment import PECM


def _make_grid_fixture(seed: int = 42) -> dict:
    """Build a minimal plausible grids dict in the live API response format."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0, 1, (10, 10)).astype(np.float32)
    return {
        "pm25_grid": g * 50 + 15,
        "no2_grid":  g * 30 + 10,
        "ndvi_grid": (1 - g) * 0.5,
        "far_grid":  g * 5,
        "gvi_grid":  (1 - g) * 0.7,
        "o3_grid":   (1 - g) * 40 + 40,
    }


def _make_node_features(seed: int = 42):
    grids = _make_grid_fixture(seed)
    nf  = _grids_to_node_features(grids, 10)
    adj = _queen_adjacency(_make_grid_centroids(10))
    return nf, adj


class TestGridCentroids:

    def test_shape(self):
        assert _make_grid_centroids(10).shape == (100, 2)

    def test_dtype_float32(self):
        assert _make_grid_centroids(10).dtype == np.float32

    def test_centre(self):
        np.testing.assert_allclose(
            _make_grid_centroids(10).mean(0), [4.5, 4.5], atol=0.01)

    def test_row_major_order(self):
        coords = _make_grid_centroids(10)
        for i in range(100):
            assert coords[i, 0] == i // 10
            assert coords[i, 1] == i % 10


class TestQueenAdjacency:

    @pytest.fixture
    def adj(self):
        return _queen_adjacency(_make_grid_centroids(10))

    def test_shape(self, adj):
        assert adj.shape == (100, 100)

    def test_symmetric(self, adj):
        np.testing.assert_allclose(adj, adj.T, atol=1e-6)

    def test_non_negative(self, adj):
        assert (adj >= 0).all()

    def test_diagonal_positive(self, adj):
        assert (np.diag(adj) > 0).all()

    def test_corner_node_3_neighbours(self, adj):
        assert (adj[0] > 0).sum() - 1 == 3

    def test_interior_node_8_neighbours(self, adj):
        assert (adj[55] > 0).sum() - 1 == 8

    def test_dtype_float32(self, adj):
        assert adj.dtype == np.float32


class TestGridsToNodeFeatures:

    def test_shape(self):
        grids = _make_grid_fixture(42)
        nf = _grids_to_node_features(grids, 10)
        assert nf.shape == (100, 6)

    def test_dtype_float32(self):
        grids = _make_grid_fixture(42)
        nf = _grids_to_node_features(grids, 10)
        assert nf.dtype == np.float32

    def test_zero_mean(self):
        grids = _make_grid_fixture(42)
        nf = _grids_to_node_features(grids, 10)
        # Each column is independently z-scored.
        for col in range(1, 6):  # NDVI, FAR, GVI, NO2, O3
            np.testing.assert_allclose(nf[:, col].mean(), 0.0, atol=0.05,
                err_msg=f"Column {FEAT_NAMES[col]} not zero-mean")

    def test_different_grids_different_features(self):
        nf1 = _grids_to_node_features(_make_grid_fixture(42), 10)
        nf2 = _grids_to_node_features(_make_grid_fixture(99), 10)
        assert not np.allclose(nf1, nf2)

    def test_feat_index_layout(self):
        assert FEAT_IDX["PM2.5"] == 0
        assert FEAT_IDX["NDVI"] == 1
        assert FEAT_IDX["FAR"]  == 2
        assert FEAT_IDX["GVI"]  == 3
        assert FEAT_IDX["NO2"]  == 4
        assert FEAT_IDX["O3"]   == 5
        assert len(FEAT_NAMES)  == 6


class TestBuildAstanaGrid:

    def test_fetches_live_builder(self, tmp_path):
        from config import Config
        cfg = Config(); cfg.paths.cache_dir = tmp_path
        grids = _make_grid_fixture(42)
        with patch("data.geo_data.AstanaGridData.build", return_value=(grids, None)) as build:
            nf, adj = build_astana_grid(cfg)
        build.assert_called_once()
        nf_expected = _grids_to_node_features(grids, 10)
        np.testing.assert_array_equal(nf, nf_expected)
        assert adj.shape == (100, 100)

    def test_live_shapes(self, tmp_path):
        from config import Config
        cfg = Config(); cfg.paths.cache_dir = tmp_path
        with patch("data.geo_data.AstanaGridData.build", return_value=(_make_grid_fixture(42), None)):
            nf, adj = build_astana_grid(cfg)
        assert nf.shape == (100, 6)
        assert adj.shape == (100, 100)

    def test_adj_symmetric(self, tmp_path):
        from config import Config
        cfg = Config(); cfg.paths.cache_dir = tmp_path
        with patch("data.geo_data.AstanaGridData.build", return_value=(_make_grid_fixture(42), None)):
            _, adj = build_astana_grid(cfg)
        np.testing.assert_allclose(adj, adj.T, atol=1e-6)

    def test_epsg_is_32641(self):
        from config import Config
        assert Config().geo.epsg_astana == 32641

    def test_does_not_write_static_cache(self, tmp_path):
        from config import Config
        cfg = Config(); cfg.paths.cache_dir = tmp_path
        with patch("data.geo_data.AstanaGridData.build", return_value=(_make_grid_fixture(42), None)):
            build_astana_grid(cfg)
        assert not (tmp_path / "geo_features_astana.npy").exists()


class TestBuildSingaporeGrid:

    def test_fetches_live_builder(self, tmp_path):
        from config import Config
        cfg = Config(); cfg.paths.cache_dir = tmp_path
        grids = _make_grid_fixture(99)
        with patch("data.geo_data.SingaporeGridData.build", return_value=(grids, None)) as build:
            nf, adj = build_singapore_grid(cfg)
        build.assert_called_once()
        nf_expected = _grids_to_node_features(grids, 10)
        np.testing.assert_array_equal(nf, nf_expected)

    def test_live_shapes(self, tmp_path):
        from config import Config
        cfg = Config(); cfg.paths.cache_dir = tmp_path
        with patch("data.geo_data.SingaporeGridData.build", return_value=(_make_grid_fixture(99), None)):
            nf, adj = build_singapore_grid(cfg)
        assert nf.shape == (100, 6) and adj.shape == (100, 100)

    def test_epsg_is_3414(self):
        from config import Config
        assert Config().geo.epsg_singapore == 3414

    def test_does_not_write_static_cache(self, tmp_path):
        from config import Config
        cfg = Config(); cfg.paths.cache_dir = tmp_path
        with patch("data.geo_data.SingaporeGridData.build", return_value=(_make_grid_fixture(99), None)):
            build_singapore_grid(cfg)
        assert not (tmp_path / "geo_features_singapore.npy").exists()

    def test_differs_from_astana(self, tmp_path):
        nfa, _ = _make_node_features(seed=42)
        nfs, _ = _make_node_features(seed=99)
        assert not np.allclose(nfa, nfs)


class TestBuildCityGrid:

    def test_invalid_city_raises(self, tmp_path):
        from config import Config
        cfg = Config(); cfg.paths.cache_dir = tmp_path
        with pytest.raises(ValueError, match="Unknown city"):
            build_city_grid(cfg, "london")

    def test_case_insensitive(self, tmp_path):
        from config import Config
        cfg = Config(); cfg.paths.cache_dir = tmp_path
        with patch("data.geo_data.AstanaGridData.build", return_value=(_make_grid_fixture(42), None)):
            nf1, _ = build_city_grid(cfg, "Astana")
            nf2, _ = build_city_grid(cfg, "astana")
        np.testing.assert_array_equal(nf1, nf2)


class TestCoarsenPollutionFeatures:

    @pytest.fixture
    def F(self, tmp_path):
        nf, _ = _make_node_features(seed=42)
        return nf

    @pytest.mark.parametrize("km", [5, 10])
    def test_shape_preserved(self, F, km):
        assert coarsen_pollution_features(F, 10, km).shape == F.shape

    @pytest.mark.parametrize("km", [5, 10])
    def test_only_pm25_no2_changed(self, F, km):
        out = coarsen_pollution_features(F, 10, km)
        for feat in ["NDVI", "FAR", "GVI", "O3"]:
            np.testing.assert_array_equal(
                out[:, FEAT_IDX[feat]], F[:, FEAT_IDX[feat]])

    @pytest.mark.parametrize("km", [5, 10])
    def test_variance_reduced(self, F, km):
        out = coarsen_pollution_features(F, 10, km)
        assert out[:, FEAT_IDX["PM2.5"]].var() <= F[:, FEAT_IDX["PM2.5"]].var()

    def test_no_input_mutation(self, F):
        orig = F.copy()
        coarsen_pollution_features(F, 10, 5)
        np.testing.assert_array_equal(F, orig)

    def test_bad_factor_raises(self, F):
        with pytest.raises(ValueError):
            coarsen_pollution_features(F, 10, 3)

    def test_dtype_float32(self, F):
        assert coarsen_pollution_features(F, 10, 5).dtype == np.float32


class TestPECM:

    @pytest.fixture
    def F_astana(self, tmp_path):
        nf, _ = _make_node_features(seed=42)
        return nf

    @pytest.fixture
    def F_singapore(self, tmp_path):
        nf, _ = _make_node_features(seed=99)
        return nf

    def test_sample_shape(self, F_astana):
        env = PECM(F_astana, random_state=42).sample(np.arange(5))
        assert env.shape == (5, 6)

    def test_dtype(self, F_astana):
        env = PECM(F_astana, random_state=42).sample(np.zeros(5, dtype=int))
        assert env.dtype == np.float32

    def test_astana_singapore_differ(self, F_astana, F_singapore):
        labels = np.tile(np.arange(5), 40)
        ea = PECM(F_astana,    random_state=42).sample(labels)
        es = PECM(F_singapore, random_state=42).sample(labels)
        assert not np.allclose(ea, es)

    def test_anger_sadness_higher_pm25(self, F_astana):
        pecm = PECM(F_astana, random_state=42)
        neg  = pecm.sample(np.array([1, 2] * 250))
        pos  = pecm.sample(np.array([3, 4] * 250))
        assert neg[:, FEAT_IDX["PM2.5"]].mean() > pos[:, FEAT_IDX["PM2.5"]].mean()

    def test_happy_relaxed_higher_ndvi(self, F_astana):
        pecm = PECM(F_astana, random_state=42)
        pos  = pecm.sample(np.array([3, 4] * 250))
        neg  = pecm.sample(np.array([1, 2] * 250))
        assert pos[:, FEAT_IDX["NDVI"]].mean() > neg[:, FEAT_IDX["NDVI"]].mean()

    def test_reverse_dose_response_lowers_pollution_prior(self, F_astana):
        labels = np.array([1, 2] * 250)
        normal = PECM(F_astana, random_state=42).sample(labels)
        reverse = PECM(F_astana, random_state=42, reverse=True).sample(labels)
        assert reverse[:, FEAT_IDX["PM2.5"]].mean() < normal[:, FEAT_IDX["PM2.5"]].mean()

    def test_reverse_dose_response_lowers_green_prior(self, F_astana):
        labels = np.array([3, 4] * 250)
        normal = PECM(F_astana, random_state=42).sample(labels)
        reverse = PECM(F_astana, random_state=42, reverse=True).sample(labels)
        assert reverse[:, FEAT_IDX["NDVI"]].mean() < normal[:, FEAT_IDX["NDVI"]].mean()
