"""
test_geo_data_integration.py
Integration tests for live API fetching.

These tests make real network requests to OpenAQ, GEE, and Overpass.
They are skipped by default and must be run explicitly:

    pytest tests/test_geo_data_integration.py -v -m integration
    --gee_project YOUR_PROJECT_ID

Set environment variables:
    EAV_ROOT=/path/to/eav
    GEE_PROJECT=your-gee-project
    OPENAQ_KEY=your-api-key
"""

import os
import pytest
import numpy as np

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def gee_project():
    proj = os.environ.get("GEE_PROJECT")
    if not proj:
        pytest.skip("GEE_PROJECT env var not set")
    return proj


@pytest.fixture(scope="session")
def openaq_key():
    key = os.environ.get("OPENAQ_KEY")
    if not key:
        pytest.skip("OPENAQ_KEY env var not set")
    return key


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    import sys; sys.path.insert(0, ".")
    from config import Config
    c = Config()
    c.paths.cache_dir = tmp_path_factory.mktemp("cache")
    return c


class TestOpenAQFetcher:
    """Test live OpenAQ v3 API."""

    def test_singapore_pm25_fetch(self, cfg, openaq_key):
        from data.geo_data import OpenAQFetcher, LAT_CENTERS, LON_CENTERS, BBOX_W, BBOX_S, BBOX_E, BBOX_N, GRID_N
        aq = OpenAQFetcher(api_key=openaq_key)
        grid = aq.fetch_pollutant_grid(
            bbox=(BBOX_W, BBOX_S, BBOX_E, BBOX_N),
            parameter="pm25", grid_n=GRID_N,
            lat_centers=LAT_CENTERS, lon_centers=LON_CENTERS, days_back=30,
        )
        assert grid.shape == (GRID_N, GRID_N)
        assert (grid > 0).any(), "Expected non-zero PM2.5 readings"
        assert grid.max() < 200, "Unrealistically high PM2.5"

    def test_astana_pm25_fetch(self, cfg, openaq_key):
        from data.geo_data import OpenAQFetcher, AstanaGridData
        aq = OpenAQFetcher(api_key=openaq_key)
        grid = aq.fetch_pollutant_grid(
            bbox=(AstanaGridData._BBOX_W, AstanaGridData._BBOX_S,
                  AstanaGridData._BBOX_E, AstanaGridData._BBOX_N),
            parameter="pm25", grid_n=10,
            lat_centers=AstanaGridData._LAT_CENTERS,
            lon_centers=AstanaGridData._LON_CENTERS, days_back=30,
        )
        assert grid.shape == (10, 10)
        assert (grid >= 0).all()


class TestGEEFetcher:
    """Test live Google Earth Engine API (requires earthengine authenticate)."""

    def test_singapore_ndvi(self, gee_project):
        from data.geo_data import GEEFetcher, LAT_CENTERS, LON_CENTERS, BBOX_W, BBOX_S, BBOX_E, BBOX_N, GRID_N
        gee = GEEFetcher(project=gee_project, bbox=(BBOX_W, BBOX_S, BBOX_E, BBOX_N),
                         grid_n=GRID_N, lat_centers=LAT_CENTERS, lon_centers=LON_CENTERS)
        ndvi = gee.fetch_ndvi(year=2024)
        assert ndvi.shape == (GRID_N, GRID_N)
        assert ndvi.max() <= 1.0 and ndvi.min() >= -1.0

    def test_singapore_no2(self, gee_project):
        from data.geo_data import GEEFetcher, LAT_CENTERS, LON_CENTERS, BBOX_W, BBOX_S, BBOX_E, BBOX_N, GRID_N
        gee = GEEFetcher(project=gee_project, bbox=(BBOX_W, BBOX_S, BBOX_E, BBOX_N),
                         grid_n=GRID_N, lat_centers=LAT_CENTERS, lon_centers=LON_CENTERS)
        no2 = gee.fetch_no2(year=2024)
        assert no2.shape == (GRID_N, GRID_N)
        assert (no2 >= 0).all()

    def test_astana_ndvi(self, gee_project):
        from data.geo_data import GEEFetcher, AstanaGridData
        gee = GEEFetcher(
            project=gee_project,
            bbox=(AstanaGridData._BBOX_W, AstanaGridData._BBOX_S,
                  AstanaGridData._BBOX_E, AstanaGridData._BBOX_N),
            grid_n=10,
            lat_centers=AstanaGridData._LAT_CENTERS,
            lon_centers=AstanaGridData._LON_CENTERS,
        )
        ndvi = gee.fetch_ndvi(year=2024)
        assert ndvi.shape == (10, 10)


class TestOverpassFetcher:
    """Test live Overpass / OSM API."""

    def test_singapore_far(self):
        from data.geo_data import OverpassFetcher, LAT_CENTERS, LON_CENTERS, BBOX_W, BBOX_S, BBOX_E, BBOX_N, GRID_N
        osm = OverpassFetcher(bbox=(BBOX_S, BBOX_W, BBOX_N, BBOX_E))
        far = osm.fetch_far_grid(grid_n=GRID_N, lat_centers=LAT_CENTERS, lon_centers=LON_CENTERS)
        assert far.shape == (GRID_N, GRID_N)
        assert (far >= 0).all()


class TestFullBuildPipeline:
    """End-to-end test of the full data fetch pipeline."""

    def test_astana_full_build(self, cfg, gee_project, openaq_key):
        from data.geo_data import AstanaGridData
        cfg.geo.gee_project    = gee_project
        cfg.geo.openaq_api_key = openaq_key
        grids, df = AstanaGridData.build(cfg=cfg, cache=False,
                                          gee_project=gee_project,
                                          openaq_api_key=openaq_key)
        assert "pm25_grid" in grids and "ndvi_grid" in grids
        assert df.shape[0] == 100
        assert (grids["pm25_grid"] >= 0).all()

    def test_singapore_full_build(self, cfg, gee_project, openaq_key):
        from data.geo_data import SingaporeGridData
        grids, df = SingaporeGridData.build(cfg=cfg, cache=False,
                                             gee_project=gee_project,
                                             openaq_api_key=openaq_key)
        assert "pm25_grid" in grids and "far_grid" in grids
        assert df.shape[0] == 100

    def test_build_astana_grid_live(self, cfg, gee_project, openaq_key):
        from data.geospatial import build_astana_grid
        cfg.geo.gee_project    = gee_project
        cfg.geo.openaq_api_key = openaq_key
        nf, adj = build_astana_grid(cfg, gee_project=gee_project,
                                    openaq_api_key=openaq_key, force_refresh=True)
        assert nf.shape == (100, 6)
        assert adj.shape == (100, 100)
        # z-scored features should be near zero mean
        np.testing.assert_allclose(nf.mean(0), np.zeros(6), atol=0.3)
