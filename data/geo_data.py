"""Fetch live OpenAQ, Google Earth Engine, and Overpass data for Astana and Singapore city grids."""

import logging
import time
import json
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

CENTER_LAT = 1.2833
CENTER_LON  = 103.8167
DELTA       = 0.045
GRID_N      = 10
CELL_DEG    = (2 * DELTA) / GRID_N

BBOX_W = CENTER_LON - DELTA
BBOX_E = CENTER_LON + DELTA
BBOX_S = CENTER_LAT - DELTA
BBOX_N = CENTER_LAT + DELTA

LAT_CENTERS = np.linspace(BBOX_S + CELL_DEG/2, BBOX_N - CELL_DEG/2, GRID_N)
LON_CENTERS = np.linspace(BBOX_W + CELL_DEG/2, BBOX_E - CELL_DEG/2, GRID_N)


class OpenAQFetcher:
    BASE = "https://api.openaq.org/v3"

    def __init__(
        self,
        api_key: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        grid_n: int = GRID_N,
        lat_centers: Optional[np.ndarray] = None,
        lon_centers: Optional[np.ndarray] = None,
    ):
        self.api_key = api_key
        self.headers = {"Accept": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
        self._bbox = bbox if bbox is not None else (BBOX_W, BBOX_S, BBOX_E, BBOX_N)
        self._grid_n = grid_n
        self._lat_centers = lat_centers if lat_centers is not None else LAT_CENTERS
        self._lon_centers = lon_centers if lon_centers is not None else LON_CENTERS

    def _get(self, endpoint: str, params: dict) -> dict:
        import requests
        url = f"{self.BASE}/{endpoint}"
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, headers=self.headers, timeout=30)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning("OpenAQ request failed (attempt %d/3): %s", attempt+1, e)
                time.sleep(2 ** attempt)

    def get_singapore_locations(self) -> pd.DataFrame:
        logger.info("OpenAQ: fetching Singapore monitoring locations …")
        data = self._get("locations", {
            "countries_id": 204,
            "limit": 200,
            "page": 1,
            "order_by": "lastUpdated",
            "sort_order": "desc",
        })

        records = []
        for loc in data.get("results", []):
            lat = lon = None
            if loc.get("coordinates"):
                lat = loc["coordinates"].get("latitude")
                lon = loc["coordinates"].get("longitude")
            if lat is None or lon is None:
                continue
            params = [s.get("parameter", {}).get("name", "")
                      for s in loc.get("sensors", [])]
            records.append({
                "id":         loc["id"],
                "name":       loc.get("name", ""),
                "lat":        float(lat),
                "lon":        float(lon),
                "parameters": params,
            })

        df = pd.DataFrame(records)
        logger.info("  Found %d active locations in Singapore.", len(df))
        return df

    @staticmethod
    def _normalise_parameter(name: str) -> str:
        return name.lower().replace(".", "").replace("_", "").replace(" ", "")

    def fetch_pollutant_grid(
        self,
        bbox: Tuple[float, float, float, float],
        parameter: str,
        grid_n: int = GRID_N,
        lat_centers: Optional[np.ndarray] = None,
        lon_centers: Optional[np.ndarray] = None,
        days_back: int = 30,
    ) -> np.ndarray:
        w, s, e, n = bbox
        lat_centers = lat_centers if lat_centers is not None else np.linspace(
            s + (n - s) / (2 * grid_n), n - (n - s) / (2 * grid_n), grid_n
        )
        lon_centers = lon_centers if lon_centers is not None else np.linspace(
            w + (e - w) / (2 * grid_n), e - (e - w) / (2 * grid_n), grid_n
        )

        center_lat = (s + n) / 2.0
        center_lon = (w + e) / 2.0
        lat_m = 111_319.5
        lon_m = lat_m * np.cos(np.radians(center_lat))
        radius_m = int(np.ceil(np.hypot((n - s) * lat_m, (e - w) * lon_m) / 2.0))

        queries = [
            {"bbox": f"{w},{s},{e},{n}", "limit": 1000, "page": 1},
            {
                "coordinates": f"{center_lat},{center_lon}",
                "radius": radius_m,
                "limit": 1000,
                "page": 1,
            },
        ]

        wanted = self._normalise_parameter(parameter)
        location_ids = []
        last_error = None
        for query_params in queries:
            try:
                data = self._get("locations", query_params)
            except Exception as exc:
                last_error = exc
                continue

            candidate_ids = []
            for loc in data.get("results", []):
                coords = loc.get("coordinates") or {}
                lat = coords.get("latitude")
                lon = coords.get("longitude")
                if lat is None or lon is None:
                    continue
                if not (s <= float(lat) <= n and w <= float(lon) <= e):
                    continue

                sensors = loc.get("sensors", [])
                params = [
                    self._normalise_parameter(
                        sensor.get("parameter", {}).get("name", "")
                    )
                    for sensor in sensors
                ]
                if any(wanted == p or wanted in p for p in params):
                    candidate_ids.append(loc["id"])

            if candidate_ids:
                location_ids = candidate_ids
                break

        if not location_ids:
            if last_error is not None:
                logger.debug("Last OpenAQ lookup error: %s", last_error)
                response = getattr(last_error, "response", None)
                status = getattr(response, "status_code", None)
                auth_hint = (
                    " Provide a valid OpenAQ API key."
                    if status in (401, 403) else ""
                )
                raise RuntimeError(
                    f"OpenAQ location lookup failed for {parameter!r} in bbox {bbox}: "
                    f"{last_error}.{auth_hint}"
                ) from last_error
            raise RuntimeError(
                f"OpenAQ returned no {parameter!r} monitoring locations in bbox {bbox}."
            )

        measurements = self.get_recent_measurements(
            location_ids, parameter, hours_back=days_back * 24
        )
        if measurements.empty:
            raise RuntimeError(
                f"OpenAQ returned no recent {parameter!r} measurements in bbox {bbox}."
            )

        old_grid_n, old_lats, old_lons = self._grid_n, self._lat_centers, self._lon_centers
        try:
            self._grid_n = grid_n
            self._lat_centers = lat_centers
            self._lon_centers = lon_centers
            return self.interpolate_to_grid(measurements)
        finally:
            self._grid_n = old_grid_n
            self._lat_centers = old_lats
            self._lon_centers = old_lons

    def get_recent_measurements(
        self,
        location_ids: list,
        parameter: str,
        hours_back: int = 72,
    ) -> pd.DataFrame:
        if not location_ids:
            return pd.DataFrame()

        date_to   = datetime.now(timezone.utc)
        date_from = date_to - timedelta(hours=hours_back)

        records = []
        for loc_id in location_ids:
            try:
                data = self._get("sensors", {
                    "locations_id": loc_id,
                    "limit": 500,
                })
                for sensor in data.get("results", []):
                    p_name = sensor.get("parameter", {}).get("name", "")
                    if p_name.lower().replace(".", "").replace("_", "") != \
                       parameter.lower().replace(".", "").replace("_", ""):
                        continue
                    sensor_id = sensor.get("id")
                    if sensor_id is None:
                        continue

                    meas = self._get(f"sensors/{sensor_id}/measurements", {
                        "date_from": date_from.isoformat(),
                        "date_to":   date_to.isoformat(),
                        "limit":     500,
                    })
                    for m in meas.get("results", []):
                        val = m.get("value")
                        if val is None or val < 0:
                            continue
                        coords = sensor.get("coordinates") or {}
                        records.append({
                            "location_id":  loc_id,
                            "lat":          coords.get("latitude", np.nan),
                            "lon":          coords.get("longitude", np.nan),
                            "value":        float(val),
                            "datetime_utc": m.get("datetime", {}).get("utc", ""),
                        })
                time.sleep(0.2)
            except Exception as e:
                logger.warning("  location %d failed: %s", loc_id, e)

        df = pd.DataFrame(records)
        logger.info("  %s: %d measurements from %d locations.",
                    parameter, len(df), len(location_ids))
        return df

    def interpolate_to_grid(
        self,
        df: pd.DataFrame,
        value_col: str = "value",
        fill_value: Optional[float] = None,
    ) -> np.ndarray:
        grid_n = self._grid_n
        lat_centers = self._lat_centers
        lon_centers = self._lon_centers
        grid = np.full((grid_n, grid_n), np.nan)

        if df.empty or value_col not in df.columns:
            if fill_value is None:
                raise RuntimeError("IDW interpolation received no measurements.")
            logger.warning("  IDW: empty DataFrame — using fill_value=%s", fill_value)
            return np.full((grid_n, grid_n), fill_value or 0.0, dtype=np.float32)

        pts = (df.dropna(subset=["lat", "lon", value_col])
                 .groupby(["lat", "lon"])[value_col]
                 .mean()
                 .reset_index())

        if pts.empty:
            if fill_value is None:
                raise RuntimeError("IDW interpolation has no valid measurement points.")
            return np.full((grid_n, grid_n), fill_value or 0.0, dtype=np.float32)

        lats = pts["lat"].values
        lons = pts["lon"].values
        vals = pts[value_col].values
        power = 2.0

        for i, lat_c in enumerate(lat_centers[::-1]):
            for j, lon_c in enumerate(lon_centers):
                dist = np.sqrt((lats - lat_c)**2 + (lons - lon_c)**2)
                if dist.min() < 1e-6:
                    grid[i, j] = vals[dist.argmin()]
                else:
                    w = 1.0 / (dist ** power)
                    grid[i, j] = np.sum(w * vals) / np.sum(w)

        return grid.astype(np.float32)


class GEEFetcher:
    def __init__(
        self,
        project: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        grid_n: int = 10,
        lat_centers: Optional[np.ndarray] = None,
        lon_centers: Optional[np.ndarray] = None,
    ):
        self._initialized = False
        self._project = project
        self._bbox_coords = bbox if bbox is not None else (BBOX_W, BBOX_S, BBOX_E, BBOX_N)
        self._grid_n      = grid_n
        self._lat_centers = lat_centers if lat_centers is not None else LAT_CENTERS
        self._lon_centers = lon_centers if lon_centers is not None else LON_CENTERS

    def _init_ee(self):
        if self._initialized:
            return
        try:
            import ee
            if self._project:
                ee.Initialize(project=self._project)
            else:
                ee.Initialize()
            self._ee = ee
            self._initialized = True
            logger.info("GEE: Earth Engine initialized.")
        except Exception as e:
            raise RuntimeError(
                f"Google Earth Engine initialization failed: {e}\n"
                "Run `earthengine authenticate` in your terminal first.\n"
                "Then pass your GEE project ID: GEEFetcher(project='my-project')"
            )

    def _bbox(self):
        import ee
        w, s, e, n = self._bbox_coords
        return ee.Geometry.Rectangle([w, s, e, n])

    def _sample_grid(self, image, band: str, scale: int = 1000) -> np.ndarray:
        import ee

        points = []
        for i, lat_c in enumerate(self._lat_centers[::-1]):
            for j, lon_c in enumerate(self._lon_centers):
                feat = ee.Feature(
                    ee.Geometry.Point([lon_c, lat_c]),
                    {"row": i, "col": j}
                )
                points.append(feat)

        fc = ee.FeatureCollection(points)
        sampled = image.select(band).sampleRegions(
            collection=fc, scale=scale, geometries=False
        )

        result = sampled.getInfo()
        grid = np.full((self._grid_n, self._grid_n), np.nan)
        for feat in result.get("features", []):
            props = feat.get("properties", {})
            r = int(props.get("row", -1))
            c = int(props.get("col", -1))
            v = props.get(band, np.nan)
            if 0 <= r < self._grid_n and 0 <= c < self._grid_n and v is not None:
                grid[r, c] = float(v)

        if np.isnan(grid).any():
            mean_val = np.nanmean(grid)
            grid = np.where(np.isnan(grid), mean_val, grid)

        return grid

    def fetch_ndvi(
        self,
        year: int = 2024,
        max_cloud_pct: float = 20.0,
    ) -> np.ndarray:
        self._init_ee()
        ee = self._ee

        logger.info("GEE: fetching Sentinel-2 NDVI for year %d …", year)

        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(self._bbox())
               .filterDate(f"{year}-01-01", f"{year}-12-31")
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
               .map(lambda img: img.normalizedDifference(["B8", "B4"])
                               .rename("NDVI")
                               .copyProperties(img, img.propertyNames()))
               .median()
               .clip(self._bbox()))

        grid = self._sample_grid(col, "NDVI", scale=500)
        logger.info("  NDVI grid: min=%.3f  mean=%.3f  max=%.3f",
                    np.nanmin(grid), np.nanmean(grid), np.nanmax(grid))
        return grid

    def fetch_no2(
        self,
        year: int = 2024,
    ) -> np.ndarray:
        self._init_ee()
        ee = self._ee

        logger.info("GEE: fetching Sentinel-5P NO2 for year %d …", year)

        col = (ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2")
               .filterBounds(self._bbox())
               .filterDate(f"{year}-01-01", f"{year}-12-31")
               .select("tropospheric_NO2_column_number_density")
               .mean()
               .clip(self._bbox()))

        grid = self._sample_grid(col, "tropospheric_NO2_column_number_density", scale=1000)

        grid = np.clip(grid * 1e5 * 46.01, 0, None)

        logger.info("  NO2 grid: min=%.1f  mean=%.1f  max=%.1f µg/m³ proxy",
                    grid.min(), grid.mean(), grid.max())
        return grid

    def fetch_o3(
        self,
        year: int = 2024,
    ) -> np.ndarray:
        self._init_ee()
        ee = self._ee

        logger.info("GEE: fetching Sentinel-5P O3 for year %d …", year)

        col = (ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_O3")
               .filterBounds(self._bbox())
               .filterDate(f"{year}-01-01", f"{year}-12-31")
               .select("O3_column_number_density")
               .mean()
               .clip(self._bbox()))

        grid = self._sample_grid(col, "O3_column_number_density", scale=1000)
        grid = np.clip(grid, 0, None)
        if grid.max() > 0:
            grid = 20.0 + (grid - grid.min()) / (grid.max() - grid.min() + 1e-9) * 40.0

        logger.info("  O3 grid: min=%.1f  mean=%.1f  max=%.1f µg/m³ proxy",
                    grid.min(), grid.mean(), grid.max())
        return grid


class OverpassFetcher:
    ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    def __init__(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        grid_n: int = GRID_N,
        lat_centers: Optional[np.ndarray] = None,
        lon_centers: Optional[np.ndarray] = None,
    ):
        self._bbox = bbox if bbox is not None else (BBOX_S, BBOX_W, BBOX_N, BBOX_E)
        self._grid_n = grid_n
        self._lat_centers = lat_centers if lat_centers is not None else LAT_CENTERS
        self._lon_centers = lon_centers if lon_centers is not None else LON_CENTERS
        s, w, n, e = self._bbox
        self._center_lat = (s + n) / 2.0
        self._center_lon = (w + e) / 2.0

    def _query(self, ql: str) -> dict:
        import requests
        headers = {
            "Accept": "application/json",
            "User-Agent": "geocognition-research/1.0",
        }
        for ep in self.ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": ql}, headers=headers, timeout=120)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.warning("Overpass %s failed: %s", ep, e)
                time.sleep(2)
        raise RuntimeError("All Overpass endpoints failed.")

    def _building_query(self, bbox_s: float, bbox_w: float,
                        bbox_n: float, bbox_e: float) -> str:
        return f"""
[out:json][timeout:90];
(
  way["building"]({bbox_s},{bbox_w},{bbox_n},{bbox_e});
  relation["building"]({bbox_s},{bbox_w},{bbox_n},{bbox_e});
);
out geom;
"""

    def _query_buildings_tiled(self, splits: int = 2) -> list:
        bbox_s, bbox_w, bbox_n, bbox_e = self._bbox
        elements_by_key = {}

        for r in range(splits):
            tile_s = bbox_s + (bbox_n - bbox_s) * r / splits
            tile_n = bbox_s + (bbox_n - bbox_s) * (r + 1) / splits
            for c in range(splits):
                tile_w = bbox_w + (bbox_e - bbox_w) * c / splits
                tile_e = bbox_w + (bbox_e - bbox_w) * (c + 1) / splits
                for el in self._query_buildings_bbox(
                    tile_s, tile_w, tile_n, tile_e
                ):
                    key = (el.get("type"), el.get("id"))
                    elements_by_key[key] = el

        return list(elements_by_key.values())

    def _query_buildings_bbox(self, bbox_s: float, bbox_w: float,
                              bbox_n: float, bbox_e: float,
                              depth: int = 0, max_depth: int = 2) -> list:
        try:
            data = self._query(self._building_query(bbox_s, bbox_w, bbox_n, bbox_e))
            return data.get("elements", [])
        except RuntimeError:
            if depth >= max_depth:
                raise

            logger.warning("Overpass tile rejected; splitting into smaller tiles.")
            mid_lat = (bbox_s + bbox_n) / 2.0
            mid_lon = (bbox_w + bbox_e) / 2.0
            subtiles = [
                (bbox_s, bbox_w, mid_lat, mid_lon),
                (bbox_s, mid_lon, mid_lat, bbox_e),
                (mid_lat, bbox_w, bbox_n, mid_lon),
                (mid_lat, mid_lon, bbox_n, bbox_e),
            ]
            elements = []
            for tile in subtiles:
                elements.extend(
                    self._query_buildings_bbox(*tile, depth=depth + 1,
                                               max_depth=max_depth)
                )
            return elements

    def _estimate_storeys(self, tags: dict) -> float:
        if "building:levels" in tags:
            try:
                return max(1.0, float(tags["building:levels"]))
            except ValueError:
                pass
        btype = tags.get("building", "yes").lower()
        return {
            "house": 2, "residential": 4, "apartments": 8,
            "commercial": 6, "retail": 3, "office": 10,
            "industrial": 2, "warehouse": 1, "school": 3,
            "university": 5, "hospital": 6, "hotel": 12,
            "yes": 4,
        }.get(btype, 4)

    def _polygon_area_m2(self, coords: list) -> float:
        if len(coords) < 3:
            return 0.0
        LAT_M = 111_319.5
        LON_M = LAT_M * np.cos(np.radians(self._center_lat))
        xs = [(c[0] - self._center_lon) * LON_M for c in coords]
        ys = [(c[1] - self._center_lat) * LAT_M for c in coords]
        n = len(xs)
        area = 0.5 * abs(
            sum(xs[i]*ys[(i+1)%n] - xs[(i+1)%n]*ys[i] for i in range(n))
        )
        return area

    def fetch_far_grid(
        self,
        grid_n: Optional[int] = None,
        lat_centers: Optional[np.ndarray] = None,
        lon_centers: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        logger.info("Overpass: querying building footprints …")
        if grid_n is not None:
            self._grid_n = grid_n
        if lat_centers is not None:
            self._lat_centers = lat_centers
        if lon_centers is not None:
            self._lon_centers = lon_centers
        bbox_s, bbox_w, bbox_n, bbox_e = self._bbox
        lat_step = (bbox_n - bbox_s) / self._grid_n
        lon_step = (bbox_e - bbox_w) / self._grid_n

        elements = self._query_buildings_tiled(splits=2)
        logger.info("  Retrieved %d building elements.", len(elements))

        LAT_M = 111_319.5
        LON_M = LAT_M * np.cos(np.radians(self._center_lat))
        cell_area_m2 = (lat_step * LAT_M) * (lon_step * LON_M)

        gfa_grid = np.zeros((self._grid_n, self._grid_n))

        for el in elements:
            tags = el.get("tags", {})
            geom = el.get("geometry", [])
            if not geom:
                continue

            lats = [n["lat"] for n in geom if "lat" in n]
            lons = [n["lon"] for n in geom if "lon" in n]
            if not lats:
                continue
            c_lat = np.mean(lats)
            c_lon = np.mean(lons)

            row = int((bbox_n - c_lat) / lat_step)
            col = int((c_lon  - bbox_w) / lon_step)
            if not (0 <= row < self._grid_n and 0 <= col < self._grid_n):
                continue

            coords = [(n["lon"], n["lat"]) for n in geom if "lat" in n]
            fp_area   = self._polygon_area_m2(coords)
            storeys   = self._estimate_storeys(tags)
            gfa_grid[row, col] += fp_area * storeys

        far_grid = gfa_grid / cell_area_m2
        logger.info("  FAR grid: min=%.2f  mean=%.2f  max=%.2f",
                    far_grid.min(), far_grid.mean(), far_grid.max())
        return far_grid.astype(np.float32)


def compute_gvi_from_ndvi(ndvi_grid: np.ndarray) -> np.ndarray:
    gvi = 1.0 / (1.0 + np.exp(-8.0 * (ndvi_grid - 0.30)))
    return np.clip(gvi, 0.0, 1.0).astype(np.float32)


class SingaporeGridData:
    @staticmethod
    def build(
        cfg,
        use_gee: bool = True,
        gee_project: Optional[str] = None,
        openaq_api_key: Optional[str] = None,
        year: int = 2024,
        cache: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        if not use_gee:
            raise RuntimeError("SingaporeGridData requires GEE and live geospatial inputs.")

        grids: Dict[str, np.ndarray] = {}

        logger.info("Step 1 — OpenAQ v3: PM2.5 …")
        aq = OpenAQFetcher(api_key=openaq_api_key)
        try:
            grids["pm25_grid"] = aq.fetch_pollutant_grid(
                bbox=(BBOX_W, BBOX_S, BBOX_E, BBOX_N),
                parameter="pm25", grid_n=GRID_N,
                lat_centers=LAT_CENTERS, lon_centers=LON_CENTERS, days_back=30,
            )

        except Exception as e:
            raise RuntimeError(
                f"OpenAQ fetch failed for Singapore: {e}. "
                "Check your API key, network connection, and OpenAQ v3 availability."
            ) from e

        logger.info("Step 2 — GEE: Sentinel-2 NDVI + Sentinel-5P NO2/O3 …")
        try:
            gee = GEEFetcher(project=gee_project)
            grids["ndvi_grid"] = gee.fetch_ndvi(year=year)
            grids["no2_grid"]  = gee.fetch_no2(year=year)
            grids["o3_grid"]   = gee.fetch_o3(year=year)
        except Exception as e:
            raise RuntimeError(
                f"GEE fetch failed for Singapore: {e}. "
                "Run 'earthengine authenticate' and set --gee_project."
            ) from e

        logger.info("Step 3 — Overpass API: building footprints → FAR …")
        try:
            osm = OverpassFetcher(bbox=(BBOX_S, BBOX_W, BBOX_N, BBOX_E))
            grids["far_grid"] = osm.fetch_far_grid(
                grid_n=GRID_N, lat_centers=LAT_CENTERS, lon_centers=LON_CENTERS
            )
        except Exception as e:
            raise RuntimeError(
                f"Overpass/OSM fetch failed for Singapore: {e}. "
                "Check network connectivity to overpass-api.de."
            ) from e

        grids["gvi_grid"]         = compute_gvi_from_ndvi(grids["ndvi_grid"])
        grids["pollution_index"]  = _pollution_index(
            grids["pm25_grid"], grids["no2_grid"]
        )

        df = _build_dataframe(grids)
        _write_live_manifest(cfg, "singapore", {
            "city": "singapore",
            "bbox_wgs84": [BBOX_W, BBOX_S, BBOX_E, BBOX_N],
            "grid_n": GRID_N,
            "satellite_year": year,
            "openaq_days_back": 30,
            "openaq_key_supplied": bool(openaq_api_key),
            "gee_project_supplied": bool(gee_project),
            "sources": ["OpenAQ v3", "Google Earth Engine", "Overpass API"],
            "note": "Live API provenance only; no geospatial feature arrays are cached.",
        })

        logger.info("SingaporeGridData: all indicators assembled.")
        _print_summary(grids, df)
        return grids, df


def _pollution_index(pm25: np.ndarray, no2: np.ndarray) -> np.ndarray:
    def zs(a): return (a - a.mean()) / (a.std() + 1e-8)
    return ((zs(pm25) + zs(no2)) / 2.0).astype(np.float32)


def _build_dataframe(grids: Dict[str, np.ndarray]) -> pd.DataFrame:
    records = []
    ndvi = grids["ndvi_grid"]
    pm25 = grids["pm25_grid"]
    no2  = grids["no2_grid"]
    o3   = grids["o3_grid"]
    far  = grids["far_grid"]
    gvi  = grids["gvi_grid"]
    pidx = grids["pollution_index"]

    for i in range(GRID_N):
        for j in range(GRID_N):
            lat_c = LAT_CENTERS[::-1][i]
            lon_c = LON_CENTERS[j]
            records.append({
                "Cell":           f"R{i+1:02d}C{j+1:02d}",
                "Row":            i + 1,
                "Col":            j + 1,
                "Lat":            round(lat_c, 5),
                "Lon":            round(lon_c, 5),
                "NDVI":           round(float(ndvi[i, j]), 4),
                "PM2_5_µg_m3":   round(float(pm25[i, j]), 2),
                "NO2_µg_m3":     round(float(no2[i, j]),  2),
                "O3_µg_m3":      round(float(o3[i, j]),   2),
                "FAR":            round(float(far[i, j]),  3),
                "GVI":            round(float(gvi[i, j]),  4),
                "Pollution_Index": round(float(pidx[i, j]), 4),
                "Vegetated":      bool(ndvi[i, j] > 0.30),
                "Urban_Canyon":   bool(far[i, j]  > 5.00),
            })

    return pd.DataFrame(records)


def _print_summary(grids: Dict[str, np.ndarray], df: pd.DataFrame):
    logger.info("─" * 62)
    logger.info("  %-20s %8s %8s %8s  Unit", "Indicator", "Min", "Mean", "Max")
    logger.info("  " + "─" * 58)
    specs = [
        ("NDVI",          "ndvi_grid",       "dimensionless"),
        ("PM2.5",         "pm25_grid",       "µg m⁻³"),
        ("NO2",           "no2_grid",        "µg m⁻³"),
        ("O3",            "o3_grid",         "µg m⁻³"),
        ("FAR",           "far_grid",        ""),
        ("GVI",           "gvi_grid",        "[0,1]"),
        ("Pollution Idx", "pollution_index", "z-score"),
    ]
    for name, key, unit in specs:
        g = grids[key]
        logger.info("  %-20s %8.3f %8.3f %8.3f  %s",
                    name, g.min(), g.mean(), g.max(), unit)
    veg = int(df["Vegetated"].sum())
    uc  = int(df["Urban_Canyon"].sum())
    logger.info("  Vegetated (NDVI>0.30): %d/100  |  Urban canyon (FAR>5): %d/100",
                veg, uc)
    logger.info("─" * 62)


def _write_live_manifest(cfg, city: str, manifest: Dict):
    if cfg is None or not hasattr(cfg, "paths"):
        return
    output_dir = getattr(cfg.paths, "output_dir", None)
    if output_dir is None:
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{city.lower()}_live_data_manifest.json"
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **manifest,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info("Live data manifest saved to %s", path)
    except Exception as exc:
        logger.warning("Could not write live data manifest: %s", exc)


class AstanaGridData:
    _BBOX_W = 71.3994 - 0.045
    _BBOX_E = 71.3994 + 0.045
    _BBOX_S = 51.0900 - 0.045
    _BBOX_N = 51.0900 + 0.045

    _LAT_CENTERS = np.linspace(_BBOX_S + 0.009, _BBOX_N - 0.009, 10)
    _LON_CENTERS = np.linspace(_BBOX_W + 0.009, _BBOX_E - 0.009, 10)

    @classmethod
    def build(
        cls,
        cfg=None,
        use_gee: bool = True,
        cache: bool = False,
        gee_project: Optional[str] = None,
        year: int = 2025,
        openaq_api_key: Optional[str] = None,
    ) -> Tuple[Dict, "pd.DataFrame"]:
        if not use_gee:
            raise RuntimeError("AstanaGridData requires GEE and live geospatial inputs.")

        logger.info("Fetching Astana geospatial data ...")
        grids = {}

        aq = OpenAQFetcher(api_key=openaq_api_key)
        grids["pm25_grid"] = aq.fetch_pollutant_grid(
            bbox=(cls._BBOX_W, cls._BBOX_S, cls._BBOX_E, cls._BBOX_N),
            parameter="pm25", grid_n=10,
            lat_centers=cls._LAT_CENTERS,
            lon_centers=cls._LON_CENTERS, days_back=365,
        )

        gee = GEEFetcher(
            project=gee_project,
            bbox=(cls._BBOX_W, cls._BBOX_S, cls._BBOX_E, cls._BBOX_N),
            grid_n=10, lat_centers=cls._LAT_CENTERS,
            lon_centers=cls._LON_CENTERS,
        )
        grids["ndvi_grid"] = gee.fetch_ndvi(year=year)
        grids["no2_grid"]  = gee.fetch_no2(year=year)
        grids["o3_grid"]   = gee.fetch_o3(year=year)

        grids["far_grid"] = OverpassFetcher(
            bbox=(cls._BBOX_S, cls._BBOX_W, cls._BBOX_N, cls._BBOX_E)
        ).fetch_far_grid(
            grid_n=10, lat_centers=cls._LAT_CENTERS,
            lon_centers=cls._LON_CENTERS,
        )

        grids["gvi_grid"]        = compute_gvi_from_ndvi(grids["ndvi_grid"])
        grids["pollution_index"] = _pollution_index(grids["pm25_grid"], grids["no2_grid"])

        rows = []
        for r in range(10):
            for c in range(10):
                rows.append({
                    "row": r, "col": c, "node_id": r * 10 + c,
                    "lat": float(cls._LAT_CENTERS[r]),
                    "lon": float(cls._LON_CENTERS[c]),
                    "pm25": float(grids["pm25_grid"][r, c]),
                    "no2":  float(grids["no2_grid"][r, c]),
                    "ndvi": float(grids["ndvi_grid"][r, c]),
                    "o3":   float(grids["o3_grid"][r, c]),
                    "far":  float(grids["far_grid"][r, c]),
                    "gvi":  float(grids["gvi_grid"][r, c]),
                    "pidx": float(grids["pollution_index"][r, c]),
                })
        raw_df = pd.DataFrame(rows)
        _write_live_manifest(cfg, "astana", {
            "city": "astana",
            "bbox_wgs84": [cls._BBOX_W, cls._BBOX_S, cls._BBOX_E, cls._BBOX_N],
            "grid_n": 10,
            "satellite_year": year,
            "openaq_days_back": 365,
            "openaq_key_supplied": bool(openaq_api_key),
            "gee_project_supplied": bool(gee_project),
            "sources": ["OpenAQ v3", "Google Earth Engine", "Overpass API"],
            "note": "Live API provenance only; no geospatial feature arrays are cached.",
        })

        return grids, raw_df
