"""Real environmental grid around Nazarbayev University, Astana, for the contrast analysis.

10 x 10 km, 1 km cells, centred on the university (51.0904 N, 71.3984 E; OpenStreetMap),
in UTM zone 42N (EPSG:32642). Layers, all from open sources and all measured:
  ndvi      Sentinel-2 L2A, June-August 2024, cloud-masked median, (B08-B04)/(B08+B04)
  ndbi      same composite, built-up index (B11-B08)/(B11+B08)
  building  OpenStreetMap building footprint area / cell area
  roads     OpenStreetMap road length (km) / cell area (km^2)
The original submission's land-use regression (6 OpenAQ stations) is not reproduced.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

CENTER = (51.0904, 71.3984)               # lat, lon
UTM = "EPSG:32642"
N_CELLS, CELL_M = 10, 1000
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]


def grid(center=CENTER, utm=UTM):
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    cx, cy = to_utm.transform(center[1], center[0])
    x0, y0 = cx - N_CELLS * CELL_M / 2, cy - N_CELLS * CELL_M / 2
    to_ll = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    rows = []
    for r in range(N_CELLS):
        for c in range(N_CELLS):
            xmin, ymin = x0 + c * CELL_M, y0 + r * CELL_M
            lon, lat = to_ll.transform(xmin + CELL_M / 2, ymin + CELL_M / 2)
            rows.append(dict(cell=r * N_CELLS + c, row=r, col=c, xmin=xmin, ymin=ymin,
                             lon=lon, lat=lat))
    g = pd.DataFrame(rows)
    lon0, lat0 = to_ll.transform(x0, y0)
    lon1, lat1 = to_ll.transform(x0 + N_CELLS * CELL_M, y0 + N_CELLS * CELL_M)
    return g, (x0, y0, x0 + N_CELLS * CELL_M, y0 + N_CELLS * CELL_M), (lon0, lat0, lon1, lat1)


def sentinel2(g, bounds_utm, bbox_ll, start="2024-06-01", end="2024-08-31", max_cloud=20, utm=UTM):
    """Per-cell mean NDVI and NDBI from a cloud-masked median composite (Planetary Computer)."""
    import planetary_computer as pc
    import pystac_client
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_origin

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=pc.sign_inplace)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox_ll,
                            datetime=f"{start}/{end}",
                            query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    res = 10
    w = int((bounds_utm[2] - bounds_utm[0]) / res)
    dst_t = from_origin(bounds_utm[0], bounds_utm[3], res, res)

    def read(item, band, resampling):
        """Read only the grid window (COG range requests), resampled to 10 m."""
        from rasterio.windows import from_bounds
        with rasterio.open(item.assets[band].href) as src:
            if str(src.crs) != utm:
                return np.full((w, w), np.nan, np.float32)   # tile in another UTM zone
            win = from_bounds(*bounds_utm, transform=src.transform)
            arr = src.read(1, window=win, out_shape=(w, w), boundless=True, fill_value=0,
                           resampling=resampling).astype(np.float32)
        arr[arr == 0] = np.nan                       # 0 is nodata in L2A bands and SCL
        return arr

    ndvi_s, ndbi_s, used = [], [], []
    for it in items:
        scl = read(it, "SCL", Resampling.nearest)
        if np.isnan(scl).mean() > 0.5:
            continue
        clear = np.isin(scl, [4, 5, 6, 7, 11])            # vegetation, bare, water, unclassified, snow
        b4, b8 = read(it, "B04", Resampling.bilinear), read(it, "B08", Resampling.bilinear)
        b11 = read(it, "B11", Resampling.bilinear)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = np.where(clear, (b8 - b4) / (b8 + b4), np.nan)
            ndbi = np.where(clear, (b11 - b8) / (b11 + b8), np.nan)
        ndvi_s.append(ndvi)
        ndbi_s.append(ndbi)
        used.append(it.id)
    ndvi = np.nanmedian(np.stack(ndvi_s), axis=0)
    ndbi = np.nanmedian(np.stack(ndbi_s), axis=0)
    k = CELL_M // res
    out = g.copy()
    # raster row 0 is the northern edge; grid row 0 is the southern edge
    out["ndvi"] = [np.nanmean(ndvi[(N_CELLS - 1 - r) * k:(N_CELLS - r) * k, c * k:(c + 1) * k])
                   for r, c in zip(g.row, g.col)]
    out["ndbi"] = [np.nanmean(ndbi[(N_CELLS - 1 - r) * k:(N_CELLS - r) * k, c * k:(c + 1) * k])
                   for r, c in zip(g.row, g.col)]
    return out, used


def overpass(query: str, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(3):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": query}, timeout=300,
                                  headers={"User-Agent": "research-script/1.0"})
                if r.ok:
                    cache.write_text(r.text)
                    return r.json()
            except requests.RequestException:
                pass
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Overpass request failed on all endpoints")


def osm(g, bbox_ll, cache_dir: Path, utm=UTM, name="astana", tiles=1):
    """Building footprint share and road length density per cell."""
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import transform as shp_transform

    s, w_, n, e = bbox_ll[1], bbox_ll[0], bbox_ll[3], bbox_ll[2]
    # dense cities exceed Overpass limits in one request: query tiles, merge by element id
    elements, k = {}, tiles
    for i in range(k):
        for j in range(k):
            ts, tn = s + (n - s) * i / k, s + (n - s) * (i + 1) / k
            tw, te = w_ + (e - w_) * j / k, w_ + (e - w_) * (j + 1) / k
            for kind in ("building", "highway"):
                q = f'[out:json][timeout:180];way["{kind}"]({ts},{tw},{tn},{te});out geom;'
                part = overpass(q, cache_dir / f"osm_{name}_{kind}_{i}{j}.json")
                for el in part["elements"]:
                    elements[el["id"]] = el
    data = {"elements": list(elements.values())}
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    cells = [box(x, y, x + CELL_M, y + CELL_M) for x, y in zip(g.xmin, g.ymin)]
    bld, road = np.zeros(len(g)), np.zeros(len(g))
    road_types = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                  "residential", "living_street", "service", "motorway_link", "trunk_link",
                  "primary_link", "secondary_link", "tertiary_link"}
    for el in data["elements"]:
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        tags = el.get("tags", {})
        if "building" in tags and len(pts) >= 4:
            geom = shp_transform(to_utm, Polygon(pts)).buffer(0)
            acc = bld
        elif tags.get("highway") in road_types and len(pts) >= 2:
            geom = shp_transform(to_utm, LineString(pts))
            acc = road
        else:
            continue
        for i, c in enumerate(cells):
            if geom.intersects(c):
                part = geom.intersection(c)
                acc[i] += part.area if acc is bld else part.length
    out = g.copy()
    out["building"] = bld / CELL_M ** 2
    out["roads"] = road / 1000 / (CELL_M / 1000) ** 2
    return out


def build(out_csv: Path, center=CENTER, utm=UTM, start="2024-06-01", end="2024-08-31",
          max_cloud=20, name="astana", tiles=1):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    g, bounds_utm, bbox_ll = grid(center, utm)
    s2_cache = out_csv.parent / f"{name}_s2.csv"
    if s2_cache.exists():
        g = pd.read_csv(s2_cache)
        scenes = json.loads((out_csv.parent / f"{name}_s2_items.json").read_text())
    else:
        g, scenes = sentinel2(g, bounds_utm, bbox_ll, start, end, max_cloud, utm)
        g.to_csv(s2_cache, index=False)
        (out_csv.parent / f"{name}_s2_items.json").write_text(json.dumps(scenes))
    g = osm(g, bbox_ll, out_csv.parent, utm, name, tiles)
    g.to_csv(out_csv, index=False)
    (out_csv.parent / f"{name}_grid_sources.json").write_text(json.dumps(
        {"sentinel2_items": scenes, "center": center, "crs": utm, "bbox_lonlat": bbox_ll,
         "dates": [start, end], "max_cloud": max_cloud}, indent=1))
    return g


# Singapore central planning area, as in the original submission (1 deg 17' N, 103 deg 49' E)
SINGAPORE = dict(center=(1.2833, 103.8167), utm="EPSG:32648", start="2024-01-01",
                 end="2024-12-31", max_cloud=30, name="singapore", tiles=3)
