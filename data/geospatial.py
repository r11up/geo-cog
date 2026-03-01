"""Build live Astana and Singapore environmental feature graphs for the neuro-geospatial pipeline."""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

FEAT_IDX = {
    "PM2.5": 0,
    "NDVI": 1,
    "FAR":  2,
    "GVI":  3,
    "NO2":  4,
    "O3":   5,
}
FEAT_NAMES = ["PM2.5", "NDVI", "FAR", "GVI", "NO2", "O3"]


def _make_grid_centroids(grid_size: int = 10) -> np.ndarray:
    rows = np.arange(grid_size)
    cols = np.arange(grid_size)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.float32)


def _queen_adjacency(coords: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    N = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    queen_mask = (dist > 0) & (dist <= np.sqrt(2) + 0.01)
    W = np.where(queen_mask, 1.0 / (dist ** 2 + epsilon), 0.0)
    A_tilde = W + np.eye(N)
    D = A_tilde.sum(axis=1)
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    A_hat = D_inv_sqrt[:, None] * A_tilde * D_inv_sqrt[None, :]
    return A_hat.astype(np.float32)


def _grids_to_node_features(grids: dict, grid_size: int = 10) -> np.ndarray:
    N = grid_size * grid_size

    def _zscore(arr: np.ndarray) -> np.ndarray:
        flat = arr.ravel().astype(np.float64)
        mu, sigma = flat.mean(), flat.std()
        return ((flat - mu) / (sigma + 1e-8)).astype(np.float32)

    pm25_z = _zscore(grids["pm25_grid"])
    no2_z  = _zscore(grids["no2_grid"])
    pm25   = pm25_z
    ndvi   = _zscore(grids["ndvi_grid"])
    far    = _zscore(grids["far_grid"])
    gvi    = _zscore(grids["gvi_grid"])
    no2    = no2_z
    o3     = _zscore(grids["o3_grid"])

    F = np.stack([pm25, ndvi, far, gvi, no2, o3], axis=1)
    assert F.shape == (N, 6)
    return F


def build_city_grid(
    cfg,
    city: str,
    use_gee: bool = True,
    gee_project: Optional[str] = None,
    openaq_api_key: Optional[str] = None,
    year: int = 2025,
    force_refresh: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    city = city.lower()
    if city not in ("astana", "singapore"):
        raise ValueError(f"Unknown city '{city}'. Choose 'astana' or 'singapore'.")

    logger.info("Fetching live geospatial data for %s ...", city.title())

    from data.geo_data import AstanaGridData, SingaporeGridData

    if city == "astana":
        grids, _ = AstanaGridData.build(
            cfg=cfg,
            use_gee=use_gee,
            cache=False,
            gee_project=gee_project,
            openaq_api_key=openaq_api_key,
            year=year,
        )
    else:
        grids, _ = SingaporeGridData.build(
            cfg=cfg,
            use_gee=use_gee,
            cache=False,
            gee_project=gee_project,
            openaq_api_key=openaq_api_key,
            year=year,
        )

    node_features = _grids_to_node_features(grids, cfg.geo.grid_size)
    coords        = _make_grid_centroids(cfg.geo.grid_size)
    adj_matrix    = _queen_adjacency(coords, epsilon=cfg.geo.epsilon)

    logger.info(
        "%s grid built: %d nodes, EPSG:%d",
        city.title(), node_features.shape[0],
        cfg.geo.epsg_astana if city == "astana" else cfg.geo.epsg_singapore,
    )
    return node_features, adj_matrix


def build_astana_grid(
    cfg,
    use_gee: bool = True,
    gee_project: Optional[str] = None,
    openaq_api_key: Optional[str] = None,
    year: int = 2025,
    force_refresh: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    return build_city_grid(
        cfg, "astana",
        use_gee=use_gee, gee_project=gee_project,
        openaq_api_key=openaq_api_key, year=year,
        force_refresh=force_refresh,
    )


def build_singapore_grid(
    cfg,
    use_gee: bool = True,
    gee_project: Optional[str] = None,
    openaq_api_key: Optional[str] = None,
    year: int = 2025,
    force_refresh: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    return build_city_grid(
        cfg, "singapore",
        use_gee=use_gee, gee_project=gee_project,
        openaq_api_key=openaq_api_key, year=year,
        force_refresh=force_refresh,
    )


def coarsen_pollution_features(
    node_features: np.ndarray,
    grid_size: int = 10,
    target_km: int = 5,
) -> np.ndarray:
    factor = target_km
    if grid_size % factor != 0:
        raise ValueError(
            f"grid_size={grid_size} not divisible by target_km={factor}."
        )

    features_out = node_features.copy()
    F_grid = features_out.reshape(grid_size, grid_size, 6)
    n_blocks = grid_size // factor

    for br in range(n_blocks):
        for bc in range(n_blocks):
            r0, r1 = br * factor, (br + 1) * factor
            c0, c1 = bc * factor, (bc + 1) * factor
            block = F_grid[r0:r1, c0:c1, :]
            for feat_i in [FEAT_IDX["PM2.5"], FEAT_IDX["NO2"]]:
                F_grid[r0:r1, c0:c1, feat_i] = block[:, :, feat_i].mean()

    return F_grid.reshape(grid_size * grid_size, 6).astype(np.float32)
