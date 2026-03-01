"""Fit simple land-use regression surfaces for PM2.5 and NO2 from station observations."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PREDICTOR_COLS = [
    "road_density",
    "far",
    "tropomi_bg",
    "ndvi",
    "dist_road",
]


class LURModel:
    def __init__(self, pollutant: str = "PM2.5", standardise: bool = True):
        if pollutant not in ("PM2.5", "NO2"):
            raise ValueError("pollutant must be 'PM2.5' or 'NO2'")
        self.pollutant = pollutant
        self.standardise = standardise
        self._model = LinearRegression()
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, station_df: pd.DataFrame, city: Optional[str] = None) -> "LURModel":
        _check_columns(station_df, PREDICTOR_COLS + ["concentration"])
        X = station_df[PREDICTOR_COLS].values.astype(np.float64)
        y = station_df["concentration"].values.astype(np.float64)
        if self.standardise:
            X = self._scaler.fit_transform(X)
        self._model.fit(X, y)
        self._fitted = True
        y_hat = np.maximum(self._model.predict(X), 0.0)
        r2 = r2_score(y, y_hat)
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        logger.info(
            "LUR [%s | %s] fitted on %d stations — train R2=%.3f RMSE=%.3f",
            city or "?", self.pollutant, len(station_df), r2, rmse,
        )
        return self

    def predict(self, grid_df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        _check_columns(grid_df, PREDICTOR_COLS)
        X = grid_df[PREDICTOR_COLS].values.astype(np.float64)
        if self.standardise:
            X = self._scaler.transform(X)
        return np.maximum(self._model.predict(X), 0.0).astype(np.float32)

    def loso_cv(self, station_df: pd.DataFrame) -> Tuple[float, np.ndarray]:
        _check_columns(station_df, PREDICTOR_COLS + ["concentration"])
        X_all = station_df[PREDICTOR_COLS].values.astype(np.float64)
        y_all = station_df["concentration"].values.astype(np.float64)
        n = len(station_df)
        y_oof = np.zeros(n, dtype=np.float64)
        for i in range(n):
            idx_tr = [j for j in range(n) if j != i]
            X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
            X_te = X_all[[i]]
            scaler = StandardScaler()
            if self.standardise:
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
            m = LinearRegression().fit(X_tr, y_tr)
            y_oof[i] = max(m.predict(X_te)[0], 0.0)
        r2 = r2_score(y_all, y_oof)
        rmse = float(np.sqrt(np.mean((y_all - y_oof) ** 2)))
        logger.info(
            "LUR [%s] LOSO CV R2=%.3f RMSE=%.3f (n=%d)",
            self.pollutant, r2, rmse, n,
        )
        return r2, (y_all - y_oof)

    @property
    def coefficients(self) -> Optional[Dict[str, float]]:
        if not self._fitted:
            return None
        coef = dict(zip(PREDICTOR_COLS, self._model.coef_))
        coef["intercept"] = float(self._model.intercept_)
        return coef


def fit_city_lur(
    city: str,
    pm25_station_df: pd.DataFrame,
    no2_station_df: pd.DataFrame,
    grid_predictors: pd.DataFrame,
) -> Dict[str, object]:
    logger.info("Fitting LUR models for %s ...", city)
    lur_pm25 = LURModel("PM2.5").fit(pm25_station_df, city=city)
    lur_no2  = LURModel("NO2").fit(no2_station_df,  city=city)
    r2_pm25, _ = lur_pm25.loso_cv(pm25_station_df)
    r2_no2,  _ = lur_no2.loso_cv(no2_station_df)
    return {
        "pm25":      lur_pm25.predict(grid_predictors),
        "no2":       lur_no2.predict(grid_predictors),
        "lur_pm25":  lur_pm25,
        "lur_no2":   lur_no2,
        "r2_pm25":   r2_pm25,
        "r2_no2":    r2_no2,
    }


def _check_columns(df: pd.DataFrame, required: list):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")
