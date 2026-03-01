"""PECM samples city-grid context for training and RMA provides optional covariance-space alignment."""

import logging
import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

CLASS_NEUTRAL   = 0
CLASS_SADNESS   = 1
CLASS_ANGER     = 2
CLASS_HAPPINESS = 3
CLASS_RELAXED   = 4

FEAT_PM25 = 0
FEAT_NDVI = 1
FEAT_FAR  = 2
FEAT_GVI  = 3
FEAT_NO2  = 4
FEAT_O3   = 5


def _select_bandwidth_cv(
    node_features: np.ndarray,
    n_splits: int = 5,
    candidates: Optional[List[float]] = None,
    random_state: int = 42,
) -> float:
    n, d = node_features.shape
    rng = np.random.default_rng(random_state)
    scott = n ** (-1.0 / (d + 4))
    if candidates is None:
        candidates = [scott * factor for factor in (0.25, 0.5, 1.0, 2.0, 4.0)]

    idx = rng.permutation(n)
    folds = np.array_split(idx, min(n_splits, n))
    best_h, best_ll = float(candidates[0]), -np.inf

    for h in candidates:
        ll_total, n_folds = 0.0, 0
        for val_idx in folds:
            train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            train_pts = node_features[train_idx]
            val_pts = node_features[val_idx]
            diff = val_pts[:, None, :] - train_pts[None, :, :]
            sq_dist = (diff ** 2).sum(axis=-1)
            log_kernel = -0.5 * sq_dist / (h ** 2)
            max_log = log_kernel.max(axis=1, keepdims=True)
            log_const = -d / 2.0 * np.log(2 * np.pi * h ** 2)
            log_p = (
                log_const
                + np.log(np.mean(np.exp(log_kernel - max_log), axis=1) + 1e-300)
                + max_log.ravel()
            )
            ll_total += float(log_p.mean())
            n_folds += 1

        mean_ll = ll_total / max(n_folds, 1)
        if mean_ll > best_ll:
            best_ll = mean_ll
            best_h = float(h)

    logger.info("PECM bandwidth CV selected h=%.4f (mean log-lik=%.4f)",
                best_h, best_ll)
    return best_h


class PECM:
    def __init__(
        self,
        node_features: np.ndarray,
        bandwidth: float | None = None,
        high_q: float = 0.75,
        low_q:  float = 0.25,
        random_state: int = 42,
        cv_candidates: Optional[List[float]] = None,
        reverse: bool = False,
    ):
        self.F    = node_features.astype(np.float64)
        self.high_q = high_q
        self.low_q  = low_q
        self.rng    = np.random.default_rng(random_state)
        self.reverse = bool(reverse)

        n, d = self.F.shape
        if bandwidth is None:
            self.h = _select_bandwidth_cv(
                self.F,
                n_splits=5,
                candidates=cv_candidates,
                random_state=random_state,
            )
        else:
            self.h = float(bandwidth)
            logger.info("PECM: using supplied bandwidth h=%.4f", self.h)

        self._build_subsets()

    def _build_subsets(self):
        F = self.F
        hi = {f: F[:, f] >= np.quantile(F[:, f], self.high_q) for f in range(6)}
        lo = {f: F[:, f] <= np.quantile(F[:, f], self.low_q)  for f in range(6)}
        iq = {f: (F[:, f] >  np.quantile(F[:, f], self.low_q)) &
                 (F[:, f] <  np.quantile(F[:, f], self.high_q)) for f in range(6)}
        high_or_low = lo if self.reverse else hi

        self._subsets: dict[int, np.ndarray] = {
            CLASS_NEUTRAL:  iq[FEAT_PM25] & iq[FEAT_NDVI],
            CLASS_SADNESS:  high_or_low[FEAT_PM25] & high_or_low[FEAT_FAR],
            CLASS_ANGER:    high_or_low[FEAT_PM25] & high_or_low[FEAT_FAR],
            CLASS_HAPPINESS:high_or_low[FEAT_NDVI] & high_or_low[FEAT_GVI],
            CLASS_RELAXED:  high_or_low[FEAT_NDVI] & high_or_low[FEAT_GVI],
        }
        for cls, mask in self._subsets.items():
            if not mask.any():
                logger.warning(
                    "PECM: empty subset for class %d — using full grid.", cls
                )
                self._subsets[cls] = np.ones(len(self.F), dtype=bool)

    def sample(self, labels: np.ndarray) -> np.ndarray:
        N = len(labels)
        env_vectors = np.zeros((N, self.F.shape[1]), dtype=np.float32)

        for cls in range(5):
            idx = np.where(labels == cls)[0]
            if not len(idx):
                continue
            subset = self.F[self._subsets[cls]]

            ref_idx  = self.rng.integers(0, len(subset), size=len(idx))
            ref_pts  = subset[ref_idx]

            noise    = self.rng.normal(0.0, self.h, size=ref_pts.shape)
            env_vectors[idx] = (ref_pts + noise).astype(np.float32)

        return env_vectors

    def sample_unconditional(self, n_samples: int) -> np.ndarray:
        ref_idx = self.rng.integers(0, len(self.F), size=n_samples)
        ref_pts = self.F[ref_idx]
        noise = self.rng.normal(0.0, self.h, size=ref_pts.shape)
        return (ref_pts + noise).astype(np.float32)

def compute_eeg_covariances(
    epochs_raw: np.ndarray,
    regularisation: float = 1e-4,
) -> np.ndarray:
    T, C, S = epochs_raw.shape

    if S < C:
        reg = max(regularisation, 1e-2)
        logger.warning(
            "compute_eeg_covariances: S=%d < C=%d → covariance is rank ≤ %d. "
            "Increasing regularisation to %.2e to stabilise Riemannian ops. "
            "For best results pass raw epoch arrays (T, C, 500) rather than "
            "feature tensors (T, C, 10).",
            S, C, S - 1, reg,
        )
    else:
        reg = regularisation

    covs = np.zeros((T, C, C), dtype=np.float64)
    for t in range(T):
        cov = np.cov(epochs_raw[t], rowvar=True)
        cov += reg * np.eye(C)
        covs[t] = cov
    return covs


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    vals, vecs = eigh(M)
    vals = np.maximum(vals, 1e-10)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def _matrix_sqrt_inv(M: np.ndarray) -> np.ndarray:
    vals, vecs = eigh(M)
    vals = np.maximum(vals, 1e-10)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def _riemannian_mean(covs: np.ndarray, max_iter: int = 50,
                     tol: float = 1e-6) -> np.ndarray:
    N, d, _ = covs.shape
    C_bar = covs.mean(axis=0)

    for iteration in range(max_iter):
        C_bar_sqrt_inv = _matrix_sqrt_inv(C_bar)
        C_bar_sqrt     = _matrix_sqrt(C_bar)

        S = np.zeros_like(C_bar)
        for i in range(N):
            tmp = C_bar_sqrt_inv @ covs[i] @ C_bar_sqrt_inv
            vals, vecs = eigh(tmp)
            vals = np.maximum(vals, 1e-10)
            log_tmp = vecs @ np.diag(np.log(vals)) @ vecs.T
            S += log_tmp

        S /= N

        vals_S, vecs_S = eigh(S)
        exp_S = vecs_S @ np.diag(np.exp(vals_S)) @ vecs_S.T
        C_bar_new = C_bar_sqrt @ exp_S @ C_bar_sqrt

        change = np.linalg.norm(C_bar_new - C_bar, "fro")
        C_bar = C_bar_new

        if change < tol:
            logger.debug("Karcher flow converged in %d iterations.", iteration + 1)
            break

    return C_bar


def _tangent_project(covs: np.ndarray, mean_cov: np.ndarray) -> np.ndarray:
    N, d, _ = covs.shape
    S_inv_sqrt = _matrix_sqrt_inv(mean_cov)
    S_sqrt     = _matrix_sqrt(mean_cov)
    Z = np.zeros((N, d * d), dtype=np.float64)

    for i in range(N):
        tmp = S_inv_sqrt @ covs[i] @ S_inv_sqrt
        vals, vecs = eigh(tmp)
        vals = np.maximum(vals, 1e-10)
        log_tmp = vecs @ np.diag(np.log(vals)) @ vecs.T
        tangent = S_sqrt @ log_tmp @ S_sqrt
        Z[i] = tangent.ravel()

    return Z


class RMA:
    def __init__(self, embed_dim: int = 256):
        self.embed_dim = embed_dim
        self._fitted = False

    def fit(self, eeg_covs: np.ndarray, env_vectors: np.ndarray):
        N, C, _ = eeg_covs.shape

        logger.info("RMA Step 1: computing Riemannian mean (Karcher flow) …")
        self.C_bar = _riemannian_mean(eeg_covs)

        C_bar_sqrt_inv = _matrix_sqrt_inv(self.C_bar)
        covs_centered = np.stack([
            C_bar_sqrt_inv @ eeg_covs[i] @ C_bar_sqrt_inv
            for i in range(N)
        ])

        logger.info("RMA Step 2: dispersion matching (Frobenius-norm, Eq. 3) …")
        Z_eeg = _tangent_project(covs_centered, np.eye(C))

        frob_eeg = np.linalg.norm(Z_eeg, ord='fro')
        env_mat  = env_vectors.astype(np.float64)
        frob_env = np.linalg.norm(env_mat, ord='fro') + 1e-12
        self.scale_env = frob_eeg / frob_env
        env_scaled = env_mat * self.scale_env

        max_pca_components = min(self.embed_dim, N, Z_eeg.shape[1])
        from sklearn.decomposition import PCA
        self.pca_eeg = PCA(n_components=max_pca_components, random_state=42)
        Z_eeg_proj = self.pca_eeg.fit_transform(Z_eeg)

        if Z_eeg_proj.shape[1] < self.embed_dim:
            pad = np.zeros((N, self.embed_dim - Z_eeg_proj.shape[1]))
            Z_eeg_proj = np.concatenate([Z_eeg_proj, pad], axis=1)

        self.rng = np.random.default_rng(42)
        self.W_env = self._random_orthogonal(6, self.embed_dim)
        Z_env_proj = env_scaled @ self.W_env

        logger.info("RMA Step 3: Procrustes rotation …")
        M = Z_eeg_proj.T @ Z_env_proj
        U, _, Vt = np.linalg.svd(M)
        self.R_hat = U @ Vt

        self._fitted = True
        logger.info("RMA fitted on %d samples.", N)

        return Z_eeg_proj, Z_env_proj @ self.R_hat

    def transform(self, eeg_covs: np.ndarray,
                  env_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        N, C, _ = eeg_covs.shape
        C_bar_sqrt_inv = _matrix_sqrt_inv(self.C_bar)
        covs_centered = np.stack([
            C_bar_sqrt_inv @ eeg_covs[i] @ C_bar_sqrt_inv
            for i in range(N)
        ])
        Z_eeg_flat = _tangent_project(covs_centered, np.eye(C))
        Z_eeg_proj = self.pca_eeg.transform(Z_eeg_flat)
        if Z_eeg_proj.shape[1] < self.embed_dim:
            pad = np.zeros((N, self.embed_dim - Z_eeg_proj.shape[1]))
            Z_eeg_proj = np.concatenate([Z_eeg_proj, pad], axis=1)

        env_scaled = env_vectors.astype(np.float64) * self.scale_env
        Z_env_proj = (env_scaled @ self.W_env) @ self.R_hat

        return Z_eeg_proj.astype(np.float32), Z_env_proj.astype(np.float32)

    @staticmethod
    def _random_orthogonal(in_dim: int, out_dim: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        if in_dim >= out_dim:
            A = rng.standard_normal((in_dim, out_dim))
            Q, _ = np.linalg.qr(A)
            return Q[:, :out_dim].astype(np.float32)
        else:
            A = rng.standard_normal((out_dim, in_dim))
            Q, _ = np.linalg.qr(A)
            return Q.T.astype(np.float32)
