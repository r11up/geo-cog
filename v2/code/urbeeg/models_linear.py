"""Ridge-regression models on fixed feature sets, plus the control models.

Contract shared by every model (the evaluation loop treats them alike):

    m.fit(data, Y, inner, final)
        Y     : (n_rows, n_targets) z-scored targets aligned with data rows (NaN outside
                the training block);
        inner : list of (train_rows, val_rows) splits inside the training block, each new in
                both people and scenes, used only to choose hyperparameters / blend weights;
        final : all training rows; the model is refitted on these at the end.
    m.predict(data, rows) -> (n_rows, n_targets)

`data` is the dict from dataset.load_pairs() plus data['scene'] (DataFrame indexed by
stimulus). Rows are integer indices.
"""
import numpy as np
from pyriemann.tangentspace import TangentSpace

from .scene import FEATURES as COMPOSITION, LOW_LEVEL

ALPHAS = np.logspace(-2, 6, 17)


def _svd_solve(X, Y, alphas):
    """Ridge coefficients, each target with its own alpha, from one SVD. X, Y centred."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    return np.column_stack([Vt.T @ ((s / (s ** 2 + a)) * UtY[:, k]) for k, a in enumerate(alphas)])


def _path_predictions(X_tr, Y_tr, X_ev):
    """(n_alphas, n_ev, n_targets) predictions for every alpha in ALPHAS. X, Y centred."""
    U, s, Vt = np.linalg.svd(X_tr, full_matrices=False)
    UtY, XV = U.T @ Y_tr, X_ev @ Vt.T
    return np.stack([XV @ ((s / (s ** 2 + a))[:, None] * UtY) for a in ALPHAS])


class Scaler:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


class FeatureRidge:
    def __init__(self, name, features):
        self.name, self.features = name, features   # features(data, rows, fit) -> X

    def select(self, data, Y, inner):
        """Choose alpha per target by summed squared error over the inner splits."""
        err = 0.0
        for tr, va in inner:
            X_tr = self.features(data, tr, fit=True)
            sc = Scaler().fit(X_tr)
            mu = Y[tr].mean(0)
            P = _path_predictions(sc(X_tr), Y[tr] - mu, sc(self.features(data, va))) + mu
            err = err + ((P - Y[va][None]) ** 2).sum(axis=1)
        self.alpha = ALPHAS[np.asarray(err).argmin(axis=0)]
        return self

    def fit_rows(self, data, Y, rows):
        X = self.features(data, rows, fit=True)
        self.scaler, self.y_mu = Scaler().fit(X), Y[rows].mean(0)
        self.W = _svd_solve(self.scaler(X), Y[rows] - self.y_mu, self.alpha)
        return self

    def fit(self, data, Y, inner, final):
        return self.select(data, Y, inner).fit_rows(data, Y, final)

    def predict(self, data, rows):
        return self.scaler(self.features(data, rows)) @ self.W + self.y_mu


class Intercept:
    name = "intercept"

    def fit(self, data, Y, inner, final):
        self.mu = Y[final].mean(0)
        return self

    def predict(self, data, rows):
        return np.tile(self.mu, (len(rows), 1))


class Stack:
    """Prediction-level fusion. Each base chooses its alpha on the inner splits; its
    out-of-sample predictions on every inner validation set are pooled, and a non-negative
    blend (plus intercept) is fitted per target on them. The bases are then refitted on
    all training rows and combined with those weights."""

    def __init__(self, name, bases):
        self.name, self.bases = name, bases

    def fit(self, data, Y, inner, final):
        from scipy.optimize import nnls

        for b in self.bases:
            b.select(data, Y, inner)
        va_all = np.concatenate([va for _, va in inner])
        P = [np.vstack([b.fit_rows(data, Y, tr).predict(data, va) for tr, va in inner])
             for b in self.bases]
        Yv = Y[va_all]
        self.w, self.b0 = [], []
        for k in range(Y.shape[1]):
            A = np.column_stack([p[:, k] for p in P])
            w, _ = nnls(A - A.mean(0), Yv[:, k] - Yv[:, k].mean())
            self.w.append(w)
            self.b0.append(Yv[:, k].mean() - A.mean(0) @ w)
        for b in self.bases:
            b.fit_rows(data, Y, final)
        return self

    def predict(self, data, rows):
        P = [b.predict(data, rows) for b in self.bases]
        return np.column_stack([sum(w[i] * P[i][:, k] for i in range(len(P))) + self.b0[k]
                                for k, w in enumerate(self.w)])


# ---------------------------------------------------------------- feature builders

def scene_composition(data, rows, fit=False):
    return data["scene"].loc[data["stimulus"][rows], COMPOSITION].to_numpy()


def scene_columns(cols):
    """Builder for a subset of the composition (per-category ablations)."""
    def build(data, rows, fit=False):
        return data["scene"].loc[data["stimulus"][rows], cols].to_numpy()
    return build


ABLATE = ["greenery", "buildings", "sky", "ground", "vehicles"]


def scene_low_level(data, rows, fit=False):
    return data["scene"].loc[data["stimulus"][rows], LOW_LEVEL].to_numpy()


def erp_components(data, rows, fit=False):
    return data["erp_comp"][rows].astype(float)


def erp_windows(data, rows, fit=False):
    return data["erp_win"][rows].astype(float)


class RiemannTangent:
    """Tangent-space projection of the per-band covariances. The reference point is the
    Riemannian mean of the rows passed with fit=True (training rows only).

    `cache` (shared by the models of one cell) memoises reference fits by the SET of
    rows and projections by (reference, rows), since several models need the same ones."""

    def __init__(self, cache=None):
        self.ref = None
        self.cache = cache if cache is not None else {}

    @staticmethod
    def _key(rows, ordered):
        r = np.asarray(rows)
        return (r if ordered else np.sort(r)).tobytes()

    def __call__(self, data, rows, fit=False):
        if fit or self.ref is None:
            self.ref = ("ref", self._key(rows, ordered=False))
            if self.ref not in self.cache:
                covs = data["cov"][np.sort(rows)].astype(float)
                self.cache[self.ref] = [TangentSpace(metric="riemann").fit(covs[:, b])
                                        for b in range(covs.shape[1])]
        key = ("proj", self.ref, self._key(rows, ordered=True))
        if key not in self.cache:
            covs = data["cov"][rows].astype(float)
            self.cache[key] = np.hstack([t.transform(covs[:, b])
                                         for b, t in enumerate(self.cache[self.ref])])
        return self.cache[key]


class Concat:
    def __init__(self, *builders):
        self.builders = builders

    def __call__(self, data, rows, fit=False):
        return np.hstack([b(data, rows, fit=fit) for b in self.builders])


class ShuffledScenes:
    """Control: each scene gets another scene's features (one fixed permutation per cell,
    applied to training and test alike)."""

    def __init__(self, builder, seed):
        self.builder, self.seed = builder, seed

    def __call__(self, data, rows, fit=False):
        scenes = np.array(sorted(data["scene"].index))
        perm = dict(zip(scenes, np.random.default_rng(self.seed).permutation(scenes)))
        shuffled = dict(data, stimulus=np.array([perm[s] for s in data["stimulus"]]))
        return self.builder(shuffled, rows, fit=fit)


class NoiseScenes:
    """Control: a pure-noise 'environment' with the same number of
    features as the composition, drawn once per scene, so it is as consistent across
    people as the real scene features but carries no environmental information."""

    def __init__(self, n_features, seed):
        self.n, self.seed = n_features, seed

    def __call__(self, data, rows, fit=False):
        scenes = sorted(data["scene"].index)
        noise = dict(zip(scenes, np.random.default_rng(self.seed).normal(size=(len(scenes), self.n))))
        return np.array([noise[s] for s in data["stimulus"][rows]])


class ShuffledTrainingRows:
    """Control: when fitting, each training row is given another training row's features,
    breaking the EEG-rating link; rows used for prediction keep their own features."""

    def __init__(self, builder, seed):
        self.builder, self.seed = builder, seed

    def __call__(self, data, rows, fit=False):
        if fit:
            rows = np.random.default_rng(self.seed).permutation(rows)
        return self.builder(data, rows, fit=fit)


def model_zoo(seed: int) -> list:
    """Linear models and controls evaluated in every cell (fresh instances per cell)."""
    cache = {}

    def eeg():
        return Concat(erp_windows, RiemannTangent(cache))

    return [
        Intercept(),
        FeatureRidge("scene", scene_composition),
        FeatureRidge("scene_greenery_only", scene_columns(["greenery"])),
        *[FeatureRidge(f"scene_minus_{c}", scene_columns([x for x in COMPOSITION if x != c]))
          for c in ABLATE],
        FeatureRidge("scene_lowlevel_only", scene_low_level),
        FeatureRidge("scene+lowlevel", Concat(scene_composition, scene_low_level)),
        FeatureRidge("eeg_erp4", erp_components),
        FeatureRidge("eeg_erpwin", erp_windows),
        FeatureRidge("eeg_riemann", RiemannTangent(cache)),
        FeatureRidge("eeg", eeg()),
        Stack("fusion", [FeatureRidge("scene", scene_composition), FeatureRidge("eeg", eeg())]),
        FeatureRidge("ctrl_scene_shuffled", ShuffledScenes(scene_composition, seed)),
        FeatureRidge("ctrl_scene_noise", NoiseScenes(len(COMPOSITION), seed)),
        FeatureRidge("ctrl_eeg_shuffled", ShuffledTrainingRows(eeg(), seed)),
        Stack("ctrl_fusion_noise", [FeatureRidge("noise", NoiseScenes(len(COMPOSITION), seed)),
                                    FeatureRidge("eeg", eeg())]),
    ]
