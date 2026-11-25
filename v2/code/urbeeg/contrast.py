"""Contrast analysis: the original submission's literature-based prior, re-run under controls.

Environment vectors come from the real Astana grid (astana_grid.py): NDVI, NDBI, building
coverage, road density, z-scored over the 100 cells. The prior follows the original rule:
  anger, sadness       -> cells in the upper quantile of the exposure score
                          (mean z of building coverage, road density, NDBI; stands in for
                          PM2.5 and floor-area ratio, which are not reproduced)
  happiness, calmness  -> cells in the upper quantile of NDVI
  neutral              -> cells inside the interquartile range of both scores
A vector is a random cell of the class subset plus Gaussian kernel noise (bandwidth h,
chosen by cross-validated likelihood over all cells). Training vectors are conditioned on
the label; test vectors are drawn from all cells, independent of the label.

Conditions
  eeg_only            no environment branch
  prior               literature prior as above
  noise               N(0, I) vectors, independent of the label, in training and test
  matched_random      synthetic vectors with the same class means and covariances as the
                      prior (no environmental data), label-conditioned in training only
  prior_removed_test  trained with the prior, environment set to the grid mean at test
  prior_reversed      prior with the anger/sadness and happiness/calmness subsets swapped
  prior_test_labels   diagnostic only: test vectors also conditioned on the true label
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .eav import CLASSES

NEG, POS, NEU = (1, 2), (3, 4), (0,)          # sadness, anger | happiness, calmness | neutral


def env_matrix(grid: pd.DataFrame) -> np.ndarray:
    E = grid[["ndvi", "ndbi", "building", "roads"]].to_numpy(float)
    return (E - E.mean(0)) / E.std(0)


def class_subsets(E, q=0.75, reversed_=False):
    exposure = E[:, 1:].mean(axis=1)
    veg = E[:, 0]
    hi_exp = np.where(exposure >= np.quantile(exposure, q))[0]
    hi_veg = np.where(veg >= np.quantile(veg, q))[0]
    mid = np.where((exposure > np.quantile(exposure, .25)) & (exposure < np.quantile(exposure, .75))
                   & (veg > np.quantile(veg, .25)) & (veg < np.quantile(veg, .75)))[0]
    neg, pos = (hi_veg, hi_exp) if reversed_ else (hi_exp, hi_veg)
    sub = {}
    for k in NEG:
        sub[k] = neg
    for k in POS:
        sub[k] = pos
    sub[0] = mid
    return sub


def select_bandwidth(E, seed=0):
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.neighbors import KernelDensity

    gs = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": np.logspace(-1.5, 0.5, 21)},
                      cv=KFold(min(5, len(E)), shuffle=True, random_state=seed))  # LOO for tiny grids
    gs.fit(E)
    return float(gs.best_params_["bandwidth"])


class EnvSampler:
    def __init__(self, E, kind, q=0.75, h=0.3, seed=0):
        self.E, self.kind, self.h = E, kind, h
        self.rng = np.random.default_rng(seed)
        self.sub = class_subsets(E, q, reversed_=(kind == "prior_reversed"))
        if kind == "matched_random":
            # moments of the prior's class-conditional distributions, nothing else
            self.mom = {k: (E[s].mean(0), np.cov(E[s].T) + h ** 2 * np.eye(E.shape[1]))
                        for k, s in self.sub.items()}
            self.mix = (E.mean(0), np.cov(E.T) + h ** 2 * np.eye(E.shape[1]))

    def train(self, y):
        d = self.E.shape[1]
        if self.kind == "unconditional":        # original 'random pairing': no label link
            return self.test(y)
        if self.kind == "shuffled_labels":      # original 'shuffled labels': prior keyed to
            y = self.rng.permutation(y)         # a random relabelling of the epochs
        if self.kind == "noise":
            return self.rng.normal(size=(len(y), d))
        if self.kind == "matched_random":
            return np.stack([self.rng.multivariate_normal(*self.mom[k]) for k in y])
        cells = np.array([self.rng.choice(self.sub[k]) for k in y])
        return self.E[cells] + self.rng.normal(scale=self.h, size=(len(y), d))

    def test(self, y):
        d = self.E.shape[1]
        if self.kind == "noise":
            return self.rng.normal(size=(len(y), d))
        if self.kind == "matched_random":
            return self.rng.multivariate_normal(*self.mix, size=len(y))
        if self.kind == "prior_removed_test":
            return np.zeros((len(y), d))
        if self.kind == "prior_test_labels":
            return self.train(y)
        cells = self.rng.integers(0, len(self.E), size=len(y))
        return self.E[cells] + self.rng.normal(scale=self.h, size=(len(y), d))


class Fusion(nn.Module):
    """EEG branch + environment branch, concatenated, as in the original design, trained with
    cross-entropy plus lambda * ||z_eeg - z_env||^2 (lambda = 0.1)."""

    def __init__(self, n_in, n_env, n_cls=5, use_env=True, drop=0.3):
        super().__init__()
        self.use_env = use_env
        self.eeg = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(128, 64), nn.ReLU())
        if use_env:
            self.env = nn.Sequential(nn.Linear(n_env, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(128 if use_env else 64, n_cls))

    def forward(self, x, v):
        ze = self.eeg(x)
        if not self.use_env:
            return self.head(ze), torch.zeros((), device=x.device)
        zv = self.env(v)
        return self.head(torch.cat([ze, zv], 1)), ((ze - zv) ** 2).mean()


def macro_f1(y, p, k=5):
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp, fn = np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def fit_predict(X, y, tr, va, te, kind, E, q, h, seed, lam=0.1, epochs=60, patience=8):
    torch.manual_seed(seed)
    use_env = kind != "eeg_only"
    s = EnvSampler(E, kind if use_env else "noise", q=q, h=h, seed=seed)
    V_tr = s.train(y[tr]) if use_env else np.zeros((len(tr), E.shape[1]))
    V_va = s.test(y[va]) if use_env else np.zeros((len(va), E.shape[1]))
    V_te = s.test(y[te]) if use_env else np.zeros((len(te), E.shape[1]))
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    Xtr, Vtr, Ytr = t(X[tr]), t(V_tr), torch.as_tensor(y[tr])
    model = Fusion(X.shape[1], E.shape[1], use_env=use_env)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best, state, bad = -1, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        for b in torch.randperm(len(tr), generator=g).split(256):
            logits, align = model(Xtr[b], Vtr[b])
            loss = ce(logits, Ytr[b]) + lam * align
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            f = macro_f1(y[va], model(t(X[va]), t(V_va))[0].argmax(1).numpy())
        if f > best + 1e-4:
            best, bad = f, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return model(t(X[te]), t(V_te))[0].argmax(1).numpy()


def subject_folds(subjects, n_folds=5, n_rep=3, seed=0):
    subs = np.unique(subjects)
    out = []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed + rep)
        perm = rng.permutation(subs)
        for f in range(n_folds):
            test = perm[f::n_folds]
            rest = np.setdiff1d(perm, test)
            val = rng.choice(rest, size=4, replace=False)
            out.append((rep, f, np.setdiff1d(rest, val), val, test))
    return out


def per_subject_scores(pred_df: pd.DataFrame) -> pd.DataFrame:
    g = pred_df.groupby(["condition", "rep", "subject"])
    s = g.apply(lambda d: pd.Series({"acc": (d.y == d.pred).mean(),
                                     "f1": macro_f1(d.y.to_numpy(), d.pred.to_numpy())}),
                include_groups=False).reset_index()
    return s.groupby(["condition", "subject"])[["acc", "f1"]].mean().reset_index()
