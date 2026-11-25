"""Metrics on out-of-fold predictions.

Primary: per-person Pearson r between predicted and observed ratings over the scenes that
person was tested on (all 56 across the four scene folds of a repetition), averaged over
people with Fisher's z. It ignores each person's overall rating level and spread, which no
model can know for an unseen person. Uncertainty: bootstrap over participants.
Secondary: consensus r, correlating the mean observed and mean predicted rating per scene.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["model", "target", "mode"]


def _r(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def per_person_r(oof: pd.DataFrame) -> pd.DataFrame:
    """One r per (model, target, mode, participant), averaged over repetitions (Fisher z).
    Pairs with a missing rating are skipped."""
    oof = oof.dropna(subset=["y_true"])
    r = (oof.groupby(KEYS + ["rep", "participant"])
             .apply(lambda g: _r(g.y_pred.to_numpy(), g.y_true.to_numpy()), include_groups=False)
             .rename("r").reset_index())
    r["z"] = np.arctanh(r["r"].clip(-0.999, 0.999))
    out = r.groupby(KEYS + ["participant"])["z"].mean().rename("z").reset_index()
    out["r"] = np.tanh(out["z"])
    return out


def consensus_r(oof: pd.DataFrame) -> pd.DataFrame:
    m = oof.groupby(KEYS + ["rep", "stimulus"])[["y_true", "y_pred"]].mean().reset_index()
    r = m.groupby(KEYS + ["rep"]).apply(lambda g: _r(g.y_pred.to_numpy(), g.y_true.to_numpy()),
                                        include_groups=False).rename("r").reset_index()
    return r.groupby(KEYS)["r"].mean().rename("consensus_r").reset_index()


def summarize(oof: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    pp = per_person_r(oof)
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in pp.groupby(KEYS):
        z = g["z"].to_numpy()
        boot = np.tanh(rng.choice(z, size=(n_boot, len(z)), replace=True).mean(axis=1))
        rows.append(dict(zip(KEYS, key), n_people=len(z), r=np.tanh(z.mean()),
                         ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5)))
    return pd.DataFrame(rows).merge(consensus_r(oof), on=KEYS, how="left")


def compare(oof: pd.DataFrame, model_a: str, model_b: str, n_boot: int = 2000, seed: int = 0):
    """Paired comparison of per-person r (a minus b), per target and mode."""
    pp = per_person_r(oof)
    rng = np.random.default_rng(seed)
    out = []
    for (t, mode), g in pp.groupby(["target", "mode"]):
        w = g.pivot(index="participant", columns="model", values="z")
        if model_a not in w or model_b not in w:
            continue
        d = (w[model_a] - w[model_b]).dropna().to_numpy()
        boot = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        p = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
        out.append(dict(target=t, mode=mode, a=model_a, b=model_b, n_people=len(d),
                        mean_dz=d.mean(), ci_low=np.percentile(boot, 2.5),
                        ci_high=np.percentile(boot, 97.5), wilcoxon_p=p))
    return pd.DataFrame(out)
