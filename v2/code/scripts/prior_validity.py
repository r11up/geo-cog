"""Round 6 (declared 26 Nov 2026, 21:57), Urban Appraisal, nothing trained.

B. Prior validity. The Study 1 prior assumes that greener places go with pleasant affect (happiness, calm)
   and built, traffic-dense places with unpleasant affect (anger, sadness). Where the environment of each
   epoch is measured, this predicts: greenery share ~ valence (+) and stress (-); built exposure score
   (mean of the z-scored shares of buildings, road and pavement, and vehicles over the 56 scenes, the
   street-level analogue of the prior's building / road / NDBI score) ~ valence (-) and stress (+).
   Arousal: no prediction (happiness and anger are high-arousal, calm and sadness low); descriptive.
   Primary: per-person Spearman correlation over the scenes, two-sided Wilcoxon against zero, Holm over
   the four directional tests. Secondary: the prior's own quartile rule applied to the 56 scenes.
C. Scene content versus low-level image information: (i) per-person out-of-sample variance partition
   (composition only, low-level only, both; saved predictions of the main linear run, raw ratings);
   (ii) greenery residualised on the eleven low-level statistics, per-person Spearman with the ratings;
   (iii) AUC of the greenery share for the HT vs LT design code.

    python -I scripts/prior_validity.py [--shares results/vision6/segmentation_shares_v6.csv]
"""
import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from urbeeg.config import RESULTS  # noqa: E402
from urbeeg.dataset import processed_participants, ratings_wide  # noqa: E402
from urbeeg.scene import LOW_LEVEL, low_level_features, scene_features  # noqa: E402

TARGETS = ["valence", "arousal", "stress"]
PREDICTED = {("greenery", "valence"): "+", ("greenery", "stress"): "-",
             ("built", "valence"): "-", ("built", "stress"): "+"}
N_BOOT, SEED = 2000, 0


def built_score(shares):
    z = shares[["buildings", "ground", "vehicles"]]
    return ((z - z.mean()) / z.std(ddof=0)).mean(axis=1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def summarise_rho(rho, rng):
    """Mean of per-person correlations via Fisher z, bootstrap interval over people, Wilcoxon vs 0."""
    z = np.arctanh(np.clip(rho, -0.999, 0.999))
    boot = np.tanh(rng.choice(z, size=(N_BOOT, len(z))).mean(1))
    return {"rho": float(np.tanh(z.mean())), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "p": float(wilcoxon(rho).pvalue),
            "n_people": int(len(rho)), "n_positive": int((rho > 0).sum())}


def prior_validity(shares, ratings, tag, rng):
    """Primary directional tests and the quartile rule, for one set of composition shares."""
    env = pd.DataFrame({"greenery": shares["greenery"], "built": built_score(shares)})
    rows = []
    for (pred, tgt) in [(p, t) for p in ("greenery", "built") for t in TARGETS]:
        rho = []
        for _, g in ratings.groupby(level="participant"):
            g = g.droplevel("participant")[tgt].dropna()
            rho.append(spearmanr(env.loc[g.index, pred], g).statistic)
        rows.append({"segmentation": tag, "predictor": pred, "target": tgt,
                     "predicted_sign": PREDICTED.get((pred, tgt), "none"), **summarise_rho(np.array(rho), rng)})
    out = pd.DataFrame(rows)
    prim = out.predicted_sign != "none"
    out.loc[prim, "p_holm"] = holm(out.loc[prim, "p"])
    out["direction_as_predicted"] = [(s == "+" and r > 0) or (s == "-" and r < 0) if s != "none" else np.nan
                                     for s, r in zip(out.predicted_sign, out.rho)]
    # quartile rule of the prior, applied to the scenes
    g, b = env["greenery"], env["built"]
    subsets = {"green": g >= g.quantile(0.75), "built": b >= b.quantile(0.75),
               "middle": (g > g.quantile(0.25)) & (g < g.quantile(0.75)) & (b > b.quantile(0.25)) & (b < b.quantile(0.75))}
    zr = ratings.groupby(level="participant").transform(lambda s: (s - s.mean()) / s.std())
    qrows = []
    for tgt in ("valence", "stress", "arousal"):
        per = {}
        for name, mask in subsets.items():
            scenes = mask[mask].index
            per[name] = zr[zr.index.get_level_values("stimulus").isin(scenes)].groupby(level="participant")[tgt].mean()
        for name in subsets:
            v = per[name].to_numpy()
            boot = rng.choice(v, size=(N_BOOT, len(v))).mean(1)
            qrows.append({"segmentation": tag, "target": tgt, "subset": name, "n_scenes": int(subsets[name].sum()),
                          "mean_z": float(v.mean()), "lo": float(np.percentile(boot, 2.5)),
                          "hi": float(np.percentile(boot, 97.5))})
        d = (per["green"] - per["built"]).dropna().to_numpy()
        qrows.append({"segmentation": tag, "target": tgt, "subset": "green_minus_built", "n_scenes": np.nan,
                      "mean_z": float(d.mean()), "lo": float(np.percentile(rng.choice(d, size=(N_BOOT, len(d))).mean(1), 2.5)),
                      "hi": float(np.percentile(rng.choice(d, size=(N_BOOT, len(d))).mean(1), 97.5)),
                      "p": float(wilcoxon(d).pvalue)})
    overlap = int((subsets["green"] & subsets["built"]).sum())
    return out, pd.DataFrame(qrows), overlap


def variance_partition(rng):
    lin = sorted(glob.glob(str(RESULTS / "linear_*")))[-1]
    oof = pd.read_csv(Path(lin) / "oof.csv.gz")
    oof = oof[(oof["mode"] == "raw") & oof.model.isin(["scene", "scene_lowlevel_only", "scene+lowlevel"])].dropna(subset=["y_true"])

    def r(g):
        a, b = g.y_pred.to_numpy(), g.y_true.to_numpy()
        return 0.0 if a.std() < 1e-12 or b.std() < 1e-12 else float(np.corrcoef(a, b)[0, 1])
    rr = oof.groupby(["model", "target", "rep", "participant"]).apply(r, include_groups=False).rename("r").reset_index()
    rr["r2"] = rr.r ** 2
    rr["r2_clipped"] = np.where(rr.r > 0, rr.r ** 2, 0.0)
    rows = []
    for col in ("r2", "r2_clipped"):
        pp = rr.groupby(["model", "target", "participant"])[col].mean().unstack("model")
        for tgt, g in pp.groupby(level="target"):
            comp, low, both = g["scene"], g["scene_lowlevel_only"], g["scene+lowlevel"]
            parts = {"r2_composition": comp, "r2_lowlevel": low, "r2_both": both,
                     "unique_composition": both - low, "unique_lowlevel": both - comp, "shared": comp + low - both}
            for k, v in parts.items():
                v = v.to_numpy()
                boot = rng.choice(v, size=(N_BOOT, len(v))).mean(1)
                rows.append({"measure": col, "target": tgt, "part": k, "mean": float(v.mean()),
                             "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))})
    n_neg = rr.groupby(["model", "target"]).r.apply(lambda s: int((s < 0).sum())).rename("n_negative_r").reset_index()
    return pd.DataFrame(rows), n_neg, Path(lin).name


def residual_greenery(shares, ratings, rng):
    low = low_level_features().reindex(shares.index)
    X = np.column_stack([np.ones(len(low)), low[LOW_LEVEL].to_numpy(float)])
    yv = shares["greenery"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    fitted = X @ beta
    r2 = 1 - ((yv - fitted) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    resid = pd.Series(yv - fitted, index=shares.index)
    rows = []
    for tgt in TARGETS:
        rho = []
        for _, g in ratings.groupby(level="participant"):
            g = g.droplevel("participant")[tgt].dropna()
            rho.append(spearmanr(resid.loc[g.index], g).statistic)
        rows.append({"target": tgt, **summarise_rho(np.array(rho), rng)})
    return pd.DataFrame(rows), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shares", default=None, help="segmentation_shares_v6.csv from the image round v6")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    subs = processed_participants()
    ratings = ratings_wide(subs)[TARGETS]
    sf = scene_features()
    assert len(subs) == 62 and len(sf) == 56
    res = RESULTS / f"priorvalidity_{datetime.now():%Y%m%d_%H%M}"
    res.mkdir(parents=True)
    tests, quart, overlap = prior_validity(sf, ratings, "released", rng)
    extra = []
    if args.shares:                                  # the same tests with other segmentation models
        sh = pd.read_csv(args.shares, header=[0, 1], index_col=0)
        for tag in sh.columns.get_level_values(0).unique():
            if tag == "released":
                continue
            s = sh[tag]
            r_sub = ratings[ratings.index.get_level_values("stimulus").isin(s.index)]
            t, q, _ = prior_validity(s, r_sub, tag, rng)
            extra.append(t)
        if extra:
            pd.concat(extra).to_csv(res / "prior_validity_other_segmentations.csv", index=False)
    tests.to_csv(res / "prior_validity_tests.csv", index=False)
    quart.to_csv(res / "prior_validity_quartile_rule.csv", index=False)
    vp, n_neg, lin = variance_partition(rng)
    vp.to_csv(res / "variance_partition.csv", index=False)
    n_neg.to_csv(res / "variance_partition_negative_r.csv", index=False)
    rg, r2 = residual_greenery(sf, ratings, rng)
    rg.to_csv(res / "residual_greenery.csv", index=False)
    auc = roc_auc_score(sf["code_HT"], sf["greenery"])
    pd.Series({"auc_greenery_HT_vs_LT": auc, "r2_greenery_on_lowlevel": r2, "linear_run": lin,
               "green_built_overlap_scenes": overlap, "n_people": len(subs), "n_scenes": len(sf)}).to_csv(res / "meta.csv")
    pd.set_option("display.width", 200)
    print(tests.round(3).to_string(index=False))
    print(quart.round(3).to_string(index=False))
    print(vp[vp.measure == "r2"].round(4).to_string(index=False))
    print(n_neg.to_string(index=False))
    print(rg.round(3).to_string(index=False), "\nR2 greenery on low-level:", round(r2, 3), "AUC HT vs LT:", round(auc, 3))
    if extra:
        print(pd.concat(extra).round(3).to_string(index=False))
    print("saved", res)


if __name__ == "__main__":
    main()
