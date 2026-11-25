"""Urban Appraisal: person-level appraisal for unseen people x unseen scenes from scene representations
(segmentation composition, low-level statistics, vision-model embeddings, and the mean of their
predictions). Same crossed 5 x 4 x 3 design as Study 2; 55 scenes with images.

For each cell a scene-level ridge model (PCA to at most 50 components fitted on the training scenes)
is trained on the training people's mean z-scored rating of each training scene; its prediction for a
test scene is the predicted rating of every test person. Per-person Pearson r over the 55 scenes,
assembled across scene folds, averaged over repetitions.

    ~/.venvs/urbeeg/bin/python -I scripts/person_level_embeddings.py
"""
import sys
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from urbeeg.config import DERIVED, RESULTS, SEED  # noqa: E402
from urbeeg.dataset import processed_participants, ratings_wide  # noqa: E402
from urbeeg.splits import crossed_cells  # noqa: E402

SCALES = ["valence", "arousal", "stress", "dominance", "openness", "safety", "beauty", "hominess", "fascination"]


def main():
    emb = np.load(RESULTS / "vision" / "embeddings.npz")
    names = [str(n) for n in emb["names"]]
    sf = pd.read_csv(DERIVED / "kaggle_bundle" / "scene_features.csv", index_col=0).reindex(names)
    comp = ["greenery", "buildings", "sky", "ground", "vehicles", "people", "other"]
    low = ["Brightness", "sdBrightness", "Saturation", "sdSaturation", "HSVHue", "StraightEdgeDensity",
           "NonStraightEdgeDensity", "Entropy", "FractalD", "depth_metric_mean", "depth_metric_sd"]
    sets = {"composition": sf[comp].to_numpy(float), "lowlevel": sf[low].to_numpy(float),
            "composition_lowlevel": sf[comp + low].to_numpy(float)}
    sets.update({k: emb[k] for k in emb.files if k != "names"})
    subs = processed_participants()
    R = ratings_wide(subs)                                  # (participant, stimulus) x scale
    Z = R.groupby(level="participant").transform(lambda c: (c - c.mean()) / c.std())
    Y = np.stack([Z.xs(s, level="participant").reindex(names)[SCALES].to_numpy(float) for s in subs])  # (P, S, 9)
    strata = sf[["code_HT", "code_HB"]].astype(str).agg("".join, axis=1).to_dict()
    cells = crossed_cells(subs, names, strata, n_p=5, n_s=4, n_rep=3, seed=SEED)
    pi = {s: i for i, s in enumerate(subs)}
    si = {s: i for i, s in enumerate(names)}
    preds = {k: np.full((3, len(subs), len(names), len(SCALES)), np.nan) for k in sets}
    for c in cells:
        tr_p, te_p = [pi[p] for p in c.train_participants], [pi[p] for p in c.test_participants]
        tr_s, te_s = [si[s] for s in c.train_scenes], [si[s] for s in c.test_scenes]
        target = np.nanmean(Y[np.ix_(tr_p, tr_s)], axis=0)          # training people's mean, (n_tr_s, 9)
        for k, F in sets.items():
            sc = StandardScaler().fit(F[tr_s])
            Ftr, Fte = sc.transform(F[tr_s]), sc.transform(F[te_s])
            if F.shape[1] > 50:
                pca = PCA(n_components=min(50, len(tr_s) - 1), random_state=SEED).fit(Ftr)
                Ftr, Fte = pca.transform(Ftr), pca.transform(Fte)
            m = RidgeCV(alphas=np.logspace(-2, 6, 33), alpha_per_target=True).fit(Ftr, target)
            out = m.predict(Fte)                                     # (n_te_s, 9)
            for p in te_p:
                preds[k][c.rep, p][te_s] = out
    preds["ensemble_all"] = np.nanmean(np.stack([preds[k] for k in sets]), axis=0)
    # consensus ceiling: each person's ratings predicted by the mean of all other people's ratings
    ceil = np.stack([np.nanmean(np.delete(Y, p, 0), 0) for p in range(len(subs))])
    preds["consensus_ceiling"] = np.stack([ceil] * 3)
    rows, rng = [], np.random.default_rng(SEED)
    for k, P in preds.items():
        per = np.full((len(subs), len(SCALES)), np.nan)
        for p in range(len(subs)):
            for j in range(len(SCALES)):
                rs = []
                for rep in range(3):
                    y, yh = Y[p, :, j], P[rep, p, :, j]
                    ok = ~np.isnan(y) & ~np.isnan(yh)
                    rs.append(np.corrcoef(yh[ok], y[ok])[0, 1])
                per[p, j] = np.mean(rs)
        for j, scale in enumerate(SCALES + ["mean_9"]):
            v = per[:, j] if scale != "mean_9" else per.mean(1)
            b = rng.choice(v, size=(2000, len(v))).mean(1)
            rows.append({"features": k, "scale": scale, "r": v.mean(), "lo": np.percentile(b, 2.5),
                         "hi": np.percentile(b, 97.5), "p": wilcoxon(v).pvalue})
    out = RESULTS / f"person_embeddings_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    show = df[df.scale.isin(["valence", "arousal", "stress", "mean_9"])].pivot(index="features", columns="scale", values="r")
    print(show.round(3).sort_values("mean_9", ascending=False).to_string())
    print("saved", out)


if __name__ == "__main__":
    main()
