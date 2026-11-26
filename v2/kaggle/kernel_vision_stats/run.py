"""Scene-level bootstrap for the group-level image benchmark (round 6 amendment, declared 26 Nov 2026, 22:15).

Out-of-fold ridge predictions are recomputed exactly as in the first image round and in round v5 (group-mean
ratings of 55 scenes, 11-fold CV x 20 partitions, ridge with the penalty chosen by leave-one-out on the training
scenes); the recomputed mean r is checked against the reported values. Then 2,000 bootstrap resamples of the
55 scenes, predictions held fixed: interval of mean r (over partitions and the nine scales) per set, and paired
differences on the same resamples."""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SCALES = ["valence", "arousal", "stress", "dominance", "openness", "safety", "beauty", "hominess", "fascination"]
COMP = ["greenery", "buildings", "sky", "ground", "vehicles", "people", "other"]
LOW = ["Brightness", "sdBrightness", "Saturation", "sdSaturation", "HSVHue", "StraightEdgeDensity",
       "NonStraightEdgeDensity", "Entropy", "FractalD", "depth_metric_mean", "depth_metric_sd"]
OUT = "/kaggle/working"


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise SystemExit(f"{name} not found")
    return hits[0]


src = find("ratings_person.csv").rsplit("/", 1)[0]
e1, e5 = np.load(find("embeddings.npz")), np.load(find("embeddings_v5.npz"))
e3, e4 = np.load(find("embeddings_v3.npz")), np.load(find("embeddings_v4.npz"))
m3, m4 = json.load(open(find("meta_v3.json"))), json.load(open(find("meta_v4.json")))
names = list(e1["names"])
for e in (e3, e4, e5):
    assert names == list(e["names"]), "image order differs between rounds"
R = pd.read_csv(f"{src}/ratings_person.csv")
Y = R.groupby("stimulus")[SCALES].mean().reindex(names).to_numpy(float)
SF = pd.read_csv(f"{src}/scene_features.csv", index_col=0).reindex(names)
seg_low = SF[COMP + LOW].to_numpy(float)
round1 = {k: e1[k] for k in ("siglip_so400m", "siglip2_so400m", "clip_vitl14", "dinov2_large")}
round1["segmentation_lowlevel"] = seg_low
v5 = {k: e5[k] for k in e5.files if k.endswith("_embedpatch")}
v5["segmentation_lowlevel"] = seg_low
extra = {"composition_only": SF[COMP].to_numpy(float), "lowlevel_only": SF[LOW].to_numpy(float)}
v3 = {k: (seg_low if k == "segmentation_lowlevel" else e3[k]) for k in m3["primary_sets"]}
v4 = {k: (seg_low if k == "segmentation_lowlevel" else e4[k]) for k in m4["primary_sets"]}


def oof(F, rep):
    P = np.zeros_like(Y)
    for tr, te in KFold(11, shuffle=True, random_state=rep).split(F):
        sc = StandardScaler().fit(F[tr])
        m = RidgeCV(alphas=np.logspace(-2, 6, 33), alpha_per_target=True).fit(sc.transform(F[tr]), Y[tr])
        P[te] = m.predict(sc.transform(F[te]))
    return P


P = {}
for tag, group in (("", round1), ("", v5), ("", extra), ("v3:", v3), ("v4:", v4)):
    for k, F in group.items():
        key = k if k == "segmentation_lowlevel" else tag + k             # rounds may reuse a set name
        if key not in P:
            P[key] = np.stack([oof(F, rep) for rep in range(20)])     # (20, 55, 9)
P["ensemble_round1"] = np.mean([P[k] for k in round1], axis=0)
P["ensemble_v5"] = np.mean([P[k] for k in v5], axis=0)
P["ensemble_v3"] = np.mean([P[k if k == "segmentation_lowlevel" else "v3:" + k] for k in v3], axis=0)
P["ensemble_v4"] = np.mean([P[k if k == "segmentation_lowlevel" else "v4:" + k] for k in v4], axis=0)


def mean_r(Pk, idx):
    Yb = Y[idx]
    out = []
    for rep in range(Pk.shape[0]):
        Pb = Pk[rep][idx]
        out.append(np.mean([np.corrcoef(Pb[:, j], Yb[:, j])[0, 1] for j in range(Y.shape[1])]))
    return float(np.mean(out))


full = np.arange(len(names))
point = {k: mean_r(v, full) for k, v in P.items()}
reported = {"siglip_so400m": 0.8717, "clip_vitl14": 0.8376, "dinov2_large": 0.8140, "ensemble_round1": 0.893,
            "siglip_so400m_embedpatch": 0.8964, "clip_vitl14_embedpatch": 0.8969, "dinov2_large_embedpatch": 0.8465,
            "ensemble_v5": 0.9109, "segmentation_lowlevel": 0.6435, "ensemble_v3": 0.9056, "ensemble_v4": 0.9047}
for k, v in reported.items():
    print(f"check {k}: recomputed {point[k]:.4f}, reported {v:.4f}", flush=True)
SETS = ["siglip_so400m", "clip_vitl14", "dinov2_large", "siglip_so400m_embedpatch", "clip_vitl14_embedpatch",
        "dinov2_large_embedpatch", "ensemble_round1", "ensemble_v3", "ensemble_v4", "ensemble_v5",
        "segmentation_lowlevel", "composition_only",
        "lowlevel_only"]
PAIRS = [("siglip_so400m_embedpatch", "siglip_so400m"), ("clip_vitl14_embedpatch", "clip_vitl14"),
         ("dinov2_large_embedpatch", "dinov2_large"), ("ensemble_round1", "siglip_so400m"),
         ("ensemble_v3", "siglip_so400m"), ("ensemble_v4", "siglip_so400m"),
         ("ensemble_v5", "siglip_so400m"), ("segmentation_lowlevel", "composition_only"),
         ("segmentation_lowlevel", "lowlevel_only")]
rng = np.random.default_rng(0)
B = 2000
boot = {k: np.empty(B) for k in SETS}
for b in range(B):
    idx = rng.integers(0, len(names), len(names))
    for k in SETS:
        boot[k][b] = mean_r(P[k], idx)
rows = [{"set": k, "mean_r": point[k], "lo": np.percentile(boot[k], 2.5), "hi": np.percentile(boot[k], 97.5)}
        for k in SETS]
for a, c in PAIRS:
    d = boot[a] - boot[c]
    rows.append({"set": f"{a} - {c}", "mean_r": point[a] - point[c], "lo": np.percentile(d, 2.5),
                 "hi": np.percentile(d, 97.5), "share_above_0": float((d > 0).mean())})
out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/scene_bootstrap.csv", index=False)
json.dump({"n_scenes": len(names), "n_boot": B, "point": point}, open(f"{OUT}/scene_bootstrap_meta.json", "w"), indent=1)
print(out.round(4).to_string(index=False), flush=True)
