"""Vision embeddings of the Urban Appraisal scenes and the group-level rating benchmark of
Tan & Deng (2026): group-mean ratings of 55 scenes, 11-fold cross-validation over scenes repeated with
20 random partitions, ridge with the penalty chosen by leave-one-out on the training scenes, Pearson r
per scale averaged over the nine scales. Also the split-half noise ceiling of the group means."""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

T0 = time.time()
SCALES = ["valence", "arousal", "stress", "dominance", "openness", "safety", "beauty", "hominess", "fascination"]
MODELS = {"siglip_so400m": "google/siglip-so400m-patch14-384",
          "siglip2_so400m": "google/siglip2-so400m-patch14-384",
          "clip_vitl14": "openai/clip-vit-large-patch14",
          "dinov2_large": "facebook/dinov2-large"}


def find(marker):
    for depth in range(1, 6):
        hits = glob.glob("/kaggle/input/" + "*/" * depth + marker)
        if hits:
            return hits[0][:-len(marker)].rstrip("/")
    raise SystemExit(f"{marker} not found")


SRC, OUT = find("ratings_person.csv"), "/kaggle/working"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
paths = sorted(glob.glob(f"{SRC}/images/*.jpg"))
names = [os.path.basename(p)[:-4] for p in paths]
images = [Image.open(p).convert("RGB") for p in paths]
print(len(names), "images; device", DEV, flush=True)


def embed(key, repo):
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    model = AutoModel.from_pretrained(repo, dtype=dtype).to(DEV).eval()
    try:
        proc = AutoProcessor.from_pretrained(repo)
    except Exception:
        proc = AutoImageProcessor.from_pretrained(repo)
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), 8):
            inp = proc(images=images[i:i + 8], return_tensors="pt")
            inp = {k: (v.to(DEV, dtype) if v.is_floating_point() else v.to(DEV)) for k, v in inp.items()
                   if k in ("pixel_values", "pixel_attention_mask", "spatial_shapes")}
            if hasattr(model, "get_image_features"):
                f = model.get_image_features(**inp)
                if not torch.is_tensor(f):                   # newer transformers return an output object
                    emb = getattr(f, "image_embeds", None)
                    f = emb if emb is not None else f.pooler_output
            else:                                            # DINOv2: CLS token after the final norm
                f = model(**inp).pooler_output
            feats.append(f.float().cpu().numpy())
    del model
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return np.concatenate(feats)


emb = {}
for key, repo in MODELS.items():
    t = time.time()
    try:
        emb[key] = embed(key, repo)
        print(f"{key}: {emb[key].shape} in {time.time() - t:.0f} s", flush=True)
    except Exception as e:                                   # keep going; report what failed
        print(f"{key}: FAILED {type(e).__name__}: {e}", flush=True)
np.savez_compressed(f"{OUT}/embeddings.npz", names=np.array(names), **emb)

# ------------------------------------------------------------------ targets and our scene features
R = pd.read_csv(f"{SRC}/ratings_person.csv")
G = R.groupby("stimulus")[SCALES].mean().reindex(names)
SF = pd.read_csv(f"{SRC}/scene_features.csv", index_col=0).reindex(names)
comp = ["greenery", "buildings", "sky", "ground", "vehicles", "people", "other"]
low = ["Brightness", "sdBrightness", "Saturation", "sdSaturation", "HSVHue", "StraightEdgeDensity",
       "NonStraightEdgeDensity", "Entropy", "FractalD", "depth_metric_mean", "depth_metric_sd"]
sets = dict(emb)
sets["segmentation_lowlevel"] = SF[comp + low].to_numpy(float)
Y = G.to_numpy(float)


def oof_predictions(F, rep):
    P = np.zeros_like(Y)
    for tr, te in KFold(11, shuffle=True, random_state=rep).split(F):
        sc = StandardScaler().fit(F[tr])
        m = RidgeCV(alphas=np.logspace(-2, 6, 33), alpha_per_target=True).fit(sc.transform(F[tr]), Y[tr])
        P[te] = m.predict(sc.transform(F[te]))
    return P


def r_per_scale(P):
    return np.array([np.corrcoef(P[:, k], Y[:, k])[0, 1] for k in range(Y.shape[1])])


rows, preds = [], {}
for key, F in sets.items():
    per_rep = []
    for rep in range(20):
        P = oof_predictions(F, rep)
        preds.setdefault(rep, {})[key] = P
        per_rep.append(r_per_scale(P))
    per_rep = np.array(per_rep)
    rows.append({"features": key, "mean_r": per_rep.mean(), "sd_over_partitions": per_rep.mean(1).std(),
                 **{f"r_{s}": per_rep[:, k].mean() for k, s in enumerate(SCALES)}})
    print(f"{key}: mean r = {per_rep.mean():.3f}", flush=True)
ens = np.array([r_per_scale(np.mean([preds[rep][k] for k in sets], axis=0)) for rep in range(20)])
rows.append({"features": "ensemble_all", "mean_r": ens.mean(), "sd_over_partitions": ens.mean(1).std(),
             **{f"r_{s}": ens[:, k].mean() for k, s in enumerate(SCALES)}})
print(f"ensemble_all: mean r = {ens.mean():.3f}", flush=True)

# ------------------------------------------------------------------ split-half noise ceiling
rng = np.random.default_rng(0)
people = R.participant.unique()
ceil = []
for _ in range(100):
    half = rng.permutation(people)[:len(people) // 2]
    a = R[R.participant.isin(half)].groupby("stimulus")[SCALES].mean().reindex(names).to_numpy(float)
    b = R[~R.participant.isin(half)].groupby("stimulus")[SCALES].mean().reindex(names).to_numpy(float)
    r = np.array([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(len(SCALES))])
    ceil.append(2 * r / (1 + r))
ceil = np.array(ceil)
rows.append({"features": "noise_ceiling_split_half", "mean_r": ceil.mean(), "sd_over_partitions": ceil.mean(1).std(),
             **{f"r_{s}": ceil[:, k].mean() for k, s in enumerate(SCALES)}})
pd.DataFrame(rows).to_csv(f"{OUT}/group_benchmark.csv", index=False)
json.dump({"n_scenes": len(names), "n_people": int(len(people)), "seconds": round(time.time() - T0)},
          open(f"{OUT}/meta.json", "w"))
print(pd.DataFrame(rows)[["features", "mean_r", "sd_over_partitions"]].to_string(index=False), flush=True)
