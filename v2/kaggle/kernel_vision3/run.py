"""Urban Appraisal group-level benchmark of Tan & Deng (2026), new methods (declared 26 Nov 2026, 09:13).
Protocol unchanged: group-mean ratings of 55 scenes, 11-fold cross-validation over scenes repeated with
20 random partitions, ridge with the penalty chosen by leave-one-out on the training scenes, Pearson r per
scale averaged over the nine scales.
New: larger encoders (SigLIP 2 giant, DINOv2 giant, CLIP ViT-bigG-14, AIMv2 large); horizontal-flip
averaging of every embedding; language-anchored features (image-text similarity to 34 fixed prompts) from
each vision-language model. Primary: equal-weight ensemble of the ridge predictions of every feature set.
Secondary: stacked ensemble with non-negative weights fitted on inner out-of-fold predictions of the
training scenes only."""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

T0 = time.time()
SCALES = ["valence", "arousal", "stress", "dominance", "openness", "safety", "beauty", "hominess", "fascination"]
MODELS = {"siglip_so400m": "google/siglip-so400m-patch14-384",
          "siglip2_so400m": "google/siglip2-so400m-patch14-384",
          "clip_vitl14": "openai/clip-vit-large-patch14",
          "dinov2_large": "facebook/dinov2-large",
          "siglip2_giant": "google/siglip2-giant-opt-patch16-384",
          "dinov2_giant": "facebook/dinov2-giant",
          "clip_bigg14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
          "aimv2_large": "apple/aimv2-large-patch14-448"}
OLD = ["siglip_so400m", "siglip2_so400m", "clip_vitl14", "dinov2_large"]
APPRAISAL = [("a pleasant street", "an unpleasant street"),
             ("an exciting, lively street", "a calm, quiet street"),
             ("a stressful street", "a relaxing street"),
             ("a street where one feels in control", "a street where one feels overwhelmed"),
             ("an open, spacious street", "a narrow, enclosed street"),
             ("a safe street", "a dangerous street"),
             ("a beautiful street", "an ugly street"),
             ("a homely, welcoming street", "an unwelcoming, alienating street"),
             ("a fascinating street", "a boring street")]
ATTRIBUTES = ["trees", "green vegetation", "a park", "parked cars", "heavy traffic", "pedestrians", "cyclists",
              "historic buildings", "modern buildings", "high-rise buildings", "graffiti", "litter", "a wide road",
              "a construction site", "shops and cafes", "open sky"]
PROMPTS = [f"a photo of {a}" for pair in APPRAISAL for a in pair] + [f"a photo of a street with {a}" for a in ATTRIBUTES]


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
flipped = [ImageOps.mirror(im) for im in images]
print(len(names), "images; device", DEV, "; prompts", len(PROMPTS), flush=True)


def image_features(model, proc, imgs):
    feats = []
    for i in range(0, len(imgs), 4):
        inp = proc(images=imgs[i:i + 4], return_tensors="pt")
        inp = {k: (v.to(DEV, model.dtype) if v.is_floating_point() else v.to(DEV)) for k, v in inp.items()
               if k in ("pixel_values", "pixel_attention_mask", "spatial_shapes")}
        if hasattr(model, "get_image_features"):
            f = model.get_image_features(**inp)
            if not torch.is_tensor(f):                       # newer transformers return an output object
                emb = getattr(f, "image_embeds", None)
                f = emb if emb is not None else f.pooler_output
        else:
            o = model(**inp)
            f = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def text_features(model, proc, prompts):
    kw = {"padding": "max_length", "max_length": 64} if "siglip" in type(model).__name__.lower() else {"padding": True}
    inp = proc(text=prompts, return_tensors="pt", truncation=True, **kw)
    inp = {k: v.to(DEV) for k, v in inp.items() if k in ("input_ids", "attention_mask")}
    f = model.get_text_features(**inp)
    if not torch.is_tensor(f):
        emb = getattr(f, "text_embeds", None)
        f = emb if emb is not None else f.pooler_output
    return f.float().cpu().numpy()


def run_model(key, repo):
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    model = AutoModel.from_pretrained(repo, dtype=dtype, trust_remote_code=key.startswith("aimv2")).to(DEV).eval()
    try:
        proc = AutoProcessor.from_pretrained(repo)
    except Exception:
        proc = AutoImageProcessor.from_pretrained(repo)
    out = {}
    with torch.no_grad():
        a, b = image_features(model, proc, images), image_features(model, proc, flipped)
        out[key] = a
        out[key + "_flip"] = (a + b) / 2
        if hasattr(model, "get_text_features") and hasattr(proc, "tokenizer"):
            t = text_features(model, proc, PROMPTS)
            unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
            out[key + "_text"] = (unit(a) + unit(b)) / 2 @ unit(t).T
    del model
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return out


emb, failed = {}, {}
for key, repo in MODELS.items():
    t = time.time()
    try:
        got = run_model(key, repo)
        emb.update(got)
        print(f"{key}: {', '.join(f'{k} {v.shape}' for k, v in got.items())} in {time.time() - t:.0f} s", flush=True)
    except Exception as e:                                   # keep going; report what failed
        failed[key] = f"{type(e).__name__}: {e}"[:300]
        print(f"{key}: FAILED {failed[key]}", flush=True)
np.savez_compressed(f"{OUT}/embeddings_v3.npz", names=np.array(names), **emb)

# ------------------------------------------------------------------ targets and scene features
R = pd.read_csv(f"{SRC}/ratings_person.csv")
G = R.groupby("stimulus")[SCALES].mean().reindex(names)
SF = pd.read_csv(f"{SRC}/scene_features.csv", index_col=0).reindex(names)
comp = ["greenery", "buildings", "sky", "ground", "vehicles", "people", "other"]
low = ["Brightness", "sdBrightness", "Saturation", "sdSaturation", "HSVHue", "StraightEdgeDensity",
       "NonStraightEdgeDensity", "Entropy", "FractalD", "depth_metric_mean", "depth_metric_sd"]
sets = dict(emb)
sets["segmentation_lowlevel"] = SF[comp + low].to_numpy(float)
Y = G.to_numpy(float)
ALPHAS = np.logspace(-2, 6, 33)


def ridge_fit_predict(F, tr, te, Yt):
    sc = StandardScaler().fit(F[tr])
    m = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(sc.transform(F[tr]), Yt)
    return m.predict(sc.transform(F[te]))


def r_per_scale(P):
    return np.array([np.corrcoef(P[:, k], Y[:, k])[0, 1] for k in range(Y.shape[1])])


def stack_weights(keys, tr, seed):
    """Non-negative weights per scale from inner 10-fold out-of-fold predictions on the training scenes."""
    inner = {k: np.zeros((len(tr), Y.shape[1])) for k in keys}
    for itr, ite in KFold(10, shuffle=True, random_state=seed).split(tr):
        for k in keys:
            inner[k][ite] = ridge_fit_predict(sets[k], tr[itr], tr[ite], Y[tr[itr]])
    Wt = np.zeros((len(keys), Y.shape[1]))
    for j in range(Y.shape[1]):
        A = np.column_stack([inner[k][:, j] for k in keys])
        w, _ = nnls(A, Y[tr, j])
        Wt[:, j] = w / w.sum() if w.sum() > 0 else np.full(len(keys), 1 / len(keys))
    return Wt


primary = [k for k in sets if k.endswith("_flip") or k.endswith("_text")] + ["segmentation_lowlevel"]
old_ens = [k for k in OLD if k in sets] + ["segmentation_lowlevel"]
rows, scores = [], {k: [] for k in sets}
scores.update({"ensemble_v1_reproduced": [], "ensemble_v3": [], "stacked_v3": []})
for rep in range(20):
    P = {k: np.zeros_like(Y) for k in sets}
    Pst = np.zeros_like(Y)
    for tr, te in KFold(11, shuffle=True, random_state=rep).split(Y):
        for k in sets:
            P[k][te] = ridge_fit_predict(sets[k], tr, te, Y[tr])
        Wt = stack_weights(primary, tr, 1000 + rep)
        Pst[te] = sum(Wt[i] * P[k][te] for i, k in enumerate(primary))
    for k in sets:
        scores[k].append(r_per_scale(P[k]))
    scores["ensemble_v1_reproduced"].append(r_per_scale(np.mean([P[k] for k in old_ens], axis=0)))
    scores["ensemble_v3"].append(r_per_scale(np.mean([P[k] for k in primary], axis=0)))
    scores["stacked_v3"].append(r_per_scale(Pst))
    print(f"partition {rep}: ensemble_v3 {scores['ensemble_v3'][-1].mean():.3f}, stacked {scores['stacked_v3'][-1].mean():.3f}",
          flush=True)
for k, v in scores.items():
    v = np.array(v)
    rows.append({"features": k, "mean_r": v.mean(), "sd_over_partitions": v.mean(1).std(),
                 "role": "primary" if k == "ensemble_v3" else "secondary",
                 **{f"r_{s}": v[:, j].mean() for j, s in enumerate(SCALES)}})

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
             "role": "ceiling", **{f"r_{s}": ceil[:, k].mean() for k, s in enumerate(SCALES)}})
pd.DataFrame(rows).to_csv(f"{OUT}/group_benchmark_v3.csv", index=False)
json.dump({"n_scenes": len(names), "n_people": int(len(people)), "primary_sets": primary, "failed": failed,
           "prompts": PROMPTS, "seconds": round(time.time() - T0)}, open(f"{OUT}/meta_v3.json", "w"))
print(pd.DataFrame(rows)[["features", "mean_r", "sd_over_partitions", "role"]].to_string(index=False), flush=True)
