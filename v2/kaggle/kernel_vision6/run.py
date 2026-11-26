"""Image round v6 (declared 26 Nov 2026, 21:57): does image-based appraisal prediction rest on scene
content or on low-level image statistics, and how robust is the scene composition to the segmentation model?

(i) Image controls under the group protocol of Tan & Deng (2026): group-mean ratings of 55 scenes,
11-fold cross-validation over scenes repeated with 20 random partitions, ridge with the penalty chosen by
leave-one-out on the training scenes, Pearson r per scale averaged over the nine scales. Four encoders
(final embedding, single view, as in the first round) on the original, grayscale and phase-scrambled images
(Fourier amplitude of each colour channel kept, one random phase field per image added to all channels,
DC phase kept, fixed seed). References: low-level statistics alone, composition alone.
(ii) Segmentation: SegFormer-B5 (ADE20K) re-run, Mask2Former Swin-L trained on ADE20K and on Cityscapes;
category shares compared with the released SegFormer-B5 shares; composition-only benchmark per model."""
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
SEGMODELS = {"segformer_b5_ade": "nvidia/segformer-b5-finetuned-ade-640-640",
             "mask2former_ade": "facebook/mask2former-swin-large-ade-semantic",
             "mask2former_cityscapes": "facebook/mask2former-swin-large-cityscapes-semantic"}
ADE_GROUPS = {  # the class grouping used for the released segmentation (urbeeg/scene.py)
    "greenery": ["tree", "grass", "plant", "flower", "field", "palm"],
    "buildings": ["building", "wall", "house", "skyscraper", "tower", "column", "door", "awning", "stairs",
                  "step", "ceiling"],
    "sky": ["sky"], "ground": ["road", "sidewalk", "earth", "base", "path"],
    "vehicles": ["car", "van", "truck", "bus", "minibike", "bicycle"], "people": ["person"]}
CITY_GROUPS = {"greenery": ["vegetation", "terrain"], "buildings": ["building", "wall", "fence"], "sky": ["sky"],
               "ground": ["road", "sidewalk"],
               "vehicles": ["car", "truck", "bus", "train", "motorcycle", "bicycle"], "people": ["person", "rider"]}
COMP = ["greenery", "buildings", "sky", "ground", "vehicles", "people", "other"]
LOW = ["Brightness", "sdBrightness", "Saturation", "sdSaturation", "HSVHue", "StraightEdgeDensity",
       "NonStraightEdgeDensity", "Entropy", "FractalD", "depth_metric_mean", "depth_metric_sd"]


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
originals = [Image.open(p).convert("RGB") for p in paths]
print(len(names), "images; device", DEV, "; size of the first", originals[0].size, flush=True)


def phase_scramble(img, seed):
    a = np.asarray(img, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise_phase = np.angle(np.fft.fft2(rng.normal(size=a.shape[:2])))
    noise_phase[0, 0] = 0.0                                  # keep the mean of every channel
    out = np.empty_like(a)
    for c in range(3):
        F = np.fft.fft2(a[:, :, c])
        out[:, :, c] = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * (np.angle(F) + noise_phase))))
    clipped = float(np.mean((out < 0) | (out > 255)))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), clipped


gray = [im.convert("L").convert("RGB") for im in originals]
scr = [phase_scramble(im, 1000 + k) for k, im in enumerate(originals)]
scrambled = [s[0] for s in scr]
print(f"phase scrambling: mean share of clipped pixel values {np.mean([s[1] for s in scr]):.4f}", flush=True)
os.makedirs(f"{OUT}/examples", exist_ok=True)
for k in (0, len(names) // 2):
    for tag, ims in (("original", originals), ("gray", gray), ("scrambled", scrambled)):
        ims[k].resize((480, int(480 * ims[k].size[1] / ims[k].size[0]))).save(f"{OUT}/examples/{names[k]}_{tag}.jpg")


def image_stats(ims):
    rows = []
    for im in ims:
        hsv = np.asarray(im.convert("HSV"), dtype=float)
        rows.append({"brightness": np.asarray(im.convert("L"), dtype=float).mean(), "saturation": hsv[..., 1].mean() / 255})
    return pd.DataFrame(rows, index=names)


stats = pd.concat({t: image_stats(ims) for t, ims in (("original", originals), ("gray", gray), ("scrambled", scrambled))}, axis=1)
stats.to_csv(f"{OUT}/image_stats_v6.csv")


def embed(repo, images):
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
                if not torch.is_tensor(f):
                    emb = getattr(f, "image_embeds", None)
                    f = emb if emb is not None else f.pooler_output
            else:
                f = model(**inp).pooler_output
            feats.append(f.float().cpu().numpy())
    del model
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return np.concatenate(feats)


emb = {}
for key, repo in MODELS.items():
    for tag, ims in (("original", originals), ("gray", gray), ("scrambled", scrambled)):
        t = time.time()
        try:
            emb[f"{key}@{tag}"] = embed(repo, ims)
            print(f"{key}@{tag}: {emb[f'{key}@{tag}'].shape} in {time.time() - t:.0f} s", flush=True)
        except Exception as e:
            print(f"{key}@{tag}: FAILED {type(e).__name__}: {e}", flush=True)
np.savez_compressed(f"{OUT}/embeddings_v6.npz", names=np.array(names), **{k.replace("@", "__"): v for k, v in emb.items()})


# ------------------------------------------------------------------ segmentation
def segment(key, repo, images):
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(repo)
    if key.startswith("segformer"):
        from transformers import SegformerForSemanticSegmentation as M
    else:
        from transformers import Mask2FormerForUniversalSegmentation as M
    model = M.from_pretrained(repo).to(DEV).eval()
    id2label = {int(k): v.split(",")[0].strip().lower() for k, v in model.config.id2label.items()}
    groups = CITY_GROUPS if "cityscapes" in key else ADE_GROUPS
    known = set(id2label.values())
    missing = [c for cs in groups.values() for c in cs if c not in known]
    print(f"{key}: {len(id2label)} classes; grouped names not in the label set: {missing}", flush=True)
    rows = []
    with torch.no_grad():
        for im in images:
            inp = proc(images=im, return_tensors="pt").to(DEV)
            out = model(**inp)
            seg = proc.post_process_semantic_segmentation(out, target_sizes=[im.size[::-1]])[0].cpu().numpy()
            counts = np.bincount(seg.ravel(), minlength=len(id2label)).astype(float)
            share = counts / counts.sum()
            row, used = {}, set()
            for g, cs in groups.items():
                ids = [i for i, n in id2label.items() if n in cs]
                used.update(ids)
                row[g] = share[ids].sum()
            row["other"] = share[[i for i in id2label if i not in used]].sum()
            rows.append(row)
    del model
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return pd.DataFrame(rows, index=names)


SF = pd.read_csv(f"{SRC}/scene_features.csv", index_col=0).reindex(names)
seg = {"released": SF[COMP]}
half = [im.resize((im.size[0] // 2, im.size[1] // 2), Image.BICUBIC) for im in originals]  # GPU memory
for key, repo in SEGMODELS.items():
    t = time.time()
    try:
        seg[key] = segment(key, repo, half)
        print(f"{key}: done in {time.time() - t:.0f} s", flush=True)
    except Exception as e:
        print(f"{key}: FAILED {type(e).__name__}: {e}", flush=True)
pd.concat(seg, axis=1).to_csv(f"{OUT}/segmentation_shares_v6.csv")
agree = []
for key in seg:
    if key == "released":
        continue
    for c in COMP:
        a, b = seg["released"][c].to_numpy(float), seg[key][c].to_numpy(float)
        agree.append({"model": key, "category": c, "pearson_r": np.corrcoef(a, b)[0, 1] if b.std() > 0 else np.nan,
                      "mean_abs_diff": np.abs(a - b).mean(), "mean_released": a.mean(), "mean_model": b.mean()})
pd.DataFrame(agree).to_csv(f"{OUT}/segmentation_agreement_v6.csv", index=False)
print(pd.DataFrame(agree).to_string(index=False), flush=True)

# ------------------------------------------------------------------ group benchmark
R = pd.read_csv(f"{SRC}/ratings_person.csv")
G = R.groupby("stimulus")[SCALES].mean().reindex(names)
Y = G.to_numpy(float)
sets = dict(emb)
sets["lowlevel_only"] = SF[LOW].to_numpy(float)
for key, S in seg.items():
    sets[f"composition_only@{key}"] = S[COMP].to_numpy(float)
sets["segmentation_lowlevel@released"] = SF[COMP + LOW].to_numpy(float)


def oof_predictions(F, rep):
    P = np.zeros_like(Y)
    for tr, te in KFold(11, shuffle=True, random_state=rep).split(F):
        sc = StandardScaler().fit(F[tr])
        m = RidgeCV(alphas=np.logspace(-2, 6, 33), alpha_per_target=True).fit(sc.transform(F[tr]), Y[tr])
        P[te] = m.predict(sc.transform(F[te]))
    return P


def r_per_scale(P):
    return np.array([np.corrcoef(P[:, k], Y[:, k])[0, 1] for k in range(Y.shape[1])])


rows = []
for key, F in sets.items():
    per_rep = np.array([r_per_scale(oof_predictions(F, rep)) for rep in range(20)])
    rows.append({"features": key, "mean_r": per_rep.mean(), "sd_over_partitions": per_rep.mean(1).std(),
                 **{f"r_{s}": per_rep[:, k].mean() for k, s in enumerate(SCALES)}})
    print(f"{key}: mean r = {per_rep.mean():.3f}", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/group_benchmark_v6.csv", index=False)
json.dump({"n_scenes": len(names), "seconds": round(time.time() - T0),
           "scramble_clipped_share": float(np.mean([s[1] for s in scr]))}, open(f"{OUT}/meta_v6.json", "w"))
print(pd.DataFrame(rows)[["features", "mean_r", "sd_over_partitions"]].to_string(index=False), flush=True)
