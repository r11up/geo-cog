"""Urban Appraisal group-level benchmark of Tan & Deng (2026), image round v4 (declared 26 Nov 2026).
Protocol unchanged: group-mean ratings of 55 scenes, 11-fold cross-validation over scenes repeated with
20 random partitions, ridge with the penalty chosen by leave-one-out on the training scenes, Pearson r per
scale averaged over the nine scales.
New in v4: four more encoders (SigLIP 2 SO400M at 512 px, CLIP ViT-H/14, EVA02-CLIP-L/14-336, AIMv2 large
without remote code); ten-view test-time averaging (five 90% crops and their mirror images) for every
encoder; language-anchored features (34 fixed prompts, as in v3) from every vision-language model.
Primary: equal-weight ensemble of the ridge predictions of all feature sets. Secondary: stacked ensemble."""
import glob
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "timm"], check=False)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

T0 = time.time()
SCALES = ["valence", "arousal", "stress", "dominance", "openness", "safety", "beauty", "hominess", "fascination"]
HF_MODELS = {"siglip_so400m": "google/siglip-so400m-patch14-384",
             "siglip2_so400m": "google/siglip2-so400m-patch14-384",
             "clip_vitl14": "openai/clip-vit-large-patch14",
             "dinov2_large": "facebook/dinov2-large",
             "siglip2_giant": "google/siglip2-giant-opt-patch16-384",
             "dinov2_giant": "facebook/dinov2-giant",
             "clip_bigg14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
             "siglip2_so400m_512": "google/siglip2-so400m-patch16-512",
             "clip_vith14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
             "aimv2_large": "apple/aimv2-large-patch14-448"}
OPEN_CLIP = {"eva02_l14_336": ("EVA02-L-14-336", "merged2b_s6b_b61k")}
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


def ten_views(im):
    """Five 90% crops (centre and corners) and their mirror images."""
    w, h = im.size
    cw, ch = int(0.9 * w), int(0.9 * h)
    boxes = [((w - cw) // 2, (h - ch) // 2), (0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch)]
    crops = [im.crop((x, y, x + cw, y + ch)) for x, y in boxes]
    return crops + [ImageOps.mirror(c) for c in crops]


VIEWS = [ten_views(im) for im in images]                     # 55 x 10
print(len(names), "images; device", DEV, "; prompts", len(PROMPTS), flush=True)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)


def hf_image_features(model, proc, imgs):
    feats = []
    for i in range(0, len(imgs), 10):
        inp = proc(images=imgs[i:i + 10], return_tensors="pt")
        inp = {k: (v.to(DEV, model.dtype) if v.is_floating_point() else v.to(DEV)) for k, v in inp.items()
               if k in ("pixel_values", "pixel_attention_mask", "spatial_shapes")}
        if hasattr(model, "get_image_features"):
            f = model.get_image_features(**inp)
            if not torch.is_tensor(f):
                emb = getattr(f, "image_embeds", None)
                f = emb if emb is not None else f.pooler_output
        else:
            o = model(**inp)
            f = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def hf_text_features(model, proc, prompts):
    kw = {"padding": "max_length", "max_length": 64} if "siglip" in type(model).__name__.lower() else {"padding": True}
    inp = proc(text=prompts, return_tensors="pt", truncation=True, **kw)
    inp = {k: v.to(DEV) for k, v in inp.items() if k in ("input_ids", "attention_mask")}
    f = model.get_text_features(**inp)
    if not torch.is_tensor(f):
        emb = getattr(f, "text_embeds", None)
        f = emb if emb is not None else f.pooler_output
    return f.float().cpu().numpy()


def summarise(per_view, text=None, key=""):
    """per_view: (55 x 10, d) image features in view order. Mean embedding, and image-text similarities."""
    E = per_view.reshape(len(images), 10, -1)
    out = {key: E.mean(1)}
    if text is not None:
        out[key + "_text"] = (unit(E) @ unit(text).T).mean(1)
    return out


def run_hf(key, repo):
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    model = AutoModel.from_pretrained(repo, dtype=dtype).to(DEV).eval()
    try:
        proc = AutoProcessor.from_pretrained(repo)
    except Exception:
        proc = AutoImageProcessor.from_pretrained(repo)
    flat = [v for vs in VIEWS for v in vs]
    with torch.no_grad():
        per_view = hf_image_features(model, proc, flat)
        text = hf_text_features(model, proc, PROMPTS) if hasattr(model, "get_text_features") and hasattr(proc, "tokenizer") else None
    del model
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return summarise(per_view, text, key)


def run_open_clip(key, arch, tag):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag, device=DEV)
    model.eval()
    tok = open_clip.get_tokenizer(arch)
    flat = [v for vs in VIEWS for v in vs]
    feats = []
    with torch.no_grad(), torch.autocast(DEV, enabled=DEV == "cuda"):
        for i in range(0, len(flat), 10):
            x = torch.stack([preprocess(im) for im in flat[i:i + 10]]).to(DEV)
            feats.append(model.encode_image(x).float().cpu().numpy())
        text = model.encode_text(tok(PROMPTS).to(DEV)).float().cpu().numpy()
    del model
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return summarise(np.concatenate(feats), text, key)


emb, failed = {}, {}
for key, spec in list(HF_MODELS.items()) + list(OPEN_CLIP.items()):
    t = time.time()
    try:
        got = run_hf(key, spec) if key in HF_MODELS else run_open_clip(key, *spec)
        emb.update(got)
        print(f"{key}: {', '.join(f'{k} {v.shape}' for k, v in got.items())} in {time.time() - t:.0f} s", flush=True)
    except Exception as e:                                   # keep going; report what failed
        failed[key] = f"{type(e).__name__}: {e}"[:300]
        print(f"{key}: FAILED {failed[key]}", flush=True)
np.savez_compressed(f"{OUT}/embeddings_v4.npz", names=np.array(names), **emb)

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
    inner = {k: np.zeros((len(tr), Y.shape[1])) for k in keys}
    for itr, ite in KFold(10, shuffle=True, random_state=seed).split(tr):
        for k in keys:
            inner[k][ite] = ridge_fit_predict(sets[k], tr[itr], tr[ite], Y[tr[itr]])
    Wt = np.zeros((len(keys), Y.shape[1]))
    for j in range(Y.shape[1]):
        w, _ = nnls(np.column_stack([inner[k][:, j] for k in keys]), Y[tr, j])
        Wt[:, j] = w / w.sum() if w.sum() > 0 else np.full(len(keys), 1 / len(keys))
    return Wt


primary = list(sets)
scores = {k: [] for k in sets}
scores.update({"ensemble_v4": [], "stacked_v4": []})
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
    scores["ensemble_v4"].append(r_per_scale(np.mean([P[k] for k in primary], axis=0)))
    scores["stacked_v4"].append(r_per_scale(Pst))
    print(f"partition {rep}: ensemble_v4 {scores['ensemble_v4'][-1].mean():.3f}, stacked {scores['stacked_v4'][-1].mean():.3f}",
          flush=True)
rows = []
for k, v in scores.items():
    v = np.array(v)
    rows.append({"features": k, "mean_r": v.mean(), "sd_over_partitions": v.mean(1).std(),
                 "role": "primary" if k == "ensemble_v4" else "secondary",
                 **{f"r_{s}": v[:, j].mean() for j, s in enumerate(SCALES)}})
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
pd.DataFrame(rows).to_csv(f"{OUT}/group_benchmark_v4.csv", index=False)
json.dump({"n_scenes": len(names), "n_people": int(len(people)), "primary_sets": primary, "failed": failed,
           "prompts": PROMPTS, "seconds": round(time.time() - T0)}, open(f"{OUT}/meta_v4.json", "w"))
print(pd.DataFrame(rows)[["features", "mean_r", "sd_over_partitions", "role"]].to_string(index=False), flush=True)
