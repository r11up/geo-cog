"""Decoding the viewed environment from EEG (Urban Appraisal, ds006850), for people and scenes held
out together. Part A: time-resolved decoding of high vs low greenery (HT/LT) and built form (HB/LB)
from person x scene averaged epochs, with a sign-flip cluster permutation test. Part B: frozen CBraMod
(EEG foundation model) features with a linear probe, for greenery and for person-level appraisal.
Part C: EEGNet and EEG-Conformer trained on single trials."""
import glob
import json
import os
import subprocess
import sys
import time

T0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "braindecode", "huggingface_hub", "pyriemann"],
               check=False)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def find(*markers):
    for marker in markers:
        for depth in range(1, 6):
            hits = glob.glob("/kaggle/input/" + "*/" * depth + marker)
            if hits:
                return hits[0][:-len(marker)].rstrip("/")
    raise SystemExit(f"none of {markers} found")


SRC = find("scene_features.csv")
CODE = find("urbeeg/splits.py", "code/urbeeg/splits.py")
CODE = CODE + "/code" if os.path.isdir(CODE + "/code/urbeeg") else CODE
sys.path.insert(0, CODE)
from urbeeg.config import EXCLUDED, SEED  # noqa: E402
from urbeeg.splits import crossed_cells  # noqa: E402

OUT = "/kaggle/working"
RUN = set(os.environ.get("DECODE_PARTS", "C"))           # parts to run (A, B, C); A and B are done
DEV = "cuda" if torch.cuda.is_available() else "cpu"
scene = pd.read_csv(f"{SRC}/scene_features.csv", index_col=0)
scenes = sorted(scene.index)
subs = sorted(os.path.basename(p)[:-len("_epochs.npz")] for p in glob.glob(f"{SRC}/eeg/sub-*_epochs.npz"))
subs = [s for s in subs if s not in EXCLUDED]
strata = scene[["code_HT", "code_HB"]].astype(str).agg("".join, axis=1).to_dict()
cells = crossed_cells(subs, scenes, strata, n_p=5, n_s=4, n_rep=3, seed=SEED)
y_scene = {"greenery_code": scene.loc[scenes, "code_HT"].to_numpy().astype(int),
           "built_code": scene.loc[scenes, "code_HB"].to_numpy().astype(int)}
SCALES = ["valence", "arousal", "stress"]
print(f"{len(subs)} participants, {len(scenes)} scenes, device {DEV}", flush=True)


def save(name, obj):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)


# ------------------------------------------------------------------ data: pair averages and trials
PA, TRIALS, RATINGS, times = [], [], [], None
for s in subs:
    z = np.load(f"{SRC}/eeg/{s}_epochs.npz")
    x = z["eeg_uv"]
    times = z["times"]
    tr = pd.read_csv(f"{SRC}/eeg/{s}_trials.csv")
    has = tr["has_eeg"].to_numpy(bool) if "has_eeg" in tr else np.ones(len(tr), bool)
    avg = np.zeros((len(scenes), x.shape[1], x.shape[2]), np.float32)
    keep = []
    for k, st in enumerate(scenes):
        g = np.where((tr.stimulus == st).to_numpy() & has)[0]
        clean = g[~tr.ptp_flag.to_numpy(bool)[g]]
        use = clean if len(clean) >= 3 else g
        avg[k] = x[use].astype(np.float32).mean(0)
        keep += [(i, k) for i in use]
    PA.append(avg)
    TRIALS.append((x, np.array(keep)))
    r = tr.pivot_table(index="stimulus", columns="scale", values="rating", aggfunc="first").reindex(scenes)[SCALES]
    RATINGS.append(((r - r.mean()) / r.std()).to_numpy(float))
PA = np.stack(PA)                                         # (P, 56, 65, T)
RATINGS = np.stack(RATINGS)                               # (P, 56, 3)
P_IDX, S_IDX = {s: i for i, s in enumerate(subs)}, {s: i for i, s in enumerate(scenes)}
print(f"data ready ({time.time() - T0:.0f} s)", flush=True)


def zscore_people(F):                                     # F (P, 56, d): z-score within person
    return (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-9)


def crossed_probe(F, target, kind="binary", reps=(0, 1, 2), C=0.01, alpha=1e3):
    """F (P, 56, d). Returns per-person metric averaged over repetitions (AUC or Pearson r)."""
    scores = {}
    for c in cells:
        if c.rep not in reps:
            continue
        pi_tr = [P_IDX[p] for p in c.train_participants]; pi_te = [P_IDX[p] for p in c.test_participants]
        si_tr = [S_IDX[s] for s in c.train_scenes]; si_te = [S_IDX[s] for s in c.test_scenes]
        Xtr = F[np.ix_(pi_tr, si_tr)].reshape(-1, F.shape[-1]); Xte = F[np.ix_(pi_te, si_te)].reshape(-1, F.shape[-1])
        sc = StandardScaler().fit(Xtr)
        if kind == "binary":
            ytr = np.tile(target[si_tr], len(pi_tr))
            m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
            out = m.decision_function(sc.transform(Xte)).reshape(len(pi_te), len(si_te))
        else:                                             # target (P, 56): person-level ratings
            ytr = target[np.ix_(pi_tr, si_tr)].reshape(-1)
            m = Ridge(alpha=alpha, solver="lsqr").fit(sc.transform(Xtr), np.nan_to_num(ytr))
            out = m.predict(sc.transform(Xte)).reshape(len(pi_te), len(si_te))
        for a, p in enumerate(pi_te):
            for b, si in enumerate(si_te):
                scores[(c.rep, p, si)] = out[a, b]
    res = []
    for p in range(F.shape[0]):
        vals = []
        for rep in reps:
            s = np.array([scores.get((rep, p, si), np.nan) for si in range(len(scenes))])
            if np.isnan(s).any():
                continue
            if kind == "binary":
                vals.append(roc_auc_score(target, s))
            else:
                y = target[p]; ok = ~np.isnan(y)
                vals.append(np.corrcoef(s[ok], y[ok])[0, 1])
        res.append(np.mean(vals) if vals else np.nan)
    return np.array(res)


def summary(v, null):
    v = v[~np.isnan(v)]
    b = np.random.default_rng(0).choice(v, size=(2000, len(v))).mean(1)
    return {"mean": v.mean(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5), "n": len(v),
            "above_null": float(np.mean(v > null))}


# ------------------------------------------------------------------ Part A: time-resolved decoding
try:
    if "A" not in RUN:
        raise RuntimeError("skipped")
    wins = [(a / 1000, (a + 50) / 1000) for a in range(-100, 1000, 50)]
    partA = {"windows_ms": [[int(a * 1000), int(b * 1000)] for a, b in wins]}
    for task in ("greenery_code", "built_code"):
        auc = []
        for a, b in wins:
            m = (times >= a) & (times < b)
            auc.append(crossed_probe(zscore_people(PA[..., m].mean(-1)), y_scene[task]))
        auc = np.array(auc).T                             # (P, W)
        d = auc - 0.5
        t_obs = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(d)))

        def clusters(t):
            out, cur = [], []
            for i, v in enumerate(t):
                if v > 2.0:
                    cur.append(i)
                elif cur:
                    out.append(cur); cur = []
            if cur:
                out.append(cur)
            return out
        obs = [(c, t_obs[c].sum()) for c in clusters(t_obs)]
        rng = np.random.default_rng(1)
        null = []
        for _ in range(5000):
            sg = rng.choice([-1, 1], size=(len(d), 1))
            dd = d * sg
            tp = dd.mean(0) / (dd.std(0, ddof=1) / np.sqrt(len(dd)))
            null.append(max([tp[c].sum() for c in clusters(tp)], default=0.0))
        null = np.array(null)
        partA[task] = {"auc_mean": auc.mean(0).tolist(), "auc_sd": auc.std(0).tolist(), "t": t_obs.tolist(),
                       "clusters": [{"windows": c, "mass": float(mss), "p": float((null >= mss).mean())} for c, mss in obs]}
        print(task, "peak AUC", round(float(auc.mean(0).max()), 3), "clusters",
              [(partA['windows_ms'][c[0]][0], partA['windows_ms'][c[-1]][1], round(float((null >= mss).mean()), 4))
               for c, mss in obs], flush=True)
    save("partA_time_resolved", partA)
except Exception as e:
    print("Part A failed:", type(e).__name__, e, flush=True)

# ------------------------------------------------------------------ Part B: CBraMod frozen features
try:
    if "B" not in RUN:
        raise RuntimeError("skipped")
    from braindecode.models import CBraMod
    model = None
    for kw in ({}, {"n_outputs": 2, "n_chans": PA.shape[2], "n_times": 600, "sfreq": 200}):
        try:
            model = CBraMod.from_pretrained("braindecode/cbramod-pretrained", **kw)
            break
        except Exception as e:
            print("from_pretrained", kw, "->", type(e).__name__, str(e)[:200], flush=True)
    model = model.to(DEV).eval()
    m0 = (times >= 0) & (times < 3.0)

    def feats(xb):                                        # xb (B, C, 375) in microvolts at 125 Hz
        xr = resample_poly(xb, 8, 5, axis=-1)[..., :600] / 100.0
        with torch.no_grad():
            out = model(torch.as_tensor(xr, dtype=torch.float32, device=DEV), return_features=True)
        f = out["features"] if isinstance(out, dict) else out
        f = f.float()
        if f.ndim == 4:                                   # (B, C, patches, emb) -> mean over channels
            f = f.mean(1).flatten(1)
        elif f.ndim == 3:
            f = f.mean(1)
        return f.cpu().numpy()
    FB = []
    for pi, (x, keep) in enumerate(TRIALS):
        F = None
        cnt = np.zeros(len(scenes))
        for i in range(0, len(keep), 64):
            rows = keep[i:i + 64]
            f = feats(x[rows[:, 0]][..., m0].astype(np.float32))
            if F is None:
                F = np.zeros((len(scenes), f.shape[1]), np.float32)
            np.add.at(F, rows[:, 1], f)
            np.add.at(cnt, rows[:, 1], 1)
        FB.append(F / cnt[:, None])
    FB = zscore_people(np.stack(FB))
    print("CBraMod features", FB.shape, f"({time.time() - T0:.0f} s)", flush=True)
    partB = {"dim": int(FB.shape[-1])}
    for task in ("greenery_code", "built_code"):
        partB[task] = summary(crossed_probe(FB, y_scene[task]), 0.5)
    for k, sc_name in enumerate(SCALES):
        partB[sc_name] = summary(crossed_probe(FB, RATINGS[..., k], kind="ratings"), 0.0)
    save("partB_cbramod", partB)
    print("Part B", json.dumps(partB, default=float)[:600], flush=True)
except Exception as e:
    print("Part B failed:", type(e).__name__, str(e)[:500], flush=True)

# ------------------------------------------------------------------ Part C: EEGNet / EEG-Conformer
try:
    from braindecode.models import EEGConformer, EEGNet
    n_ch, n_t = PA.shape[2], PA.shape[3]
    partC = {}
    for name in ("eegnet", "conformer"):
        scores = {}
        for c in [c for c in cells if c.rep == 0]:
          try:
            rng = np.random.default_rng(SEED + 10 * c.p_fold + c.s_fold)
            val_p = set(rng.choice(c.train_participants, 6, replace=False))
            cls = {s: y_scene["greenery_code"][S_IDX[s]] for s in c.train_scenes}   # both classes in validation
            val_s = set(rng.choice([s for s in c.train_scenes if cls[s] == 1], 3, replace=False)) | \
                set(rng.choice([s for s in c.train_scenes if cls[s] == 0], 3, replace=False))

            def gather(people, scs):
                X, Y, own = [], [], []
                for p in people:
                    x, keep = TRIALS[P_IDX[p]]
                    sel = keep[np.isin(keep[:, 1], [S_IDX[s] for s in scs])]
                    X.append(x[sel[:, 0]]); Y.append(y_scene["greenery_code"][sel[:, 1]])
                    own += [(P_IDX[p], k) for k in sel[:, 1]]
                return np.concatenate(X), np.concatenate(Y), own
            tr_p = [p for p in c.train_participants if p not in val_p]
            tr_s = [s for s in c.train_scenes if s not in val_s]
            Xtr, Ytr, _ = gather(tr_p, tr_s)
            Xva, Yva, _ = gather(sorted(val_p), sorted(val_s))
            Xte, Yte, own = gather(c.test_participants, c.test_scenes)
            torch.manual_seed(SEED)
            net = (EEGNet(n_chans=n_ch, n_outputs=2, n_times=n_t) if name == "eegnet" else
                   EEGConformer(n_chans=n_ch, n_outputs=2, n_times=n_t, final_fc_length="auto")).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
            T = lambda a: torch.as_tensor(a.astype(np.float32) / 10.0)
            Xtr_t, Ytr_t = T(Xtr), torch.as_tensor(Ytr)

            def predict(X):
                net.eval(); out = []
                with torch.no_grad():
                    for i in range(0, len(X), 512):
                        out.append(torch.softmax(net(T(X[i:i + 512]).to(DEV)), 1)[:, 1].cpu().numpy())
                return np.concatenate(out)
            best, bad = -1, 0
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            for ep in range(30):
                net.train()
                for b in torch.randperm(len(Xtr_t)).split(128):
                    loss = torch.nn.functional.cross_entropy(net(Xtr_t[b].to(DEV)), Ytr_t[b].to(DEV))
                    opt.zero_grad(); loss.backward(); opt.step()
                pv = predict(Xva)
                f = roc_auc_score(Yva, pv) if len(np.unique(Yva)) == 2 and np.isfinite(pv).all() else -1
                if f > best + 1e-4:
                    best, bad, state = f, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 5:
                        break
            net.load_state_dict(state)
            pr = predict(Xte)
            agg = pd.DataFrame(own, columns=["p", "s"]).assign(score=pr).groupby(["p", "s"]).score.mean()
            for (p, s), v in agg.items():
                scores[(p, s)] = v
            print(name, "cell", c.p_fold, c.s_fold, "val AUC", round(best, 3), f"({time.time() - T0:.0f} s)", flush=True)
          except Exception:
            import traceback
            print(name, "cell", c.p_fold, c.s_fold, "FAILED", traceback.format_exc()[-800:], flush=True)
        per = []
        for p in range(len(subs)):
            s = np.array([scores.get((p, si), np.nan) for si in range(len(scenes))])
            per.append(roc_auc_score(y_scene["greenery_code"], s) if not np.isnan(s).any() else np.nan)
        partC[name] = summary(np.array(per), 0.5)
        save("partC_deep", partC)
        print("Part C", name, partC[name], flush=True)
except Exception as e:
    print("Part C failed:", type(e).__name__, str(e)[:500], flush=True)
print(f"done in {time.time() - T0:.0f} s", flush=True)
