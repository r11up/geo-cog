"""EAV (Lee et al. 2024) for the contrast analysis: the design of the original submission.

Label coding verified against the EAV paper's published class sequence
[A,A,C,C,S,S,H,A,C,H,H,S,S,A,A,C,C,H,H,S]: trigger rows 0/1 neutral, 2/3 sadness,
4/5 anger, 6/7 happiness, 8/9 calmness; even = listening, odd = speaking.
We use the five speaking conditions (20 trials each, 20 s at 500 Hz), as before.

Preprocessing (minimal, stated in the manuscript): common average reference,
1-45 Hz zero-phase band-pass, resample to 100 Hz, four non-overlapping 5 s epochs per
trial. Features per epoch: differential entropy and log Welch power in five bands for each
of 30 channels (30 x 10), z-scored within participant using that participant's own epochs
(no labels involved).
"""
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

EAV_ROOT = Path(os.environ.get("EAV_ROOT", Path.home() / "Desktop/Research/datasets/EAV-EEG"))
SPEAKING = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}            # trigger row -> class
CLASSES = ["neutral", "sadness", "anger", "happiness", "calmness"]
FS_IN, FS = 500, 100
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
EPOCH_S, N_EPOCHS = 5, 4


def subject_features(subject: int):
    base = EAV_ROOT / f"subject{subject}" / "EEG"
    m = sio.loadmat(base / f"subject{subject}_eeg.mat")
    seg = m["seg"] if "seg" in m else m["seg1"]     # variable is named seg1 in 31 of 42 files
    lab = sio.loadmat(base / f"subject{subject}_eeg_label.mat")["label"].argmax(0)
    trials = [t for t in range(seg.shape[2]) if lab[t] in SPEAKING]
    x = seg[:, :, trials].transpose(2, 1, 0).astype(np.float64)          # (100, 30, 10000)
    x = x - x.mean(axis=1, keepdims=True)                                 # average reference
    sos = butter(4, [1, 45], btype="bandpass", fs=FS_IN, output="sos")
    x = resample_poly(sosfiltfilt(sos, x, axis=-1), 1, FS_IN // FS, axis=-1)   # 100 Hz
    n = EPOCH_S * FS
    ep = np.stack([x[..., i * n:(i + 1) * n] for i in range(N_EPOCHS)], axis=1)  # (100,4,30,500)
    ep = ep.reshape(-1, ep.shape[2], n)
    feats = []
    band_sos = [butter(4, b, btype="bandpass", fs=FS, output="sos") for b in BANDS]
    f, p = welch(ep, fs=FS, nperseg=200, noverlap=100, axis=-1)
    for (lo, hi), bs in zip(BANDS, band_sos):
        var = sosfiltfilt(bs, ep, axis=-1).var(axis=-1)
        feats.append(0.5 * np.log(2 * np.pi * np.e * var + 1e-20))          # differential entropy
    for lo, hi in BANDS:
        feats.append(np.log(p[..., (f >= lo) & (f < hi)].mean(axis=-1) + 1e-20))
    X = np.stack(feats, axis=-1)                                          # (400, 30, 10)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)                     # within-person
    y = np.repeat([SPEAKING[lab[t]] for t in trials], N_EPOCHS)
    trial = np.repeat(trials, N_EPOCHS)
    return X.astype(np.float32), y, trial


def build(out: Path, n_jobs: int = 3):
    from joblib import Parallel, delayed

    res = Parallel(n_jobs=n_jobs)(delayed(subject_features)(s) for s in range(1, 43))
    X = np.concatenate([r[0] for r in res])
    y = np.concatenate([r[1] for r in res])
    trial = np.concatenate([r[2] for r in res])
    subject = np.concatenate([np.full(len(r[1]), s) for s, r in zip(range(1, 43), res)])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, subject=subject, trial=trial)
    return X.shape


# ---------------------------------------------------------------------------------------
# v2: adds band-wise covariances, re-centred per participant
# at that participant's Riemannian mean and mapped to the tangent space (no labels used).
def subject_features_v2(subject: int, epoch_s: int = EPOCH_S):
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    base = EAV_ROOT / f"subject{subject}" / "EEG"
    m = sio.loadmat(base / f"subject{subject}_eeg.mat")
    seg = m["seg"] if "seg" in m else m["seg1"]
    lab = sio.loadmat(base / f"subject{subject}_eeg_label.mat")["label"].argmax(0)
    trials = [t for t in range(seg.shape[2]) if lab[t] in SPEAKING]
    x = seg[:, :, trials].transpose(2, 1, 0).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    sos = butter(4, [1, 45], btype="bandpass", fs=FS_IN, output="sos")
    x = resample_poly(sosfiltfilt(sos, x, axis=-1), 1, FS_IN // FS, axis=-1)
    n, n_ep = epoch_s * FS, 20 // epoch_s
    ep = np.stack([x[..., i * n:(i + 1) * n] for i in range(n_ep)], axis=1).reshape(-1, x.shape[1], n)
    feats, tang = [], []
    nper = 200 if epoch_s >= 4 else 100
    f, p = welch(ep, fs=FS, nperseg=nper, noverlap=nper // 2, axis=-1)
    for lo, hi in BANDS:
        xb = sosfiltfilt(butter(4, (lo, hi), btype="bandpass", fs=FS, output="sos"), ep, axis=-1)
        feats.append(0.5 * np.log(2 * np.pi * np.e * xb.var(axis=-1) + 1e-20))
        C = Covariances("oas").fit_transform(xb)
        tang.append(TangentSpace(metric="riemann").fit_transform(C))      # reference = own mean
    for lo, hi in BANDS:
        feats.append(np.log(p[..., (f >= lo) & (f < hi)].mean(axis=-1) + 1e-20))
    X = np.stack(feats, axis=-1)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    T = np.concatenate(tang, axis=1)
    T = (T - T.mean(axis=0)) / (T.std(axis=0) + 1e-8)
    y = np.repeat([SPEAKING[lab[t]] for t in trials], n_ep)
    trial = np.repeat(trials, n_ep)
    return X.astype(np.float32), T.astype(np.float32), y, trial


def build_v2(out: Path, n_jobs: int = 6, epoch_s: int = EPOCH_S):
    from joblib import Parallel, delayed

    res = Parallel(n_jobs=n_jobs)(delayed(subject_features_v2)(s, epoch_s) for s in range(1, 43))
    np.savez_compressed(out, X=np.concatenate([r[0] for r in res]), T=np.concatenate([r[1] for r in res]),
                        y=np.concatenate([r[2] for r in res]), trial=np.concatenate([r[3] for r in res]),
                        subject=np.concatenate([np.full(len(r[2]), s) for s, r in zip(range(1, 43), res)]))


def subject_raw(subject: int):
    """Preprocessed 5 s epochs (CAR, 1-45 Hz, 100 Hz) in microvolt-scale float16, for raw-signal models."""
    base = EAV_ROOT / f"subject{subject}" / "EEG"
    m = sio.loadmat(base / f"subject{subject}_eeg.mat")
    seg = m["seg"] if "seg" in m else m["seg1"]
    lab = sio.loadmat(base / f"subject{subject}_eeg_label.mat")["label"].argmax(0)
    trials = [t for t in range(seg.shape[2]) if lab[t] in SPEAKING]
    x = seg[:, :, trials].transpose(2, 1, 0).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    sos = butter(4, [1, 45], btype="bandpass", fs=FS_IN, output="sos")
    x = resample_poly(sosfiltfilt(sos, x, axis=-1), 1, FS_IN // FS, axis=-1)
    n = EPOCH_S * FS
    ep = np.stack([x[..., i * n:(i + 1) * n] for i in range(N_EPOCHS)], axis=1).reshape(-1, x.shape[1], n)
    ep = ep / (ep.std() + 1e-12)                       # one scale per participant (no labels)
    return ep.astype(np.float16)


def build_raw(out: Path, n_jobs: int = 6):
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs)(delayed(subject_raw)(s) for s in range(1, 43))
    np.save(out, np.concatenate(res))
