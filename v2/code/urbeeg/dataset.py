"""Pair-level data: one row per (participant, scene), with that person's 9 ratings
and features computed from the 9 EEG epochs recorded while they viewed the scene.

Per-pair EEG features do not depend on any other pair, so they are computed once per
participant and cached. Anything that must be fitted (scaling, Riemannian reference,
ridge weights) is fitted later, inside each cross-validation cell, on training data only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .config import DERIVED, EXCLUDED, OCCIPITAL, PARIETAL, PRIMARY_TARGETS, SECONDARY_TARGETS

ALL_TARGETS = PRIMARY_TARGETS + SECONDARY_TARGETS
EEG_DIR = DERIVED / "eeg"
FEAT_DIR = DERIVED / "features"

BASELINE = (-0.2, 0.0)
# Mean-amplitude windows (s) covering early visual, mid and late evaluative activity.
ERP_WINDOWS = [(0.08, 0.13), (0.13, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 0.80),
               (0.80, 1.20), (1.20, 2.00), (2.00, 3.00)]
# Component definitions follow the dataset authors' clusters.
COMPONENTS = {"P1": (OCCIPITAL, (0.08, 0.13)), "N1": (OCCIPITAL, (0.13, 0.20)),
              "P3": (PARIETAL, (0.30, 0.50)), "LPP": (PARIETAL, (0.50, 1.00))}
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
COV_WINDOW = (0.0, 3.0)
MIN_CLEAN_EPOCHS = 3


def processed_participants() -> list:
    """Preprocessed participants, minus those excluded in config.EXCLUDED."""
    subs = (p.name.replace("_epochs.npz", "") for p in EEG_DIR.glob("sub-*_epochs.npz"))
    return sorted(s for s in subs if s not in EXCLUDED)


def load_participant(sub: str):
    z = np.load(EEG_DIR / f"{sub}_epochs.npz")
    trials = pd.read_csv(EEG_DIR / f"{sub}_trials.csv")
    return z["eeg"], trials, [str(c) for c in z["ch_names"]], z["times"], float(z["sfreq"])


def epochs_per_scene(trials: pd.DataFrame) -> dict:
    """Scene -> indices of its epochs, dropping peak-to-peak-flagged epochs when at least
    MIN_CLEAN_EPOCHS clean ones remain."""
    out = {}
    has = trials["has_eeg"] if "has_eeg" in trials else pd.Series(True, index=trials.index)
    for stim, g in trials.groupby("stimulus"):
        with_eeg = g.index[has.loc[g.index]].to_numpy()          # never use trials without EEG
        clean = g.index[has.loc[g.index] & ~g["ptp_flag"]].to_numpy()
        out[stim] = clean if len(clean) >= MIN_CLEAN_EPOCHS else with_eeg
    return out


def _covariances(x: np.ndarray, sfreq: float, times: np.ndarray) -> np.ndarray:
    """x: (n_epochs, n_ch, n_t) volts. Returns (n_bands, n_ch, n_ch) shrinkage covariances
    of the band-passed signal, epochs concatenated in time."""
    from pyriemann.utils.covariance import covariances

    w = (times >= COV_WINDOW[0]) & (times <= COV_WINDOW[1])
    covs = []
    for lo, hi in BANDS.values():
        sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
        xf = sosfiltfilt(sos, x, axis=-1)[..., w]
        cat = np.concatenate(list(xf), axis=-1)[None] * 1e6          # microvolts
        covs.append(covariances(cat, estimator="oas")[0])
    return np.stack(covs)


def participant_features(sub: str) -> dict:
    eeg, trials, ch, times, sfreq = load_participant(sub)
    groups = epochs_per_scene(trials)
    base = (times >= BASELINE[0]) & (times < BASELINE[1])
    scenes = sorted(groups)
    erp_win, erp_comp, covs, n_used = [], [], [], []
    for s in scenes:
        x = eeg[groups[s]]
        avg = x.mean(axis=0)
        avg = (avg - avg[:, base].mean(axis=1, keepdims=True)) * 1e6     # microvolts
        erp_win.append(np.concatenate([avg[:, (times >= a) & (times < b)].mean(axis=1)
                                       for a, b in ERP_WINDOWS]))
        erp_comp.append([avg[[ch.index(c) for c in chans]][:, (times >= a) & (times < b)].mean()
                         for chans, (a, b) in COMPONENTS.values()])
        covs.append(_covariances(x, sfreq, times))
        n_used.append(len(groups[s]))
    return {"participant": np.array([sub] * len(scenes)), "stimulus": np.array(scenes),
            "erp_win": np.array(erp_win, np.float32), "erp_comp": np.array(erp_comp, np.float32),
            "cov": np.array(covs, np.float32), "n_epochs": np.array(n_used),
            "ch_names": np.array(ch)}


def cache_features(subs=None, overwrite=False) -> list:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    done = []
    for sub in subs or processed_participants():
        out = FEAT_DIR / f"{sub}_pairfeat.npz"
        if overwrite or not out.exists():
            np.savez_compressed(out, **participant_features(sub))
        done.append(sub)
    return done


def ratings_wide(subs) -> pd.DataFrame:
    """(participant, stimulus) x scale, recoded ratings (higher = more of the quality)."""
    frames = [pd.read_csv(EEG_DIR / f"{s}_trials.csv") for s in subs]
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["participant", "stimulus"], columns="scale",
                            values="rating", aggfunc="first", dropna=False)
    # A few trials have no response (e.g. sub-31 LTLB02 openness, sub-78 HTHB09 stress).
    # They stay NaN here; evaluate.py imputes the person's mean for training and skips
    # them when scoring.
    return wide.reindex(columns=ALL_TARGETS).astype(float)


def load_pairs(subs=None) -> dict:
    """Everything the models need, aligned row by row."""
    subs = subs or processed_participants()
    cache_features(subs)
    parts = [dict(np.load(FEAT_DIR / f"{s}_pairfeat.npz")) for s in subs]
    feat = {k: np.concatenate([p[k] for p in parts]) for k in
            ["participant", "stimulus", "erp_win", "erp_comp", "cov", "n_epochs"]}
    idx = pd.MultiIndex.from_arrays([feat["participant"], feat["stimulus"]],
                                    names=["participant", "stimulus"])
    ratings = ratings_wide(subs).reindex(idx)
    if ratings.isna().all(axis=1).any():
        raise ValueError("ratings and EEG pairs do not line up")
    feat["ratings"] = ratings.reset_index(drop=True)
    feat["ch_names"] = list(parts[0]["ch_names"])
    return feat
