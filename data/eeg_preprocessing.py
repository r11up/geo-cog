"""Preprocess EAV EEG files with filtering, downsampling, artifact cleanup, and 5-second epoching; the 3 Hz high-pass leaves only 3-4 Hz of delta."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import scipy.io
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.decomposition import FastICA

logger = logging.getLogger(__name__)


def bandpass_filter(signal: np.ndarray, low: float, high: float,
                    fs: int, order: int = 6) -> np.ndarray:
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def polyphase_downsample(signal: np.ndarray, up: int, down: int) -> np.ndarray:
    return resample_poly(signal, up, down, axis=-1)


def asr_clean(signal: np.ndarray, threshold: float = 5.0,
              ref_fraction: float = 0.1) -> np.ndarray:
    n_channels, n_samples = signal.shape
    ref_end = max(1, int(n_samples * ref_fraction))
    ref_std = np.std(signal[:, :ref_end], axis=1, keepdims=True) + 1e-8

    bad_mask = np.any(np.abs(signal / ref_std) > threshold, axis=0)
    cleaned  = signal.copy()
    bad_idx  = np.where(bad_mask)[0]
    good_idx = np.where(~bad_mask)[0]

    if len(good_idx) > 1 and len(bad_idx) > 0:
        for ch in range(n_channels):
            cleaned[ch, bad_idx] = np.interp(
                bad_idx, good_idx, signal[ch, good_idx]
            )
    return cleaned


def ica_artifact_removal(signal: np.ndarray, n_components: int = 20,
                         random_state: int = 42) -> np.ndarray:
    n_channels = signal.shape[0]
    n_components = min(n_components, n_channels)

    ica = FastICA(n_components=n_components, random_state=random_state,
                  max_iter=500, tol=1e-4, whiten="unit-variance")
    sources = ica.fit_transform(signal.T)
    mixing  = ica.mixing_

    mu    = sources.mean(axis=0)
    sigma = sources.std(axis=0) + 1e-8
    kurt  = ((((sources - mu) / sigma) ** 4).mean(axis=0)) - 3.0
    sources_clean = sources.copy()
    sources_clean[:, np.abs(kurt) > 5.0] = 0.0

    return (sources_clean @ mixing.T).T


def _kurtosis(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x)
    mu = x.mean(axis=axis, keepdims=True)
    sigma = x.std(axis=axis, keepdims=True) + 1e-8
    return ((((x - mu) / sigma) ** 4).mean(axis=axis)) - 3.0


def segment_epochs(signal: np.ndarray, epoch_samples: int) -> np.ndarray:
    n_channels, n_samples = signal.shape
    n_epochs = n_samples // epoch_samples
    return np.stack([
        signal[:, i * epoch_samples:(i + 1) * epoch_samples]
        for i in range(n_epochs)
    ], axis=0)


def _load_mat(mat_path: Path) -> dict:
    try:
        mat = scipy.io.loadmat(str(mat_path))
        return mat
    except NotImplementedError:
        pass
    except Exception:
        pass

    try:
        import mat73
        return mat73.loadmat(str(mat_path))
    except ImportError:
        pass
    except Exception as e:
        logger.debug("mat73 failed: %s", e)

    try:
        import h5py

        def _read(f, path="/"):
            out = {}
            for key in f[path].keys():
                item = f[path][key]
                if hasattr(item, 'keys'):
                    out[key] = _read(f, f"{path}{key}/")
                else:
                    out[key] = item[()]
            return out

        with h5py.File(str(mat_path), "r") as hf:
            return _read(hf)
    except ImportError:
        pass

    raise RuntimeError(
        f"Cannot load {mat_path}. Install mat73 (pip install mat73) or h5py "
        "to read MATLAB v7.3 HDF5 files used by the EAV dataset."
    )


def _find_key(mat: dict, candidates: List[str]) -> str:
    for k in candidates:
        if k in mat:
            return k
    for k in mat:
        if not k.startswith("__"):
            return k
    raise KeyError(f"No EEG data key found. Available: {list(mat.keys())}")


def _normalize_eeg_shape(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 3:
        raise ValueError(f"Expected 3-D EEG array, got shape {raw.shape}")

    sizes = list(raw.shape)
    trial_ax = sizes.index(200) if 200 in sizes else 0
    channel_ax = sizes.index(30) if 30 in sizes else None
    if channel_ax is None or channel_ax == trial_ax:
        remaining = [i for i in range(3) if i != trial_ax]
        channel_ax = remaining[0]

    sample_candidates = [i for i in range(3) if i not in (trial_ax, channel_ax)]
    if not sample_candidates:
        raise ValueError(f"Could not infer sample axis from shape {raw.shape}")
    sample_ax = sample_candidates[0]

    return raw.transpose(trial_ax, channel_ax, sample_ax)


def load_subject_raw(
    mat_path: Path,
    speaking_indices: Tuple[int, ...] = (1, 3, 5, 7, 9),
) -> Tuple[np.ndarray, np.ndarray]:
    mat = _load_mat(mat_path)

    data_key  = _find_key(mat, ["data", "EEG", "eeg", "Data"])
    label_key = _find_key(mat, ["labels", "label", "Labels", "label_SPE"])

    raw = np.array(mat[data_key])
    lab = np.array(mat[label_key])

    raw = _normalize_eeg_shape(raw)

    py_indices    = [i - 1 for i in speaking_indices]
    trial_labels  = np.argmax(lab, axis=0)
    speaking_mask = np.isin(trial_labels, py_indices)

    eeg_speak    = raw[speaking_mask]
    raw_labels   = trial_labels[speaking_mask]
    label_map    = {idx: j for j, idx in enumerate(sorted(py_indices))}
    labels_out   = np.array([label_map[l] for l in raw_labels])

    return eeg_speak, labels_out


def preprocess_subject(
    eeg_raw: np.ndarray,
    labels:  np.ndarray,
    fs_native:        int   = 500,
    fs_target:        int   = 100,
    bp_low:           float = 3.0,
    bp_high:          float = 50.0,
    butter_order:     int   = 5,
    asr_threshold:    float = 5.0,
    n_ica_components: int   = 20,
    ica_random_state: int   = 42,
    epoch_samples:    int   = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    decimation = fs_native // fs_target
    all_epochs, all_labels = [], []

    for i in range(eeg_raw.shape[0]):
        trial = eeg_raw[i].astype(np.float32)

        trial = bandpass_filter(trial, bp_low, bp_high, fs_native, butter_order)
        trial = polyphase_downsample(trial, up=1, down=decimation)
        trial = asr_clean(trial, threshold=asr_threshold)

        try:
            trial = ica_artifact_removal(
                trial, n_components=n_ica_components,
                random_state=ica_random_state,
            )
        except Exception as e:
            logger.warning("ICA failed for trial %d: %s — skipping ICA.", i, e)

        epochs = segment_epochs(trial, epoch_samples)
        all_epochs.append(epochs)
        all_labels.extend([labels[i]] * epochs.shape[0])

    return (
        np.concatenate(all_epochs, axis=0).astype(np.float32),
        np.array(all_labels, dtype=np.int64),
    )


def preprocess_all(cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_e = cfg.paths.cache_dir / "all_epochs.npy"
    cache_l = cfg.paths.cache_dir / "all_labels.npy"
    cache_s = cfg.paths.cache_dir / "subject_ids.npy"

    if cache_e.exists() and cache_l.exists() and cache_s.exists():
        logger.info("Loading preprocessed EEG from cache …")
        return (
            np.load(str(cache_e)),
            np.load(str(cache_l)),
            np.load(str(cache_s)),
        )

    eav_root = cfg.paths.eav_root
    logger.info("Preprocessing EAV dataset from %s …", eav_root)

    subject_dirs = sorted(eav_root.glob("subject_*/EEG"))
    if not subject_dirs:
        subject_dirs = sorted(eav_root.glob("Subject_*/EEG"))
    if not subject_dirs:
        subject_dirs = sorted(eav_root.glob("subject_*"))

    mat_paths: List[Path] = []
    for d in subject_dirs[:cfg.eav.n_subjects]:
        if d.is_dir():
            mats = sorted(d.glob("*.mat"))
            if mats:
                mat_paths.append(mats[0])
        elif d.suffix == ".mat":
            mat_paths.append(d)

    if not mat_paths:
        raise FileNotFoundError(
            f"No .mat files found under {eav_root}. "
            "Expected layout: subject_01/EEG/subject_01.mat"
        )

    all_epochs, all_labels, subject_ids = [], [], []

    for subj_idx, mat_path in enumerate(mat_paths):
        logger.info("Subject %02d / %02d : %s",
                    subj_idx + 1, len(mat_paths), mat_path.name)
        try:
            eeg_raw, labels = load_subject_raw(
                mat_path, cfg.eav.speaking_label_indices
            )
            epochs, ep_labels = preprocess_subject(
                eeg_raw, labels,
                fs_native        = cfg.preprocess.native_fs,
                fs_target        = cfg.preprocess.target_fs,
                bp_low           = cfg.preprocess.bp_low,
                bp_high          = cfg.preprocess.bp_high,
                butter_order     = cfg.preprocess.butter_order,
                asr_threshold    = cfg.preprocess.asr_threshold,
                n_ica_components = cfg.preprocess.n_ica_components,
                ica_random_state = cfg.preprocess.ica_random_state,
                epoch_samples    = cfg.preprocess.epoch_samples,
            )
        except Exception as e:
            logger.error("Subject %02d failed: %s — skipping.", subj_idx, e)
            continue

        all_epochs.append(epochs)
        all_labels.append(ep_labels)
        subject_ids.extend([subj_idx] * len(ep_labels))

    if not all_epochs:
        raise RuntimeError(
            f"No EAV subjects could be preprocessed from {eav_root}. "
            "Check the dataset layout and .mat files."
        )

    all_epochs   = np.concatenate(all_epochs, axis=0).astype(np.float32)
    all_labels   = np.concatenate(all_labels, axis=0).astype(np.int64)
    subject_ids  = np.array(subject_ids, dtype=np.int64)

    np.save(str(cache_e), all_epochs)
    np.save(str(cache_l), all_labels)
    np.save(str(cache_s), subject_ids)
    logger.info("Saved cache: %d epochs from %d subjects.",
                len(all_epochs), len(mat_paths))

    return all_epochs, all_labels, subject_ids
