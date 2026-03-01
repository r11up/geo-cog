"""Extract differential-entropy and Welch PSD features; delta is defined as 1-4 Hz, while the current high-pass leaves only 3-4 Hz."""

import numpy as np
from scipy.signal import welch
from typing import Tuple, List, Dict


FREQ_BANDS: List[Tuple[str, float, float]] = [
    ("delta", 1.0,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0,  13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 50.0),
]

N_BANDS = len(FREQ_BANDS)

def _band_variance(signal_1d: np.ndarray, fs: float,
                   low: float, high: float) -> float:
    n = len(signal_1d)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(signal_1d)

    mask = (freqs >= low) & (freqs < high)
    spectrum_masked = np.zeros_like(spectrum)
    spectrum_masked[mask] = spectrum[mask]

    band_signal = np.fft.irfft(spectrum_masked, n=n)
    return float(np.var(band_signal))


def extract_de(epoch: np.ndarray, fs: float = 100.0,
               bands: List[Tuple[str, float, float]] = FREQ_BANDS
               ) -> np.ndarray:
    n_channels = epoch.shape[0]
    n_bands = len(bands)
    de = np.zeros((n_channels, n_bands), dtype=np.float32)

    for ch in range(n_channels):
        for b, (_, low, high) in enumerate(bands):
            var = _band_variance(epoch[ch], fs, low, high)
            var = max(var, 1e-10)
            de[ch, b] = 0.5 * np.log(2.0 * np.pi * np.e * var)

    return de


def extract_psd(epoch: np.ndarray, fs: float = 100.0,
                nperseg: int = 256,
                bands: List[Tuple[str, float, float]] = FREQ_BANDS
                ) -> np.ndarray:
    n_channels, n_samples = epoch.shape
    n_bands = len(bands)
    psd = np.zeros((n_channels, n_bands), dtype=np.float32)

    seg = min(nperseg, n_samples)

    for ch in range(n_channels):
        freqs, pxx = welch(epoch[ch], fs=fs, window="hann",
                           nperseg=seg, noverlap=seg // 2)
        for b, (_, low, high) in enumerate(bands):
            band_mask = (freqs >= low) & (freqs < high)
            if band_mask.any():
                mean_pxx = np.mean(pxx[band_mask])
                mean_pxx = max(mean_pxx, 1e-10)
                psd[ch, b] = float(np.log(mean_pxx))
            else:
                psd[ch, b] = 0.0

    return psd


def extract_features(
    epochs: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 256,
    bands: List[Tuple[str, float, float]] = FREQ_BANDS,
) -> np.ndarray:
    T, n_channels, _ = epochs.shape
    n_bands = len(bands)
    X = np.zeros((T, n_channels, 2 * n_bands), dtype=np.float32)

    for t in range(T):
        epoch = epochs[t]
        de  = extract_de(epoch, fs, bands)
        psd = extract_psd(epoch, fs, nperseg=nperseg, bands=bands)
        X[t, :, :n_bands]      = de
        X[t, :, n_bands:]      = psd

    return X


def extract_features_cached(epochs: np.ndarray, cfg) -> np.ndarray:
    import logging
    logger = logging.getLogger(__name__)

    cache_path = cfg.paths.cache_dir / "features_X.npy"
    if cache_path.exists():
        logger.info("Loading feature tensor X from cache …")
        return np.load(str(cache_path))

    logger.info("Extracting DE + PSD features for %d epochs …", len(epochs))
    X = extract_features(
        epochs,
        fs=cfg.preprocess.target_fs,
        nperseg=256,
        bands=FREQ_BANDS,
    )
    np.save(str(cache_path), X)
    logger.info("Feature tensor X shape: %s — saved to cache.", X.shape)
    return X
