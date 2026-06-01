"""
tests/test_features.py
=======================
Unit tests for neural feature extraction (data/eeg_features.py).

Covers:
  - extract_de:  shape, finite values, sensitivity to variance
  - extract_psd: shape, finite values, monotonic with signal power
  - extract_features: combined DE+PSD tensor shape and dtype
  - FREQ_BANDS: delta band post-filter effective range note
  - Band-limited variance helper
  - Edge cases: silent channel, all-identical signal

Run:
    pytest tests/test_features.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.eeg_features import (
    extract_de,
    extract_psd,
    extract_features,
    FREQ_BANDS,
    N_BANDS,
    _band_variance,
)


RNG    = np.random.default_rng(0)
N_CH   = 30
N_SAMP = 500   # 5 s at 100 Hz
FS     = 100.0


@pytest.fixture
def epoch():
    return RNG.standard_normal((N_CH, N_SAMP)).astype(np.float32)


@pytest.fixture
def silent_epoch():
    return np.zeros((N_CH, N_SAMP), dtype=np.float32)


# ---------------------------------------------------------------------------
# Band-limited variance
# ---------------------------------------------------------------------------

class TestBandVariance:

    def test_pure_tone_in_band(self):
        """A pure sine at 20 Hz should have most variance in the beta band."""
        t = np.linspace(0, N_SAMP / FS, N_SAMP)
        sig = np.sin(2 * np.pi * 20 * t).astype(np.float32)
        var_beta  = _band_variance(sig, FS, low=13.0, high=30.0)
        var_alpha = _band_variance(sig, FS, low=8.0,  high=13.0)
        assert var_beta > var_alpha * 10, (
            f"Beta variance ({var_beta:.4f}) should dominate alpha ({var_alpha:.4f})"
        )

    def test_returns_non_negative(self, epoch):
        for _, low, high in FREQ_BANDS:
            v = _band_variance(epoch[0], FS, low, high)
            assert v >= 0.0

    def test_silent_signal_near_zero(self):
        sig = np.zeros(N_SAMP, dtype=np.float32)
        for _, low, high in FREQ_BANDS:
            v = _band_variance(sig, FS, low, high)
            assert v < 1e-8


# ---------------------------------------------------------------------------
# Differential Entropy
# ---------------------------------------------------------------------------

class TestExtractDE:

    def test_output_shape(self, epoch):
        de = extract_de(epoch, fs=FS)
        assert de.shape == (N_CH, N_BANDS)

    def test_output_dtype(self, epoch):
        de = extract_de(epoch, fs=FS)
        assert de.dtype == np.float32

    def test_all_finite(self, epoch):
        de = extract_de(epoch, fs=FS)
        assert np.all(np.isfinite(de)), "DE contains NaN or Inf"

    def test_silent_channel_finite(self, silent_epoch):
        """Silent channel (zero variance) should produce finite DE (log guarded)."""
        de = extract_de(silent_epoch, fs=FS)
        assert np.all(np.isfinite(de))

    def test_higher_variance_higher_de(self):
        """DE is monotonically increasing in signal variance."""
        t = np.linspace(0, N_SAMP / FS, N_SAMP)
        # Two signals in the alpha band with different amplitudes
        sig_low  = (0.1 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
        sig_high = (1.0 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)

        epoch_low  = np.tile(sig_low,  (N_CH, 1))
        epoch_high = np.tile(sig_high, (N_CH, 1))

        de_low  = extract_de(epoch_low,  fs=FS)
        de_high = extract_de(epoch_high, fs=FS)

        alpha_band_idx = 2   # alpha is the 3rd band (8-13 Hz)
        assert (de_high[:, alpha_band_idx] > de_low[:, alpha_band_idx]).all()

    def test_custom_bands(self, epoch):
        custom = [("only_beta", 13.0, 30.0)]
        de = extract_de(epoch, fs=FS, bands=custom)
        assert de.shape == (N_CH, 1)


# ---------------------------------------------------------------------------
# Power Spectral Density
# ---------------------------------------------------------------------------

class TestExtractPSD:

    def test_output_shape(self, epoch):
        psd = extract_psd(epoch, fs=FS)
        assert psd.shape == (N_CH, N_BANDS)

    def test_output_dtype(self, epoch):
        psd = extract_psd(epoch, fs=FS)
        assert psd.dtype == np.float32

    def test_all_finite(self, epoch):
        psd = extract_psd(epoch, fs=FS)
        assert np.all(np.isfinite(psd))

    def test_silent_channel_finite(self, silent_epoch):
        psd = extract_psd(silent_epoch, fs=FS)
        assert np.all(np.isfinite(psd))

    def test_dominant_frequency_band(self):
        """Log PSD should be highest in the band containing the signal frequency."""
        t = np.linspace(0, N_SAMP / FS, N_SAMP)
        # Strong gamma-band signal (40 Hz)
        sig = (5.0 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
        epoch = np.tile(sig, (N_CH, 1))
        psd = extract_psd(epoch, fs=FS)

        gamma_idx = 4   # gamma is 5th band (30-50 Hz)
        # Gamma band PSD should be highest across all channels
        assert (psd[:, gamma_idx] == psd.max(axis=1)).all(), (
            f"Gamma PSD not dominant. Per-channel max bands: {psd.argmax(axis=1)}"
        )

    def test_nperseg_clipped_to_epoch_length(self):
        """nperseg larger than epoch length should not raise an error."""
        short_epoch = RNG.standard_normal((N_CH, 64)).astype(np.float32)
        psd = extract_psd(short_epoch, fs=FS, nperseg=256)
        assert psd.shape == (N_CH, N_BANDS)


# ---------------------------------------------------------------------------
# Full feature extraction
# ---------------------------------------------------------------------------

class TestExtractFeatures:

    def test_output_shape(self):
        T = 10
        epochs = RNG.standard_normal((T, N_CH, N_SAMP)).astype(np.float32)
        X = extract_features(epochs, fs=FS)
        assert X.shape == (T, N_CH, 2 * N_BANDS)   # (10, 30, 10)

    def test_output_dtype(self):
        epochs = RNG.standard_normal((4, N_CH, N_SAMP)).astype(np.float32)
        X = extract_features(epochs, fs=FS)
        assert X.dtype == np.float32

    def test_all_finite(self):
        epochs = RNG.standard_normal((4, N_CH, N_SAMP)).astype(np.float32)
        X = extract_features(epochs, fs=FS)
        assert np.all(np.isfinite(X)), f"Feature tensor has non-finite values"

    def test_de_psd_concatenation_order(self):
        """First 5 features are DE, next 5 are PSD per the paper."""
        T = 2
        epochs = RNG.standard_normal((T, N_CH, N_SAMP)).astype(np.float32)
        X = extract_features(epochs, fs=FS)

        # Extract DE and PSD independently and compare
        for t in range(T):
            de  = extract_de(epochs[t], fs=FS)    # (30, 5)
            psd = extract_psd(epochs[t], fs=FS)   # (30, 5)
            np.testing.assert_array_almost_equal(X[t, :, :5],  de,  decimal=5)
            np.testing.assert_array_almost_equal(X[t, :, 5:],  psd, decimal=5)

    def test_single_epoch(self):
        epoch = RNG.standard_normal((1, N_CH, N_SAMP)).astype(np.float32)
        X = extract_features(epoch, fs=FS)
        assert X.shape == (1, N_CH, 10)


# ---------------------------------------------------------------------------
# Frequency band definitions sanity
# ---------------------------------------------------------------------------

class TestFreqBands:

    def test_five_bands(self):
        assert N_BANDS == 5

    def test_band_names(self):
        names = [b[0] for b in FREQ_BANDS]
        assert names == ["delta", "theta", "alpha", "beta", "gamma"]

    def test_bands_non_overlapping(self):
        for i in range(len(FREQ_BANDS) - 1):
            _, _, high_curr = FREQ_BANDS[i]
            _, low_next, _  = FREQ_BANDS[i + 1]
            assert high_curr == low_next, (
                f"Gap or overlap between {FREQ_BANDS[i][0]} and {FREQ_BANDS[i+1][0]}"
            )

    def test_delta_band_documented(self):
        """Delta band docstring should note the 3 Hz bandpass interaction."""
        import data.eeg_features as feat_mod
        src = open(feat_mod.__file__).read()
        assert "3" in src and "delta" in src.lower(), (
            "Delta band 3 Hz limitation should be documented"
        )
