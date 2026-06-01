"""
tests/test_preprocessing.py
============================
Unit tests for the EEG preprocessing pipeline (data/eeg_preprocessing.py).

Covers:
  - bandpass_filter: zero-phase property, correct frequency attenuation
  - polyphase_downsample: output length and shape
  - asr_clean: artifact suppression reduces extreme values
  - ica_artifact_removal: output shape preserved
  - segment_epochs: correct shape and no sample loss within complete epochs
  - load_subject_raw: shape normalisation logic (_normalize_eeg_shape)

Run:
    pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.eeg_preprocessing import (
    bandpass_filter,
    polyphase_downsample,
    asr_clean,
    ica_artifact_removal,
    segment_epochs,
    _normalize_eeg_shape,
    _kurtosis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N_CH = 30
N_SAMP = 2000       # 20 s at 100 Hz
FS = 500   # native rate for 2000-sample test signals


@pytest.fixture
def white_noise():
    """White noise signal (30 channels, 2000 samples)."""
    return RNG.standard_normal((N_CH, N_SAMP)).astype(np.float32)


@pytest.fixture
def sinusoidal_signal():
    """
    Signal with known frequency components:
      - 1 Hz (below 3 Hz passband → should be attenuated)
      - 20 Hz (inside passband → should pass)
      - 60 Hz (above 50 Hz passband → should be attenuated)
    """
    t = np.linspace(0, N_SAMP / FS, N_SAMP)
    sig = (
        np.sin(2 * np.pi * 1  * t) +   # 1 Hz — should be suppressed
        np.sin(2 * np.pi * 20 * t) +   # 20 Hz — should pass
        np.sin(2 * np.pi * 60 * t)     # 60 Hz — should be suppressed
    )
    return np.tile(sig, (N_CH, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:

    def test_output_shape(self, white_noise):
        out = bandpass_filter(white_noise, low=3.0, high=50.0, fs=FS)
        assert out.shape == white_noise.shape

    def test_output_dtype(self, white_noise):
        out = bandpass_filter(white_noise, low=3.0, high=50.0, fs=FS)
        assert out.dtype in (np.float32, np.float64)

    def test_attenuates_below_passband(self, sinusoidal_signal):
        """Power at 1 Hz (below 3 Hz cutoff) should be largely attenuated."""
        out = bandpass_filter(sinusoidal_signal, low=3.0, high=50.0, fs=FS)
        freqs = np.fft.rfftfreq(N_SAMP, d=1.0 / FS)
        spectrum_in  = np.abs(np.fft.rfft(sinusoidal_signal[0]))
        spectrum_out = np.abs(np.fft.rfft(out[0]))
        bin_1hz = np.argmin(np.abs(freqs - 1.0))
        # Output power at 1 Hz must be < 10% of input power at 1 Hz
        assert spectrum_out[bin_1hz] < 0.1 * spectrum_in[bin_1hz], (
            f"1 Hz not sufficiently attenuated: "
            f"in={spectrum_in[bin_1hz]:.2f} out={spectrum_out[bin_1hz]:.2f}"
        )

    def test_attenuates_above_passband(self, sinusoidal_signal):
        """Power at 60 Hz (above 50 Hz cutoff) should be largely attenuated."""
        out = bandpass_filter(sinusoidal_signal, low=3.0, high=50.0, fs=FS)
        freqs = np.fft.rfftfreq(N_SAMP, d=1.0 / FS)
        spectrum_in  = np.abs(np.fft.rfft(sinusoidal_signal[0]))
        spectrum_out = np.abs(np.fft.rfft(out[0]))
        bin_60hz = np.argmin(np.abs(freqs - 60.0))
        assert spectrum_out[bin_60hz] < 0.1 * spectrum_in[bin_60hz]

    def test_preserves_passband(self, sinusoidal_signal):
        """Power at 20 Hz (inside passband) should be largely preserved."""
        out = bandpass_filter(sinusoidal_signal, low=3.0, high=50.0, fs=FS)
        freqs = np.fft.rfftfreq(N_SAMP, d=1.0 / FS)
        spectrum_in  = np.abs(np.fft.rfft(sinusoidal_signal[0]))
        spectrum_out = np.abs(np.fft.rfft(out[0]))
        bin_20hz = np.argmin(np.abs(freqs - 20.0))
        # At least 80% of power preserved inside the passband
        assert spectrum_out[bin_20hz] >= 0.8 * spectrum_in[bin_20hz]

    def test_zero_phase_symmetry(self, white_noise):
        """
        Zero-phase filter (sosfiltfilt) introduces no group delay.
        A pure cosine at a passband frequency should remain a cosine
        (not be shifted) after filtering.
        """
        t = np.linspace(0, N_SAMP / FS, N_SAMP)
        cos_10hz = np.cos(2 * np.pi * 10 * t).astype(np.float32)
        sig = np.tile(cos_10hz, (N_CH, 1))
        out = bandpass_filter(sig, low=3.0, high=50.0, fs=FS)
        # Cross-correlation peak should be near lag=0 (no phase shift)
        mid = N_SAMP // 2
        xcorr = np.correlate(sig[0], out[0], mode="full")
        lag = xcorr.argmax() - (N_SAMP - 1)
        assert abs(lag) <= 2, f"Phase lag too large: {lag} samples"


# ---------------------------------------------------------------------------
# polyphase_downsample
# ---------------------------------------------------------------------------

class TestPolyphaseDownsample:

    def test_output_length(self, white_noise):
        out = polyphase_downsample(white_noise, up=1, down=5)
        expected_len = N_SAMP // 5
        assert out.shape[-1] == expected_len, f"Expected {expected_len}, got {out.shape[-1]}"

    def test_channel_dimension_preserved(self, white_noise):
        out = polyphase_downsample(white_noise, up=1, down=5)
        assert out.shape[0] == N_CH

    def test_bandlimited_energy_preserved_approximately(self):
        """A passband sinusoid should keep approximately the same RMS energy."""
        t = np.arange(N_SAMP) / FS
        sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        bandlimited = np.tile(sig, (N_CH, 1))
        out = polyphase_downsample(bandlimited, up=1, down=5)
        e_in  = np.mean(bandlimited ** 2)
        e_out = np.mean(out ** 2)
        assert 0.8 * e_in <= e_out <= 1.2 * e_in


# ---------------------------------------------------------------------------
# asr_clean
# ---------------------------------------------------------------------------

class TestASRClean:

    def test_output_shape(self, white_noise):
        out = asr_clean(white_noise, threshold=5.0)
        assert out.shape == white_noise.shape

    def test_extreme_values_suppressed(self):
        """Signal with injected spike should have reduced max amplitude."""
        sig = RNG.standard_normal((N_CH, N_SAMP)).astype(np.float32)
        # Inject a large spike in the second half
        sig[:, N_SAMP // 2] = 100.0
        out = asr_clean(sig, threshold=5.0, ref_fraction=0.3)
        assert np.max(np.abs(out)) < np.max(np.abs(sig)), \
            "ASR should suppress extreme spike"

    def test_clean_signal_mostly_preserved(self, white_noise):
        """Clean signal should be mostly unchanged (< 5% RMS change)."""
        out = asr_clean(white_noise, threshold=10.0)
        rms_in  = np.sqrt(np.mean(white_noise ** 2))
        rms_out = np.sqrt(np.mean(out ** 2))
        assert abs(rms_in - rms_out) / rms_in < 0.05


# ---------------------------------------------------------------------------
# ica_artifact_removal
# ---------------------------------------------------------------------------

class TestICARemoval:

    def test_output_shape(self, white_noise):
        out = ica_artifact_removal(white_noise, n_components=10, random_state=42)
        assert out.shape == white_noise.shape

    def test_non_gaussian_component_removed(self):
        """
        A signal with one heavily kurtotic (non-Gaussian) channel mixed in
        should have reduced kurtosis after ICA removal.
        """
        sig = RNG.standard_normal((N_CH, N_SAMP)).astype(np.float32)
        # Inject a high-kurtosis artifact (squared Gaussian → kurtosis = 6)
        artifact = (RNG.standard_normal(N_SAMP) ** 2).astype(np.float32)
        artifact = artifact / artifact.std() * 10   # amplify
        sig[0] += artifact

        kurt_before = float(np.abs(_kurtosis(sig.T)).max())
        out = ica_artifact_removal(sig, n_components=20, random_state=42)
        kurt_after  = float(np.abs(_kurtosis(out.T)).max())
        # Kurtosis should decrease (artifact removed)
        assert kurt_after <= kurt_before, (
            f"Kurtosis increased after ICA: before={kurt_before:.1f}, after={kurt_after:.1f}"
        )


# ---------------------------------------------------------------------------
# segment_epochs
# ---------------------------------------------------------------------------

class TestSegmentEpochs:

    def test_output_shape(self, white_noise):
        epoch_len = 500   # 5 s at 100 Hz
        epochs = segment_epochs(white_noise, epoch_len)
        n_expected = N_SAMP // epoch_len
        assert epochs.shape == (n_expected, N_CH, epoch_len)

    def test_no_overlap(self, white_noise):
        """First sample of epoch i+1 == sample right after last sample of epoch i."""
        epoch_len = 500
        epochs = segment_epochs(white_noise, epoch_len)
        for i in range(epochs.shape[0] - 1):
            last_sample  = white_noise[:, (i + 1) * epoch_len - 1]
            first_sample = white_noise[:, (i + 1) * epoch_len]
            # First sample of next epoch comes from the expected position
            np.testing.assert_array_equal(epochs[i + 1, :, 0], first_sample)

    def test_trailing_samples_discarded(self):
        """Samples that don't fill a complete epoch should be silently discarded."""
        sig = RNG.standard_normal((N_CH, 1050)).astype(np.float32)
        epoch_len = 500
        epochs = segment_epochs(sig, epoch_len)
        # 1050 // 500 = 2 complete epochs
        assert epochs.shape[0] == 2


# ---------------------------------------------------------------------------
# _normalize_eeg_shape
# ---------------------------------------------------------------------------

class TestNormalizeEEGShape:

    def test_trials_last_axis(self):
        """(10000, 30, 200) → (200, 30, 10000)"""
        raw = np.zeros((10000, 30, 200))
        out = _normalize_eeg_shape(raw)
        assert out.shape == (200, 30, 10000)

    def test_trials_first_axis_unchanged(self):
        """(200, 30, 10000) → (200, 30, 10000) — already correct."""
        raw = np.zeros((200, 30, 10000))
        out = _normalize_eeg_shape(raw)
        assert out.shape == (200, 30, 10000)

    def test_unique_axis_sizes(self):
        """Function correctly identifies which axis is trials (200)."""
        # Arbitrary permutation with distinct sizes
        raw = np.zeros((30, 200, 10000))   # channels, trials, samples
        out = _normalize_eeg_shape(raw)
        assert out.shape[0] == 200, f"Trial axis not found: {out.shape}"
