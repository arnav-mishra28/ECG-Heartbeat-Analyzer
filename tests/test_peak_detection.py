"""
Tests for R-peak detection.

Uses a synthetic ECG-like signal where we know exactly where the peaks are,
so we can verify the detector without depending on real data.
"""

import numpy as np
import pytest

from src.peak_detection import detect_r_peaks


FS = 360  # sampling rate


def _make_synthetic_ecg(
    fs: float = FS,
    duration_sec: float = 30.0,
    heart_rate_bpm: float = 72.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic ECG-like signal with known R-peak locations.

    Returns
    -------
    signal : np.ndarray
        Synthetic ECG signal.
    true_peaks : np.ndarray
        Sample indices of the true R-peaks.
    """
    N = int(duration_sec * fs)
    t = np.arange(N) / fs
    signal = np.zeros(N)

    # Place Gaussian "QRS complexes" at regular intervals
    rr_sec = 60.0 / heart_rate_bpm
    rr_samples = int(rr_sec * fs)

    # QRS width: ~80 ms
    qrs_width = 0.04  # std dev in seconds

    true_peaks = []
    peak_sample = rr_samples  # start after one RR interval
    while peak_sample < N - rr_samples:
        true_peaks.append(peak_sample)
        # Add a Gaussian pulse centered at the peak
        for i in range(max(0, peak_sample - int(0.2 * fs)),
                       min(N, peak_sample + int(0.2 * fs))):
            dt = (i - peak_sample) / fs
            signal[i] += 1.5 * np.exp(-(dt ** 2) / (2 * qrs_width ** 2))
        peak_sample += rr_samples

    true_peaks = np.array(true_peaks, dtype=int)

    # Add a small amount of noise
    rng = np.random.default_rng(42)
    signal += 0.05 * rng.standard_normal(N)

    return signal, true_peaks


class TestRPeakDetection:
    """Tests for the Pan-Tompkins-style R-peak detector."""

    def test_returns_array_of_ints(self):
        signal, _ = _make_synthetic_ecg(duration_sec=10.0)
        peaks = detect_r_peaks(signal, FS)
        assert isinstance(peaks, np.ndarray)
        assert peaks.dtype in [np.int32, np.int64, int]

    def test_valid_indices(self):
        """All detected peaks should be valid sample indices."""
        signal, _ = _make_synthetic_ecg(duration_sec=10.0)
        peaks = detect_r_peaks(signal, FS)
        assert np.all(peaks >= 0)
        assert np.all(peaks < len(signal))

    def test_sorted_output(self):
        """Peaks should be in ascending order."""
        signal, _ = _make_synthetic_ecg(duration_sec=10.0)
        peaks = detect_r_peaks(signal, FS)
        assert np.all(np.diff(peaks) > 0)

    def test_detects_most_peaks(self):
        """On a clean synthetic signal, we should detect >90% of true peaks."""
        signal, true_peaks = _make_synthetic_ecg(duration_sec=30.0)
        detected = detect_r_peaks(signal, FS)

        # Match with tolerance of ±50 ms
        tolerance = int(0.05 * FS)
        matched = 0
        for tp in true_peaks:
            diffs = np.abs(detected.astype(np.int64) - int(tp))
            if np.min(diffs) <= tolerance:
                matched += 1

        sensitivity = matched / len(true_peaks)
        assert sensitivity > 0.90, f"Sensitivity too low: {sensitivity:.3f}"

    def test_low_false_positives(self):
        """On a clean synthetic signal, >90% of detections should match a true peak."""
        signal, true_peaks = _make_synthetic_ecg(duration_sec=30.0)
        detected = detect_r_peaks(signal, FS)

        tolerance = int(0.05 * FS)
        true_pos = 0
        for d in detected:
            diffs = np.abs(true_peaks.astype(np.int64) - int(d))
            if np.min(diffs) <= tolerance:
                true_pos += 1

        ppv = true_pos / len(detected) if len(detected) > 0 else 0
        assert ppv > 0.90, f"PPV too low: {ppv:.3f}"

    def test_respects_refractory_period(self):
        """No two detected peaks should be closer than the refractory period."""
        signal, _ = _make_synthetic_ecg(duration_sec=10.0)
        peaks = detect_r_peaks(signal, FS, refractory_sec=0.200)

        min_gap = int(0.200 * FS)
        if len(peaks) > 1:
            gaps = np.diff(peaks)
            assert np.all(gaps >= min_gap), f"Min gap: {np.min(gaps)}, expected >= {min_gap}"

    def test_empty_signal(self):
        """An all-zeros signal should produce no peaks."""
        signal = np.zeros(3600)
        peaks = detect_r_peaks(signal, FS)
        assert len(peaks) == 0
