"""
Tests for digital filters.

Verifies:
  - Output length matches input length (zero-phase filtering preserves length)
  - Filters actually attenuate the expected frequency ranges
  - Filters work on synthetic signals without errors
"""

import numpy as np
import pytest

from src.filters import highpass_filter, notch_filter, lowpass_filter


FS = 360  # MIT-BIH sampling rate


def _make_sine(freq: float, duration: float = 5.0, fs: float = FS) -> np.ndarray:
    """Generate a pure sine wave at the given frequency."""
    t = np.arange(int(duration * fs)) / fs
    return np.sin(2 * np.pi * freq * t)


class TestHighpassFilter:
    """Tests for the high-pass (baseline wander removal) filter."""

    def test_output_length(self):
        signal = np.random.randn(3600)
        out = highpass_filter(signal, FS, cutoff=0.5, order=4)
        assert len(out) == len(signal)

    def test_attenuates_dc(self):
        """A DC signal (0 Hz) should be almost entirely removed."""
        signal = np.ones(3600)
        out = highpass_filter(signal, FS, cutoff=0.5, order=4)
        assert np.max(np.abs(out)) < 0.01

    def test_passes_qrs_frequencies(self):
        """A 10 Hz sine (typical QRS energy) should pass through."""
        signal = _make_sine(10.0)
        out = highpass_filter(signal, FS, cutoff=0.5, order=4)
        # Energy should be largely preserved (>90%)
        assert np.std(out) / np.std(signal) > 0.9

    def test_attenuates_baseline_wander(self):
        """A 0.1 Hz sine (baseline wander) should be strongly attenuated."""
        signal = _make_sine(0.1, duration=30.0)
        out = highpass_filter(signal, FS, cutoff=0.5, order=4)
        assert np.std(out) / np.std(signal) < 0.1


class TestNotchFilter:
    """Tests for the notch (power-line interference) filter."""

    def test_output_length(self):
        signal = np.random.randn(3600)
        out = notch_filter(signal, FS, freq=50.0, quality=30.0)
        assert len(out) == len(signal)

    def test_removes_50hz(self):
        """A pure 50 Hz sine should be almost completely removed."""
        signal = _make_sine(50.0)
        out = notch_filter(signal, FS, freq=50.0, quality=30.0)
        # Threshold set to 0.10 to account for edge transients from
        # zero-phase (filtfilt) filtering on short signals.
        assert np.std(out) / np.std(signal) < 0.10

    def test_preserves_nearby_frequencies(self):
        """Frequencies well away from 50 Hz should pass through."""
        signal = _make_sine(10.0)
        out = notch_filter(signal, FS, freq=50.0, quality=30.0)
        assert np.std(out) / np.std(signal) > 0.95


class TestLowpassFilter:
    """Tests for the low-pass (high-frequency noise removal) filter."""

    def test_output_length(self):
        signal = np.random.randn(3600)
        out = lowpass_filter(signal, FS, cutoff=40.0, order=4)
        assert len(out) == len(signal)

    def test_attenuates_high_frequency(self):
        """A 100 Hz sine should be strongly attenuated."""
        signal = _make_sine(100.0)
        out = lowpass_filter(signal, FS, cutoff=40.0, order=4)
        assert np.std(out) / np.std(signal) < 0.05

    def test_passes_ecg_band(self):
        """A 10 Hz sine should pass through."""
        signal = _make_sine(10.0)
        out = lowpass_filter(signal, FS, cutoff=40.0, order=4)
        assert np.std(out) / np.std(signal) > 0.95
