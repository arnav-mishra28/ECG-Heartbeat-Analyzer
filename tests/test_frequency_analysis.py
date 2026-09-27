"""Tests for frequency domain analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.frequency_analysis import compute_fft, compute_power_spectrum


class TestFrequencyAnalysis:
    def test_pure_sine_peak_frequency(self):
        fs = 360.0
        duration = 2.0
        t = np.arange(int(fs * duration)) / fs
        f0 = 20.0
        signal = np.sin(2 * np.pi * f0 * t)

        freqs, magnitude = compute_fft(signal, fs)

        # Peak frequency should match f0 closely
        peak_freq = freqs[np.argmax(magnitude)]
        assert abs(peak_freq - f0) < 0.6
        # Sine amplitude 1 -> single-sided magnitude ~1.0
        assert magnitude[np.argmax(magnitude)] == pytest.approx(1.0, rel=0.05)

    def test_dc_component(self):
        fs = 100.0
        signal = np.ones(500) * 3.5

        freqs, magnitude = compute_fft(signal, fs)
        assert freqs[0] == 0.0
        assert magnitude[0] == pytest.approx(3.5, rel=1e-3)
        assert np.max(magnitude[1:]) == pytest.approx(0.0, abs=1e-5)

    def test_power_spectrum(self):
        fs = 200.0
        signal = np.sin(2 * np.pi * 15 * np.arange(400) / fs)

        freqs, psd = compute_power_spectrum(signal, fs)
        assert len(freqs) == len(psd)
        assert np.all(psd >= 0.0)
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 15.0) < 1.0
