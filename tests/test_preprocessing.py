"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import preprocess_ecg


class TestPreprocessing:
    def test_output_shape_and_type(self):
        fs = 360.0
        signal = np.random.randn(1000)
        clean = preprocess_ecg(signal, fs=fs)

        assert clean.shape == signal.shape
        assert isinstance(clean, np.ndarray)

    def test_removes_dc_and_50hz(self):
        fs = 360.0
        t = np.arange(int(fs * 4)) / fs
        # Baseline wander (0.2 Hz) + 50 Hz powerline + HF noise (100 Hz)
        noise = 5.0 + 2.0 * np.sin(2 * np.pi * 0.2 * t) + 1.5 * np.sin(2 * np.pi * 50 * t) + 0.8 * np.sin(2 * np.pi * 100 * t)
        clean = preprocess_ecg(noise, fs=fs)

        # Standard deviation and mean should be drastically reduced
        assert abs(np.mean(clean)) < 0.1
        assert np.std(clean) < np.std(noise) * 0.2
