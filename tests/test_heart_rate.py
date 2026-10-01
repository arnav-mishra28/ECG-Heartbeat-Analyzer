"""
Tests for heart-rate and HRV calculations.

Uses deterministic inputs where the correct answers are known analytically.
"""

import numpy as np
import pytest

from src.heart_rate import compute_rr_intervals, compute_heart_rate


FS = 360  # sampling rate


class TestRRIntervals:
    """Tests for RR interval computation."""

    def test_regular_rhythm(self):
        """Evenly spaced peaks at 1-second intervals → RR = 1.0 s for all."""
        peaks = np.array([360, 720, 1080, 1440])  # every 360 samples = 1 sec
        rr, rr_times = compute_rr_intervals(peaks, fs=FS)
        np.testing.assert_allclose(rr, [1.0, 1.0, 1.0], atol=1e-10)

    def test_output_length(self):
        """N peaks → N-1 RR intervals."""
        peaks = np.arange(0, 3600, 360)
        rr, _ = compute_rr_intervals(peaks, fs=FS)
        assert len(rr) == len(peaks) - 1

    def test_fewer_than_two_peaks(self):
        """Fewer than 2 peaks → empty arrays."""
        rr, rr_times = compute_rr_intervals(np.array([100]), fs=FS)
        assert len(rr) == 0
        assert len(rr_times) == 0


class TestHeartRate:
    """Tests for BPM and HRV computation."""

    def test_sixty_bpm(self):
        """RR = 1.0 s → 60 BPM exactly."""
        peaks = np.arange(0, 3600, 360)  # 1 beat per second
        stats = compute_heart_rate(peaks, fs=FS)
        assert abs(stats.mean_bpm - 60.0) < 0.1

    def test_120_bpm(self):
        """RR = 0.5 s → 120 BPM."""
        peaks = np.arange(0, 3600, 180)  # 0.5 s intervals
        stats = compute_heart_rate(peaks, fs=FS)
        assert abs(stats.mean_bpm - 120.0) < 0.1

    def test_sdnn_constant_rhythm(self):
        """Perfectly regular rhythm → SDNN ≈ 0."""
        peaks = np.arange(0, 36000, 360)
        stats = compute_heart_rate(peaks, fs=FS)
        assert stats.sdnn < 0.01  # essentially zero

    def test_rmssd_constant_rhythm(self):
        """Perfectly regular rhythm → RMSSD ≈ 0."""
        peaks = np.arange(0, 36000, 360)
        stats = compute_heart_rate(peaks, fs=FS)
        assert stats.rmssd < 0.01

    def test_sdnn_variable_rhythm(self):
        """Variable RR intervals should produce a non-zero SDNN."""
        # Alternate between 0.8 s and 1.2 s intervals
        peaks = [0]
        intervals_samples = [int(0.8 * FS), int(1.2 * FS)] * 20
        for interval in intervals_samples:
            peaks.append(peaks[-1] + interval)
        peaks = np.array(peaks)

        stats = compute_heart_rate(peaks, fs=FS)
        assert stats.sdnn > 50  # should be substantial (in ms)

    def test_min_max_bpm(self):
        """Min and max BPM should bracket the mean."""
        peaks = np.array([0, 360, 900, 1260, 1800])
        stats = compute_heart_rate(peaks, fs=FS)
        assert stats.min_bpm <= stats.mean_bpm <= stats.max_bpm
