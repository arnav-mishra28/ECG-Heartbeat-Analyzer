"""
Heart-rate and heart-rate-variability (HRV) analysis.

Given detected R-peak locations, this module calculates:
  - RR intervals (time between consecutive R-peaks)
  - Instantaneous and average heart rate (BPM)
  - Basic time-domain HRV statistics (SDNN, RMSSD)

Disclaimer
----------
This is an educational/research prototype, **not** a diagnostic medical
device.  Clinical HRV analysis requires additional preprocessing, artifact
rejection, and standardized recording conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src import config

logger = logging.getLogger(__name__)


@dataclass
class HeartRateStats:
    """Container for heart-rate and HRV results."""

    # RR intervals
    rr_intervals: np.ndarray        # seconds
    rr_times: np.ndarray            # time of each RR interval (midpoint, seconds)

    # BPM
    instantaneous_bpm: np.ndarray   # BPM for each RR interval
    mean_bpm: float
    median_bpm: float
    min_bpm: float
    max_bpm: float
    std_bpm: float

    # HRV — time-domain
    sdnn: float   # Standard Deviation of NN (≈RR) intervals (ms)
    rmssd: float  # Root Mean Square of Successive Differences (ms)

    def __repr__(self) -> str:
        return (
            f"HeartRateStats("
            f"mean_bpm={self.mean_bpm:.1f}, "
            f"median_bpm={self.median_bpm:.1f}, "
            f"SDNN={self.sdnn:.1f} ms, "
            f"RMSSD={self.rmssd:.1f} ms)"
        )


def compute_rr_intervals(
    r_peak_samples: np.ndarray,
    fs: float = config.SAMPLING_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert R-peak sample indices to RR intervals (seconds).

    Parameters
    ----------
    r_peak_samples : np.ndarray
        Sorted 1-D array of sample indices of detected R-peaks.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    rr_intervals : np.ndarray
        Time differences between consecutive R-peaks (seconds).
    rr_times : np.ndarray
        Midpoint time of each interval (seconds), useful for plotting
        heart rate vs. time.
    """
    if len(r_peak_samples) < 2:
        return np.array([]), np.array([])

    # RR interval = difference in sample indices / fs
    rr_samples = np.diff(r_peak_samples)
    rr_intervals = rr_samples / fs

    # Midpoint time for each interval
    rr_times = (r_peak_samples[:-1] + rr_samples / 2.0) / fs

    return rr_intervals, rr_times


def compute_heart_rate(
    r_peak_samples: np.ndarray,
    fs: float = config.SAMPLING_RATE,
) -> HeartRateStats:
    """Compute heart rate and basic HRV statistics from R-peak locations.

    Parameters
    ----------
    r_peak_samples : np.ndarray
        Sorted 1-D array of sample indices.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    HeartRateStats
        Complete heart-rate and HRV analysis results.
    """
    rr_intervals, rr_times = compute_rr_intervals(r_peak_samples, fs)

    if len(rr_intervals) == 0:
        logger.warning("No RR intervals — fewer than 2 R-peaks detected.")
        empty = np.array([])
        return HeartRateStats(
            rr_intervals=empty, rr_times=empty,
            instantaneous_bpm=empty,
            mean_bpm=0, median_bpm=0, min_bpm=0, max_bpm=0, std_bpm=0,
            sdnn=0, rmssd=0,
        )

    # ── BPM ──────────────────────────────────────────────────────────
    instantaneous_bpm = 60.0 / rr_intervals

    mean_bpm = float(np.mean(instantaneous_bpm))
    median_bpm = float(np.median(instantaneous_bpm))
    min_bpm = float(np.min(instantaneous_bpm))
    max_bpm = float(np.max(instantaneous_bpm))
    std_bpm = float(np.std(instantaneous_bpm, ddof=1)) if len(instantaneous_bpm) > 1 else 0.0

    # ── HRV — time-domain ────────────────────────────────────────────
    # SDNN: standard deviation of NN intervals (in milliseconds)
    rr_ms = rr_intervals * 1000.0
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0

    # RMSSD: root mean square of successive RR differences (in ms)
    if len(rr_intervals) > 1:
        successive_diffs = np.diff(rr_ms)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
    else:
        rmssd = 0.0

    logger.info(
        "Heart rate: mean=%.1f BPM, median=%.1f BPM, SDNN=%.1f ms, RMSSD=%.1f ms",
        mean_bpm, median_bpm, sdnn, rmssd,
    )

    return HeartRateStats(
        rr_intervals=rr_intervals,
        rr_times=rr_times,
        instantaneous_bpm=instantaneous_bpm,
        mean_bpm=mean_bpm,
        median_bpm=median_bpm,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        std_bpm=std_bpm,
        sdnn=sdnn,
        rmssd=rmssd,
    )


def print_heart_rate_summary(stats: HeartRateStats) -> None:
    """Print a human-readable summary of heart-rate analysis."""
    print("-" * 50)
    print("  Heart Rate Analysis")
    print("-" * 50)
    print(f"  RR intervals     : {len(stats.rr_intervals):,}")
    print(f"  Mean BPM         : {stats.mean_bpm:.1f}")
    print(f"  Median BPM       : {stats.median_bpm:.1f}")
    print(f"  Min BPM          : {stats.min_bpm:.1f}")
    print(f"  Max BPM          : {stats.max_bpm:.1f}")
    print(f"  Std BPM          : {stats.std_bpm:.2f}")
    print(f"  SDNN             : {stats.sdnn:.1f} ms")
    print(f"  RMSSD            : {stats.rmssd:.1f} ms")
    print("-" * 50)
