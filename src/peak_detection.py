"""
R-peak detection using a Pan–Tompkins-style algorithm.

Pipeline
--------
Filtered ECG
    → Bandpass (5–15 Hz)
    → Differentiation
    → Squaring
    → Moving-window integration
    → Adaptive thresholding
    → Refractory period enforcement
    → R-peak refinement (back-search in original ECG)
    → Final R-peak sample indices

Design rationale
----------------
* **Bandpass 5–15 Hz** emphasises QRS energy and suppresses P/T waves.
* **Differentiation** highlights rapid signal changes (the QRS complex).
* **Squaring** makes all values positive and amplifies large slopes.
* **Moving-window integration** smooths the energy envelope so each
  QRS complex produces a single broad hump.
* **Adaptive threshold** adjusts to signal amplitude — works across
  recordings with different gain levels.
* **Refractory period** prevents double-detections (the heart cannot
  produce two R-peaks within ~200 ms).
* **Refinement** maps detection-stage candidates back to the true
  R-peak (maximum) in the original/cleaned ECG.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, filtfilt

from src import config

logger = logging.getLogger(__name__)


def bandpass_filter(
    signal: np.ndarray,
    fs: float = config.SAMPLING_RATE,
    low: float = config.DETECTION_LOW_CUTOFF,
    high: float = config.DETECTION_HIGH_CUTOFF,
    order: int = 2,
) -> np.ndarray:
    """Apply a 2nd-order Butterworth bandpass filter (5-15 Hz)."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def derivative(signal: np.ndarray, fs: float = config.SAMPLING_RATE) -> np.ndarray:
    """Five-point central difference derivative (Pan-Tompkins)."""
    N = len(signal)
    diff_signal = np.zeros(N)
    for i in range(2, N - 2):
        diff_signal[i] = (
            -signal[i - 2]
            - 2 * signal[i - 1]
            + 2 * signal[i + 1]
            + signal[i + 2]
        ) / 8.0
    return diff_signal


def squaring(signal: np.ndarray) -> np.ndarray:
    """Non-linear squaring operator to amplify large slopes."""
    return signal ** 2


def moving_window_integration(
    signal: np.ndarray,
    fs: float = config.SAMPLING_RATE,
    window_sec: float = config.INTEGRATION_WINDOW_SEC,
) -> np.ndarray:
    """Moving-window integrator with a rectangular window."""
    win_samples = max(1, int(round(window_sec * fs)))
    kernel = np.ones(win_samples) / win_samples
    return np.convolve(signal, kernel, mode="same")


def detect_r_peaks(
    ecg_cleaned: np.ndarray,
    fs: float = config.SAMPLING_RATE,
    bandpass_low: float = config.DETECTION_LOW_CUTOFF,
    bandpass_high: float = config.DETECTION_HIGH_CUTOFF,
    integration_window_sec: float = config.INTEGRATION_WINDOW_SEC,
    refractory_sec: float = config.REFRACTORY_PERIOD_SEC,
    refinement_window_sec: float = config.REFINEMENT_WINDOW_SEC,
) -> np.ndarray:
    """Detect R-peak locations in a preprocessed ECG signal.

    Parameters
    ----------
    ecg_cleaned : np.ndarray
        1-D cleaned ECG signal (output of ``preprocess_ecg``).
    fs : float
        Sampling frequency (Hz).
    bandpass_low, bandpass_high : float
        Bandpass filter cutoffs for the detection stage (Hz).
    integration_window_sec : float
        Moving-window integration width (seconds).
    refractory_sec : float
        Minimum inter-beat interval (seconds).
    refinement_window_sec : float
        Half-width of the search window for R-peak refinement (seconds).

    Returns
    -------
    np.ndarray
        Sorted 1-D array of sample indices where R-peaks were detected.
    """
    N = len(ecg_cleaned)
    if N == 0:
        return np.array([], dtype=int)

    # ── Step 1: Bandpass filter (5–15 Hz) ────────────────────────────
    bp_signal = bandpass_filter(ecg_cleaned, fs, low=bandpass_low, high=bandpass_high)

    # ── Step 2: Differentiation ──────────────────────────────────────
    diff_signal = derivative(bp_signal, fs)

    # ── Step 3: Squaring ─────────────────────────────────────────────
    squared_signal = squaring(diff_signal)

    # ── Step 4: Moving-window integration ────────────────────────────
    integrated = moving_window_integration(squared_signal, fs, window_sec=integration_window_sec)

    # ── Step 5: Adaptive thresholding ────────────────────────────────
    # We use a two-level adaptive threshold inspired by Pan–Tompkins.
    # - SPKI tracks the running estimate of the signal peak level.
    # - NPKI tracks the running estimate of the noise peak level.
    # - Threshold = NPKI + 0.25 * (SPKI - NPKI)

    # Initialize with the first 2 seconds of data
    init_samples = min(int(2 * fs), N)
    spki = np.max(integrated[:init_samples])   # signal peak estimate
    npki = np.mean(integrated[:init_samples])   # noise peak estimate
    threshold = npki + 0.25 * (spki - npki)

    refractory_samples = int(round(refractory_sec * fs))
    candidates = []
    last_peak_idx = -refractory_samples  # allow detection from the start

    # Scan through the integrated signal looking for peaks above threshold
    i = 0
    while i < N:
        if integrated[i] > threshold and (i - last_peak_idx) > refractory_samples:
            # Find the local maximum in this region
            # Look ahead until the signal drops below threshold
            j = i
            while j < N and integrated[j] > threshold:
                j += 1
            # The peak is the maximum in [i, j)
            peak_idx = i + np.argmax(integrated[i:j])
            candidates.append(peak_idx)
            last_peak_idx = peak_idx

            # Update signal peak estimate
            spki = 0.125 * integrated[peak_idx] + 0.875 * spki
            threshold = npki + 0.25 * (spki - npki)

            i = j  # skip ahead past this detection
        else:
            # Update noise peak estimate
            npki = 0.125 * integrated[i] + 0.875 * npki
            threshold = npki + 0.25 * (spki - npki)
            i += 1

    logger.info("Pan-Tompkins candidates: %d", len(candidates))

    # ── Step 6: R-peak refinement ────────────────────────────────────
    # Map each candidate back to the actual R-peak (local max in the
    # cleaned ECG within a small search window).
    half_win = int(round(refinement_window_sec * fs))
    refined_peaks = []

    for c in candidates:
        lo = max(0, c - half_win)
        hi = min(N, c + half_win + 1)
        # R-peak = the sample with the maximum absolute amplitude
        local_idx = np.argmax(np.abs(ecg_cleaned[lo:hi]))
        refined_peaks.append(lo + local_idx)

    # Remove any duplicates and sort
    refined_peaks = sorted(set(refined_peaks))

    # Final refractory enforcement after refinement
    final_peaks = []
    for p in refined_peaks:
        if len(final_peaks) == 0 or (p - final_peaks[-1]) >= refractory_samples:
            final_peaks.append(p)

    logger.info("Final R-peaks after refinement: %d", len(final_peaks))
    return np.array(final_peaks, dtype=int)
