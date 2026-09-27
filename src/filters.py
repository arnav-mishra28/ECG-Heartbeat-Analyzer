"""
Digital filter implementations for ECG preprocessing.

Three core filters:
  1. High-pass Butterworth — removes baseline wander (< 0.5 Hz)
  2. Notch (band-stop)     — removes power-line interference (50 Hz)
  3. Low-pass Butterworth  — removes high-frequency noise (> 40 Hz)

All filters use **zero-phase** (forward-backward) filtering via
``scipy.signal.filtfilt`` to avoid introducing time shifts, which is
critical for accurate R-peak localization.

Design notes
------------
* Butterworth was chosen because its passband is maximally flat —
  it does not introduce ripple that could distort ECG morphology.
* Filter order controls roll-off steepness.  Higher order = sharper
  transition band but greater risk of ringing and numerical instability.
  Order 4 is a reasonable default for ECG work.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, freqz

from src import config


# ──────────────────────────────────────────────────────────────────────
# High-pass filter  (baseline wander removal)
# ──────────────────────────────────────────────────────────────────────
def highpass_filter(
    signal: np.ndarray,
    fs: float = config.SAMPLING_RATE,
    cutoff: float = config.HIGH_PASS_CUTOFF,
    order: int = config.FILTER_ORDER,
) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    signal : np.ndarray
        Input 1-D ECG signal.
    fs : float
        Sampling frequency (Hz).
    cutoff : float
        -3 dB cutoff frequency (Hz).
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered signal (same length as input).
    """
    nyq = fs / 2.0
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype="high")
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────────────────────────────
# Notch filter  (power-line interference removal)
# ──────────────────────────────────────────────────────────────────────
def notch_filter(
    signal: np.ndarray,
    fs: float = config.SAMPLING_RATE,
    freq: float = config.NOTCH_FREQUENCY,
    quality: float = config.NOTCH_QUALITY_FACTOR,
) -> np.ndarray:
    """Apply a zero-phase IIR notch (band-reject) filter.

    Parameters
    ----------
    signal : np.ndarray
        Input 1-D ECG signal.
    fs : float
        Sampling frequency (Hz).
    freq : float
        Center frequency to suppress (Hz).
    quality : float
        Quality factor Q — higher = narrower notch.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────────────────────────────
# Low-pass filter  (high-frequency noise removal)
# ──────────────────────────────────────────────────────────────────────
def lowpass_filter(
    signal: np.ndarray,
    fs: float = config.SAMPLING_RATE,
    cutoff: float = config.LOW_PASS_CUTOFF,
    order: int = config.FILTER_ORDER,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    signal : np.ndarray
        Input 1-D ECG signal.
    fs : float
        Sampling frequency (Hz).
    cutoff : float
        -3 dB cutoff frequency (Hz).
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = fs / 2.0
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────────────────────────────
# Filter response helpers  (for visualization / debugging)
# ──────────────────────────────────────────────────────────────────────
def get_highpass_response(
    fs: float = config.SAMPLING_RATE,
    cutoff: float = config.HIGH_PASS_CUTOFF,
    order: int = config.FILTER_ORDER,
    num_points: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_Hz, magnitude_dB) for the high-pass filter."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="high")
    w, h = freqz(b, a, worN=num_points, fs=fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    return w, mag_db


def get_notch_response(
    fs: float = config.SAMPLING_RATE,
    freq: float = config.NOTCH_FREQUENCY,
    quality: float = config.NOTCH_QUALITY_FACTOR,
    num_points: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_Hz, magnitude_dB) for the notch filter."""
    b, a = iirnotch(freq, quality, fs)
    w, h = freqz(b, a, worN=num_points, fs=fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    return w, mag_db


def get_lowpass_response(
    fs: float = config.SAMPLING_RATE,
    cutoff: float = config.LOW_PASS_CUTOFF,
    order: int = config.FILTER_ORDER,
    num_points: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_Hz, magnitude_dB) for the low-pass filter."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    w, h = freqz(b, a, worN=num_points, fs=fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    return w, mag_db
