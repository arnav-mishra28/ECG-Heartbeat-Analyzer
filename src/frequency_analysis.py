"""
Frequency-domain analysis of ECG signals.

Implements FFT-based spectral analysis to understand the frequency content
of raw and filtered ECG signals.  This is essential for:
  - Identifying noise components (baseline wander, 50 Hz power-line, HF noise)
  - Verifying that our filters work correctly
  - Producing publication-quality spectral plots
"""

from __future__ import annotations

import numpy as np


def compute_fft(
    signal: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided amplitude spectrum of a signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-domain signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis (Hz), from 0 to fs/2.
    magnitude : np.ndarray
        Magnitude spectrum (absolute value, normalized by N).
    """
    N = len(signal)

    # Compute the FFT
    fft_vals = np.fft.fft(signal)

    # Take only the positive-frequency half (DC to Nyquist)
    # For N samples, the single-sided spectrum has N//2 + 1 points
    n_one_side = N // 2 + 1
    fft_one_side = fft_vals[:n_one_side]

    # Magnitude, normalized by the number of samples
    magnitude = np.abs(fft_one_side) / N

    # Double the non-DC, non-Nyquist components to account for
    # the energy in the negative-frequency half
    magnitude[1:-1] *= 2

    # Frequency axis
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    return freqs, magnitude


def compute_power_spectrum(
    signal: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided power spectral density (PSD) estimate.

    Uses a simple periodogram (|FFT|² / N).  For more sophisticated
    PSD estimation (Welch, etc.) see ``scipy.signal.welch``.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-domain signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis (Hz).
    psd : np.ndarray
        Power spectral density (V² / Hz or equivalent).
    """
    freqs, magnitude = compute_fft(signal, fs)
    psd = magnitude ** 2
    return freqs, psd
