"""
Signal preprocessing pipeline.

Orchestrates the three-stage filtering:
    Raw ECG → High-pass → Notch → Low-pass → Clean ECG

Each filter is individually accessible (via ``src.filters``) for
experimentation.  This module provides the combined pipeline.
"""

from __future__ import annotations

import logging

import numpy as np

from src import config
from src.filters import highpass_filter, notch_filter, lowpass_filter

logger = logging.getLogger(__name__)


def preprocess_ecg(
    signal: np.ndarray,
    fs: float = config.SAMPLING_RATE,
    highpass_cutoff: float = config.HIGH_PASS_CUTOFF,
    notch_freq: float = config.NOTCH_FREQUENCY,
    notch_q: float = config.NOTCH_QUALITY_FACTOR,
    lowpass_cutoff: float = config.LOW_PASS_CUTOFF,
    filter_order: int = config.FILTER_ORDER,
) -> np.ndarray:
    """Run the full three-stage ECG cleaning pipeline.

    Parameters
    ----------
    signal : np.ndarray
        Raw 1-D ECG signal.
    fs : float
        Sampling frequency (Hz).
    highpass_cutoff : float
        High-pass cutoff frequency (Hz).  Removes baseline wander.
    notch_freq : float
        Notch center frequency (Hz).  Removes power-line interference.
    notch_q : float
        Quality factor for the notch filter.
    lowpass_cutoff : float
        Low-pass cutoff frequency (Hz).  Removes HF noise.
    filter_order : int
        Butterworth filter order for HP and LP stages.

    Returns
    -------
    np.ndarray
        Cleaned ECG signal (same length as input).
    """
    logger.info(
        "Preprocessing: HP=%.1f Hz, Notch=%.0f Hz (Q=%.0f), LP=%.0f Hz, order=%d",
        highpass_cutoff, notch_freq, notch_q, lowpass_cutoff, filter_order,
    )

    # Stage 1: Remove baseline wander
    ecg = highpass_filter(signal, fs, cutoff=highpass_cutoff, order=filter_order)
    logger.debug("  ✓ High-pass filter applied.")

    # Stage 2: Remove power-line interference
    ecg = notch_filter(ecg, fs, freq=notch_freq, quality=notch_q)
    logger.debug("  ✓ Notch filter applied.")

    # Stage 3: Remove high-frequency noise
    ecg = lowpass_filter(ecg, fs, cutoff=lowpass_cutoff, order=filter_order)
    logger.debug("  ✓ Low-pass filter applied.")

    return ecg
