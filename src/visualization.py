"""
Publication-quality visualization for the ECG analysis pipeline.

Generates the following figure types:
  1. Raw ECG waveform
  2. Frequency spectrum (FFT)
  3. Filter frequency responses
  4. Before/after filtering comparison
  5. R-peak detection overlay
  6. Heart rate over time
  7. RR interval histogram
  8. Ground truth vs. detected peaks comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from src import config

logger = logging.getLogger(__name__)

# Use a non-interactive backend when saving figures from CLI
matplotlib.use("Agg")

# Global style
plt.rcParams.update({
    "figure.dpi": config.FIGURE_DPI,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "lines.linewidth": 0.8,
})


def _save_or_show(fig: plt.Figure, filename: Optional[str], output_dir: Path) -> None:
    """Save figure to disk or display interactively."""
    if filename:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        fig.savefig(path, bbox_inches="tight", dpi=config.FIGURE_DPI)
        logger.info("Saved figure: %s", path)
        plt.close(fig)
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — Raw ECG
# ──────────────────────────────────────────────────────────────────────
def plot_raw_ecg(
    signal: np.ndarray,
    time: np.ndarray,
    title: str = "Raw ECG Signal",
    start_sec: float = 0,
    duration_sec: float = config.DEFAULT_PLOT_DURATION_SEC,
    filename: Optional[str] = None,
    output_dir: Path = config.OUTPUT_FIGURES_DIR,
) -> None:
    """Plot a segment of raw ECG amplitude vs. time."""
    fs = 1.0 / (time[1] - time[0]) if len(time) > 1 else config.SAMPLING_RATE
    start_idx = int(start_sec * fs)
    end_idx = int((start_sec + duration_sec) * fs)
    end_idx = min(end_idx, len(signal))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time[start_idx:end_idx], signal[start_idx:end_idx], color="#2563EB", linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, filename, output_dir)


# ──────────────────────────────────────────────────────────────────────
# Figure 2 — FFT spectrum
# ──────────────────────────────────────────────────────────────────────
def plot_spectrum(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    title: str = "Frequency Spectrum",
    max_freq: float = 100.0,
    filename: Optional[str] = None,
    output_dir: Path = config.OUTPUT_FIGURES_DIR,
) -> None:
    """Plot the magnitude spectrum."""
    mask = freqs <= max_freq

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freqs[mask], magnitude[mask], color="#DC2626", linewidth=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, filename, output_dir)


# ──────────────────────────────────────────────────────────────────────
# Figure 3 — Filter responses
# ──────────────────────────────────────────────────────────────────────
def plot_filter_responses(
    hp_response: tuple[np.ndarray, np.ndarray],
    notch_response: tuple[np.ndarray, np.ndarray],
    lp_response: tuple[np.ndarray, np.ndarray],
    filename: Optional[str] = None,
    output_dir: Path = config.OUTPUT_FIGURES_DIR,
) -> None:
    """Plot the frequency response of all three filters."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    titles = ["High-pass Filter", "Notch Filter", "Low-pass Filter"]
    responses = [hp_response, notch_response, lp_response]
    colors = ["#16A34A", "#CA8A04", "#9333EA"]

    for ax, (freqs, mag_db), title, color in zip(axes, responses, titles, colors):
        ax.plot(freqs, mag_db, color=color, linewidth=1.2)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(title)
        ax.set_ylim(-60, 5)
        ax.axhline(-3, color="gray", linestyle="--", linewidth=0.5, label="-3 dB")
        ax.legend(loc="lower right", fontsize=8)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    _save_or_show(fig, filename, output_dir)


# ──────────────────────────────────────────────────────────────────────
# Figure 4 — Before / after filtering
# ──────────────────────────────────────────────────────────────────────
def plot_before_after(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    time: np.ndarray,
    start_sec: float = 0,
    duration_sec: float = config.DEFAULT_PLOT_DURATION_SEC,
    filename: Optional[str] = None,
    output_dir: Path = config.OUTPUT_FIGURES_DIR,
) -> None:
    """Plot raw and filtered ECG side by side."""
    fs = 1.0 / (time[1] - time[0]) if len(time) > 1 else config.SAMPLING_RATE
    s = int(start_sec * fs)
    e = min(int((start_sec + duration_sec) * fs), len(raw_signal))

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(time[s:e], raw_signal[s:e], color="#2563EB", linewidth=0.6)
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("Raw ECG")

    axes[1].plot(time[s:e], filtered_signal[s:e], color="#16A34A", linewidth=0.6)
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].set_title("Filtered ECG")
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    _save_or_show(fig, filename, output_dir)


# ──────────────────────────────────────────────────────────────────────
# Figure 5 — R-peak detection
# ──────────────────────────────────────────────────────────────────────
def plot_r_peaks(
    signal: np.ndarray,
    time: np.ndarray,
    r_peaks: np.ndarray,
    start_sec: float = 0,
    duration_sec: float = config.DEFAULT_PLOT_DURATION_SEC,
    title: str = "R-Peak Detection",
    filename: Optional[str] = None,
    output_dir: Path = config.OUTPUT_FIGURES_DIR,
) -> None:
    """Plot ECG with R-peak markers overlaid."""
    fs = 1.0 / (time[1] - time[0]) if len(time) > 1 else config.SAMPLING_RATE
    s = int(start_sec * fs)
    e = min(int((start_sec + duration_sec) * fs), len(signal))

    # Filter peaks that fall in the visible window
    visible_peaks = r_peaks[(r_peaks >= s) & (r_peaks < e)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time[s:e], signal[s:e], color="#2563EB", linewidth=0.6, label="ECG")
    ax.plot(
        time[visible_peaks], signal[visible_peaks],
        "rv", markersize=8, label=f"R-peaks ({len(visible_peaks)})",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_or_show(fig, filename, output_dir)


# ──────────────────────────────────────────────────────────────────────
# Figure 6 — Heart rate over time
# ──────────────────────────────────────────────────────────────────────
def plot_heart_rate(
    bpm: np.ndarray,
    rr_times: np.ndarray,
    title: str = "Heart Rate Over Time",
    filename: Optional[str] = None,
    output_dir: Path = config.OUTPUT_FIGURES_DIR,
) -> None:
    """Plot instantaneous heart rate (BPM) vs. time."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rr_times, bpm, color="#DC2626", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, filename, output_dir)


# ──────────────────────────────────────────────────────────────────────
# Figure 7 — RR interval histogram
# ──────────────────────────────────────────────────────────────────────
def plot_rr_histogram(
    rr_intervals: np.ndarray,
    title: str = "RR Interval Distribution",
    filename: Optional[str] = None,
    output_dir: Path = config.OUTPUT_FIGURES_DIR,
) -> None:
    """Plot a histogram of RR intervals."""
    fig, ax = plt.subplots(figsize=(10, 5))
    rr_ms = rr_intervals * 1000  # convert to milliseconds
    ax.hist(rr_ms, bins=50, color="#7C3AED", edgecolor="white", alpha=0.85)
    ax.set_xlabel("RR Interval (ms)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(np.median(rr_ms), color="#DC2626", linestyle="--", linewidth=1.2,
               label=f"Median = {np.median(rr_ms):.0f} ms")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, filename, output_dir)


# ──────────────────────────────────────────────────────────────────────
# Figure 8 — Ground truth vs. detected
# ──────────────────────────────────────────────────────────────────────
def plot_gt_vs_detected(
    signal: np.ndarray,
    time: np.ndarray,
    gt_samples: np.ndarray,
    det_samples: np.ndarray,
    start_sec: float = 0,
    duration_sec: float = config.DEFAULT_PLOT_DURATION_SEC,
    title: str = "Ground Truth vs. Detected R-Peaks",
    filename: Optional[str] = None,
    output_dir: Path = config.OUTPUT_FIGURES_DIR,
) -> None:
    """Overlay ground-truth and detected peaks on the ECG."""
    fs = 1.0 / (time[1] - time[0]) if len(time) > 1 else config.SAMPLING_RATE
    s = int(start_sec * fs)
    e = min(int((start_sec + duration_sec) * fs), len(signal))

    gt_vis = gt_samples[(gt_samples >= s) & (gt_samples < e)]
    det_vis = det_samples[(det_samples >= s) & (det_samples < e)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time[s:e], signal[s:e], color="#94A3B8", linewidth=0.5, label="ECG")
    ax.plot(
        time[gt_vis], signal[gt_vis],
        "g^", markersize=10, alpha=0.7, label=f"Ground truth ({len(gt_vis)})",
    )
    ax.plot(
        time[det_vis], signal[det_vis],
        "rv", markersize=8, alpha=0.7, label=f"Detected ({len(det_vis)})",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_or_show(fig, filename, output_dir)
