"""
Data loader for the MIT-BIH Arrhythmia Database.

Responsibilities
----------------
1. Download records from PhysioNet (if not already cached locally).
2. Read the ECG signal and associated metadata via ``wfdb``.
3. Read the annotation file (expert-labeled beat locations).
4. Return everything in a clean, consistent data structure.

Key concepts
------------
* **Record** — One continuous ECG recording (e.g. record "100").
  Each record consists of a ``.dat`` (signal), ``.hea`` (header),
  and ``.atr`` (annotation) file.
* **Sampling frequency** — MIT-BIH uses 360 Hz (360 samples / second).
* **Channels** — Most MIT-BIH records have **two** leads (MLII + V1/V5).
  We default to channel 0 (MLII) for R-peak detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb

from src import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ECGRecord:
    """Immutable container for everything we need from one MIT-BIH record."""

    record_name: str
    signal: np.ndarray          # 1-D array of ECG amplitudes (single channel)
    sampling_frequency: float   # Hz
    num_samples: int
    num_channels: int           # of the *original* record (before channel selection)
    channel_names: list[str]
    units: list[str]
    time: np.ndarray            # 1-D time axis in seconds

    # Annotations
    annotation_samples: np.ndarray   # sample indices of annotated beats
    annotation_symbols: list[str]    # beat-type labels (e.g. "N", "V", …)
    annotation_times: np.ndarray     # times (seconds) of annotated beats

    # Beat-only annotations (filtered by BEAT_ANNOTATION_SYMBOLS)
    beat_samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    beat_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    num_beats: int = 0

    @property
    def duration_sec(self) -> float:
        """Total duration of the record in seconds."""
        return self.num_samples / self.sampling_frequency

    def __repr__(self) -> str:
        duration_min = self.num_samples / self.sampling_frequency / 60
        return (
            f"ECGRecord('{self.record_name}', "
            f"fs={self.sampling_frequency:.0f} Hz, "
            f"samples={self.num_samples:,}, "
            f"duration={duration_min:.1f} min, "
            f"channels={self.num_channels}, "
            f"annotations={len(self.annotation_samples):,}, "
            f"beats={self.num_beats:,})"
        )


# ──────────────────────────────────────────────────────────────────────
# Download helper
# ──────────────────────────────────────────────────────────────────────
def download_record(
    record_name: str,
    save_dir: Optional[Path] = None,
    db: str = config.PHYSIONET_DB,
) -> Path:
    """Download a single MIT-BIH record from PhysioNet if not already present.

    Parameters
    ----------
    record_name : str
        Record identifier, e.g. ``"100"``.
    save_dir : Path, optional
        Directory to store the downloaded files.
        Defaults to ``config.DATA_RAW_DIR``.
    db : str
        PhysioNet database name.

    Returns
    -------
    Path
        Directory where the record files were saved.
    """
    save_dir = save_dir or config.DATA_RAW_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if the record already exists locally (need .dat, .hea, .atr)
    expected_files = [
        save_dir / f"{record_name}.dat",
        save_dir / f"{record_name}.hea",
        save_dir / f"{record_name}.atr",
    ]
    if all(f.exists() for f in expected_files):
        logger.info("Record %s already exists in %s — skipping download.", record_name, save_dir)
        return save_dir

    logger.info("Downloading record %s from PhysioNet (%s)...", record_name, db)
    wfdb.dl_database(
        db,
        dl_dir=str(save_dir),
        records=[record_name],
    )
    logger.info("Download complete: %s", record_name)
    return save_dir


# ──────────────────────────────────────────────────────────────────────
# Main loading function
# ──────────────────────────────────────────────────────────────────────
def load_record(
    record_name: str = config.DEFAULT_RECORD,
    data_dir: Optional[Path] = None,
    channel: int = 0,
    download: bool = True,
) -> ECGRecord:
    """Load a complete MIT-BIH record (signal + annotations).

    Parameters
    ----------
    record_name : str
        Record identifier (e.g. ``"100"``).
    data_dir : Path, optional
        Directory containing (or to download to) the record files.
        Defaults to ``config.DATA_RAW_DIR``.
    channel : int
        Which ECG lead to extract (0 = first lead, usually MLII).
    download : bool
        If ``True``, automatically download the record if missing.

    Returns
    -------
    ECGRecord
        Fully populated data container.
    """
    data_dir = data_dir or config.DATA_RAW_DIR

    # ── Step 1: ensure data is available ──────────────────────────────
    if download:
        download_record(record_name, save_dir=data_dir)

    record_path = str(data_dir / record_name)

    # ── Step 2: read the signal ───────────────────────────────────────
    logger.info("Reading record %s...", record_name)
    record = wfdb.rdrecord(record_path)

    fs = float(record.fs)
    num_samples = record.sig_len
    num_channels = record.n_sig
    channel_names = list(record.sig_name)
    units = list(record.units)

    # Extract the requested channel as a 1-D float64 array
    signal = record.p_signal[:, channel].astype(np.float64)

    # Build the time axis: t[i] = i / fs
    time = np.arange(num_samples) / fs

    logger.info(
        "Record %s: fs=%d Hz, %d samples (%.1f min), %d channels %s",
        record_name, int(fs), num_samples,
        num_samples / fs / 60, num_channels, channel_names,
    )

    # ── Step 3: read annotations ──────────────────────────────────────
    annotation = wfdb.rdann(record_path, "atr")

    ann_samples = np.array(annotation.sample)
    ann_symbols = list(annotation.symbol)
    ann_times = ann_samples / fs

    logger.info("Loaded %d annotations for record %s.", len(ann_samples), record_name)

    # ── Step 4: filter to beat-only annotations ───────────────────────
    beat_mask = np.array(
        [s in config.BEAT_ANNOTATION_SYMBOLS for s in ann_symbols],
        dtype=bool,
    )
    beat_samples = ann_samples[beat_mask]
    beat_times = ann_times[beat_mask]
    num_beats = int(beat_samples.shape[0])

    logger.info(
        "  → %d beats (out of %d annotations) match beat symbols.",
        num_beats, len(ann_samples),
    )

    # ── Step 5: assemble and return ───────────────────────────────────
    return ECGRecord(
        record_name=record_name,
        signal=signal,
        sampling_frequency=fs,
        num_samples=num_samples,
        num_channels=num_channels,
        channel_names=channel_names,
        units=units,
        time=time,
        annotation_samples=ann_samples,
        annotation_symbols=ann_symbols,
        annotation_times=ann_times,
        beat_samples=beat_samples,
        beat_times=beat_times,
        num_beats=num_beats,
    )


# ──────────────────────────────────────────────────────────────────────
# Summary printer (useful for CLI and notebooks)
# ──────────────────────────────────────────────────────────────────────
def print_record_summary(ecg: ECGRecord) -> None:
    """Print a human-readable summary of a loaded ECG record."""
    duration_sec = ecg.num_samples / ecg.sampling_frequency
    duration_min = duration_sec / 60

    print("=" * 60)
    print(f"  ECG Record: {ecg.record_name}")
    print("=" * 60)
    print(f"  Sampling frequency : {ecg.sampling_frequency:.0f} Hz")
    print(f"  Nyquist frequency  : {ecg.sampling_frequency / 2:.0f} Hz")
    print(f"  Number of samples  : {ecg.num_samples:,}")
    print(f"  Duration           : {duration_sec:,.1f} s  ({duration_min:.1f} min)")
    print(f"  Number of channels : {ecg.num_channels}")
    print(f"  Channel names      : {ecg.channel_names}")
    print(f"  Units              : {ecg.units}")
    print(f"  Signal shape       : {ecg.signal.shape}")
    print(f"  Signal range       : [{ecg.signal.min():.4f}, {ecg.signal.max():.4f}]")
    print("-" * 60)
    print(f"  Total annotations  : {len(ecg.annotation_samples):,}")
    print(f"  Beat annotations   : {ecg.num_beats:,}")
    print(f"  Non-beat annotations: {len(ecg.annotation_samples) - ecg.num_beats:,}")

    # Show annotation symbol distribution
    from collections import Counter
    counts = Counter(ecg.annotation_symbols)
    print(f"  Annotation types   :")
    for sym, cnt in counts.most_common():
        label = "beat" if sym in config.BEAT_ANNOTATION_SYMBOLS else "non-beat"
        print(f"    '{sym}' ({label}): {cnt}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# CLI convenience
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    record_name = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_RECORD
    ecg = load_record(record_name)
    print_record_summary(ecg)
