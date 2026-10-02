"""
Ground-truth evaluation: compare detected R-peaks against expert annotations.

Implements a tolerance-based matching algorithm:
  1. For each reference beat, check if any detected peak falls within
     ±MATCHING_TOLERANCE_SEC.
  2. Classify every (reference, detection) pair as TP, FP, or FN.
  3. Compute Sensitivity, PPV (Positive Predictive Value), and F1 score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src import config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for beat-detection performance metrics."""

    record_name: str
    total_reference_beats: int
    total_detected_beats: int

    true_positives: int
    false_positives: int
    false_negatives: int

    sensitivity: float   # TP / (TP + FN)
    ppv: float           # TP / (TP + FP)   (positive predictive value)
    f1: float            # 2 * (PPV * Se) / (PPV + Se)

    tolerance_sec: float

    def __repr__(self) -> str:
        return (
            f"EvaluationResult('{self.record_name}', "
            f"Se={self.sensitivity:.4f}, "
            f"PPV={self.ppv:.4f}, "
            f"F1={self.f1:.4f}, "
            f"TP={self.true_positives}, "
            f"FP={self.false_positives}, "
            f"FN={self.false_negatives})"
        )


def evaluate_detection(
    detected_samples: np.ndarray,
    reference_samples: np.ndarray,
    fs: float = config.SAMPLING_RATE,
    tolerance_sec: float = config.MATCHING_TOLERANCE_SEC,
    record_name: str = "",
) -> EvaluationResult:
    """Compare detected peaks against reference annotations.

    Parameters
    ----------
    detected_samples : np.ndarray
        Sample indices of detected R-peaks.
    reference_samples : np.ndarray
        Sample indices of reference (expert) beat annotations.
    fs : float
        Sampling frequency (Hz).
    tolerance_sec : float
        Maximum allowed time difference (seconds) for a match.
    record_name : str
        Record identifier (for logging / reporting).

    Returns
    -------
    EvaluationResult
        Performance metrics.
    """
    tolerance_samples = int(round(tolerance_sec * fs))

    detected = np.sort(detected_samples)
    reference = np.sort(reference_samples)

    n_det = len(detected)
    n_ref = len(reference)

    # Greedy matching: for each reference beat, find the closest detection
    matched_det = set()
    tp = 0
    fn = 0

    for ref in reference:
        # Find all detections within the tolerance window
        diffs = np.abs(detected.astype(np.int64) - int(ref))
        within = np.where(diffs <= tolerance_samples)[0]

        if len(within) > 0:
            # Pick the closest one that hasn't been matched yet
            sorted_by_dist = within[np.argsort(diffs[within])]
            found = False
            for idx in sorted_by_dist:
                if idx not in matched_det:
                    matched_det.add(idx)
                    tp += 1
                    found = True
                    break
            if not found:
                fn += 1  # all nearby detections were already matched
        else:
            fn += 1

    fp = n_det - len(matched_det)

    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * sensitivity * ppv / (sensitivity + ppv)
        if (sensitivity + ppv) > 0
        else 0.0
    )

    logger.info(
        "Evaluation [%s]: TP=%d, FP=%d, FN=%d, Se=%.4f, PPV=%.4f, F1=%.4f",
        record_name, tp, fp, fn, sensitivity, ppv, f1,
    )

    return EvaluationResult(
        record_name=record_name,
        total_reference_beats=n_ref,
        total_detected_beats=n_det,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        sensitivity=sensitivity,
        ppv=ppv,
        f1=f1,
        tolerance_sec=tolerance_sec,
    )


def print_evaluation_summary(result: EvaluationResult) -> None:
    """Print a human-readable evaluation summary."""
    print("-" * 55)
    print(f"  Evaluation — Record {result.record_name}")
    print("-" * 55)
    print(f"  Tolerance          : ±{result.tolerance_sec * 1000:.0f} ms")
    print(f"  Reference beats    : {result.total_reference_beats:,}")
    print(f"  Detected beats     : {result.total_detected_beats:,}")
    print(f"  True Positives     : {result.true_positives:,}")
    print(f"  False Positives    : {result.false_positives:,}")
    print(f"  False Negatives    : {result.false_negatives:,}")
    print(f"  Sensitivity (Se)   : {result.sensitivity:.4f}  ({result.sensitivity * 100:.2f}%)")
    print(f"  PPV (Precision)    : {result.ppv:.4f}  ({result.ppv * 100:.2f}%)")
    print(f"  F1 Score           : {result.f1:.4f}  ({result.f1 * 100:.2f}%)")
    print("-" * 55)
