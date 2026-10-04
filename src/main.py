"""
ECG Heartbeat Analyzer — CLI entry point.

Usage
-----
    python -m src.main --record 100
    python -m src.main --record 100 101 102 103
    python -m src.main --all

The pipeline:
    Load → Preprocess → Detect R-peaks → Heart rate → Evaluate → Visualize → Report
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time as time_module
from pathlib import Path

import numpy as np

from src import config
from src.data_loader import load_record, print_record_summary
from src.preprocessing import preprocess_ecg
from src.frequency_analysis import compute_fft
from src.filters import get_highpass_response, get_notch_response, get_lowpass_response
from src.peak_detection import detect_r_peaks
from src.heart_rate import compute_heart_rate, print_heart_rate_summary
from src.evaluation import evaluate_detection, print_evaluation_summary, EvaluationResult
from src import visualization as viz

logger = logging.getLogger(__name__)


def run_pipeline(record_name: str) -> EvaluationResult:
    """Run the full ECG analysis pipeline on a single record.

    Parameters
    ----------
    record_name : str
        MIT-BIH record identifier (e.g. "100").

    Returns
    -------
    EvaluationResult
        Detection performance metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Processing record: {record_name}")
    print(f"{'='*60}\n")

    # ── 1. Load ──────────────────────────────────────────────────────
    ecg = load_record(record_name)
    print_record_summary(ecg)

    # ── 2. Frequency analysis (raw) ──────────────────────────────────
    freqs_raw, mag_raw = compute_fft(ecg.signal, ecg.sampling_frequency)

    # ── 3. Preprocessing ─────────────────────────────────────────────
    print("\nFiltering...")
    t0 = time_module.perf_counter()
    ecg_clean = preprocess_ecg(ecg.signal, ecg.sampling_frequency)
    t_filter = time_module.perf_counter() - t0
    print(f"  Filtering completed in {t_filter:.3f} s")

    # ── 4. Frequency analysis (filtered) ─────────────────────────────
    freqs_filt, mag_filt = compute_fft(ecg_clean, ecg.sampling_frequency)

    # ── 5. R-peak detection ──────────────────────────────────────────
    print("\nDetecting R-peaks...")
    t0 = time_module.perf_counter()
    r_peaks = detect_r_peaks(ecg_clean, ecg.sampling_frequency)
    t_detect = time_module.perf_counter() - t0
    print(f"  Detected {len(r_peaks):,} beats in {t_detect:.3f} s")

    # ── 6. Heart rate ────────────────────────────────────────────────
    hr_stats = compute_heart_rate(r_peaks, ecg.sampling_frequency)
    print_heart_rate_summary(hr_stats)

    # ── 7. Evaluation ────────────────────────────────────────────────
    print("\nEvaluating against ground-truth annotations...")
    result = evaluate_detection(
        detected_samples=r_peaks,
        reference_samples=ecg.beat_samples,
        fs=ecg.sampling_frequency,
        record_name=record_name,
    )
    print_evaluation_summary(result)

    # ── 8. Visualizations ────────────────────────────────────────────
    prefix = f"rec{record_name}_"

    # Figure 1 — Raw ECG
    viz.plot_raw_ecg(
        ecg.signal, ecg.time,
        title=f"Raw ECG — Record {record_name}",
        filename=f"{prefix}01_raw_ecg.{config.FIGURE_FORMAT}",
    )

    # Figure 2 — Raw spectrum
    viz.plot_spectrum(
        freqs_raw, mag_raw,
        title=f"Frequency Spectrum (Raw) — Record {record_name}",
        filename=f"{prefix}02_spectrum_raw.{config.FIGURE_FORMAT}",
    )

    # Figure 3 — Filter responses
    hp_resp = get_highpass_response(ecg.sampling_frequency)
    notch_resp = get_notch_response(ecg.sampling_frequency)
    lp_resp = get_lowpass_response(ecg.sampling_frequency)
    viz.plot_filter_responses(
        hp_resp, notch_resp, lp_resp,
        filename=f"{prefix}03_filter_responses.{config.FIGURE_FORMAT}",
    )

    # Figure 4 — Before / after
    viz.plot_before_after(
        ecg.signal, ecg_clean, ecg.time,
        filename=f"{prefix}04_before_after.{config.FIGURE_FORMAT}",
    )

    # Figure 5 — R-peaks
    viz.plot_r_peaks(
        ecg_clean, ecg.time, r_peaks,
        title=f"R-Peak Detection — Record {record_name}",
        filename=f"{prefix}05_r_peaks.{config.FIGURE_FORMAT}",
    )

    # Figure 6 — Heart rate over time
    if len(hr_stats.instantaneous_bpm) > 0:
        viz.plot_heart_rate(
            hr_stats.instantaneous_bpm, hr_stats.rr_times,
            title=f"Heart Rate — Record {record_name}",
            filename=f"{prefix}06_heart_rate.{config.FIGURE_FORMAT}",
        )

    # Figure 7 — RR histogram
    if len(hr_stats.rr_intervals) > 0:
        viz.plot_rr_histogram(
            hr_stats.rr_intervals,
            title=f"RR Interval Distribution — Record {record_name}",
            filename=f"{prefix}07_rr_histogram.{config.FIGURE_FORMAT}",
        )

    # Figure 8 — Ground truth vs. detected
    viz.plot_gt_vs_detected(
        ecg_clean, ecg.time, ecg.beat_samples, r_peaks,
        title=f"Ground Truth vs. Detected — Record {record_name}",
        filename=f"{prefix}08_gt_vs_detected.{config.FIGURE_FORMAT}",
    )

    # Filtered spectrum
    viz.plot_spectrum(
        freqs_filt, mag_filt,
        title=f"Frequency Spectrum (Filtered) — Record {record_name}",
        filename=f"{prefix}02b_spectrum_filtered.{config.FIGURE_FORMAT}",
    )

    print(f"\nFigures saved to: {config.OUTPUT_FIGURES_DIR}")
    return result


def save_results_csv(results: list[EvaluationResult]) -> Path:
    """Write evaluation results to a CSV file."""
    config.OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUTPUT_REPORTS_DIR / "evaluation_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Record", "Reference_Beats", "Detected_Beats",
            "TP", "FP", "FN",
            "Sensitivity", "PPV", "F1",
            "Tolerance_ms",
        ])
        for r in results:
            writer.writerow([
                r.record_name, r.total_reference_beats, r.total_detected_beats,
                r.true_positives, r.false_positives, r.false_negatives,
                f"{r.sensitivity:.6f}", f"{r.ppv:.6f}", f"{r.f1:.6f}",
                f"{r.tolerance_sec * 1000:.0f}",
            ])

    logger.info("Results saved to %s", csv_path)
    return csv_path


def print_final_report(results: list[EvaluationResult]) -> None:
    """Print an aggregate summary across all processed records."""
    print("\n" + "=" * 70)
    print("  FINAL REPORT — ECG Heartbeat Analyzer")
    print("=" * 70)
    print(f"  Records processed  : {len(results)}")
    print(f"  Matching tolerance : ±{results[0].tolerance_sec * 1000:.0f} ms")
    print()

    total_ref = sum(r.total_reference_beats for r in results)
    total_det = sum(r.total_detected_beats for r in results)
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)

    overall_se = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_f1 = 2 * overall_se * overall_ppv / (overall_se + overall_ppv) if (overall_se + overall_ppv) > 0 else 0

    # Per-record table
    print(f"  {'Record':<10} {'Beats':>7} {'Det':>7} {'TP':>7} {'FP':>5} {'FN':>5} "
          f"{'Se':>8} {'PPV':>8} {'F1':>8}")
    print("  " + "-" * 66)
    for r in results:
        print(
            f"  {r.record_name:<10} {r.total_reference_beats:>7,} "
            f"{r.total_detected_beats:>7,} {r.true_positives:>7,} "
            f"{r.false_positives:>5,} {r.false_negatives:>5,} "
            f"{r.sensitivity:>8.4f} {r.ppv:>8.4f} {r.f1:>8.4f}"
        )
    print("  " + "-" * 66)

    # Aggregates
    sensitivities = [r.sensitivity for r in results]
    ppvs = [r.ppv for r in results]
    f1s = [r.f1 for r in results]

    print(f"\n  {'Aggregate':>30}")
    print(f"  {'Total reference beats':>30} : {total_ref:,}")
    print(f"  {'Total detected beats':>30} : {total_det:,}")
    print(f"  {'Overall Sensitivity':>30} : {overall_se:.4f}  ({overall_se*100:.2f}%)")
    print(f"  {'Overall PPV':>30} : {overall_ppv:.4f}  ({overall_ppv*100:.2f}%)")
    print(f"  {'Overall F1':>30} : {overall_f1:.4f}  ({overall_f1*100:.2f}%)")
    print(f"  {'Mean Sensitivity':>30} : {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
    print(f"  {'Mean PPV':>30} : {np.mean(ppvs):.4f} ± {np.std(ppvs):.4f}")
    print(f"  {'Mean F1':>30} : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print("=" * 70)


def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="ECG Heartbeat Analyzer — Classical DSP Pipeline",
    )
    parser.add_argument(
        "--record", "-r",
        nargs="+",
        default=[config.DEFAULT_RECORD],
        help="MIT-BIH record name(s), e.g. 100 101 102",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all 48 MIT-BIH records.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    records = config.ALL_RECORDS if args.all else args.record

    print(f"\nECG Heartbeat Analyzer v1.0")
    print(f"Records to process: {records}\n")

    results: list[EvaluationResult] = []
    for rec in records:
        try:
            result = run_pipeline(rec)
            results.append(result)
        except Exception:
            logger.exception("Failed to process record %s", rec)

    if results:
        # Save CSV
        csv_path = save_results_csv(results)
        print(f"\nCSV results saved to: {csv_path}")

        # Final aggregate report
        print_final_report(results)


if __name__ == "__main__":
    main()
