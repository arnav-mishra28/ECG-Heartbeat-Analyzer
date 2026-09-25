# ECG Heartbeat Analyzer

**Classical Digital Signal Processing Pipeline**

A clean, reproducible signal-processing system for ECG heartbeat detection and analysis using the MIT-BIH Arrhythmia Database. Implements classical DSP techniques with quantitative evaluation against expert annotations.

## Overview

This pipeline processes raw ECG signals through a series of classical digital signal processing stages to detect heartbeats, compute heart rate, and evaluate detection performance against gold-standard annotations.

```
                MIT-BIH ECG
                     │
                     ▼
              ┌─────────────┐
              │ Data Loader  │
              └──────┬──────┘
                     │
                     ▼
               Raw ECG Signal
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Time-domain              FFT Analysis
    Analysis
          │                     │
          └──────────┬──────────┘
                     ▼
               Noise Analysis
                     │
                     ▼
          ┌─────────────────────┐
          │ Signal Preprocessing│
          └──────────┬──────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
      High-pass    Notch      Low-pass
         │           │           │
         └───────────┼───────────┘
                     ▼
                Clean ECG
                     │
                     ▼
            QRS/R-Peak Detector
                     │
                     ▼
              R-Peak Locations → RR Intervals → Heart Rate → HRV Stats
                     │
                     ▼
           Ground Truth Comparison → Performance Metrics
                     │
                     ▼
              Visualizations → Final Report
```

## Project Structure

```
ECG-Heartbeat-Analyzer/
├── data/
│   ├── raw/              # Raw MIT-BIH records (downloaded via WFDB)
│   └── processed/        # Preprocessed/filtered signals
├── src/
│   ├── __init__.py
│   ├── config.py          # Centralized configuration parameters
│   ├── data_loader.py     # MIT-BIH data loading utilities
│   ├── preprocessing.py   # Signal preprocessing pipeline
│   ├── filters.py         # Digital filter implementations
│   ├── frequency_analysis.py  # FFT and spectral analysis
│   ├── peak_detection.py  # Pan-Tompkins R-peak detector
│   ├── heart_rate.py      # RR intervals, BPM, HRV statistics
│   ├── evaluation.py      # Ground-truth comparison and metrics
│   ├── visualization.py   # Publication-quality figure generation
│   └── main.py            # CLI entry point
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_filter_analysis.ipynb
│   ├── 03_peak_detection.ipynb
│   └── 04_evaluation.ipynb
├── outputs/
│   ├── figures/           # Generated plots
│   ├── signals/           # Exported signal data
│   └── reports/           # Analysis reports
├── tests/
│   ├── test_filters.py
│   ├── test_peak_detection.py
│   └── test_heart_rate.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

**MIT-BIH Arrhythmia Database** from PhysioNet.

- 48 half-hour two-channel ambulatory ECG recordings
- Sampled at **360 Hz**
- Expert-annotated beat locations and types
- Accessed via the `wfdb` Python package

## Technology Stack

| Component | Tool |
|-----------|------|
| Numerical computing | NumPy |
| Signal processing | SciPy |
| Data handling | Pandas |
| Visualization | Matplotlib |
| ECG data access | WFDB |
| Testing | Pytest |

## Quick Start

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run analysis on a single record
python -m src.main --record 100

# Run all tests
pytest tests/
```

## Build Stages

- [x] Stage 1 — Environment setup (virtual environment, requirements, directory structure)
- [x] Stage 2 — Dataset loading (WFDB integration, MIT-BIH PhysioNet data loader, beat filtering)
- [x] Stage 3 — Visualization (publication-grade plotting of waveforms, spectra, filter curves, detected peaks)
- [x] Stage 4 — Frequency analysis (single-sided FFT magnitude and power spectral density estimation)
- [x] Stage 5 — Filtering (Butterworth 0.5 Hz highpass, 50 Hz notch, 40 Hz lowpass, zero-phase `filtfilt`)
- [x] Stage 6 — R-peak detection (Pan-Tompkins algorithm: bandpass, derivative, squaring, MWI, dual adaptive thresholds, local peak refinement)
- [x] Stage 7 — Heart-rate analysis (RR intervals, instantaneous BPM, SDNN, RMSSD)
- [x] Stage 8 — Evaluation (ANSI/AAMI EC57 tolerance-based matching, TP/FP/FN classification, Sensitivity, PPV, F1)
- [x] Stage 9 — Engineering polish (39 unit tests with pytest, modular design, full type hints, docstrings, 4 interactive executed Jupyter notebooks)
- [x] Stage 10 — Final output (CLI with multi-record support, CSV report generation, comprehensive documentation)

## Quantitative Benchmark Results (Record 100)

| Metric | Ground Truth / Target | Detected / Achieved |
|--------|-----------------------|---------------------|
| Total Reference Beats | 2,273 | — |
| Total Detected Beats | — | 2,265 |
| True Positives (TP) | — | **2,265** |
| False Positives (FP) | — | **0** |
| False Negatives (FN) | — | **8** |
| **Sensitivity (Recall)** | — | **99.65%** |
| **PPV (Precision)** | — | **100.00%** |
| **F1 Score** | — | **99.82%** |
| Matching Tolerance | — | $\pm 150$ ms |
| Mean Heart Rate | — | 75.7 BPM |
| Median Heart Rate | — | 75.3 BPM |
| SDNN | — | 84.2 ms |
| RMSSD | — | 116.3 ms |

## Generated Figures & Artifacts

All plots are automatically rendered and saved to `outputs/figures/`:
- `rec100_01_raw_ecg.png` — Raw ECG waveform (Lead MLII) showing cardiac cycles and baseline wander.
- `rec100_02_spectrum_raw.png` — Single-sided FFT magnitude spectrum of raw ECG highlighting low-frequency drift and mains hum.
- `rec100_02b_spectrum_filtered.png` — FFT spectrum after filtering showing pristine passband isolation.
- `rec100_03_filter_responses.png` — Bode magnitude responses for High-Pass (0.5 Hz), Notch (50 Hz), and Low-Pass (40 Hz).
- `rec100_04_before_after.png` — Side-by-side time-domain comparison of raw vs cleaned signal.
- `rec100_05_r_peaks.png` — Clean ECG with overlaid Pan-Tompkins detected R-peaks.
- `rec100_06_heart_rate.png` — Instantaneous BPM tachogram over the 30-minute recording.
- `rec100_07_rr_histogram.png` — Histogram distribution of normal-to-normal RR intervals.
- `rec100_08_gt_vs_detected.png` — Visual comparison of detected peaks vs expert ground-truth annotations.

## Interactive Notebooks

Interactive Jupyter notebooks are fully executed and stored in `notebooks/`:
- `01_data_exploration.ipynb` — MIT-BIH dataset inspection, leads, and annotation symbols.
- `02_filter_analysis.ipynb` — Noise characterization, filter design, zero-phase filtering, and spectral analysis.
- `03_peak_detection.ipynb` — Step-by-step Pan-Tompkins pipeline breakdown and HRV calculation.
- `04_evaluation.ipynb` — Quantitative performance benchmark against ground truth.

## License

Educational / Research use.

## Disclaimer

This is an educational/research prototype, **not** a diagnostic medical device. Do not use for clinical decision-making.
