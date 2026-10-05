"""Script to generate all 4 Jupyter notebooks with rich content and structure."""

from __future__ import annotations
from pathlib import Path
import nbformat as nbf

def generate_notebooks():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Notebook 1: 01_data_exploration.ipynb
    # =========================================================================
    nb1 = nbf.v4.new_notebook()
    nb1.cells = [
        nbf.v4.new_markdown_cell(
            "# 01. MIT-BIH Arrhythmia Database Exploration\n\n"
            "This notebook demonstrates loading, inspecting, and visualizing ECG records "
            "from the **MIT-BIH Arrhythmia Database** (PhysioNet `mitdb`).\n\n"
            "### Objectives:\n"
            "1. Load standard MIT-BIH records using WFDB.\n"
            "2. Inspect sampling rate ($f_s = 360$ Hz), channel assignments (MLII, V5), and signal metadata.\n"
            "3. Analyze expert beat annotations (Normal 'N', Ventricular 'V', Atrial 'A', etc.).\n"
            "4. Visualize multi-second ECG waveforms with annotation markers."
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "# Ensure project root is in sys.path\n"
            "project_root = Path('..').resolve()\n"
            "if str(project_root) not in sys.path:\n"
            "    sys.path.insert(0, str(project_root))\n\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from collections import Counter\n"
            "from src.data_loader import load_record, print_record_summary\n"
            "from src import config"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Load Record 100\n"
            "We load MIT-BIH record 100 into our structured `ECGRecord` dataclass."
        ),
        nbf.v4.new_code_cell(
            "ecg = load_record('100')\n"
            "print_record_summary(ecg)"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Signal Properties and Lead Metadata"
        ),
        nbf.v4.new_code_cell(
            "print(f'Sampling frequency : {ecg.sampling_frequency} Hz')\n"
            "print(f'Total samples      : {ecg.num_samples:,}')\n"
            "print(f'Total duration     : {ecg.duration_sec:.1f} s ({ecg.duration_sec / 60:.1f} min)')\n"
            "print(f'Channels           : {ecg.channel_names}')\n"
            "print(f'Signal amplitude   : min={ecg.signal.min():.3f} mV, max={ecg.signal.max():.3f} mV')\n"
            "print(f'Signal mean        : {ecg.signal.mean():.4f} mV, std={ecg.signal.std():.4f} mV')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Beat vs Non-Beat Annotations\n"
            "MIT-BIH annotations include both heartbeat classifications and auxiliary rhythm/signal comments."
        ),
        nbf.v4.new_code_cell(
            "sym_counts = Counter(ecg.annotation_symbols)\n"
            "print(f'Total annotations: {len(ecg.annotation_symbols):,}')\n"
            "print(f'Beat annotations : {ecg.num_beats:,}')\n"
            "print('\\nSymbol breakdown:')\n"
            "for sym, count in sym_counts.most_common():\n"
            "    is_beat = sym in config.BEAT_ANNOTATION_SYMBOLS\n"
            "    kind = 'beat' if is_beat else 'non-beat marker'\n"
            "    print(f'  [{sym}] : {count:>5} occurrences ({kind})')"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Visualizing a 10-Second Window of Raw ECG\n"
            "Observe the characteristic P-wave, QRS complex, and T-wave alongside baseline wander."
        ),
        nbf.v4.new_code_cell(
            "duration = 10.0\n"
            "mask = ecg.time <= duration\n"
            "t_win = ecg.time[mask]\n"
            "sig_win = ecg.signal[mask]\n\n"
            "# Reference beats within window\n"
            "beat_mask = ecg.beat_times <= duration\n"
            "b_times = ecg.beat_times[beat_mask]\n"
            "b_samples = ecg.beat_samples[beat_mask]\n"
            "b_amps = ecg.signal[b_samples]\n\n"
            "plt.figure(figsize=(14, 4.5), dpi=120)\n"
            "plt.plot(t_win, sig_win, color='#1f77b4', lw=1.2, label='Raw ECG (Lead MLII)')\n"
            "plt.scatter(b_times, b_amps, color='#d62728', s=45, zorder=5, label='Expert Beat Annotation')\n"
            "plt.title('MIT-BIH Record 100 — Raw Lead MLII (First 10 Seconds)', fontsize=13, fontweight='bold')\n"
            "plt.xlabel('Time (seconds)', fontsize=11)\n"
            "plt.ylabel('Amplitude (mV)', fontsize=11)\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.legend(loc='upper right')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    ]

    with open(nb_dir / "01_data_exploration.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb1, f)
    print("Generated 01_data_exploration.ipynb")

    # =========================================================================
    # Notebook 2: 02_filter_analysis.ipynb
    # =========================================================================
    nb2 = nbf.v4.new_notebook()
    nb2.cells = [
        nbf.v4.new_markdown_cell(
            "# 02. Digital Filter Design & Frequency-Domain Analysis\n\n"
            "This notebook explores the spectral content of raw ECG signals and examines the "
            "classical digital filters designed to isolate clinical cardiac features.\n\n"
            "### Noise Sources in ECG:\n"
            "1. **Baseline Wander** ($< 0.5$ Hz): Caused by respiration and body movement.\n"
            "2. **Power-Line Interference** ($50$ Hz): Electromagnetic interference from mains electricity.\n"
            "3. **High-Frequency Muscle Noise / EMG** ($> 40$ Hz): Skeletal muscle tremors and RF noise.\n\n"
            "### Filter Pipeline:\n"
            "$$\\text{Raw ECG} \\xrightarrow{\\text{High-pass (0.5 Hz)}} \\xrightarrow{\\text{Notch (50 Hz)}} \\xrightarrow{\\text{Low-pass (40 Hz)}} \\text{Clean ECG}$$\n"
            "Using zero-phase bidirectional filtering (`filtfilt`) to prevent phase distortion."
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\n"
            "import sys\n"
            "from pathlib import Path\n"
            "project_root = Path('..').resolve()\n"
            "if str(project_root) not in sys.path:\n"
            "    sys.path.insert(0, str(project_root))\n\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from src.data_loader import load_record\n"
            "from src.frequency_analysis import compute_fft, compute_power_spectrum\n"
            "from src.filters import get_highpass_response, get_notch_response, get_lowpass_response\n"
            "from src.preprocessing import preprocess_ecg\n"
            "from src import config"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Frequency Spectrum of Raw ECG"
        ),
        nbf.v4.new_code_cell(
            "ecg = load_record('100')\n"
            "freqs_raw, mag_raw = compute_fft(ecg.signal, ecg.sampling_frequency)\n\n"
            "plt.figure(figsize=(12, 4.5), dpi=120)\n"
            "plt.plot(freqs_raw, mag_raw, color='#d62728', lw=1.0)\n"
            "plt.title('Magnitude Spectrum of Raw ECG (Record 100)', fontsize=13, fontweight='bold')\n"
            "plt.xlabel('Frequency (Hz)', fontsize=11)\n"
            "plt.ylabel('Magnitude (mV)', fontsize=11)\n"
            "plt.xlim(0, 100)\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Frequency Response of Digital Filters\n"
            "We inspect the magnitude responses of the High-pass (0.5 Hz), Notch (50 Hz), and Low-pass (40 Hz) stages."
        ),
        nbf.v4.new_code_cell(
            "hp_freqs, hp_mag = get_highpass_response(ecg.sampling_frequency)\n"
            "notch_freqs, notch_mag = get_notch_response(ecg.sampling_frequency)\n"
            "lp_freqs, lp_mag = get_lowpass_response(ecg.sampling_frequency)\n\n"
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=120)\n\n"
            "axes[0].plot(hp_freqs, hp_mag, color='#1f77b4', lw=1.5)\n"
            "axes[0].set_title('High-Pass Filter (0.5 Hz Cutoff)', fontweight='bold')\n"
            "axes[0].set_xlabel('Frequency (Hz)')\n"
            "axes[0].set_ylabel('Gain (dB)')\n"
            "axes[0].set_xlim(0, 5)\n"
            "axes[0].set_ylim(-40, 5)\n"
            "axes[0].grid(True, alpha=0.3)\n\n"
            "axes[1].plot(notch_freqs, notch_mag, color='#ff7f0e', lw=1.5)\n"
            "axes[1].set_title('Notch Filter (50 Hz, Q=30)', fontweight='bold')\n"
            "axes[1].set_xlabel('Frequency (Hz)')\n"
            "axes[1].set_ylabel('Gain (dB)')\n"
            "axes[1].set_xlim(30, 70)\n"
            "axes[1].set_ylim(-40, 5)\n"
            "axes[1].grid(True, alpha=0.3)\n\n"
            "axes[2].plot(lp_freqs, lp_mag, color='#2ca02c', lw=1.5)\n"
            "axes[2].set_title('Low-Pass Filter (40 Hz Cutoff)', fontweight='bold')\n"
            "axes[2].set_xlabel('Frequency (Hz)')\n"
            "axes[2].set_ylabel('Gain (dB)')\n"
            "axes[2].set_xlim(0, 100)\n"
            "axes[2].set_ylim(-60, 5)\n"
            "axes[2].grid(True, alpha=0.3)\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Comparison Before and After Filtering\n"
            "Clean the signal and compare the waveform and power spectrum."
        ),
        nbf.v4.new_code_cell(
            "ecg_clean = preprocess_ecg(ecg.signal, ecg.sampling_frequency)\n"
            "freqs_clean, mag_clean = compute_fft(ecg_clean, ecg.sampling_frequency)\n\n"
            "t_mask = ecg.time <= 10.0\n\n"
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, dpi=120)\n"
            "ax1.plot(ecg.time[t_mask], ecg.signal[t_mask], color='#7f7f7f', lw=1.2, label='Raw ECG')\n"
            "ax1.set_title('Raw ECG (Lead MLII)', fontweight='bold')\n"
            "ax1.set_ylabel('Amplitude (mV)')\n"
            "ax1.grid(True, alpha=0.3)\n"
            "ax1.legend()\n\n"
            "ax2.plot(ecg.time[t_mask], ecg_clean[t_mask], color='#1f77b4', lw=1.2, label='Filtered ECG')\n"
            "ax2.set_title('Filtered ECG (0.5–40 Hz Bandpass + 50 Hz Notch)', fontweight='bold')\n"
            "ax2.set_xlabel('Time (seconds)')\n"
            "ax2.set_ylabel('Amplitude (mV)')\n"
            "ax2.grid(True, alpha=0.3)\n"
            "ax2.legend()\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    ]

    with open(nb_dir / "02_filter_analysis.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb2, f)
    print("Generated 02_filter_analysis.ipynb")

    # =========================================================================
    # Notebook 3: 03_peak_detection.ipynb
    # =========================================================================
    nb3 = nbf.v4.new_notebook()
    nb3.cells = [
        nbf.v4.new_markdown_cell(
            "# 03. Pan-Tompkins R-Peak Detection & Heart-Rate Analysis\n\n"
            "This notebook details the classical **Pan-Tompkins algorithm** for real-time QRS detection "
            "and calculates subsequent heart rate variability (HRV) metrics.\n\n"
            "### Pan-Tompkins Algorithm Stages:\n"
            "1. **Bandpass Filtering (5–15 Hz)**: Isolates energy dominated by the QRS complex.\n"
            "2. **Five-Point Derivative**: Accentuates the steep slopes of the QRS wave.\n"
            "3. **Squaring**: Non-linear operator making all points positive and amplifying high slopes.\n"
            "4. **Moving Window Integration (MWI)**: Accumulates waveform duration (~150 ms).\n"
            "5. **Dual Adaptive Thresholding**: Distinguishes signal peaks from noise peaks.\n"
            "6. **Refinement & Refractory Period**: Searches local maximum in filtered ECG, enforcing 200 ms minimum RR spacing."
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\n"
            "import sys\n"
            "from pathlib import Path\n"
            "project_root = Path('..').resolve()\n"
            "if str(project_root) not in sys.path:\n"
            "    sys.path.insert(0, str(project_root))\n\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from src.data_loader import load_record\n"
            "from src.preprocessing import preprocess_ecg\n"
            "from src.peak_detection import detect_r_peaks, bandpass_filter, derivative, squaring, moving_window_integration\n"
            "from src.heart_rate import compute_heart_rate, print_heart_rate_summary\n"
            "from src import config"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Step-by-Step Transformation of the ECG"
        ),
        nbf.v4.new_code_cell(
            "ecg = load_record('100')\n"
            "fs = ecg.sampling_frequency\n"
            "ecg_clean = preprocess_ecg(ecg.signal, fs)\n\n"
            "# Run intermediate stages on a 5-second slice\n"
            "idx_slice = slice(0, int(5 * fs))\n"
            "t_sub = ecg.time[idx_slice]\n"
            "sig_sub = ecg_clean[idx_slice]\n\n"
            "bpass = bandpass_filter(sig_sub, fs, low=5.0, high=15.0)\n"
            "deriv = derivative(bpass, fs)\n"
            "sq = squaring(deriv)\n"
            "mwi = moving_window_integration(sq, fs, window_sec=0.150)\n\n"
            "fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True, dpi=120)\n\n"
            "axes[0].plot(t_sub, sig_sub, color='#1f77b4')\n"
            "axes[0].set_title('1. Clean ECG (Lead MLII)', fontweight='bold')\n"
            "axes[0].grid(True, alpha=0.3)\n\n"
            "axes[1].plot(t_sub, bpass, color='#2ca02c')\n"
            "axes[1].set_title('2. Bandpass Filtered (5–15 Hz)', fontweight='bold')\n"
            "axes[1].grid(True, alpha=0.3)\n\n"
            "axes[2].plot(t_sub, deriv, color='#ff7f0e')\n"
            "axes[2].set_title('3. Derivative', fontweight='bold')\n"
            "axes[2].grid(True, alpha=0.3)\n\n"
            "axes[3].plot(t_sub, sq, color='#9467bd')\n"
            "axes[3].set_title('4. Squaring Function', fontweight='bold')\n"
            "axes[3].grid(True, alpha=0.3)\n\n"
            "axes[4].plot(t_sub, mwi, color='#d62728')\n"
            "axes[4].set_title('5. Moving-Window Integration (150 ms)', fontweight='bold')\n"
            "axes[4].set_xlabel('Time (seconds)')\n"
            "axes[4].grid(True, alpha=0.3)\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Detect R-Peaks on Full 30-Minute Record"
        ),
        nbf.v4.new_code_cell(
            "r_peaks = detect_r_peaks(ecg_clean, fs)\n"
            "print(f'Total R-peaks detected: {len(r_peaks):,}')\n"
            "print(f'Ground-truth reference beats: {len(ecg.beat_samples):,}')"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. Heart Rate and HRV Analysis"
        ),
        nbf.v4.new_code_cell(
            "hr_stats = compute_heart_rate(r_peaks, fs)\n"
            "print_heart_rate_summary(hr_stats)\n\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.5), dpi=120)\n\n"
            "# Heart rate over time\n"
            "ax1.plot(hr_stats.rr_times / 60, hr_stats.instantaneous_bpm, color='#1f77b4', lw=1.2)\n"
            "ax1.axhline(hr_stats.mean_bpm, color='#d62728', linestyle='--', label=f'Mean: {hr_stats.mean_bpm:.1f} BPM')\n"
            "ax1.set_title('Instantaneous Heart Rate Over Time', fontweight='bold')\n"
            "ax1.set_xlabel('Time (minutes)')\n"
            "ax1.set_ylabel('Heart Rate (BPM)')\n"
            "ax1.grid(True, alpha=0.3)\n"
            "ax1.legend()\n\n"
            "# RR interval histogram\n"
            "ax2.hist(hr_stats.rr_intervals * 1000, bins=50, color='#2ca02c', edgecolor='black', alpha=0.8)\n"
            "ax2.axvline(np.mean(hr_stats.rr_intervals) * 1000, color='#d62728', linestyle='--', label=f'Mean RR: {np.mean(hr_stats.rr_intervals)*1000:.0f} ms')\n"
            "ax2.set_title('RR Interval Distribution', fontweight='bold')\n"
            "ax2.set_xlabel('RR Interval (ms)')\n"
            "ax2.set_ylabel('Beat Count')\n"
            "ax2.grid(True, alpha=0.3)\n"
            "ax2.legend()\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    ]

    with open(nb_dir / "03_peak_detection.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb3, f)
    print("Generated 03_peak_detection.ipynb")

    # =========================================================================
    # Notebook 4: 04_evaluation.ipynb
    # =========================================================================
    nb4 = nbf.v4.new_notebook()
    nb4.cells = [
        nbf.v4.new_markdown_cell(
            "# 04. Quantitative Evaluation Against PhysioNet Annotations\n\n"
            "This notebook evaluates the detector's accuracy against expert cardiologist annotations "
            "using standard ANSI/AAMI EC57 tolerance-matching criteria.\n\n"
            "### Evaluation Metrics:\n"
            "- **True Positives (TP)**: Detected peaks that match a reference beat within $\\pm 150$ ms.\n"
            "- **False Positives (FP)**: Detected peaks with no corresponding reference beat.\n"
            "- **False Negatives (FN)**: Reference beats missed by the detector.\n"
            "- **Sensitivity (Recall)**: $Se = \\frac{TP}{TP + FN}$\n"
            "- **Positive Predictive Value (Precision)**: $PPV = \\frac{TP}{TP + FP}$\n"
            "- **$F_1$ Score**: $F_1 = 2 \\times \\frac{Se \\times PPV}{Se + PPV}$"
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\n"
            "import sys\n"
            "from pathlib import Path\n"
            "project_root = Path('..').resolve()\n"
            "if str(project_root) not in sys.path:\n"
            "    sys.path.insert(0, str(project_root))\n\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from src.data_loader import load_record\n"
            "from src.preprocessing import preprocess_ecg\n"
            "from src.peak_detection import detect_r_peaks\n"
            "from src.evaluation import evaluate_detection, print_evaluation_summary\n"
            "from src import config"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Run Pipeline and Compute Quantitative Evaluation"
        ),
        nbf.v4.new_code_cell(
            "ecg = load_record('100')\n"
            "clean_ecg = preprocess_ecg(ecg.signal, ecg.sampling_frequency)\n"
            "peaks = detect_r_peaks(clean_ecg, ecg.sampling_frequency)\n\n"
            "result = evaluate_detection(\n"
            "    detected_samples=peaks,\n"
            "    reference_samples=ecg.beat_samples,\n"
            "    fs=ecg.sampling_frequency,\n"
            "    tolerance_sec=0.150,\n"
            "    record_name='100'\n"
            ")\n"
            "print_evaluation_summary(result)"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Metric Visualizations (TP, FP, FN breakdown)"
        ),
        nbf.v4.new_code_cell(
            "metrics = ['Sensitivity', 'PPV', 'F1 Score']\n"
            "values = [result.sensitivity * 100, result.ppv * 100, result.f1 * 100]\n\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=120)\n\n"
            "# Bar chart of percentages\n"
            "bars = ax1.bar(metrics, values, color=['#2ca02c', '#1f77b4', '#9467bd'], width=0.5)\n"
            "ax1.set_ylim(95, 101)\n"
            "ax1.set_ylabel('Percentage (%)', fontsize=11)\n"
            "ax1.set_title('Detection Performance (Record 100)', fontweight='bold')\n"
            "ax1.grid(True, axis='y', alpha=0.3)\n"
            "for bar in bars:\n"
            "    y = bar.get_height()\n"
            "    ax1.text(bar.get_x() + bar.get_width()/2, y + 0.3, f'{y:.2f}%', ha='center', fontweight='bold')\n\n"
            "# Classification counts\n"
            "counts = [result.true_positives, result.false_positives, result.false_negatives]\n"
            "labels = [f'TP ({result.true_positives:,})', f'FP ({result.false_positives})', f'FN ({result.false_negatives})']\n"
            "colors = ['#2ca02c', '#ff7f0e', '#d62728']\n"
            "ax2.bar(labels, counts, color=colors, width=0.5)\n"
            "ax2.set_title('Beat Classification Breakdown', fontweight='bold')\n"
            "ax2.set_ylabel('Number of Beats', fontsize=11)\n"
            "ax2.grid(True, axis='y', alpha=0.3)\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 3. High-Resolution Visual Inspection of Detections vs Ground Truth\n"
            "Overlays detected peaks (circles) against reference annotations (vertical crosses)."
        ),
        nbf.v4.new_code_cell(
            "t_limit = 15.0\n"
            "mask = ecg.time <= t_limit\n"
            "t_plot = ecg.time[mask]\n"
            "sig_plot = clean_ecg[mask]\n\n"
            "# Detected peaks in slice\n"
            "det_in_mask = peaks[peaks < int(t_limit * ecg.sampling_frequency)]\n"
            "det_times = det_in_mask / ecg.sampling_frequency\n"
            "det_amps = clean_ecg[det_in_mask]\n\n"
            "# Ground truth beats in slice\n"
            "ref_in_mask = ecg.beat_samples[ecg.beat_samples < int(t_limit * ecg.sampling_frequency)]\n"
            "ref_times = ref_in_mask / ecg.sampling_frequency\n"
            "ref_amps = clean_ecg[ref_in_mask]\n\n"
            "plt.figure(figsize=(15, 5), dpi=120)\n"
            "plt.plot(t_plot, sig_plot, color='#1f77b4', lw=1.2, label='Filtered ECG')\n"
            "plt.scatter(ref_times, ref_amps, color='#2ca02c', s=100, marker='x', lw=2.5, zorder=4, label='Ground Truth Annotation')\n"
            "plt.scatter(det_times, det_amps, facecolors='none', edgecolors='#d62728', s=120, lw=2, zorder=5, label='Detected R-Peak')\n"
            "plt.title('Ground Truth vs. Detected Peaks (0–15 Seconds)', fontsize=13, fontweight='bold')\n"
            "plt.xlabel('Time (seconds)', fontsize=11)\n"
            "plt.ylabel('Amplitude (mV)', fontsize=11)\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.legend(loc='upper right')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    ]

    with open(nb_dir / "04_evaluation.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb4, f)
    print("Generated 04_evaluation.ipynb")

if __name__ == "__main__":
    generate_notebooks()
