"""
Centralized configuration for the ECG Heartbeat Analyzer.

All tunable parameters live here so they are easy to find, change,
and document.  No magic numbers should appear in other modules.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_SIGNALS_DIR = PROJECT_ROOT / "outputs" / "signals"
OUTPUT_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
# PhysioNet database name used by wfdb.dl_database / wfdb.rdrecord
PHYSIONET_DB = "mitdb"

# Default record to start with
DEFAULT_RECORD = "100"

# MIT-BIH records (all 48)
ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209",
    "210", "212", "213", "214", "215", "217", "219",
    "220", "221", "222", "223", "228", "230", "231", "232",
    "233", "234",
]

# ──────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────
SAMPLING_RATE = 360          # Hz  (MIT-BIH standard)
NYQUIST_FREQ = SAMPLING_RATE / 2  # 180 Hz

# ──────────────────────────────────────────────────────────────────────
# Filtering
# ──────────────────────────────────────────────────────────────────────
HIGH_PASS_CUTOFF = 0.5       # Hz — removes baseline wander
LOW_PASS_CUTOFF = 40.0       # Hz — removes high-frequency noise
NOTCH_FREQUENCY = 50.0       # Hz — power-line interference (India)
NOTCH_QUALITY_FACTOR = 30.0  # Q of the notch filter
FILTER_ORDER = 4             # Butterworth filter order

# ──────────────────────────────────────────────────────────────────────
# R-Peak Detection  (Pan–Tompkins-style)
# ──────────────────────────────────────────────────────────────────────
# Bandpass for the detection stage (different from the cleaning filters)
DETECTION_LOW_CUTOFF = 5.0   # Hz
DETECTION_HIGH_CUTOFF = 15.0 # Hz

# Moving-window integration window duration
INTEGRATION_WINDOW_SEC = 0.150  # seconds

# Refractory period — minimum time between two R-peaks
REFRACTORY_PERIOD_SEC = 0.200   # seconds  (~300 BPM ceiling)

# R-peak refinement: search window half-width around a candidate
REFINEMENT_WINDOW_SEC = 0.075   # seconds

# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
# Tolerance for matching a detected peak to a reference annotation
MATCHING_TOLERANCE_SEC = 0.150  # seconds  (±150 ms)

# Annotation symbols that count as "beat" in MIT-BIH
# See https://www.physionet.org/physiobank/annotations.shtml
BEAT_ANNOTATION_SYMBOLS = [
    "N",  # Normal beat
    "L",  # Left bundle branch block beat
    "R",  # Right bundle branch block beat
    "A",  # Atrial premature beat
    "a",  # Aberrated atrial premature beat
    "J",  # Junctional premature beat
    "S",  # Supraventricular premature beat
    "V",  # Premature ventricular contraction
    "F",  # Fusion of ventricular and normal beat
    "e",  # Atrial escape beat
    "j",  # Junctional escape beat
    "E",  # Ventricular escape beat
    "/",  # Paced beat
    "f",  # Fusion of paced and normal beat
    "Q",  # Unclassifiable beat
]

# ──────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
DEFAULT_PLOT_DURATION_SEC = 10  # seconds of ECG to show by default

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
