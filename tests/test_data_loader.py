"""Tests for the data_loader module."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_loader import load_record, ECGRecord
from src import config


class TestDataLoader:
    def test_load_existing_record_100(self):
        # Record 100 is present in data/raw
        if not (config.DATA_RAW_DIR / "100.dat").exists():
            pytest.skip("Record 100 raw data not downloaded.")

        ecg = load_record("100", download=False)

        assert isinstance(ecg, ECGRecord)
        assert ecg.record_name == "100"
        assert ecg.sampling_frequency == 360
        assert ecg.num_samples == 650000
        assert len(ecg.signal) == 650000
        assert len(ecg.time) == 650000
        assert ecg.num_beats > 2000
        assert len(ecg.beat_samples) == ecg.num_beats
        assert np.all(np.diff(ecg.beat_samples) > 0)
