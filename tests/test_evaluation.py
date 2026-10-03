"""Tests for the evaluation module."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import evaluate_detection, EvaluationResult


class TestEvaluation:
    def test_perfect_detection(self):
        ref = np.array([100, 300, 500, 700], dtype=int)
        det = np.array([100, 300, 500, 700], dtype=int)
        res = evaluate_detection(det, ref, fs=360, tolerance_sec=0.15)

        assert res.true_positives == 4
        assert res.false_positives == 0
        assert res.false_negatives == 0
        assert res.sensitivity == pytest.approx(1.0)
        assert res.ppv == pytest.approx(1.0)
        assert res.f1 == pytest.approx(1.0)

    def test_within_tolerance(self):
        ref = np.array([100, 300, 500], dtype=int)
        # 10 samples at 360 Hz is ~27.7 ms, well within 150 ms (54 samples)
        det = np.array([110, 295, 505], dtype=int)
        res = evaluate_detection(det, ref, fs=360, tolerance_sec=0.15)

        assert res.true_positives == 3
        assert res.false_positives == 0
        assert res.false_negatives == 0
        assert res.sensitivity == pytest.approx(1.0)

    def test_outside_tolerance(self):
        ref = np.array([100], dtype=int)
        # 100 samples at 360 Hz is ~277 ms, outside 150 ms tolerance
        det = np.array([200], dtype=int)
        res = evaluate_detection(det, ref, fs=360, tolerance_sec=0.15)

        assert res.true_positives == 0
        assert res.false_positives == 1
        assert res.false_negatives == 1
        assert res.sensitivity == pytest.approx(0.0)
        assert res.ppv == pytest.approx(0.0)
        assert res.f1 == pytest.approx(0.0)

    def test_empty_detected(self):
        ref = np.array([100, 200, 300], dtype=int)
        det = np.array([], dtype=int)
        res = evaluate_detection(det, ref, fs=360)

        assert res.true_positives == 0
        assert res.false_positives == 0
        assert res.false_negatives == 3
        assert res.sensitivity == pytest.approx(0.0)
        assert res.ppv == pytest.approx(0.0)
        assert res.f1 == pytest.approx(0.0)

    def test_empty_reference(self):
        ref = np.array([], dtype=int)
        det = np.array([100, 200], dtype=int)
        res = evaluate_detection(det, ref, fs=360)

        assert res.true_positives == 0
        assert res.false_positives == 2
        assert res.false_negatives == 0
        assert res.sensitivity == pytest.approx(0.0)
        assert res.ppv == pytest.approx(0.0)
        assert res.f1 == pytest.approx(0.0)

    def test_greedy_one_to_one_matching(self):
        # Two detections very close to a single reference beat
        ref = np.array([100], dtype=int)
        det = np.array([98, 102], dtype=int)
        res = evaluate_detection(det, ref, fs=360, tolerance_sec=0.15)

        # Only one detection can be matched to the single reference beat
        assert res.true_positives == 1
        assert res.false_positives == 1
        assert res.false_negatives == 0
        assert res.sensitivity == pytest.approx(1.0)
        assert res.ppv == pytest.approx(0.5)

    def test_representation(self):
        res = EvaluationResult(
            record_name="100",
            total_reference_beats=10,
            total_detected_beats=10,
            true_positives=10,
            false_positives=0,
            false_negatives=0,
            sensitivity=1.0,
            ppv=1.0,
            f1=1.0,
            tolerance_sec=0.15,
        )
        assert "EvaluationResult" in repr(res)
