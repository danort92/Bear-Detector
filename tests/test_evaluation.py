"""Tests for detection and classification evaluation metrics."""

from __future__ import annotations

import pytest
import numpy as np


class TestDetectionEvaluation:
    def test_perfect_predictions(self, sample_detections, tmp_path):
        from src.training.evaluate import evaluate_detection

        gt, pred = sample_detections
        metrics = evaluate_detection(gt, pred, iou_threshold=0.5,
                                     output_dir=tmp_path, save_plots=False)
        assert metrics["mAP"] == pytest.approx(1.0, abs=0.01)
        assert metrics["precision"] == pytest.approx(1.0, abs=0.01)
        assert metrics["recall"] == pytest.approx(1.0, abs=0.01)

    def test_no_predictions(self, tmp_path):
        from src.training.evaluate import evaluate_detection

        gt = [{"boxes": [[10, 10, 50, 50]]}]
        pred = [{"boxes": [], "scores": []}]
        metrics = evaluate_detection(gt, pred, output_dir=tmp_path, save_plots=False)
        assert metrics["mAP"] == 0.0

    def test_all_false_positives(self, tmp_path):
        from src.training.evaluate import evaluate_detection

        gt = [{"boxes": []}]
        pred = [{"boxes": [[10, 10, 50, 50]], "scores": [0.9]}]
        metrics = evaluate_detection(gt, pred, output_dir=tmp_path, save_plots=False)
        assert metrics["precision"] == 0.0

    def test_metrics_saved_to_json(self, sample_detections, tmp_path):
        import json
        from src.training.evaluate import evaluate_detection

        gt, pred = sample_detections
        evaluate_detection(gt, pred, output_dir=tmp_path, save_plots=False)
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) >= 1
        with open(json_files[0]) as f:
            data = json.load(f)
        assert "mAP" in data


class TestComputeAP:
    def test_perfect_ap(self):
        from src.training.evaluate import compute_ap

        prec = np.array([1.0, 1.0, 1.0])
        rec = np.array([0.3, 0.6, 1.0])
        ap = compute_ap(prec, rec)
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_zero_ap(self):
        from src.training.evaluate import compute_ap

        prec = np.array([0.0, 0.0, 0.0])
        rec = np.array([0.3, 0.6, 1.0])
        ap = compute_ap(prec, rec)
        assert ap == pytest.approx(0.0, abs=0.01)
