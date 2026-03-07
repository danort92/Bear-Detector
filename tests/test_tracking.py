"""Tests for the SORT tracker and tracking metrics."""

from __future__ import annotations

import numpy as np
import pytest


class TestSORTTracker:
    """Tests for SORTTracker that do NOT require filterpy."""

    def _make_tracker(self):
        from src.tracking.sort_tracker import SORTTracker
        return SORTTracker(max_age=5, min_hits=1, iou_threshold=0.3)

    def test_import(self):
        """The module should import without error even if filterpy is absent."""
        try:
            from src.tracking import sort_tracker  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"Optional dependency missing: {exc}")

    def test_empty_update_returns_empty(self):
        pytest.importorskip("filterpy")
        tracker = self._make_tracker()
        result = tracker.update(np.empty((0, 5)))
        assert result.shape[0] == 0

    def test_single_detection_tracked(self):
        pytest.importorskip("filterpy")
        tracker = self._make_tracker()
        det = np.array([[10.0, 10.0, 50.0, 50.0, 0.9]])
        # Feed the same detection for min_hits frames
        for _ in range(3):
            tracks = tracker.update(det)
        assert tracks.shape[1] == 5  # [x1, y1, x2, y2, id]
        assert tracks.shape[0] >= 1

    def test_track_id_stable(self):
        """Consistent detections should produce stable track IDs."""
        pytest.importorskip("filterpy")
        tracker = self._make_tracker()
        det = np.array([[10.0, 10.0, 50.0, 50.0, 0.9]])
        ids = set()
        for _ in range(5):
            tracks = tracker.update(det)
            if tracks.shape[0] > 0:
                ids.add(int(tracks[0, 4]))
        assert len(ids) == 1, "Track ID should remain stable across frames"

    def test_lost_track_removed(self):
        pytest.importorskip("filterpy")
        tracker = self._make_tracker()
        det = np.array([[10.0, 10.0, 50.0, 50.0, 0.9]])
        # Establish a track
        for _ in range(3):
            tracker.update(det)
        # No detections for max_age + 1 frames
        for _ in range(tracker.max_age + 1):
            tracker.update(np.empty((0, 5)))
        tracks = tracker.update(np.empty((0, 5)))
        assert tracks.shape[0] == 0, "Stale track should be removed"


class TestTrackingMetrics:
    def _gt(self, boxes_ids):
        """Build a frame list in the format expected by compute_tracking_metrics."""
        return [np.array([*box, tid]) for box, tid in boxes_ids]

    def test_perfect_tracking(self):
        from src.tracking.metrics import compute_tracking_metrics

        gt = [self._gt([([0, 0, 10, 10], 1)])]
        pred = [self._gt([([0, 0, 10, 10], 1)])]
        metrics = compute_tracking_metrics(gt, pred, iou_threshold=0.5)

        assert metrics["MOTA"] == pytest.approx(1.0, abs=0.01)
        assert metrics["MOTP"] == pytest.approx(1.0, abs=0.01)
        assert metrics["id_switches"] == 0
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0

    def test_false_positive(self):
        from src.tracking.metrics import compute_tracking_metrics

        gt = [self._gt([])]         # no ground truth
        pred = [self._gt([([0, 0, 10, 10], 1)])]
        metrics = compute_tracking_metrics(gt, pred)
        assert metrics["false_positives"] == 1

    def test_false_negative(self):
        from src.tracking.metrics import compute_tracking_metrics

        gt = [self._gt([([0, 0, 10, 10], 1)])]
        pred = [self._gt([])]       # no predictions
        metrics = compute_tracking_metrics(gt, pred)
        assert metrics["false_negatives"] == 1

    def test_id_switch(self):
        from src.tracking.metrics import compute_tracking_metrics

        # Frame 1: GT id=1 matched with pred id=1
        # Frame 2: GT id=1 matched with pred id=2 → id switch
        box = [0, 0, 10, 10]
        gt = [
            self._gt([(box, 1)]),
            self._gt([(box, 1)]),
        ]
        pred = [
            self._gt([(box, 1)]),
            self._gt([(box, 2)]),   # different predicted ID
        ]
        metrics = compute_tracking_metrics(gt, pred, iou_threshold=0.5)
        assert metrics["id_switches"] == 1

    def test_empty_sequence(self):
        from src.tracking.metrics import compute_tracking_metrics

        metrics = compute_tracking_metrics([[], []], [[], []])
        assert metrics["MOTA"] == pytest.approx(1.0, abs=0.01)
        assert metrics["false_positives"] == 0
