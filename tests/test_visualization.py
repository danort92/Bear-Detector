"""Tests for visualization utilities."""

from __future__ import annotations

import numpy as np
import pytest


class TestDrawDetection:
    def _make_frame(self, h=100, w=100):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_returns_array(self):
        from src.utils.visualization import draw_detection

        frame = self._make_frame()
        result = draw_detection(frame, [10, 10, 50, 50], "bear", 0.9)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

    def test_modifies_frame(self):
        from src.utils.visualization import draw_detection

        frame = self._make_frame()
        original = frame.copy()
        draw_detection(frame, [10, 10, 50, 50], "bear", 0.9)
        assert not np.array_equal(frame, original)


class TestDrawMask:
    def test_mask_applied(self):
        from src.utils.visualization import draw_mask

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1

        result = draw_mask(frame, mask, track_id=0, alpha=0.5)
        # Masked region should no longer be all zeros
        assert result[30, 30].sum() > 0


class TestSaveTrainingCurves:
    def test_saves_png(self, tmp_path):
        from src.utils.visualization import save_training_curves

        history = {
            "loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
        }
        out = tmp_path / "curves.png"
        save_training_curves(history, out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestSaveConfusionMatrix:
    def test_saves_png(self, tmp_path):
        from src.utils.visualization import save_confusion_matrix

        cm = np.array([[50, 5], [3, 42]])
        out = tmp_path / "cm.png"
        save_confusion_matrix(cm, ["other", "bear"], out)
        assert out.exists()
