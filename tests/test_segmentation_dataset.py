"""Tests for the segmentation dataset."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def tiny_seg_dataset(tmp_path):
    """Create a tiny YOLO-seg format dataset."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(images_dir / f"img_{i:03d}.jpg")
        # YOLO-seg label: class_id x1 y1 x2 y2 x3 y3 x4 y4 (polygon)
        (labels_dir / f"img_{i:03d}.txt").write_text(
            "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        )

    return {"images_dir": images_dir, "labels_dir": labels_dir}


class TestBearSegmentationDataset:
    def test_len(self, tiny_seg_dataset):
        from src.segmentation.dataset import BearSegmentationDataset

        ds = BearSegmentationDataset(
            tiny_seg_dataset["images_dir"],
            tiny_seg_dataset["labels_dir"],
        )
        assert len(ds) == 3

    def test_getitem_has_segments(self, tiny_seg_dataset):
        from src.segmentation.dataset import BearSegmentationDataset

        ds = BearSegmentationDataset(
            tiny_seg_dataset["images_dir"],
            tiny_seg_dataset["labels_dir"],
        )
        item = ds[0]
        assert "segments" in item
        assert len(item["segments"]) == 1
        seg = item["segments"][0]
        assert seg["class_id"] == 0
        assert len(seg["polygon"]) == 4  # 4 polygon points

    def test_polygon_to_mask(self):
        from src.segmentation.dataset import BearSegmentationDataset

        polygon = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        mask = BearSegmentationDataset.polygon_to_mask(polygon, (100, 100))
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        # Centre pixel should be masked
        assert mask[50, 50] == 1
        # Corner pixel should not be masked
        assert mask[0, 0] == 0
