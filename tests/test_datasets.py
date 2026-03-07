"""Tests for dataset loading."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from src.datasets.classification_dataset import BearClassificationDataset
from src.datasets.detection_dataset import BearDetectionDataset


class TestBearClassificationDataset:
    def test_len(self, tiny_bear_dataset):
        ds = BearClassificationDataset(
            tiny_bear_dataset["bear_dir"],
            tiny_bear_dataset["other_dir"],
        )
        assert len(ds) == 8  # 4 bear + 4 other

    def test_getitem_no_transform(self, tiny_bear_dataset):
        ds = BearClassificationDataset(
            tiny_bear_dataset["bear_dir"],
            tiny_bear_dataset["other_dir"],
        )
        img, label = ds[0]
        assert label in (0, 1)
        # Without transform, PIL Image is returned
        from PIL.Image import Image
        assert isinstance(img, Image)

    def test_class_counts(self, tiny_bear_dataset):
        ds = BearClassificationDataset(
            tiny_bear_dataset["bear_dir"],
            tiny_bear_dataset["other_dir"],
        )
        counts = ds.class_counts
        assert counts[0] == 4
        assert counts[1] == 4

    def test_class_weights_shape(self, tiny_bear_dataset):
        ds = BearClassificationDataset(
            tiny_bear_dataset["bear_dir"],
            tiny_bear_dataset["other_dir"],
        )
        weights = ds.class_weights
        assert weights.shape == (2,)
        assert np.all(weights > 0)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, ValueError)):
            ds = BearClassificationDataset(
                tmp_path / "nonexistent_bears",
                tmp_path / "nonexistent_other",
            )
            # Access is lazy for some errors; trigger load
            _ = len(ds)

    def test_with_transform(self, tiny_bear_dataset):
        pytest.importorskip("torchvision")
        import torchvision.transforms as T
        import torch

        transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        ds = BearClassificationDataset(
            tiny_bear_dataset["bear_dir"],
            tiny_bear_dataset["other_dir"],
            transform=transform,
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)


class TestBearDetectionDataset:
    def test_len(self, tiny_detection_dataset):
        ds = BearDetectionDataset(
            tiny_detection_dataset["images_dir"],
            tiny_detection_dataset["labels_dir"],
        )
        assert len(ds) == 3

    def test_getitem_keys(self, tiny_detection_dataset):
        ds = BearDetectionDataset(
            tiny_detection_dataset["images_dir"],
            tiny_detection_dataset["labels_dir"],
        )
        item = ds[0]
        assert "image_path" in item
        assert "boxes" in item
        assert len(item["boxes"]) == 1
        # Each box should be [class_id, cx, cy, w, h]
        assert len(item["boxes"][0]) == 5

    def test_empty_images_dir_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No images found"):
            BearDetectionDataset(empty_dir, empty_dir)
