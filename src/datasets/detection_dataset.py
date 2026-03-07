"""Detection dataset helpers — thin wrappers for YOLO-format data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # type: ignore[no-redef]
        pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class BearDetectionDataset(Dataset):
    """YOLO-format object detection dataset for bears.

    Expects images and labels in separate directories following the standard
    YOLOv8 split layout::

        split/
            images/
                *.jpg
            labels/
                *.txt

    Each label file contains one row per bounding box in YOLO format::

        <class_id> <cx> <cy> <w> <h>

    All coordinates are normalised to [0, 1].

    Parameters
    ----------
    images_dir:
        Path to the directory containing image files.
    labels_dir:
        Path to the directory containing label ``.txt`` files.
    transform:
        Optional image transform callable.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        transform=None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        self.image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / image_path.with_suffix(".txt").name

        from PIL import Image
        import numpy as np

        image = Image.open(image_path).convert("RGB")

        boxes: list[list[float]] = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append([float(v) for v in parts[:5]])

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_path": str(image_path),
            "boxes": boxes,  # list of [class_id, cx, cy, w, h]
        }
