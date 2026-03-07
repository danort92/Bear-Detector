"""Segmentation dataset for bear instance segmentation.

Expects YOLO-segmentation format where each label file contains rows of::

    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

All polygon coordinates are normalised to [0, 1].
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # type: ignore[no-redef]
        pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class BearSegmentationDataset(Dataset):
    """YOLO-segmentation format dataset for bear instance segmentation.

    Parameters
    ----------
    images_dir:
        Directory containing image files.
    labels_dir:
        Directory containing YOLO-seg ``.txt`` label files.
    image_size:
        Target (H, W) for resizing images.
    transform:
        Optional image transform callable.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        image_size: tuple[int, int] = (640, 640),
        transform: Optional[Callable] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
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

        image = Image.open(image_path).convert("RGB")
        if self.image_size:
            image = image.resize(self.image_size[::-1])  # PIL uses (W, H)

        segments: list[dict] = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        coords = [float(v) for v in parts[1:]]
                        # Pair up x, y coordinates
                        polygon = [
                            (coords[i], coords[i + 1])
                            for i in range(0, len(coords) - 1, 2)
                        ]
                        segments.append({"class_id": class_id, "polygon": polygon})

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_path": str(image_path),
            "segments": segments,
        }

    @staticmethod
    def polygon_to_mask(
        polygon: list[tuple[float, float]],
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Convert a normalised polygon to a binary mask.

        Parameters
        ----------
        polygon:
            List of (x, y) tuples with coordinates in [0, 1].
        image_size:
            (H, W) of the target mask.

        Returns
        -------
        np.ndarray
            Binary mask of shape (H, W) with dtype ``np.uint8``.
        """
        import cv2

        h, w = image_size
        pts = np.array(
            [(int(x * w), int(y * h)) for x, y in polygon],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return mask
