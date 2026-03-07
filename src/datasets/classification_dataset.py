"""PyTorch Dataset and DataLoader factory for bear binary classification.

Folder structure expected::

    root/
        bear_ct/    # positive class — bears
        other_ct/   # negative class — non-bears
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

# Gracefully handle missing torchvision
try:
    import torchvision.transforms as T
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class BearClassificationDataset(Dataset):
    """Binary classification dataset: bear (1) vs other (0).

    Parameters
    ----------
    bear_dir:
        Directory containing bear images (positive class).
    other_dir:
        Directory containing non-bear images (negative class).
    transform:
        Optional callable that transforms a PIL image into a tensor.
        If ``None``, the raw PIL image is returned.
    """

    def __init__(
        self,
        bear_dir: str | Path,
        other_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for path in sorted(Path(bear_dir).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                self.samples.append((path, 1))

        for path in sorted(Path(other_dir).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                self.samples.append((path, 0))

        if not self.samples:
            raise ValueError(
                f"No images found in '{bear_dir}' or '{other_dir}'. "
                "Check your data paths."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def class_counts(self) -> dict[int, int]:
        """Return counts per class label."""
        from collections import Counter
        return dict(Counter(label for _, label in self.samples))

    @property
    def class_weights(self) -> np.ndarray:
        """Balanced class weights for use with weighted loss functions."""
        counts = self.class_counts
        total = sum(counts.values())
        n_classes = len(counts)
        weights = np.array(
            [total / (n_classes * counts[c]) for c in sorted(counts.keys())],
            dtype=np.float32,
        )
        return weights

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ──────────────────────────────────────────────────────────────────────────────
# Transform factories
# ──────────────────────────────────────────────────────────────────────────────

def _default_train_transform(image_size: tuple[int, int] = (224, 224)) -> Callable:
    """Standard augmentation transform for training."""
    if not _HAS_TORCHVISION:
        raise ImportError("torchvision is required. Install it with: pip install torchvision")
    return T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _default_val_transform(image_size: tuple[int, int] = (224, 224)) -> Callable:
    """Deterministic transform for validation and inference."""
    if not _HAS_TORCHVISION:
        raise ImportError("torchvision is required. Install it with: pip install torchvision")
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_classification_dataloaders(
    bear_dir: str | Path,
    other_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """Build train and validation DataLoaders for classification.

    Parameters
    ----------
    bear_dir:
        Directory of bear images.
    other_dir:
        Directory of non-bear images.
    image_size:
        Target (H, W) for resizing.
    val_split:
        Fraction of data used for validation.
    batch_size:
        Number of samples per batch.
    num_workers:
        Number of DataLoader worker processes.
    seed:
        Random seed for the train/val split.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, class_weights)`` where *class_weights* is
        a NumPy array of shape ``(2,)`` for use in a weighted loss.
    """
    import torch
    from src.utils.seed import worker_init_fn

    full_dataset = BearClassificationDataset(bear_dir, other_dir)
    class_weights = full_dataset.class_weights

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Apply different transforms to train and val splits
    train_subset.dataset = BearClassificationDataset(
        bear_dir, other_dir, transform=_default_train_transform(image_size)
    )
    val_subset.dataset = BearClassificationDataset(
        bear_dir, other_dir, transform=_default_val_transform(image_size)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, class_weights


class BearDetectionDataset(Dataset):
    """Minimal wrapper around a YOLO-format detection dataset.

    This class is lightweight: it only reads image paths and their label
    files so the dataset size and structure can be verified without loading
    every image into memory.

    Parameters
    ----------
    images_dir:
        Directory containing ``.jpg``/``.png`` images.
    labels_dir:
        Directory containing YOLO-format ``.txt`` label files.
    """

    def __init__(self, images_dir: str | Path, labels_dir: str | Path) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / image_path.with_suffix(".txt").name
        return {"image_path": str(image_path), "label_path": str(label_path)}
