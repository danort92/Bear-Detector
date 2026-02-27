"""Shared utility functions for image loading, preprocessing, and visualisation."""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

# ── Image helpers ────────────────────────────────────────────────────────────


def load_and_preprocess(
    image_path: str | Path,
    target_size: tuple[int, int] = (224, 224),
) -> npt.NDArray[np.float32] | None:
    """Load an image from *image_path*, resize and normalise to [0, 1].

    Args:
        image_path: Path to a JPEG or PNG file.
        target_size: *(width, height)* to resize to.

    Returns:
        Float32 array of shape ``(H, W, 3)`` or ``None`` if loading fails.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Could not load image: %s", image_path)
        return None
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image


def batch_from_path(
    image_path: str | Path,
    target_size: tuple[int, int] = (224, 224),
) -> npt.NDArray[np.float32] | None:
    """Return a ``(1, H, W, 3)`` batch ready for ``model.predict()``.

    Returns ``None`` if the image cannot be loaded.
    """
    image = load_and_preprocess(image_path, target_size)
    if image is None:
        return None
    return np.expand_dims(image, axis=0)


def collect_images_from_dir(directory: str | Path) -> list[Path]:
    """Return all JPEG / PNG paths in *directory* (non-recursive)."""
    directory = Path(directory)
    extensions = {".jpg", ".jpeg", ".png"}
    return [p for p in directory.iterdir() if p.suffix.lower() in extensions]


# ── ZIP helpers ──────────────────────────────────────────────────────────────


def extract_zip(zip_path: str | Path, dest_dir: str | Path) -> Path:
    """Extract *zip_path* into *dest_dir* and return the destination path."""
    zip_path, dest_dir = Path(zip_path), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extracted %s → %s", zip_path.name, dest_dir)
    return dest_dir


# ── YAML helpers ─────────────────────────────────────────────────────────────


def find_file(directory: str | Path, filename: str) -> Path | None:
    """Recursively search for *filename* inside *directory*.

    Returns the first match as a :class:`pathlib.Path`, or ``None``.
    """
    for root, _dirs, files in os.walk(directory):
        if filename in files:
            return Path(root) / filename
    return None


def find_images_dir(base_dir: str | Path, split: str) -> Path | None:
    """Return the images sub-directory for a given dataset *split* name.

    Looks for a directory whose path contains both *split* and ``images``.
    """
    for dirpath, _dirnames, _filenames in os.walk(base_dir):
        p = Path(dirpath)
        if split in p.parts and "images" in p.parts:
            return p
    return None
