"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path so 'src' is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> dict:
    """Load the default YAML config."""
    from src.utils.config import load_config
    return load_config(PROJECT_ROOT / "config" / "default.yaml")


@pytest.fixture
def tiny_bear_dataset(tmp_path: Path) -> dict:
    """Create a tiny in-memory bear/other classification dataset on disk."""
    from PIL import Image

    bear_dir = tmp_path / "bear_ct"
    other_dir = tmp_path / "other_ct"
    bear_dir.mkdir()
    other_dir.mkdir()

    # Create 4 bear images and 4 other images (solid colour, small size)
    for i in range(4):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(bear_dir / f"bear_{i:03d}.jpg")
        img.save(other_dir / f"other_{i:03d}.jpg")

    return {"bear_dir": bear_dir, "other_dir": other_dir}


@pytest.fixture
def tiny_detection_dataset(tmp_path: Path) -> dict:
    """Create a tiny YOLO-format detection dataset on disk."""
    from PIL import Image

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    for i in range(3):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(images_dir / f"img_{i:03d}.jpg")
        # YOLO label: class_id cx cy w h (normalised)
        (labels_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.3 0.3\n")

    return {"images_dir": images_dir, "labels_dir": labels_dir}


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a tiny synthetic video (10 frames) for inference tests."""
    import cv2

    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (64, 64))
    for _ in range(10):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def sample_detections() -> tuple[list, list]:
    """Return paired ground-truth and prediction dicts for evaluation tests."""
    ground_truths = [
        {"boxes": [[10, 10, 50, 50], [60, 60, 100, 100]]},
        {"boxes": [[20, 20, 80, 80]]},
    ]
    predictions = [
        {"boxes": [[12, 12, 48, 48], [62, 62, 98, 98]], "scores": [0.9, 0.8]},
        {"boxes": [[18, 18, 82, 82]], "scores": [0.7]},
    ]
    return ground_truths, predictions
