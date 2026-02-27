"""YOLOv8 detection pipeline — training, inference and video processing."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import yaml

from .utils import find_file, find_images_dir

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

_BBOX_COLOR = (0, 255, 0)   # BGR green
_TEXT_COLOR = (0, 255, 0)
_LABEL_TEXT = "Bear"
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Dataset setup ─────────────────────────────────────────────────────────────


def patch_yaml_paths(
    yaml_path: Path,
    train_path: Path,
    val_path: Path,
) -> None:
    """Overwrite ``train`` and ``val`` keys in a YOLOv8 *data.yaml* file.

    A backup copy (``data.yaml.bak``) is written before any modification.

    Args:
        yaml_path: Path to the ``data.yaml`` file.
        train_path: Absolute path to the training images directory.
        val_path: Absolute path to the validation images directory.
    """
    backup = yaml_path.with_suffix(".yaml.bak")
    shutil.copy(yaml_path, backup)
    logger.info("Backed up %s → %s", yaml_path.name, backup.name)

    with yaml_path.open("r") as fh:
        data = yaml.safe_load(fh)

    data["train"] = str(train_path)
    data["val"] = str(val_path)

    with yaml_path.open("w") as fh:
        yaml.safe_dump(data, fh)
    logger.info("Patched %s with train/val paths", yaml_path.name)


def setup_local_dataset(base_dir: str | Path) -> Path:
    """Locate and patch the ``data.yaml`` inside a local YOLOv8 dataset.

    Args:
        base_dir: Root of the Roboflow-style dataset directory.

    Returns:
        Path to the patched ``data.yaml`` file.

    Raises:
        FileNotFoundError: If ``data.yaml``, ``train/images``, or
            ``val/images`` (or ``test/images``) cannot be found.
    """
    base_dir = Path(base_dir).resolve()

    yaml_path = find_file(base_dir, "data.yaml")
    if yaml_path is None:
        raise FileNotFoundError(f"data.yaml not found inside {base_dir}")

    train_path = find_images_dir(base_dir, "train")
    # Accept both "val" and "test" directory names
    val_path = find_images_dir(base_dir, "val") or find_images_dir(base_dir, "test")

    if train_path is None or val_path is None:
        raise FileNotFoundError(
            f"Could not locate train/val image directories inside {base_dir}"
        )

    patch_yaml_paths(yaml_path, train_path, val_path)
    return yaml_path


def setup_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    download_dir: str | Path = ".",
) -> Path:
    """Download a Roboflow dataset and return the patched ``data.yaml`` path.

    Args:
        api_key: Roboflow API key (read from environment in the notebook).
        workspace: Roboflow workspace name.
        project: Roboflow project name.
        version: Dataset version number.
        download_dir: Directory where the dataset will be downloaded.

    Returns:
        Path to the patched ``data.yaml``.
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=str(download_dir))

    return setup_local_dataset(dataset.location)


# ── Training ──────────────────────────────────────────────────────────────────


def train_detection_model(
    data_yaml: str | Path,
    *,
    model_name: str = "yolov8n.pt",
    epochs: int = 50,
    image_size: int = 640,
    batch_size: int = 16,
    optimizer: str = "AdamW",
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    augment: bool = True,
    half_precision: bool = True,
) -> "YOLO":
    """Train a YOLOv8 model and return the trained instance.

    Args:
        data_yaml: Path to the (patched) ``data.yaml`` file.
        model_name: Base YOLOv8 model checkpoint (e.g. ``"yolov8n.pt"``).
        epochs: Number of training epochs (≥50 recommended for convergence).
        image_size: Training image resolution.
        batch_size: Mini-batch size.
        optimizer: Optimiser name supported by Ultralytics.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularisation coefficient.
        augment: Whether to apply Ultralytics built-in augmentation.
        half_precision: Use FP16 mixed precision when GPU is available.

    Returns:
        Trained :class:`ultralytics.YOLO` model instance.
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        optimizer=optimizer,
        lr0=learning_rate,
        weight_decay=weight_decay,
        augment=augment,
        half=half_precision,
    )
    return model


# ── Video inference ───────────────────────────────────────────────────────────


def process_video(
    video_path: str | Path,
    model: "YOLO",
    output_path: str | Path | None = None,
    conf_threshold: float = 0.25,
) -> int:
    """Run YOLOv8 detection on every frame of *video_path*.

    Bounding boxes are drawn on the BGR frame (OpenCV native format) and
    optionally written to *output_path* as an MP4 file.

    Args:
        video_path: Input video file.
        model: Loaded :class:`ultralytics.YOLO` model.
        output_path: Path for the annotated output video. If ``None``,
            no video is written (useful for live preview).
        conf_threshold: Minimum confidence to display a detection.

    Returns:
        Number of frames processed.

    Raises:
        OSError: If *video_path* cannot be opened.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer: cv2.VideoWriter | None = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # YOLOv8 accepts BGR natively — no colour conversion needed
            results = model.predict(frame, verbose=False, conf=conf_threshold)

            for bbox in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), _BBOX_COLOR, 2)
                cv2.putText(
                    frame,
                    _LABEL_TEXT,
                    (x1, y1 - 10),
                    _FONT,
                    0.6,
                    _TEXT_COLOR,
                    2,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(frame)

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    logger.info("Processed %d frames from %s", frame_count, video_path.name)
    return frame_count
