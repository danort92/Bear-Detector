"""Configuration dataclasses for classification and detection pipelines."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassificationConfig:
    """Hyper-parameters and paths for the MobileNetV2 classification pipeline."""

    # Data
    bear_dir: str = "ct/bear_ct"
    other_dir: str = "ct/other_ct"
    train_dir: str = "train"
    val_dir: str = "val"
    val_split: float = 0.2
    random_seed: int = 42
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32

    # Training
    epochs: int = 20
    learning_rate: float = 1e-4
    early_stopping_patience: int = 5
    dropout_rate: float = 0.3
    dense_units: int = 128

    # Inference
    threshold: float = 0.3  # P(bear) >= threshold → "Bear"

    # Augmentation
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True


@dataclass
class DetectionConfig:
    """Hyper-parameters for the YOLOv8 detection pipeline."""

    model_name: str = "yolov8n.pt"
    image_size: int = 640
    batch_size: int = 16
    epochs: int = 50
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    augment: bool = True
    half_precision: bool = True

    # Paths (relative to repo root)
    dataset_dir: str = "Bear detection.v3i.yolov8-obb"
    output_dir: str = "runs"
