"""MobileNetV2-based binary classification pipeline for bear detection."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

if TYPE_CHECKING:
    import numpy.typing as npt
    import tensorflow as tf

logger = logging.getLogger(__name__)


# ── Data preparation ─────────────────────────────────────────────────────────


def prepare_split_dirs(
    bear_dir: str | Path,
    other_dir: str | Path,
    train_dir: str | Path,
    val_dir: str | Path,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> tuple[int, int, int, int]:
    """Copy images from the raw *bear_dir* / *other_dir* into stratified
    train/val directory trees consumed by :class:`ImageDataGenerator`.

    Args:
        bear_dir: Source directory with bear images.
        other_dir: Source directory with non-bear images.
        train_dir: Destination root for training images.
        val_dir: Destination root for validation images.
        val_split: Fraction of images reserved for validation.
        random_seed: NumPy random seed for reproducibility.

    Returns:
        Tuple ``(n_train_bear, n_val_bear, n_train_other, n_val_other)``.
    """
    bear_dir, other_dir = Path(bear_dir), Path(other_dir)
    train_dir, val_dir = Path(train_dir), Path(val_dir)

    for split_root in (train_dir, val_dir):
        for cls in ("bear", "other"):
            (split_root / cls).mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png"}
    bear_files = [p for p in bear_dir.iterdir() if p.suffix.lower() in image_exts]
    other_files = [p for p in other_dir.iterdir() if p.suffix.lower() in image_exts]

    train_bear, val_bear = train_test_split(
        bear_files, test_size=val_split, random_state=random_seed
    )
    train_other, val_other = train_test_split(
        other_files, test_size=val_split, random_state=random_seed
    )

    for src in train_bear:
        shutil.copy(src, train_dir / "bear" / src.name)
    for src in val_bear:
        shutil.copy(src, val_dir / "bear" / src.name)
    for src in train_other:
        shutil.copy(src, train_dir / "other" / src.name)
    for src in val_other:
        shutil.copy(src, val_dir / "other" / src.name)

    logger.info(
        "Split: train bear=%d, val bear=%d | train other=%d, val other=%d",
        len(train_bear),
        len(val_bear),
        len(train_other),
        len(val_other),
    )
    return len(train_bear), len(val_bear), len(train_other), len(val_other)


# ── Model building ────────────────────────────────────────────────────────────


def build_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dense_units: int = 128,
    dropout_rate: float = 0.3,
) -> "tf.keras.Model":
    """Build a MobileNetV2 transfer-learning model for binary classification.

    The base model weights are frozen. A lightweight classification head with
    Dropout and Batch Normalisation is appended.

    Args:
        input_shape: ``(H, W, C)`` expected by the model.
        dense_units: Units in the hidden dense layer.
        dropout_rate: Dropout probability for regularisation.

    Returns:
        An uncompiled :class:`tf.keras.Model`.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import (
        BatchNormalization,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
    )
    from tensorflow.keras.models import Model

    tf.keras.backend.clear_session()

    base = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation="sigmoid")(x)

    return Model(inputs=base.input, outputs=output)


# ── Training ─────────────────────────────────────────────────────────────────


def train_classification_model(
    train_dir: str | Path,
    val_dir: str | Path,
    *,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    early_stopping_patience: int = 5,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
) -> tuple["tf.keras.Model", "tf.keras.callbacks.History"]:
    """End-to-end training for the classification model.

    Args:
        train_dir: Root with ``bear/`` and ``other/`` subdirectories.
        val_dir: Root with ``bear/`` and ``other/`` subdirectories.
        image_size: *(W, H)* to resize images to.
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        learning_rate: Adam optimiser initial learning rate.
        early_stopping_patience: Epochs without improvement before stopping.
        dense_units: Hidden layer units in the classification head.
        dropout_rate: Dropout probability.

    Returns:
        Tuple of ``(trained_model, history)``.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.metrics import Recall
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
    )
    val_gen = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    weights = class_weight.compute_class_weight(
        "balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weights = dict(enumerate(weights))
    logger.info("Class weights: %s", class_weights)

    model = build_model(
        input_shape=(*image_size, 3),
        dense_units=dense_units,
        dropout_rate=dropout_rate,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", Recall(name="recall")],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    return model, history


# ── Inference ─────────────────────────────────────────────────────────────────


def predict_image(
    model: "tf.keras.Model",
    image_batch: "npt.NDArray[np.float32]",
    threshold: float = 0.3,
) -> tuple[str, float]:
    """Run inference on a pre-processed image batch.

    The model output is ``P(other)``.  A lower *threshold* increases bear
    recall (more sensitive); a higher value increases precision.

    Args:
        model: Trained Keras model.
        image_batch: Array of shape ``(1, H, W, 3)`` in [0, 1].
        threshold: Minimum ``P(bear)`` required to label an image as "Bear".
                   ``P(bear) = 1 - model_output``.

    Returns:
        Tuple ``("Bear" | "Other", bear_probability)``.
    """
    raw: float = float(model.predict(image_batch, verbose=0)[0, 0])
    bear_prob = 1.0 - raw
    label = "Bear" if bear_prob >= threshold else "Other"
    return label, bear_prob
