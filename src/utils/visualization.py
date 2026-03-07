"""Visualization helpers: bounding boxes, masks, training curves, PR curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Drawing primitives
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (200, 149, 255),
    (199, 55, 255),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR colour for a given tracking ID."""
    return PALETTE[int(track_id) % len(PALETTE)]


def draw_detection(
    frame: np.ndarray,
    bbox: Sequence[float],
    label: str = "bear",
    conf: float = 1.0,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a single detection bounding box onto *frame*.

    Parameters
    ----------
    frame:
        BGR image (modified in-place and returned).
    bbox:
        [x1, y1, x2, y2] in pixel coordinates.
    label:
        Class label string.
    conf:
        Confidence score (shown in the overlay text).
    color:
        BGR colour tuple.
    thickness:
        Rectangle line thickness.

    Returns
    -------
    np.ndarray
        Annotated frame.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
    cv2.putText(
        frame, text, (x1, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_tracked_detection(
    frame: np.ndarray,
    bbox: Sequence[float],
    track_id: int,
    conf: float = 1.0,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a tracked bounding box with a unique colour per track ID.

    Parameters
    ----------
    frame:
        BGR image (modified in-place and returned).
    bbox:
        [x1, y1, x2, y2] in pixel coordinates.
    track_id:
        Integer tracking ID.
    conf:
        Detection confidence score.
    thickness:
        Rectangle line thickness.

    Returns
    -------
    np.ndarray
        Annotated frame.
    """
    color = _track_color(track_id)
    label = f"Bear #{track_id}"
    return draw_detection(frame, bbox, label, conf, color, thickness)


def draw_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    track_id: int = 0,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary instance segmentation mask onto *frame*.

    Parameters
    ----------
    frame:
        BGR image (modified in-place and returned).
    mask:
        Binary mask of shape (H, W) with values in {0, 1}.
    track_id:
        Used to select a consistent colour.
    alpha:
        Mask transparency (0 = fully transparent, 1 = opaque).

    Returns
    -------
    np.ndarray
        Frame with the semi-transparent mask overlay.
    """
    color = np.array(_track_color(track_id), dtype=np.uint8)
    colored = np.zeros_like(frame)
    colored[mask > 0] = color
    frame = cv2.addWeighted(frame, 1.0, colored, alpha, 0)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers (matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def save_training_curves(
    history: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    """Plot and save training / validation curves.

    Parameters
    ----------
    history:
        Dictionary mapping metric names to lists of per-epoch values.
        Keys with the prefix ``"val_"`` are treated as validation metrics.
    output_path:
        Destination path for the saved figure (PNG).
    """
    import matplotlib.pyplot as plt

    metrics = [k for k in history if not k.startswith("val_")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(history[metric], label="train")
        val_key = f"val_{metric}"
        if val_key in history:
            ax.plot(history[val_key], label="val")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_curve(
    precisions: Sequence[float],
    recalls: Sequence[float],
    ap: float,
    output_path: str | Path,
    class_name: str = "bear",
) -> None:
    """Plot and save a Precision-Recall curve.

    Parameters
    ----------
    precisions:
        Precision values at each threshold.
    recalls:
        Recall values at each threshold.
    ap:
        Average Precision (area under the PR curve).
    output_path:
        Destination path for the saved figure (PNG).
    class_name:
        Class label for the plot title.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {class_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    cm: Any,
    class_names: Sequence[str],
    output_path: str | Path,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm:
        Square confusion matrix array (true labels × predicted labels).
    class_names:
        Ordered list of class names.
    output_path:
        Destination path for the saved figure (PNG).
    """
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(max(4, len(class_names)), max(4, len(class_names))))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
