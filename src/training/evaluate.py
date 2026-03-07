"""Detection evaluation metrics: mAP, Precision, Recall, IoU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.utils.logging import get_logger
from src.utils.visualization import save_precision_recall_curve, save_confusion_matrix

logger = get_logger(__name__)


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(
    precisions: np.ndarray,
    recalls: np.ndarray,
) -> float:
    """Compute Average Precision using the 11-point interpolation method.

    Parameters
    ----------
    precisions:
        Precision values at decreasing confidence thresholds.
    recalls:
        Recall values at decreasing confidence thresholds.

    Returns
    -------
    float
        Average Precision score.
    """
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])

    # Ensure precision is monotonically decreasing
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Find recall level changes
    change_idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = float(np.sum((recalls[change_idx + 1] - recalls[change_idx]) * precisions[change_idx + 1]))
    return ap


def evaluate_detection(
    ground_truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    output_dir: str | Path = "outputs/metrics",
    save_plots: bool = True,
) -> dict[str, float]:
    """Evaluate detection predictions against ground-truth annotations.

    Parameters
    ----------
    ground_truths:
        List of dicts with keys ``"boxes"`` (list of [x1,y1,x2,y2]) per image.
    predictions:
        List of dicts with keys ``"boxes"``, ``"scores"`` per image,
        in the same image order as *ground_truths*.
    iou_threshold:
        IoU threshold for a detection to be considered a true positive.
    output_dir:
        Directory to save metric files and plots.
    save_plots:
        Whether to generate and save PR-curve plots.

    Returns
    -------
    dict
        Dictionary with ``"mAP"``, ``"precision"``, ``"recall"`` keys.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[float] = []
    all_tp: list[int] = []
    total_gt = 0

    for gt, pred in zip(ground_truths, predictions):
        gt_boxes = gt.get("boxes", [])
        pred_boxes = pred.get("boxes", [])
        pred_scores = pred.get("scores", [])
        total_gt += len(gt_boxes)

        matched_gt: set[int] = set()
        # Sort by descending confidence
        order = np.argsort(pred_scores)[::-1]

        for pi in order:
            pb = pred_boxes[pi]
            sc = pred_scores[pi]
            best_iou = 0.0
            best_gi = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = _iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_threshold and best_gi >= 0:
                matched_gt.add(best_gi)
                all_tp.append(1)
            else:
                all_tp.append(0)
            all_scores.append(sc)

    if not all_scores:
        logger.warning("No predictions found — returning zero metrics.")
        return {"mAP": 0.0, "precision": 0.0, "recall": 0.0}

    order = np.argsort(all_scores)[::-1]
    tp_cumsum = np.cumsum(np.array(all_tp)[order])
    fp_cumsum = np.cumsum(1 - np.array(all_tp)[order])

    recalls = tp_cumsum / (total_gt + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    ap = compute_ap(precisions, recalls)

    # Scalar P/R at max F1
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-6)
    best_idx = int(np.argmax(f1))

    metrics = {
        "mAP": round(float(ap), 4),
        "precision": round(float(precisions[best_idx]), 4),
        "recall": round(float(recalls[best_idx]), 4),
        "iou_threshold": iou_threshold,
        "total_gt": total_gt,
        "total_predictions": len(all_scores),
    }

    # Save metrics JSON
    metrics_file = output_dir / f"detection_metrics_iou{int(iou_threshold*100)}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Detection metrics saved to {metrics_file}: {metrics}")

    # Save PR curve
    if save_plots:
        pr_path = output_dir / "precision_recall_curve.png"
        save_precision_recall_curve(
            precisions.tolist(), recalls.tolist(), ap, pr_path, class_name="bear"
        )
        logger.info(f"PR curve saved to {pr_path}")

    return metrics


def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_scores: Sequence[float],
    output_dir: str | Path = "outputs/metrics",
    save_plots: bool = True,
) -> dict[str, float]:
    """Evaluate binary classification predictions.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0 or 1).
    y_pred:
        Predicted labels (0 or 1).
    y_scores:
        Predicted probabilities for the positive class.
    output_dir:
        Directory to save metric files and plots.
    save_plots:
        Whether to generate and save the confusion matrix.

    Returns
    -------
    dict
        ``{"accuracy", "precision", "recall", "f1", "auc_roc"}``
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    ap = average_precision_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
    }

    metrics_file = output_dir / "classification_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Classification metrics: {metrics}")

    if save_plots:
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_path = output_dir / "confusion_matrix.png"
        save_confusion_matrix(cm, ["other", "bear"], cm_path)

        # PR curve
        prs, recs, _ = precision_recall_curve(y_true, y_scores)
        pr_path = output_dir / "classification_pr_curve.png"
        save_precision_recall_curve(prs.tolist(), recs.tolist(), ap, pr_path, "bear")
        logger.info(f"Confusion matrix and PR curve saved to {output_dir}")

    return metrics
