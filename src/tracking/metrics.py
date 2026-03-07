"""Tracking evaluation metrics: MOTA, MOTP, and ID switches.

These metrics follow the MOTChallenge conventions.

Reference:
    Bernardin & Stiefelhagen, "Evaluating Multiple Object Tracking
    Performance: The CLEAR MOT Metrics", J. Image Video Proc. 2008.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_tracking_metrics(
    ground_truths: list[list[np.ndarray]],
    predictions: list[list[np.ndarray]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute MOTA, MOTP, and ID switches over a sequence of frames.

    Parameters
    ----------
    ground_truths:
        Per-frame list of ground-truth boxes. Each frame is a list of
        arrays with shape (5,): [x1, y1, x2, y2, gt_id].
    predictions:
        Per-frame list of predicted tracks. Each frame is a list of
        arrays with shape (5,): [x1, y1, x2, y2, track_id].
    iou_threshold:
        Minimum IoU to consider a detection a true positive.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``"MOTA"`` — Multiple Object Tracking Accuracy
        - ``"MOTP"`` — Multiple Object Tracking Precision
        - ``"id_switches"`` — Total number of ID switches
        - ``"false_positives"`` — Total FP count
        - ``"false_negatives"`` — Total FN count
        - ``"precision"`` — Detection precision
        - ``"recall"`` — Detection recall
    """
    assert len(ground_truths) == len(predictions), (
        "ground_truths and predictions must have the same number of frames"
    )

    total_fp = 0
    total_fn = 0
    total_id_sw = 0
    total_gt = 0
    total_tp = 0
    total_iou = 0.0

    # Map gt_id -> last matched track_id to detect ID switches
    id_map: dict[int, int] = {}

    for gt_frame, pred_frame in zip(ground_truths, predictions):
        gt_boxes = np.array([g[:4] for g in gt_frame]) if gt_frame else np.empty((0, 4))
        gt_ids = [int(g[4]) for g in gt_frame] if gt_frame else []
        pred_boxes = np.array([p[:4] for p in pred_frame]) if pred_frame else np.empty((0, 4))
        pred_ids = [int(p[4]) for p in pred_frame] if pred_frame else []

        total_gt += len(gt_frame)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            # Build IoU matrix
            iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))
            for gi, gb in enumerate(gt_boxes):
                for pi, pb in enumerate(pred_boxes):
                    iou_mat[gi, pi] = _iou(gb, pb)

            # Greedy matching by highest IoU
            while True:
                best = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                gi, pi = best
                if iou_mat[gi, pi] < iou_threshold:
                    break

                # Check ID switch
                gt_id = gt_ids[gi]
                pred_id = pred_ids[pi]
                if gt_id in id_map and id_map[gt_id] != pred_id:
                    total_id_sw += 1
                id_map[gt_id] = pred_id

                total_tp += 1
                total_iou += iou_mat[gi, pi]
                matched_gt.add(gi)
                matched_pred.add(pi)
                iou_mat[gi, :] = -1
                iou_mat[:, pi] = -1

        total_fn += len(gt_frame) - len(matched_gt)
        total_fp += len(pred_frame) - len(matched_pred)

    # MOTA = 1 - (FN + FP + ID_sw) / GT
    denom = total_gt if total_gt > 0 else 1
    mota = 1.0 - (total_fn + total_fp + total_id_sw) / denom

    # MOTP = sum(IoU for all TPs) / total TPs
    motp = total_iou / total_tp if total_tp > 0 else 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "MOTA": round(mota, 4),
        "MOTP": round(motp, 4),
        "id_switches": total_id_sw,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
