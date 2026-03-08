"""SORT (Simple Online and Realtime Tracking) implementation.

Reference:
    Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
    https://arxiv.org/abs/1602.00763

This is a self-contained NumPy implementation that does not require
additional external packages beyond NumPy and SciPy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────────────────────────────────────
# Kalman filter for a single track
# ──────────────────────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """Track a single bounding box using a constant-velocity Kalman filter.

    State vector: [cx, cy, s, r, dcx, dcy, ds]
    where (cx, cy) is the box centre, s is the area, r is the aspect ratio,
    and (dcx, dcy, ds) are their velocities.
    """

    count = 0

    def __init__(self, bbox: np.ndarray, conf: float = 1.0) -> None:
        """Initialise a tracker from a detection bbox [x1, y1, x2, y2]."""
        from filterpy.kalman import KalmanFilter  # optional but preferred

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 0
        self.no_loss_streak = 0
        self.time_since_update = 0
        self.history: list[np.ndarray] = []
        self.age = 0
        self.conf = conf  # confidence of the last matched detection

        # Kalman filter matrices
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _bbox_to_z(bbox)

    def predict(self) -> np.ndarray:
        """Advance state estimate. Return predicted [x1, y1, x2, y2]."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        self.history.append(_z_to_bbox(self.kf.x))
        return self.history[-1]

    def update(self, bbox: np.ndarray, conf: float = 1.0) -> None:
        """Update the state with an observed bounding box."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.no_loss_streak += 1
        self.conf = conf
        self.kf.update(_bbox_to_z(bbox))

    def get_state(self) -> np.ndarray:
        """Return the current state estimate as [x1, y1, x2, y2]."""
        return _z_to_bbox(self.kf.x)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to Kalman measurement [cx, cy, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def _z_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert Kalman state to [x1, y1, x2, y2]."""
    s = x[2, 0]
    r = x[3, 0]
    w = np.sqrt(s * r)
    h = s / w if w > 0 else 0.0
    cx, cy = x[0, 0], x[1, 0]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ──────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iou_batch(bb_det: np.ndarray, bb_trk: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between N detections and M tracks.

    Parameters
    ----------
    bb_det:
        (N, 4) array of detection boxes [x1, y1, x2, y2].
    bb_trk:
        (M, 4) array of track boxes [x1, y1, x2, y2].

    Returns
    -------
    np.ndarray
        (N, M) IoU matrix.
    """
    bb_det = np.expand_dims(bb_det, 1)   # (N, 1, 4)
    bb_trk = np.expand_dims(bb_trk, 0)  # (1, M, 4)

    ix1 = np.maximum(bb_det[..., 0], bb_trk[..., 0])
    iy1 = np.maximum(bb_det[..., 1], bb_trk[..., 1])
    ix2 = np.minimum(bb_det[..., 2], bb_trk[..., 2])
    iy2 = np.minimum(bb_det[..., 3], bb_trk[..., 3])

    inter_w = np.maximum(0.0, ix2 - ix1)
    inter_h = np.maximum(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_det = (bb_det[..., 2] - bb_det[..., 0]) * (bb_det[..., 3] - bb_det[..., 1])
    area_trk = (bb_trk[..., 2] - bb_trk[..., 0]) * (bb_trk[..., 3] - bb_trk[..., 1])

    iou = inter / (area_det + area_trk - inter + 1e-6)
    return iou


def _associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Use the Hungarian algorithm to match detections to existing tracks.

    Parameters
    ----------
    detections:
        (N, 4) detection boxes.
    trackers:
        (M, 4) predicted track boxes.
    iou_threshold:
        Minimum IoU for a valid match.

    Returns
    -------
    tuple
        ``(matches, unmatched_detections, unmatched_trackers)``
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(len(detections))),
            [],
        )

    iou_matrix = _iou_batch(detections, trackers)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)

    unmatched_dets = [
        d for d in range(len(detections))
        if d not in matched_indices[:, 0]
    ]
    unmatched_trks = [
        t for t in range(len(trackers))
        if t not in matched_indices[:, 1]
    ]

    # Filter out matches with low IoU
    matches = matched_indices[
        iou_matrix[matched_indices[:, 0], matched_indices[:, 1]] >= iou_threshold
    ]
    rejected = matched_indices[
        iou_matrix[matched_indices[:, 0], matched_indices[:, 1]] < iou_threshold
    ]
    unmatched_dets += rejected[:, 0].tolist()
    unmatched_trks += rejected[:, 1].tolist()

    return matches, unmatched_dets, unmatched_trks


# ──────────────────────────────────────────────────────────────────────────────
# SORT tracker
# ──────────────────────────────────────────────────────────────────────────────

class SORTTracker:
    """SORT multi-object tracker.

    Parameters
    ----------
    max_age:
        Maximum frames a track can be unmatched before deletion.
    min_hits:
        Minimum matched frames before a track is confirmed.
    iou_threshold:
        Minimum IoU for detection-to-track matching.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0  # reset IDs

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Update tracker state with new detections.

        Parameters
        ----------
        detections:
            (N, 5) array of [x1, y1, x2, y2, confidence] detections.
            Pass an empty array when there are no detections in a frame.

        Returns
        -------
        np.ndarray
            (M, 5) array of confirmed tracks: [x1, y1, x2, y2, track_id].
        """
        self.frame_count += 1

        # Step 1: Get predictions from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        # Step 2: Associate detections to tracks
        dets = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))
        matched, unmatched_dets, unmatched_trks = _associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # Step 3: Update matched trackers (propagate confidence)
        for m in matched:
            conf = float(detections[m[0], 4]) if detections.shape[1] > 4 else 1.0
            self.trackers[m[1]].update(dets[m[0]], conf)

        # Step 4: Create new tracks for unmatched detections
        for i in unmatched_dets:
            conf = float(detections[i, 4]) if detections.shape[1] > 4 else 1.0
            self.trackers.append(KalmanBoxTracker(dets[i], conf))

        # Step 5: Delete stale tracks
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        # Step 6: Return confirmed tracks as [x1, y1, x2, y2, track_id, conf]
        ret = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                pos = trk.get_state()
                ret.append(np.concatenate([pos, [trk.id, trk.conf]]))

        return np.array(ret) if ret else np.empty((0, 6))
