"""YOLOv8 bear detection inference on images and videos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np

from src.utils.logging import get_logger
from src.utils.visualization import draw_detection

logger = get_logger(__name__)


class BearDetector:
    """Run YOLOv8 detection on images or video frames.

    Parameters
    ----------
    weights_path:
        Path to a YOLOv8 ``.pt`` weights file.
    conf_threshold:
        Minimum confidence to keep a detection.
    iou_threshold:
        NMS IoU threshold.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required: pip install ultralytics") from exc

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = YOLO(str(weights_path))
        logger.info(f"Loaded detector from {weights_path}")

    # ------------------------------------------------------------------
    # Image inference
    # ------------------------------------------------------------------

    def predict_image(
        self,
        image: Union[str, Path, np.ndarray],
        annotate: bool = False,
    ) -> dict[str, Any]:
        """Run detection on a single image.

        Parameters
        ----------
        image:
            File path or BGR NumPy array.
        annotate:
            If ``True``, return an annotated copy of the image.

        Returns
        -------
        dict
            Keys: ``"boxes"`` (list of [x1, y1, x2, y2]), ``"scores"``,
            ``"labels"``, and optionally ``"annotated_image"``.
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy().tolist() if results.boxes else []
        scores = results.boxes.conf.cpu().numpy().tolist() if results.boxes else []
        class_ids = results.boxes.cls.cpu().numpy().astype(int).tolist() if results.boxes else []
        labels = [results.names[c] for c in class_ids]

        output: dict[str, Any] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

        if annotate:
            frame = results.orig_img.copy()
            for box, score, label in zip(boxes, scores, labels):
                draw_detection(frame, box, label, score)
            output["annotated_image"] = frame

        return output

    # ------------------------------------------------------------------
    # Video inference
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        tracker=None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Process a video file, drawing detections (and optionally tracks).

        Parameters
        ----------
        video_path:
            Input video file path.
        output_path:
            Destination path for the annotated output video.
        tracker:
            Optional tracker instance (e.g. ``SORTTracker``). When provided,
            tracking IDs are overlaid on the output.
        show_progress:
            Log per-frame progress to the console.

        Returns
        -------
        dict
            Processing summary: frame count, detection counts, output path.
        """
        from src.utils.visualization import draw_tracked_detection

        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        total_detections = 0

        logger.info(f"Processing {video_path.name} ({total_frames} frames) ...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.predict_image(frame)
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            total_detections += len(boxes)

            if tracker is not None and boxes:
                detections_np = np.array(
                    [[*b, s] for b, s in zip(boxes, scores)], dtype=np.float32
                )
                tracks = tracker.update(detections_np)
                for trk in tracks:
                    x1, y1, x2, y2, track_id, conf = trk
                    draw_tracked_detection(frame, [x1, y1, x2, y2], int(track_id), float(conf))
            else:
                for box, score, label in zip(boxes, scores, labels):
                    draw_detection(frame, box, label, score)

            writer.write(frame)
            frame_idx += 1

            if show_progress and frame_idx % 100 == 0:
                logger.info(f"  Frame {frame_idx}/{total_frames}")

        cap.release()
        writer.release()

        summary = {
            "frames_processed": frame_idx,
            "total_detections": total_detections,
            "output_path": str(output_path),
        }
        logger.info(f"Video saved to {output_path}. Summary: {summary}")
        return summary
