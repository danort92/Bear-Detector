"""YOLOv8-seg bear segmentation inference on images and videos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np

from src.utils.logging import get_logger
from src.utils.visualization import draw_detection, draw_mask

logger = get_logger(__name__)


class BearSegmentor:
    """Run YOLOv8-seg instance segmentation on images or video frames.

    Parameters
    ----------
    weights_path:
        Path to a YOLOv8-seg ``.pt`` weights file.
    conf_threshold:
        Minimum confidence threshold.
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
        logger.info(f"Loaded segmentor from {weights_path}")

    def predict_image(
        self,
        image: Union[str, Path, np.ndarray],
        annotate: bool = False,
    ) -> dict[str, Any]:
        """Run segmentation on a single image.

        Parameters
        ----------
        image:
            File path or BGR NumPy array.
        annotate:
            If ``True``, return an annotated copy of the image.

        Returns
        -------
        dict
            Keys: ``"boxes"``, ``"scores"``, ``"labels"``, ``"masks"``,
            and optionally ``"annotated_image"``.
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

        # Instance masks (H, W, N) → list of (H, W) binary masks
        masks: list[np.ndarray] = []
        if results.masks is not None:
            raw_masks = results.masks.data.cpu().numpy()  # (N, H, W)
            for m in raw_masks:
                masks.append((m > 0.5).astype(np.uint8))
        else:
            masks = [None] * len(boxes)

        output: dict[str, Any] = {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "masks": masks,
        }

        if annotate:
            frame = results.orig_img.copy()
            for i, (box, score, label, mask) in enumerate(
                zip(boxes, scores, labels, masks)
            ):
                if mask is not None:
                    # Resize mask to original image size if needed
                    orig_h, orig_w = frame.shape[:2]
                    if mask.shape != (orig_h, orig_w):
                        mask = cv2.resize(
                            mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                        )
                    frame = draw_mask(frame, mask, track_id=i)
                draw_detection(frame, box, label, score)
            output["annotated_image"] = frame

        return output

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Process a video file with instance segmentation.

        Parameters
        ----------
        video_path:
            Input video file path.
        output_path:
            Destination path for the annotated output video.
        show_progress:
            Log per-frame progress to the console.

        Returns
        -------
        dict
            Processing summary: frame count, detection counts, output path.
        """
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

            result = self.predict_image(frame, annotate=True)
            total_detections += len(result["boxes"])
            writer.write(result["annotated_image"])
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
        logger.info(f"Segmentation video saved to {output_path}.")
        return summary
