#!/usr/bin/env python3
"""Run bear detection (and optional tracking) on a video file.

Example
-------
.. code-block:: bash

    # Detection only
    python scripts/infer_video.py --video input.mp4 --model best.pt

    # With SORT tracking
    python scripts/infer_video.py --video input.mp4 --model best.pt --track

    # With segmentation
    python scripts/infer_video.py --video input.mp4 --model best_seg.pt --segment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_merged_config
from src.utils.device import get_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bear detection / tracking inference on video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", required=True, help="Path to YOLOv8 weights (.pt)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (default: outputs/videos/<stem>_annotated.mp4)",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold (overrides config)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold (overrides config)",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable SORT multi-object tracking",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Use instance segmentation model instead of detection",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto | cpu | cuda | mps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_merged_config(args.config)

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        out_dir = Path(cfg["experiment"]["output_dir"]) / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_seg_annotated" if args.segment else "_annotated"
        output_path = out_dir / f"{video_path.stem}{suffix}.mp4"

    infer_cfg = cfg["inference"]
    conf = args.conf if args.conf is not None else infer_cfg["detection_conf_threshold"]
    iou = args.iou if args.iou is not None else infer_cfg["detection_iou_threshold"]

    if args.segment:
        from src.segmentation.infer_segmentation import BearSegmentor

        segmentor = BearSegmentor(
            weights_path=args.model,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
        )
        summary = segmentor.process_video(
            video_path=video_path,
            output_path=output_path,
        )
    else:
        from src.inference.detector import BearDetector
        from src.tracking.sort_tracker import SORTTracker

        detector = BearDetector(
            weights_path=args.model,
            conf_threshold=conf,
            iou_threshold=iou,
            device=device,
        )

        tracker = None
        if args.track:
            track_cfg = cfg["tracking"]
            tracker = SORTTracker(
                max_age=track_cfg["max_age"],
                min_hits=track_cfg["min_hits"],
                iou_threshold=track_cfg["iou_threshold"],
            )
            logger.info("SORT tracker enabled.")

        summary = detector.process_video(
            video_path=video_path,
            output_path=output_path,
            tracker=tracker,
        )

    print("\n=== Video Processing Summary ===")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()
