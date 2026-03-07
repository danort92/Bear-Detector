#!/usr/bin/env python3
"""Run bear detection or classification inference on a single image.

Example
-------
.. code-block:: bash

    # Detection
    python scripts/infer_image.py --image bear.jpg --model best.pt

    # Classification
    python scripts/infer_image.py --image bear.jpg --model classifier.pt --classify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_merged_config
from src.utils.device import get_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bear detection/classification inference on a single image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to model weights (.pt)")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save annotated image",
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--classify", action="store_true", help="Use classifier instead of detector")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_merged_config(args.config)
    device = get_device(args.device)

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    if args.classify:
        from src.inference.classifier import BearClassifier

        threshold = args.conf or cfg["inference"]["classification_threshold"]
        classifier = BearClassifier(args.model, threshold=threshold, device=device)
        result = classifier.predict(image_path)

        print(f"\nClassification Result:")
        print(f"  Label      : {result['label']}")
        print(f"  Confidence : {result['confidence']:.4f}")
        print(f"  Is bear    : {result['is_bear']}")
    else:
        from src.inference.detector import BearDetector

        conf = args.conf or cfg["inference"]["detection_conf_threshold"]
        detector = BearDetector(args.model, conf_threshold=conf, device=device)
        result = detector.predict_image(image_path, annotate=bool(args.output))

        print(f"\nDetection Result:")
        print(f"  Detections : {len(result['boxes'])}")
        for i, (box, score, label) in enumerate(
            zip(result["boxes"], result["scores"], result["labels"])
        ):
            print(f"  [{i}] {label:10s} conf={score:.3f}  box={[round(v, 1) for v in box]}")

        if args.output and "annotated_image" in result:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), result["annotated_image"])
            print(f"\n  Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
