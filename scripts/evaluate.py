#!/usr/bin/env python3
"""Evaluate a trained detection model and report metrics.

Example
-------
.. code-block:: bash

    python scripts/evaluate.py --model outputs/models/detection/best.pt \\
                               --data "Bear detection.v3i.yolov8-obb/data.yaml"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_merged_config
from src.utils.logging import get_logger
from src.training.train_detection import DetectionTrainer

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model weights (.pt)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to data.yaml (overrides config)",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/metrics",
        help="Directory to save evaluation results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_merged_config(args.config)

    if args.data:
        cfg["data"]["detection"]["yaml_path"] = args.data

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating model: {args.model}")
    trainer = DetectionTrainer(cfg)
    metrics = trainer.validate(args.model)

    metrics_file = output_dir / "detection_evaluation.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {metrics_file}")
    print("\n=== Detection Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
