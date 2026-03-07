#!/usr/bin/env python3
"""Train a YOLOv8 bear detection model.

Example
-------
.. code-block:: bash

    python scripts/train_detection.py
    python scripts/train_detection.py --config config/detection_finetune.yaml
    python scripts/train_detection.py --epochs 100 --model yolov8s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_merged_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.training.train_detection import DetectionTrainer
from src.training.experiment_tracker import ExperimentTracker

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 bear detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to base YAML configuration file",
    )
    parser.add_argument(
        "--override",
        default=None,
        help="Optional experiment-specific YAML config to merge on top of base config",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--model", default=None, help="Override YOLOv8 architecture (e.g. yolov8s)")
    parser.add_argument("--data", default=None, help="Override path to data.yaml")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_merged_config(args.config, args.override)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg["training"]["detection"]["epochs"] = args.epochs
    if args.model is not None:
        cfg["model"]["detection"]["architecture"] = args.model
    if args.data is not None:
        cfg["data"]["detection"]["yaml_path"] = args.data
    if args.batch is not None:
        cfg["data"]["detection"]["batch_size"] = args.batch
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.no_mlflow:
        cfg["mlflow"]["enabled"] = False

    seed = cfg["experiment"]["seed"]
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    tracker = ExperimentTracker(cfg)

    with tracker.start_run() as run:
        run.log_params({
            "architecture": cfg["model"]["detection"]["architecture"],
            "epochs": cfg["training"]["detection"]["epochs"],
            "lr": cfg["training"]["detection"]["learning_rate"],
            "batch_size": cfg["data"]["detection"]["batch_size"],
            "image_size": cfg["data"]["detection"]["image_size"],
            "seed": seed,
        })

        trainer = DetectionTrainer(cfg, mlflow_run=run.mlflow_run)
        output = trainer.train()
        logger.info(f"Training complete. Best weights: {output['best_weights']}")

        # Run validation
        try:
            metrics = trainer.validate(output["best_weights"])
            run.log_metrics(metrics)
            logger.info(f"Validation metrics: {metrics}")
        except Exception as exc:
            logger.warning(f"Validation failed: {exc}")


if __name__ == "__main__":
    main()
