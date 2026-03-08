#!/usr/bin/env python3
"""Train a YOLOv8-seg bear instance segmentation model.

Example
-------
.. code-block:: bash

    python scripts/train_segmentation.py
    python scripts/train_segmentation.py --epochs 100 --model yolov8s-seg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_merged_config
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.segmentation.train_segmentation import SegmentationTrainer
from src.training.experiment_tracker import ExperimentTracker

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg bear segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--model", default=None, help="e.g. yolov8s-seg")
    parser.add_argument("--data-dir", default=None, help="Override segmentation data directory")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_merged_config(args.config, args.override)

    if args.epochs is not None:
        cfg["training"]["segmentation"]["epochs"] = args.epochs
    if args.model is not None:
        cfg["model"]["segmentation"]["architecture"] = args.model
    if args.data_dir is not None:
        cfg["data"]["segmentation"]["data_dir"] = args.data_dir
    if args.batch is not None:
        cfg["data"]["segmentation"]["batch_size"] = args.batch
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    seed = cfg["experiment"]["seed"]
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    tracker = ExperimentTracker(cfg)

    with tracker.start_run() as run:
        run.log_params({
            "architecture": cfg["model"]["segmentation"]["architecture"],
            "epochs": cfg["training"]["segmentation"]["epochs"],
            "lr": cfg["training"]["segmentation"]["learning_rate"],
            "batch_size": cfg["data"]["segmentation"]["batch_size"],
            "seed": seed,
        })

        trainer = SegmentationTrainer(cfg)
        output = trainer.train()
        logger.info(f"Segmentation training complete. Best weights: {output['best_weights']}")

        try:
            metrics = trainer.validate(output["best_weights"])
            run.log_metrics(metrics)
            logger.info(f"Validation metrics: {metrics}")
        except Exception as exc:
            logger.warning(f"Validation failed: {exc}")


if __name__ == "__main__":
    main()
