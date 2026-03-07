"""YOLOv8 detection trainer for bear detection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DetectionTrainer:
    """Wraps Ultralytics YOLOv8 training for bear detection.

    Parameters
    ----------
    cfg:
        Configuration dictionary (see ``config/default.yaml``).
    mlflow_run:
        Optional MLflow active run for logging metrics and artifacts.
    """

    def __init__(self, cfg: dict[str, Any], mlflow_run=None) -> None:
        self.cfg = cfg
        self.mlflow_run = mlflow_run
        self.output_dir = Path(cfg["experiment"]["output_dir"])

        model_cfg = cfg["model"]["detection"]
        self.architecture = model_cfg["architecture"]
        self.pretrained_weights = model_cfg.get("pretrained_weights") or self.architecture + ".pt"

        train_cfg = cfg["training"]["detection"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        self.warmup_epochs = train_cfg.get("warmup_epochs", 3)
        self.cos_lr = train_cfg.get("cos_lr", True)
        self.amp = train_cfg.get("amp", True)
        self.workers = train_cfg.get("workers", 4)

        data_cfg = cfg["data"]["detection"]
        self.data_yaml = data_cfg["yaml_path"]
        self.image_size = data_cfg["image_size"]
        self.batch_size = data_cfg["batch_size"]

        self.experiment_name = cfg["experiment"]["name"]
        self.project_dir = str(self.output_dir / "models" / "detection")

    def _resolve_data_yaml(self) -> str:
        """Return a path to a data.yaml whose image paths are absolute.

        Ultralytics resolves the ``path`` key in data.yaml relative to its
        internal datasets directory, not to the yaml file itself.  This method
        rewrites the yaml in a temporary file using absolute paths so training
        works from any working directory.

        If the ``valid/images`` split is missing the ``val`` key falls back to
        the ``test`` split.
        """
        yaml_path = Path(self.data_yaml).resolve()
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        dataset_root = yaml_path.parent.resolve()

        # Build absolute split paths
        def _abs(rel: str) -> str:
            return str(dataset_root / rel)

        data["path"] = str(dataset_root)
        data["train"] = _abs(data.get("train", "train/images"))
        test_abs = _abs(data.get("test", "test/images"))

        val_rel = data.get("val", "valid/images")
        val_abs = _abs(val_rel)
        if not Path(val_abs).exists():
            logger.warning(
                f"Validation split not found at {val_abs}. Falling back to test split."
            )
            val_abs = test_abs
        data["val"] = val_abs
        data["test"] = test_abs

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_data.yaml", delete=False
        )
        yaml.dump(data, tmp)
        tmp.close()
        logger.info(f"Resolved data yaml written to {tmp.name}")
        return tmp.name

    def train(self) -> dict[str, Any]:
        """Run the YOLOv8 training pipeline.

        Returns
        -------
        dict
            Dictionary containing the path to the best weights and
            the final training metrics exported by Ultralytics.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            ) from exc

        logger.info(f"Loading model: {self.pretrained_weights}")
        model = YOLO(self.pretrained_weights)

        logger.info(
            f"Starting detection training: {self.architecture}, "
            f"{self.epochs} epochs, imgsz={self.image_size}"
        )

        resolved_yaml = self._resolve_data_yaml()
        results = model.train(
            data=resolved_yaml,
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            lr0=self.lr,
            weight_decay=self.weight_decay,
            warmup_epochs=self.warmup_epochs,
            cos_lr=self.cos_lr,
            amp=self.amp,
            workers=self.workers,
            project=self.project_dir,
            name=self.experiment_name,
            exist_ok=True,
            verbose=True,
        )

        # Locate the best weights saved by Ultralytics
        best_pt = Path(self.project_dir) / self.experiment_name / "weights" / "best.pt"
        if not best_pt.exists():
            logger.warning("best.pt not found at expected path.")

        output = {
            "best_weights": str(best_pt),
            "results_dir": str(Path(self.project_dir) / self.experiment_name),
        }

        # Log artifact path to MLflow
        if self.mlflow_run and best_pt.exists():
            import mlflow
            mlflow.log_artifact(str(best_pt), artifact_path="detection_weights")
            logger.info("Logged best.pt to MLflow.")

        # Export key metrics to outputs/metrics/
        metrics_path = self.output_dir / "metrics" / "detection_results.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Detection results saved to {metrics_path}")

        return output

    def validate(self, weights_path: str | None = None) -> dict[str, float]:
        """Run validation on the test set and return mAP metrics.

        Parameters
        ----------
        weights_path:
            Path to a ``.pt`` file. Defaults to the best checkpoint.

        Returns
        -------
        dict
            Dictionary with keys ``"mAP50"`` and ``"mAP50-95"``.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required.") from exc

        if weights_path is None:
            weights_path = str(
                Path(self.project_dir) / self.experiment_name / "weights" / "best.pt"
            )

        model = YOLO(weights_path)
        resolved_yaml = self._resolve_data_yaml()
        metrics = model.val(data=resolved_yaml, imgsz=self.image_size, workers=self.workers)

        result = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        }
        logger.info(f"Validation metrics: {result}")

        if self.mlflow_run:
            import mlflow
            mlflow.log_metrics(result)

        return result
