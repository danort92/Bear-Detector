"""YOLOv8-seg training pipeline for bear instance segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SegmentationTrainer:
    """Wraps Ultralytics YOLOv8-seg for bear instance segmentation training.

    Parameters
    ----------
    cfg:
        Configuration dictionary (see ``config/default.yaml``).
    mlflow_run:
        Optional MLflow active run for metric and artifact logging.
    """

    def __init__(self, cfg: dict[str, Any], mlflow_run=None) -> None:
        self.cfg = cfg
        self.mlflow_run = mlflow_run
        self.output_dir = Path(cfg["experiment"]["output_dir"])

        model_cfg = cfg["model"]["segmentation"]
        self.architecture = model_cfg["architecture"]
        self.pretrained_weights = (
            model_cfg.get("pretrained_weights") or self.architecture + ".pt"
        )

        train_cfg = cfg["training"]["segmentation"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        self.amp = train_cfg.get("amp", True)
        self.workers = train_cfg.get("workers", 4)

        data_cfg = cfg["data"]["segmentation"]
        self.data_dir = data_cfg["data_dir"]
        self.image_size = data_cfg["image_size"]
        self.batch_size = data_cfg["batch_size"]

        self.experiment_name = cfg["experiment"]["name"] + "_seg"
        self.project_dir = str(self.output_dir / "models" / "segmentation")

    def train(self) -> dict[str, Any]:
        """Run the YOLOv8-seg training pipeline.

        Expects ``data_dir`` to contain a ``data.yaml`` file that describes
        the segmentation dataset in Ultralytics format.

        Returns
        -------
        dict
            Paths to the best weights and the results directory.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required: pip install ultralytics"
            ) from exc

        data_yaml = str(Path(self.data_dir) / "data.yaml")
        if not Path(data_yaml).exists():
            raise FileNotFoundError(
                f"Segmentation data.yaml not found: {data_yaml}\n"
                "Please prepare your YOLO-seg dataset and update "
                "config/default.yaml → data.segmentation.data_dir"
            )

        logger.info(f"Loading segmentation model: {self.pretrained_weights}")
        model = YOLO(self.pretrained_weights)

        logger.info(
            f"Starting segmentation training: {self.architecture}, "
            f"{self.epochs} epochs, imgsz={self.image_size}"
        )

        model.train(
            data=data_yaml,
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            lr0=self.lr,
            weight_decay=self.weight_decay,
            amp=self.amp,
            workers=self.workers,
            project=self.project_dir,
            name=self.experiment_name,
            exist_ok=True,
            verbose=True,
        )

        best_pt = Path(self.project_dir) / self.experiment_name / "weights" / "best.pt"
        output = {
            "best_weights": str(best_pt),
            "results_dir": str(Path(self.project_dir) / self.experiment_name),
        }

        if self.mlflow_run and best_pt.exists():
            import mlflow
            mlflow.log_artifact(str(best_pt), artifact_path="segmentation_weights")

        metrics_path = self.output_dir / "metrics" / "segmentation_results.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Segmentation training complete. Best weights: {best_pt}")
        return output

    def validate(self, weights_path: Optional[str] = None) -> dict[str, float]:
        """Validate the segmentation model on the test split.

        Parameters
        ----------
        weights_path:
            Path to weights ``.pt`` file. Defaults to the best checkpoint.

        Returns
        -------
        dict
            Keys: ``"mask_mAP50"``, ``"mask_mAP50-95"``, ``"box_mAP50"``.
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
        data_yaml = str(Path(self.data_dir) / "data.yaml")
        metrics = model.val(data=data_yaml, imgsz=self.image_size, workers=self.workers)

        result = {
            "mask_mAP50": float(metrics.seg.map50),
            "mask_mAP50-95": float(metrics.seg.map),
            "box_mAP50": float(metrics.box.map50),
        }
        logger.info(f"Segmentation validation metrics: {result}")

        if self.mlflow_run:
            import mlflow
            mlflow.log_metrics(result)

        return result
