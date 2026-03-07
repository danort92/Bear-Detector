"""Experiment tracking utilities using MLflow.

Provides a thin wrapper that gracefully falls back to console logging when
MLflow is not installed.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Lightweight experiment tracker backed by MLflow.

    Falls back to JSON-file logging when MLflow is unavailable.

    Parameters
    ----------
    cfg:
        Configuration dictionary (see ``config/default.yaml``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.experiment_name = cfg["experiment"]["name"]
        self.output_dir = Path(cfg["experiment"]["output_dir"]) / "experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        mlflow_cfg = cfg.get("mlflow", {})
        self.mlflow_enabled = mlflow_cfg.get("enabled", False)
        self.tracking_uri = mlflow_cfg.get("tracking_uri", str(self.output_dir / "mlruns"))
        self.mlflow_experiment = mlflow_cfg.get("experiment_name", self.experiment_name)

        self._run = None
        self._metrics_buffer: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
    ) -> Generator["ExperimentTracker", None, None]:
        """Context manager that starts an MLflow run (or no-op).

        Usage::

            with tracker.start_run() as run:
                run.log_params({"lr": 0.001})
                run.log_metric("val_acc", 0.95)

        Yields
        ------
        ExperimentTracker
            ``self`` so callers can chain calls inside the block.
        """
        if self.mlflow_enabled:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.mlflow_experiment)
                with mlflow.start_run(run_name=run_name or self.experiment_name) as run:
                    self._run = run
                    logger.info(
                        f"MLflow run started: {run.info.run_id} "
                        f"(experiment={self.mlflow_experiment})"
                    )
                    yield self
                    self._run = None
            except ImportError:
                logger.warning("MLflow not installed — falling back to JSON logging.")
                yield self
                self._flush_json()
        else:
            yield self
            self._flush_json()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters.

        Parameters
        ----------
        params:
            Dictionary of parameter names → values.
        """
        if self.mlflow_enabled and self._run:
            import mlflow
            mlflow.log_params(params)
        self._metrics_buffer.setdefault("params", {}).update(params)
        logger.info(f"Params: {params}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar metric.

        Parameters
        ----------
        key:
            Metric name.
        value:
            Metric value.
        step:
            Optional training step / epoch.
        """
        if self.mlflow_enabled and self._run:
            import mlflow
            mlflow.log_metric(key, value, step=step)
        self._metrics_buffer.setdefault("metrics", {}).setdefault(key, []).append(
            {"value": value, "step": step}
        )

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar metrics at once.

        Parameters
        ----------
        metrics:
            Dictionary of metric name → value.
        step:
            Optional training step / epoch.
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        """Log a file artifact.

        Parameters
        ----------
        local_path:
            Path to the file to log.
        """
        if self.mlflow_enabled and self._run:
            import mlflow
            mlflow.log_artifact(str(local_path))
        logger.info(f"Artifact: {local_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_json(self) -> None:
        """Write buffered metrics to a JSON file in the output directory."""
        out_file = self.output_dir / f"{self.experiment_name}_metrics.json"
        with open(out_file, "w") as f:
            json.dump(self._metrics_buffer, f, indent=2)
        logger.info(f"Experiment log saved to {out_file}")
