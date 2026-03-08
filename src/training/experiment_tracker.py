"""Experiment tracking utilities — JSON file logging.

Ultralytics' built-in MLflow callback handles MLflow tracking for
detection and segmentation training runs.  This module provides a
lightweight JSON-based fallback so params and metrics are always
persisted locally regardless of whether MLflow is configured.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Lightweight experiment tracker that writes params/metrics to JSON.

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
        self._metrics_buffer: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
    ) -> Generator["ExperimentTracker", None, None]:
        """Context manager that wraps a training run.

        Usage::

            with tracker.start_run() as run:
                run.log_params({"lr": 0.001})
                run.log_metric("val_acc", 0.95)

        Yields
        ------
        ExperimentTracker
            ``self`` so callers can chain calls inside the block.
        """
        yield self
        self._flush_json()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        self._metrics_buffer.setdefault("params", {}).update(params)
        logger.info(f"Params: {params}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar metric."""
        self._metrics_buffer.setdefault("metrics", {}).setdefault(key, []).append(
            {"value": value, "step": step}
        )

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        """Record an artifact path in the JSON log."""
        self._metrics_buffer.setdefault("artifacts", []).append(str(local_path))
        logger.info(f"Artifact: {local_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_json(self) -> None:
        """Write buffered metrics to a timestamped JSON file so each run is preserved."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = self.output_dir / f"{self.experiment_name}_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(self._metrics_buffer, f, indent=2)
        logger.info(f"Experiment log saved to {out_file}")
