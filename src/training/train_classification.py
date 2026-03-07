"""MobileNetV2-based binary classification trainer for bear detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logging import get_logger
from src.utils.visualization import save_training_curves

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_classifier(
    backbone: str = "mobilenet_v2",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    hidden_dim: int = 128,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a binary bear classifier on top of a torchvision backbone.

    Parameters
    ----------
    backbone:
        Torchvision backbone name (``"mobilenet_v2"`` or ``"resnet50"``).
    pretrained:
        Load ImageNet pretrained weights.
    freeze_backbone:
        If ``True``, freeze backbone parameters and only train the head.
    hidden_dim:
        Number of units in the intermediate FC layer.
    dropout:
        Dropout rate before the final classification layer.

    Returns
    -------
    nn.Module
        The classifier model.
    """
    import torchvision.models as models

    weights_arg = "DEFAULT" if pretrained else None

    if backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
    elif backbone == "resnet50":
        model = models.resnet50(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
    else:
        raise ValueError(f"Unsupported backbone: '{backbone}'")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class ClassificationTrainer:
    """Manages the full training loop for binary bear classification.

    Parameters
    ----------
    cfg:
        Configuration dictionary (see ``config/default.yaml``).
    device:
        Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
    mlflow_run:
        Optional MLflow active run for logging metrics.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        device: str = "cpu",
        mlflow_run=None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.mlflow_run = mlflow_run
        self.output_dir = Path(cfg["experiment"]["output_dir"])

        model_cfg = cfg["model"]["classification"]
        train_cfg = cfg["training"]["classification"]

        self.model = build_classifier(
            backbone=model_cfg["backbone"],
            pretrained=model_cfg["pretrained"],
            freeze_backbone=model_cfg["freeze_backbone"],
            hidden_dim=model_cfg["hidden_dim"],
            dropout=model_cfg.get("dropout", 0.0),
        ).to(self.device)

        self.lr = train_cfg["learning_rate"]
        self.epochs = train_cfg["epochs"]
        self.patience = train_cfg["early_stopping_patience"]

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )
        self.history: dict[str, list[float]] = {
            "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        """Run one epoch and return (loss, accuracy)."""
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.BCEWithLogitsLoss()

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                logits = self.model(images).squeeze(1)
                loss = criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(labels)
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += len(labels)

        return total_loss / total, correct / total

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Run the full training loop with early stopping.

        Parameters
        ----------
        train_loader:
            DataLoader for the training split.
        val_loader:
            DataLoader for the validation split.

        Returns
        -------
        dict
            Training history with per-epoch metrics.
        """
        best_val_acc = 0.0
        epochs_no_improve = 0
        best_model_path = self.output_dir / "models" / "best_classifier.pt"
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)

            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if self.mlflow_run:
                import mlflow
                mlflow.log_metrics(
                    {"train_loss": train_loss, "train_acc": train_acc,
                     "val_loss": val_loss, "val_acc": val_acc},
                    step=epoch,
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"  -> New best val_acc={best_val_acc:.4f}. Saved checkpoint.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping after {epoch} epochs.")
                    break

        # Save final training curves
        curves_path = self.output_dir / "metrics" / "classification_training_curves.png"
        save_training_curves(self.history, curves_path)
        logger.info(f"Training curves saved to {curves_path}")

        # Persist history as JSON
        history_path = self.output_dir / "metrics" / "classification_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def load_best(self) -> None:
        """Restore the best checkpoint saved during training."""
        path = self.output_dir / "models" / "best_classifier.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Loaded best classifier from {path}")
