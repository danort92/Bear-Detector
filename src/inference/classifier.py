"""MobileNetV2 bear classification inference."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torchvision.transforms as T
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


class BearClassifier:
    """Run bear/non-bear binary classification on single images.

    Parameters
    ----------
    model_path:
        Path to a saved classifier state-dict (``.pt``).
    threshold:
        Decision threshold for the positive (bear) class.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, etc.).
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required: pip install torchvision")

        self.threshold = threshold
        self.device = torch.device(device)

        from src.training.train_classification import build_classifier
        self.model = build_classifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Loaded classifier from {model_path}")

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> dict:
        """Classify a single image.

        Parameters
        ----------
        image:
            A file path, PIL Image, or NumPy array (RGB, uint8).

        Returns
        -------
        dict
            ``{"label": "bear"|"other", "confidence": float, "is_bear": bool}``
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor).squeeze()
            prob = torch.sigmoid(logit).item()

        is_bear = prob >= self.threshold
        return {
            "label": "bear" if is_bear else "other",
            "confidence": prob,
            "is_bear": is_bear,
        }

    def predict_batch(
        self, images: list[Union[str, Path, Image.Image]]
    ) -> list[dict]:
        """Classify a list of images.

        Parameters
        ----------
        images:
            List of file paths or PIL Images.

        Returns
        -------
        list[dict]
            One result dict per input image.
        """
        return [self.predict(img) for img in images]
