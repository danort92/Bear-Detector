# Bear Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-pytest-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A research-grade, open-source pipeline for bear detection, multi-object tracking, and instance segmentation in wildlife camera trap imagery.**

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Visual Results](#visual-results)
3. [Pipeline](#pipeline)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Installation](#installation)
7. [Google Colab](#google-colab)
8. [Configuration](#configuration)
9. [Training](#training)
10. [Inference](#inference)
11. [Evaluation](#evaluation)
12. [Experiment Tracking](#experiment-tracking)
13. [Tests](#tests)
14. [Contributing](#contributing)

---

## Visual Results

### Detection — bounding boxes + confidence scores

![Detection examples](results/detection/detection_banner.jpg)

### Multi-object tracking — unique colour per bear ID

![Tracking examples](results/tracking/tracking_banner.jpg)

### Instance segmentation — per-pixel masks

![Segmentation examples](results/segmentation/segmentation_banner.jpg)

### Binary classification — bear vs. no bear (MobileNetV2)

![Classification examples](results/classification/classification_banner.jpg)

> **Note:** The detection and tracking examples above use ground-truth label boxes from the Roboflow dataset to illustrate the overlay style. Segmentation masks are elliptical approximations; actual YOLOv8-seg output produces precise polygon masks. Classification confidence scores are shown after training.

---

## Overview

Bear Detector is a modular, fully reproducible machine-learning pipeline for wildlife conservation.
It processes images and videos from camera traps to:

- **Detect** bears with bounding boxes and confidence scores using YOLOv8.
- **Track** individual bears across video frames using SORT (Simple Online and Realtime Tracking).
- **Segment** individual bear instances with pixel-level masks using YOLOv8-seg.
- **Classify** whether an image contains a bear using a MobileNetV2 binary classifier.

All pipelines are configurable via YAML, reproducible via seeded random number generators, and logged with MLflow for experiment tracking.

---

## Pipeline

```
Camera trap image / video
        │
        ▼
┌───────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Classification │    │    Detection    │    │  Segmentation    │
│  MobileNetV2  │    │    YOLOv8       │    │  YOLOv8-seg      │
│  Bear / Other │    │  Bounding Boxes │    │  Instance Masks  │
└───────────────┘    └────────┬────────┘    └──────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │    Tracking     │
                     │  SORT / Kalman  │
                     │  Track IDs      │
                     └─────────────────┘
                              │
                              ▼
                     Annotated output video
                  (bounding boxes + IDs + scores)
```

---

## Project Structure

```
Bear-Detector/
├── config/
│   ├── default.yaml               # Master configuration file
│   └── detection_finetune.yaml    # Example override config
│
├── src/
│   ├── datasets/
│   │   ├── classification_dataset.py   # Bear/other classification dataset
│   │   └── detection_dataset.py        # YOLO-format detection dataset
│   ├── training/
│   │   ├── train_classification.py     # MobileNetV2 trainer
│   │   ├── train_detection.py          # YOLOv8 detection trainer
│   │   ├── evaluate.py                 # mAP, PR curves, confusion matrices
│   │   └── experiment_tracker.py       # MLflow experiment tracker
│   ├── inference/
│   │   ├── classifier.py               # Classification inference
│   │   └── detector.py                 # Detection + video inference
│   ├── tracking/
│   │   ├── sort_tracker.py             # SORT multi-object tracker
│   │   └── metrics.py                  # MOTA, MOTP, ID-switch metrics
│   ├── segmentation/
│   │   ├── dataset.py                  # YOLO-seg dataset
│   │   ├── train_segmentation.py       # Segmentation trainer
│   │   └── infer_segmentation.py       # Segmentation inference + video
│   └── utils/
│       ├── config.py                   # YAML config loading & merging
│       ├── seed.py                     # Global reproducibility seed
│       ├── logging.py                  # Centralised logger
│       ├── visualization.py            # Bounding boxes, masks, plots
│       └── device.py                   # Auto device selection
│
├── scripts/
│   ├── train_detection.py         # CLI: train detection model
│   ├── train_segmentation.py      # CLI: train segmentation model
│   ├── evaluate.py                # CLI: evaluate detection model
│   ├── infer_video.py             # CLI: run detection/tracking on video
│   └── infer_image.py             # CLI: run detection/classification on image
│
├── data/
│   ├── raw/                       # Raw datasets (not tracked by git)
│   ├── processed/                 # Processed / formatted datasets
│   └── sample/                    # Sample video/images for quick demos
│
├── models/
│   ├── detection/                 # Detection model weights
│   ├── tracking/
│   └── segmentation/              # Segmentation model weights
│
├── outputs/
│   ├── models/                    # Saved checkpoints
│   ├── metrics/                   # JSON metric files + plots
│   ├── visualizations/            # Example annotated images
│   ├── videos/                    # Annotated output videos
│   └── experiments/               # MLflow experiment logs
│
├── results/                       # Qualitative visual results
│   ├── detection/
│   ├── tracking/
│   ├── segmentation/
│   └── videos/
│
├── notebooks/
│   └── Bear_detection.ipynb       # Original exploratory notebook
│
├── tests/                         # pytest unit tests
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Dataset

### Detection Dataset (Roboflow)

- **Source:** [Roboflow](https://roboflow.com) — exported in YOLOv8 OBB format
- **Total images:** 1,172 (after augmentation)
- **Split:** train / valid / test
- **Classes:** `bear face` (class 0)
- **Augmentations:** horizontal flip, 90° rotations, random crop (0–25 %), shear (±12°), Gaussian blur, salt-and-pepper noise
- **License:** CC BY 4.0

### Classification Dataset

- Two folders: `ct/bear_ct/` (positive) and `ct/other_ct/` (negative)
- Used for binary bear / non-bear classification with MobileNetV2

### Segmentation Dataset

- YOLO-segmentation format (polygon annotations)
- Configure path in `config/default.yaml` → `data.segmentation.data_dir`

---

## Installation

### Prerequisites

- Python ≥ 3.10
- NVIDIA GPU (optional but strongly recommended)
- `ffmpeg` (for video processing)

### 1. Clone the repository

```bash
git clone https://github.com/danort92/Bear-Detector.git
cd Bear-Detector
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e ".[dev]"
```

---

## Google Colab

The fastest way to try Bear Detector without any local setup is Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/Bear-Detector/blob/claude/improve-bear-detection-tKY3Y/notebooks/Bear_Detector_Colab.ipynb)

The Colab notebook (`notebooks/Bear_Detector_Colab.ipynb`) walks through the full pipeline:

| Section | What it does |
|---------|-------------|
| 1. Setup | Clone repo, install deps, check GPU |
| 2. Config | Load `config/default.yaml`, override any parameter |
| 3. Detection training | Train YOLOv8n on the Roboflow bear dataset |
| 4. Image inference | **Upload your own image** or pick from test set; displays annotated result |
| 5. Video inference | **Upload your own video** (or use `data/sample/bear_sample.mp4`); runs detection + tracking and downloads H.264 output |
| 6. Segmentation | Run mask inference on an image and on the video from Section 5 |
| 7. Evaluation | Compute mAP, PR curve, MOTA/MOTP |
| 8. MLflow | Browse logged experiments (JSON by default; full UI via ngrok) |

**Adding your own sample video to the repo:**

Place a short bear video (MP4, ≤ 50 MB recommended for GitHub) at:

```
data/sample/bear_sample.mp4
```

The notebook will automatically detect and use it (Section 5). To push it to GitHub:

```bash
# For files ≤ 50 MB — commit normally
git add data/sample/bear_sample.mp4
git commit -m "add sample bear video"
git push

# For files > 50 MB — use Git LFS
git lfs install
git lfs track "data/sample/*.mp4"
git add .gitattributes data/sample/bear_sample.mp4
git commit -m "add sample bear video via LFS"
git push
```

**Quick start in Colab:**

```python
# Cell 1 — clone and install
!git clone https://github.com/danort92/Bear-Detector.git
%cd Bear-Detector
!pip install -r requirements.txt

# Cell 2 — run detection on a sample image
import sys; sys.path.insert(0, '.')
from src.inference.detector import BearDetector
from IPython.display import display
import cv2, numpy as np
from PIL import Image

detector = BearDetector("yolov8n.pt", conf_threshold=0.25)   # uses COCO pretrained
result = detector.predict_image("path/to/bear.jpg", annotate=True)
display(Image.fromarray(cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)))
```

---

## Configuration

All hyperparameters live in `config/default.yaml`.
You can override any parameter with an experiment-specific YAML:

```yaml
# config/my_experiment.yaml
experiment:
  name: "my_experiment"

training:
  detection:
    epochs: 100
    learning_rate: 0.0005
```

Pass it to any script with `--override`:

```bash
python scripts/train_detection.py --override config/my_experiment.yaml
```

Key configuration sections:

| Section | Key Parameters |
|---------|---------------|
| `experiment` | `name`, `seed`, `output_dir` |
| `data` | Dataset paths, image sizes, batch sizes |
| `model` | Architecture, backbone, pretrained weights |
| `training` | Epochs, learning rate, optimizer, early stopping |
| `inference` | Confidence threshold, IoU threshold |
| `tracking` | `max_age`, `min_hits`, IoU threshold |
| `mlflow` | Tracking URI, experiment name |

---

## Training

### Detection (YOLOv8)

```bash
# Train with default config
python scripts/train_detection.py

# Override specific parameters
python scripts/train_detection.py \
    --epochs 100 \
    --model yolov8s \
    --batch 16 \
    --seed 42

# Use a fine-tuning config
python scripts/train_detection.py \
    --override config/detection_finetune.yaml
```

### Segmentation (YOLOv8-seg)

```bash
# Prepare your YOLO-seg dataset and set data.segmentation.data_dir in config
python scripts/train_segmentation.py \
    --model yolov8n-seg \
    --epochs 50
```

### Classification (MobileNetV2)

```python
# See notebooks/Bear_detection.ipynb or use the Python API directly:
from src.utils.config import load_merged_config
from src.utils.seed import set_seed
from src.datasets import build_classification_dataloaders
from src.training import ClassificationTrainer

cfg = load_merged_config("config/default.yaml")
set_seed(cfg["experiment"]["seed"])

train_loader, val_loader, class_weights = build_classification_dataloaders(
    bear_dir="ct/bear_ct",
    other_dir="ct/other_ct",
)

trainer = ClassificationTrainer(cfg)
history = trainer.train(train_loader, val_loader)
```

---

## Inference

### Video inference (detection + tracking)

Place your test video in `data/sample/` — for example `data/sample/bears.mp4`.
A short clip demonstrating the pipeline is also tracked there (see `data/sample/`).

```bash
# Detection only
python scripts/infer_video.py \
    --video data/sample/bears.mp4 \
    --model outputs/models/detection/best.pt

# Detection + SORT tracking (draws unique IDs)
python scripts/infer_video.py \
    --video input.mp4 \
    --model outputs/models/detection/best.pt \
    --track

# Instance segmentation
python scripts/infer_video.py \
    --video input.mp4 \
    --model outputs/models/segmentation/best.pt \
    --segment
```

Output is saved to `outputs/videos/<name>_annotated.mp4` by default.

### Single image inference

```bash
# Detection
python scripts/infer_image.py \
    --image bear.jpg \
    --model outputs/models/detection/best.pt \
    --output outputs/visualizations/bear_annotated.jpg

# Classification
python scripts/infer_image.py \
    --image bear.jpg \
    --model outputs/models/classification/best_classifier.pt \
    --classify
```

### Python API

```python
from src.inference.detector import BearDetector
from src.tracking.sort_tracker import SORTTracker

detector = BearDetector("best.pt", conf_threshold=0.25)
tracker = SORTTracker(max_age=30, min_hits=3)

result = detector.predict_image("bear.jpg", annotate=True)
print(f"Found {len(result['boxes'])} bears")

# Process a full video with tracking
summary = detector.process_video("input.mp4", "output.mp4", tracker=tracker)
```

---

## Evaluation

```bash
# Evaluate a trained detection model on the test split
python scripts/evaluate.py \
    --model outputs/models/detection/best.pt \
    --data "Bear detection.v3i.yolov8-obb/data.yaml"
```

Results are saved to `outputs/metrics/detection_evaluation.json`.

### Detection Metrics

| Metric | Description |
|--------|-------------|
| mAP@0.5 | Mean Average Precision at IoU=0.5 |
| mAP@0.5:0.95 | COCO-style mAP |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |

### Tracking Metrics

| Metric | Description |
|--------|-------------|
| MOTA | Multiple Object Tracking Accuracy |
| MOTP | Multiple Object Tracking Precision |
| ID switches | Number of identity changes |

### Baseline Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Notes |
|-------|---------|--------------|-----------|--------|-------|
| YOLOv8n (COCO, zero-shot) | 0.735 | 0.524 | 0.949 | 0.715 | Pretrained on COCO, no fine-tuning |

Evaluated on 164 test images (207 ground-truth boxes). Full results in
`outputs/metrics/detection_evaluation.json`.

---

## Experiment Tracking

MLflow is enabled by default. To start the MLflow UI:

```bash
mlflow ui --backend-store-uri outputs/experiments/mlruns
```

Then open `http://127.0.0.1:5000` in your browser.

Each training run logs:
- Hyperparameters (architecture, learning rate, epochs, seed, …)
- Per-epoch training and validation metrics
- Best model weights as an artifact

To disable MLflow:

```bash
python scripts/train_detection.py --no-mlflow
```

---

## Tests

Run the full test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html
```

Test coverage includes:
- Config loading and merging
- Reproducibility seeds
- Classification and detection dataset loaders
- Tracking metrics (MOTA, MOTP, ID switches)
- Detection evaluation metrics (mAP, AP, PR curve)
- Visualization utilities
- Segmentation dataset

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your changes
4. Ensure all tests pass: `pytest tests/ -v`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Danilo Ortelli**
- GitHub: [@danort92](https://github.com/danort92)
- LinkedIn: [daniloortelli](https://www.linkedin.com/in/daniloortelli/)

---

<div align="center">
  <i>Built for wildlife conservation</i>
</div>
