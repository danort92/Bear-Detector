# 🐻 Bear-Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13--2.15-orange?logo=tensorflow&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A deep learning framework for detecting and classifying bears in wildlife camera-trap images and videos — built to support conservation efforts.**

</div>

---

## 📖 Overview

Bear-Detector provides a complete end-to-end pipeline for wildlife monitoring via two complementary deep learning models:

| Model | Architecture | Task |
|-------|-------------|------|
| **Classifier** | MobileNetV2 + custom head | Binary image classification (Bear / Other) |
| **Detector** | YOLOv8n | Object detection with bounding boxes in video |

Both models are designed to be fine-tuned on new species or custom datasets with minimal effort.

<div align="center">
  <img src="images/cameratrap_feat.png" alt="Camera trap bear detection" width="700"/>
</div>

---

## 🎯 Key Features

- **Binary image classification** using MobileNetV2 with transfer learning
- **Real-time object detection** in video using YOLOv8
- **Configurable confidence threshold** for recall / precision trade-off (default: `0.3`)
- **Roboflow integration** for dataset download and setup
- **Support for pre-trained weights** to skip training and run inference directly
- **Batch image processing** via ZIP file upload
- **End-to-end video pipeline** with annotated output

---

## 📁 Project Structure

```
Bear-Detector/
├── notebooks/
│   └── Bear_detection.ipynb       # Main notebook (classification + detection)
├── src/
│   └── bear_detector/             # Importable Python package
│       ├── __init__.py
│       ├── config.py              # Dataclass-based configuration
│       ├── classification.py      # MobileNetV2 training & inference
│       ├── detection.py           # YOLOv8 training & video processing
│       └── utils.py               # Shared helpers (image, zip, yaml)
├── ct/                            # Camera-trap classification dataset
│   ├── bear_ct/                   # Bear images (1 002)
│   └── other_ct/                  # Non-bear images (2 002)
├── Bear detection.v3i.yolov8-obb/ # YOLOv8 detection dataset (Roboflow)
├── images/                        # Sample output images and GIFs
├── requirements.txt               # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- pip
- Google Colab (recommended) or a local environment with GPU support

### Clone the Repository

```bash
git clone https://github.com/danort92/Bear-Detector.git
cd Bear-Detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Open `notebooks/Bear_detection.ipynb` in Google Colab or Jupyter and follow the numbered sections.

### Threshold Semantics

The model sigmoid output represents `P(other)`. The *threshold* controls the
minimum `P(bear) = 1 - output` required to classify an image as "Bear":

```python
# Consistent rule used throughout the notebook
is_bear = (1 - model_output) >= threshold   # i.e. model_output < 1 - threshold
```

A **lower** threshold → higher recall (fewer missed bears, more false positives).
A **higher** threshold → higher precision (fewer false alarms, more missed bears).

### Classification Inference (Single Image)

```python
from bear_detector.utils import batch_from_path
from bear_detector.classification import predict_image

batch = batch_from_path("your_image.jpg")
label, bear_prob = predict_image(model, batch, threshold=0.3)
print(f"{label}  (P(bear) = {bear_prob:.3f})")
```

### Detection Inference (Video)

```python
from ultralytics import YOLO
from bear_detector.detection import process_video

yolo = YOLO("best.pt")
frames = process_video("input.mp4", yolo, output_path="output_detected.mp4")
print(f"Processed {frames} frames")
```

---

## 📊 Model Details

### Classification Model

| Component | Detail |
|-----------|--------|
| Architecture | MobileNetV2 (frozen) + GAP + BN + Dense(128) + Dropout + Sigmoid |
| Input size | 224 × 224 px |
| Output | `P(other)` in [0, 1] |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Metrics | Accuracy, Recall |
| Class Balancing | `sklearn.utils.class_weight.compute_class_weight` |
| Regularisation | Dropout (0.3), Batch Normalisation, ReduceLROnPlateau |
| Early Stopping | `val_accuracy`, patience = 5 |

### Detection Model

| Component | Detail |
|-----------|--------|
| Architecture | YOLOv8n |
| Dataset source | Roboflow (or local `Bear detection.v3i.yolov8-obb/`) |
| Task | Object detection (bounding boxes) |
| Output | Annotated video (.mp4) |

---

## 🎥 Detection Examples

<div align="center">

**High accuracy — no false positives or negatives**

<img src="images/358739111-ea41e5cd-641e-4a36-87c7-b3cdf565cd6e.gif" width="600"/>

**Good precision — room to improve recall on standing bears**

<img src="images/358741448-9c947f16-ff53-4aa6-b254-58b9583aade8.gif" width="600"/>

</div>

---

## 🔭 Future Developments

- [ ] Multi-species classification and detection (beyond bears)
- [ ] Individual animal re-identification (tracking specific animals over time)
- [ ] Streamlit or Hugging Face Spaces demo for interactive inference
- [ ] Support for night-vision / infrared camera trap images
- [ ] Model performance benchmarking across species

---

## 📦 Dataset

### Classification

- **Bear**: 1 002 images (`ct/bear_ct/`)
- **Other**: 2 002 images (`ct/other_ct/`)
- **Split**: 80 / 20 train / val (stratified, `random_seed=42`)
- **Augmentation**: rotation, flips, zoom, shear (on-the-fly)

### Detection

- Sourced from Roboflow, YOLOv8 OBB format
- **Train**: 1 008 images | **Test**: 164 images
- Substitute your own dataset by updating `data.yaml` paths (handled automatically by `setup_local_dataset`)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Danilo Ortelli**
- GitHub: [@danort92](https://github.com/danort92)
- LinkedIn: [daniloortelli](https://www.linkedin.com/in/daniloortelli/)

---

<div align="center">
  <i>Built with ❤️ for wildlife conservation</i>
</div>
