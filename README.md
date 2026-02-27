# 🐻 Bear-Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A deep learning framework for detecting and classifying bears in wildlife camera trap images and videos — built to support conservation efforts.**

</div>

---

## 📖 Overview

Bear-Detector provides a comprehensive end-to-end pipeline for wildlife monitoring using two complementary deep learning models:

- **Classification Model** — A MobileNetV2-based binary classifier that identifies whether a camera trap image contains a bear or not.
- **Detection Model** — A YOLOv8-based object detector that localizes bears in video frames with bounding boxes, ideal for processing trail camera footage.

Both models are designed with flexibility in mind: they can be fine-tuned on new species or different datasets with minimal effort, making this framework reusable for a wide range of wildlife conservation projects.

<div align="center">
  <img src="images/cameratrap_feat.png" alt="Camera trap bear detection" width="700"/>
</div>

---

## 🎯 Key Features

- **Binary image classification** using MobileNetV2 pretrained on ImageNet
- **Real-time object detection** in video using YOLOv8
- **Customizable threshold** for sensitivity tuning (default: 0.3)
- **Roboflow integration** for seamless dataset download and setup
- **Support for pre-trained weights** to skip training and run inference directly
- **Batch image processing** via zip file upload
- **End-to-end video pipeline** with automatic frame-by-frame detection and annotated output

---

## 📁 Project Structure

```
Bear-Detector/
├── Bear_detection.ipynb            # Main notebook (classification + detection)
├── Bear detection.v3i.yolov8-obb/  # YOLOv8 dataset (Roboflow format)
├── ct/                             # Camera trap data
├── images/                         # Sample output images and GIFs
├── requirements.txt                # Python dependencies
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip
- Google Colab (recommended) or a local environment with GPU support

### Install Dependencies

```bash
pip install tensorflow keras opencv-python
pip install tensorflow-addons
pip install --upgrade typeguard
pip install ultralytics==8.0.196
pip install roboflow
pip install pyyaml
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/danort92/Bear-Detector.git
cd Bear-Detector
```

---

## 🚀 Usage

Open `Bear_detection.ipynb` in Google Colab or Jupyter Notebook and follow the steps below.

### 1. Bear Classification (Image)

The classification pipeline uses MobileNetV2 to classify images as **Bear** or **Other**.

```python
# Load and preprocess an image
image = cv2.imread("your_image.jpg")
image_resized = cv2.resize(image, (224, 224)) / 255.0
image_batch = np.expand_dims(image_resized, axis=0)

# Predict
prediction = model.predict(image_batch)
label = 'Bear' if prediction < (1 - THRESHOLD) else 'Other'
print(f"Prediction: {label}")
```

You can also upload a zip file of images — the model will sort them automatically into `predicted_bears/` and `predicted_others/` folders.

### 2. Bear Detection (Video)

The detection pipeline uses YOLOv8 to process video files frame by frame, drawing bounding boxes around detected bears.

```python
from ultralytics import YOLO

model = YOLO("your_weights.pt")
process_video_with_yolo("input_video.mp4", model, output_path="output_video.mp4")
```

### 3. Using Pre-Trained Weights

When prompted in the notebook, choose to upload pre-trained weights (`.pt` file or `.zip`) to skip training and run inference directly.

---

## 📊 Model Details

### Classification Model

| Component        | Detail                        |
|-----------------|-------------------------------|
| Architecture     | MobileNetV2 (pretrained on ImageNet) |
| Input size       | 224 × 224 px                  |
| Output           | Binary (Bear / Other)         |
| Optimizer        | Adam                          |
| Loss             | Binary Cross-Entropy          |
| Metrics          | Accuracy, Recall              |
| Class Balancing  | Sklearn `compute_class_weight` |
| Early Stopping   | Monitors `val_accuracy` (patience=2) |

### Detection Model

| Component        | Detail                        |
|-----------------|-------------------------------|
| Architecture     | YOLOv8n                       |
| Dataset source   | Roboflow (customizable)       |
| Task             | Object detection (bounding boxes) |
| Output format    | Annotated video (.mp4)        |

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

The classification dataset consists of images in two categories: **bear** and **other**, split into training and validation sets with data augmentation (rotation, flips, zoom, shear).

The detection dataset is sourced from Roboflow and labeled in YOLOv8 OBB format. You can substitute it with your own labeled dataset by updating the `data.yaml` file paths.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Danilo Ortelli**
- GitHub: [@danort92](https://github.com/danort92)
- LinkedIn: [daniloortelli](https://www.linkedin.com/in/daniloortelli/)

---

<div align="center">
  <i>Built with ❤️ for wildlife conservation</i>
</div>
