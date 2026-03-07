# ============================================================
# Bear Detector — Dockerfile
# Reproducible environment for training and inference.
# ============================================================
#
# Build:
#   docker build -t bear-detector .
#
# Train detection:
#   docker run --gpus all -v $(pwd)/data:/app/data bear-detector \
#       python scripts/train_detection.py
#
# Infer on video:
#   docker run --gpus all \
#       -v $(pwd)/outputs:/app/outputs \
#       -v /path/to/video.mp4:/app/input.mp4 \
#       bear-detector \
#       python scripts/infer_video.py --video /app/input.mp4 --model best.pt --track
# ============================================================

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (leverages Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create necessary output directories
RUN mkdir -p \
    data/raw data/processed \
    outputs/models outputs/metrics outputs/visualizations outputs/videos outputs/experiments \
    results/detection results/tracking results/segmentation results/videos

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command: show help
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
