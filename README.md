# face-keypoints-overlay

A small deep learning project that detects facial keypoints and overlays graphics on top of faces.

Features:
- CNN model for keypoint detection
- Albumentations augmentation pipeline
- Mixed precision training
- Early stopping
- Modular training pipeline

## Installation

### macOS / Linux

git clone git@github.com:hyppocritt/face-keypoints-overlay.git
cd face-keypoints-overlay

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -r requirements.txt


### Windows (PowerShell)

git clone git@github.com:hyppocritt/face-keypoints-overlay.git
cd face-keypoints-overlay

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

## Training

python -m src