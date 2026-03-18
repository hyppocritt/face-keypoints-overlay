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

## Usage

### Training

Run training:

```bash
python -m src train --data path/to/data
```

With overrides:

```bash
python -m src train \
    --data path/to/data \
    training.lr=0.01 \
    dataloader.batch_size=32
```

### Inference 

Run inference:

```bash
python -m src inference --data path/to/data
```

With overrides:

```bash
python -m src inference \
    --data path/to/data \
    inference.vis=first \
    detect.use_amp=true
```

### Configuration

You can also use YAML configuration files:

```bash
python -m src training --config path/to/yaml/config
```

```bash
python -m src inference --config path/to/yaml/config
```

**CLI overrides have priority over YAML configuration**