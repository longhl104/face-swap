# My Face Swap System

Replace a face in an image or video with another face while preserving natural expressions, skin tone, and seamless blending.

## Features

- **Training pipeline** on the [LFW dataset](https://www.kaggle.com/datasets/atulanandjha/lfwpeople) via KaggleHub
- **Dataset update/augmentation** script for adding new faces
- **Production REST API** (FastAPI) for upload and swap operations
- **Image & video inference**
- **Training metrics**: loss and accuracy graphs saved automatically after **every epoch** to `outputs/training/` (`training_metrics.json`, `training_metrics.csv`, `training_metrics.png`)

## Development Environment

This project is developed and tested on the following machine:

- **OS:** Windows 11 (64-bit)
- **Device:** ASUS ProArt
- **CPU:** AMD Ryzen AI 9 HX 370 w/ Radeon 890M (2.00 GHz)
- **RAM:** 64 GB
- **GPU (training):** NVIDIA GeForce RTX 5070 Laptop GPU (8 GB)
- **GPU (integrated):** AMD Radeon 890M
- **Python:** 3.14
- **PyTorch:** 2.13.0+cu132 (CUDA 13.2)

PyTorch uses the NVIDIA GPU for training and inference. Set `training.num_workers: 0` in `config/config.yaml` on Windows to avoid multiprocessing issues with DataLoader.

The LFW dataset is already downloaded and preprocessed on this machine (`data/raw/`, `data/processed/`). You can go straight to training.

## Project Structure

- `config/`: YAML configuration
- `data/`: Raw uploads, processed tensors
- `models/`: Neural network definitions
- `src/data/`: Download, preprocess, dataset update
- `src/inference/`: Face swap engine
- `src/training/`: Training loop and metrics
- `src/api/`: FastAPI application
- `scripts/`: CLI entry points
- `docs/`: Architecture and requirements
- `outputs/`: Training graphs and inference results

## Quick Start

```bash
# 1. Create virtual environment
py -3.14 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt --no-deps

# 3. Train the model (dataset already in data/processed/)
python scripts/train.py

# 4. Run inference (CLI)
python scripts/inference.py --source path/to/source.jpg --target path/to/target.jpg

# 5. Start the API server
python scripts/serve.py

# Open the interactive API docs in your browser:
# http://localhost:8000/docs
#
# When testing the API, create a session first and reuse its id for
# source, target, and swap requests (see API Endpoints below).
```

To download and preprocess LFW from scratch on a new machine:

```bash
python scripts/download_dataset.py
python scripts/preprocess_dataset.py
```

## API Endpoints

- `POST /sessions`: Create a face-swap session
- `PUT /sessions/{session_id}/source`: Upload source face image
- `PUT /sessions/{session_id}/target`: Upload target image/video
- `POST /sessions/{session_id}/swaps`: Run face swap and return result
- `POST /datasets/faces`: Add face images to the training dataset
- `GET /health`: Health check

## Third-Party Components

- **Identity extractor**: Pretrained **FaceNet** (`facenet-pytorch`, VGGFace2), frozen
- **Face detection**: OpenCV YuNet

## Custom Components

- **Generator (U-Net)**: Custom `torch.nn`, trained on LFW
- **Discriminator**: Custom PatchGAN, trained via adversarial training
- **Optimizer**: `torch.optim.Adam`

## Results

Training metrics and graphs are saved to `outputs/training/` after every epoch:

- `training_metrics.json`: full loss history
- `training_metrics.csv`: same data in CSV form
- `training_metrics.png`: loss and identity accuracy plots

After 34 epochs on LFW (256×256 crops, batch size 8):

- **Best validation loss:** 2.03 (epoch 31)
- **Final validation loss:** 2.08 (epoch 34)
- **Identity accuracy:** ~89% (cosine similarity between source and swapped FaceNet embeddings)

### Inference quality

Works best on **low-to-medium resolution** images and video up to about **480p**. At that range, swaps preserve identity and blend reasonably with target pose and skin tone.

On **high-resolution** images or video (720p and above), obvious **white patches** can appear in random areas of the face. This is most visible after the swapped 256×256 crop is resized back into a large frame.

### Possible improvements

This can be improved if:

- **`image_size` is increased** (e.g. 512) and the generator is retrained with enough GPU memory. Reduces upscale artifacts on large faces.
- **A refinement pass** is added after the initial swap (e.g. a small super-resolution or color-correction network on the face crop only).
