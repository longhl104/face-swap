# AI-Powered Face Swap System

Replace a face in an image or video with another face while preserving natural expressions, skin tone, and seamless blending.

## Features

- **Training pipeline** on the [LFW dataset](https://www.kaggle.com/datasets/atulanandjha/lfwpeople) via KaggleHub
- **Dataset update/augmentation** script for adding new faces
- **Production REST API** (FastAPI) for upload and swap operations
- **Image & video inference** with temporal smoothing for videos
- **Training metrics** — loss and accuracy graphs saved automatically after **every epoch** to `outputs/training/` (`training_metrics.json`, `training_metrics.csv`, `training_metrics.png`)

## Project Structure

```
face-swap/
├── config/           # YAML configuration
├── data/             # Raw uploads, processed tensors (gitignored)
├── models/           # Neural network definitions
├── src/
│   ├── data/         # Download, preprocess, dataset update
│   ├── inference/    # Face swap engine
│   ├── training/     # Training loop & metrics
│   └── api/          # FastAPI application
├── scripts/          # CLI entry points
├── docs/             # Architecture & requirements
└── outputs/          # Training graphs & inference results (gitignored)
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download & preprocess LFW dataset
python scripts/download_dataset.py
python scripts/preprocess_dataset.py

# 4. Train the model
python scripts/train.py

# 5. Run inference (CLI)
python scripts/inference.py --source path/to/source.jpg --target path/to/target.jpg

# 6. Start the API server
python scripts/serve.py
```

## Git Learning Milestones

Each major step is tagged for reference:

| Tag     | Milestone                          |
|---------|------------------------------------|
| v0.1.0  | Project scaffold                   |
| v0.2.0  | Config & storage layer             |
| v0.3.0  | Data pipeline                      |
| v0.4.0  | Model architecture                 |
| v0.5.0  | Training script & metrics          |
| v0.6.0  | Inference engine                   |
| v0.7.0  | FastAPI production interface       |
| v0.8.0  | Video support & temporal smoothing |

```bash
git tag -l          # list all milestone tags
git show v0.3.0     # inspect a specific step
```

## API Endpoints

| Method | Endpoint          | Description              |
|--------|-------------------|--------------------------|
| POST   | `/upload-source`  | Upload source face image |
| POST   | `/upload-target`  | Upload target image/video |
| POST   | `/swap`           | Trigger face swap        |
| GET    | `/health`         | Health check             |

## Third-Party vs Custom Components

| Component | Source | Trained by you? |
|-----------|--------|-----------------|
| **Generator (U-Net)** | Custom `torch.nn` | Yes |
| **Identity extractor** | Pretrained **FaceNet** (`facenet-pytorch`, VGGFace2) | No — frozen |
| **Face detection** | OpenCV YuNet | No |
| **Face mask (inference)** | BiSeNet face parsing ONNX (CelebAMask-HQ) | No — frozen |
| **Discriminator** | Custom PatchGAN | Yes (adversarial training) |
| **Optimizer** | `torch.optim.Adam` | N/A |

Only the **generator** is trained on LFW. FaceNet converts the source face into a 512-d identity vector; the generator learns to synthesize the target pose with that identity.

## License

Educational project — use responsibly and ethically.
