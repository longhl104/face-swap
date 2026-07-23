"""Generate U-Net intermediate images for the Part 2 video graphic.

Runs a single forward pass through the trained FaceSwapGenerator and
saves encoder / bottleneck / decoder feature visualizations that the
Manim scene can load as real training imagery.

Usage (from repo root):

    python scripts/generate_unet_presentation.py
    python scripts/generate_unet_presentation.py \\
        --source video_graphics/assets/images/source-1.jpg \\
        --target video_graphics/assets/images/target-1.JPG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.face_swap_model import FaceSwapModel
from src.config import StoragePaths, load_config
from src.data.preprocess import FacePreprocessor

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "video_graphics"
    / "assets"
    / "images"
)
DEFAULT_SOURCE = OUT_DIR / "source-1.jpg"
DEFAULT_TARGET = OUT_DIR / "target-1.JPG"

# Display sizes match the U-shape: large → small → large.
DISPLAY_SIZES = {
    "unet-enc-1": 256,
    "unet-enc-2": 192,
    "unet-enc-3": 128,
    "unet-bottleneck": 96,
    "unet-dec-3": 128,
    "unet-dec-2": 192,
    "unet-dec-1": 256,
}


def features_to_rgb(feat: torch.Tensor) -> np.ndarray:
    """Project multi-channel features to an RGB image via PCA (HWC, uint8 BGR)."""
    if feat.dim() == 4:
        feat = feat[0]
    c, h, w = feat.shape
    x = feat.detach().float().cpu().permute(1, 2, 0).reshape(-1, c).numpy()
    x = x - x.mean(axis=0, keepdims=True)
    # Low-rank PCA via SVD; pad if fewer than 3 channels.
    rank = min(3, c, x.shape[0])
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:rank].T
    if rank < 3:
        pad = np.zeros((proj.shape[0], 3 - rank), dtype=proj.dtype)
        proj = np.concatenate([proj, pad], axis=1)
    rgb = proj.reshape(h, w, 3)
    for i in range(3):
        ch = rgb[:, :, i]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi <= lo:
            hi = lo + 1e-6
        rgb[:, :, i] = np.clip((ch - lo) / (hi - lo), 0, 1)
    # Swap R/B so OpenCV writes a natural-looking face-ish image.
    bgr = (rgb[..., ::-1] * 255).astype(np.uint8)
    return bgr


def energy_overlay(feat: torch.Tensor, base_bgr: np.ndarray) -> np.ndarray:
    """Blend |mean| activation heatmap onto a face crop (same spatial size)."""
    if feat.dim() == 4:
        feat = feat[0]
    energy = feat.detach().float().abs().mean(dim=0).cpu().numpy()
    energy = cv2.resize(energy, (base_bgr.shape[1], base_bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
    lo, hi = np.percentile(energy, 5), np.percentile(energy, 95)
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((energy - lo) / (hi - lo), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = 0.45
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heat, alpha, 0)


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def to_tensor(face_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(face_bgr).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor.unsqueeze(0).to(device)


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    face = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.clip((face + 1.0) * 127.5, 0, 255).astype(np.uint8)


@torch.no_grad()
def capture_stages(
    model: FaceSwapModel,
    source_face: np.ndarray,
    target_face: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """One forward pass; return BGR images for each U-Net stage."""
    gen = model.generator
    source_t = to_tensor(source_face, device)
    target_t = to_tensor(target_face, device)
    identity = model.extract_identity(source_t)

    e1 = gen.enc1(target_t)
    e2 = gen.enc2(e1)
    e3 = gen.enc3(e2)
    e4 = gen.enc4(e3)

    id_map = gen.identity_proj(identity).unsqueeze(-1).unsqueeze(-1)
    id_map = id_map.expand(-1, -1, e4.size(2), e4.size(3))
    bottleneck = gen.bottleneck(e4 + id_map)

    d4 = gen.up4(bottleneck)
    d4 = torch.cat([d4, e3], dim=1)
    d3 = gen.up3(d4)
    d3 = torch.cat([d3, e2], dim=1)
    d2 = gen.up2(d3)
    d2 = torch.cat([d2, e1], dim=1)
    out = gen.dec1(d2)

    target_256 = resize_square(target_face, 256)
    # Encoder: face + activation energy (reads as "compress").
    enc1 = energy_overlay(e1, target_256)
    enc2 = energy_overlay(e2, resize_square(target_face, 128))
    enc3 = energy_overlay(e3, resize_square(target_face, 64))
    # Bottleneck: abstract PCA of compressed features.
    bot = features_to_rgb(bottleneck)
    # Decoder: PCA of skip-fused features → final RGB output.
    dec3 = features_to_rgb(d4)
    dec2 = features_to_rgb(d3)
    dec1 = from_tensor(out)

    return {
        "unet-enc-1": enc1,
        "unet-enc-2": enc2,
        "unet-enc-3": enc3,
        "unet-bottleneck": bot,
        "unet-dec-3": dec3,
        "unet-dec-2": dec2,
        "unet-dec-1": dec1,
    }


def load_aligned_faces(
    source_path: Path, target_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    source = cv2.imread(str(source_path))
    target = cv2.imread(str(target_path))
    if source is None or target is None:
        raise FileNotFoundError(f"Could not read {source_path} or {target_path}")

    with FacePreprocessor() as prep:
        source_face = prep.process_image(source)
        target_face = prep.process_image(target)
    if source_face is None or target_face is None:
        raise RuntimeError("Could not detect a face in source or target image")
    return source_face, target_face


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate U-Net stage images for the video graphic")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    cfg = load_config()
    paths = StoragePaths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaceSwapModel().to(device)
    weights = paths.best_model_path
    if not weights.exists():
        raise FileNotFoundError(
            f"No weights at {weights}. Train first with scripts/train.py.")
    model.load_trainable(weights, map_location=device)
    model.eval()
    print(f"Loaded weights from {weights}")

    source_face, target_face = load_aligned_faces(args.source, args.target)
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")

    stages = capture_stages(model, source_face, target_face, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, img in stages.items():
        size = DISPLAY_SIZES[name]
        out = resize_square(img, size)
        path = args.output_dir / f"{name}.jpg"
        cv2.imwrite(str(path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"  wrote {path}  ({size}x{size})")

    print("Done — one forward pass saved for the U-Net presentation.")


if __name__ == "__main__":
    main()
