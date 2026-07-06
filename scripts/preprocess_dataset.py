"""Batch preprocessing of raw images into normalized face tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.config import StoragePaths, load_config
from src.data.preprocess import FacePreprocessor, collect_image_paths


def preprocess_dataset(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    image_size: int = 128,
) -> int:
    """Process all raw images and save aligned face arrays as .npy files."""
    paths = StoragePaths()
    paths.ensure_dirs()

    raw = raw_dir or paths.raw_data
    out = output_dir or paths.processed_data
    out.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(raw)
    if not image_paths:
        print(f"No images found in {raw}")
        return 0

    saved = 0
    with FacePreprocessor(image_size=image_size) as preprocessor:
        for img_path in tqdm(image_paths, desc="Preprocessing faces"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            face = preprocessor.process_image(image)
            if face is None:
                continue

            rel = img_path.relative_to(raw)
            out_path = out / rel.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, face.astype(np.float32))
            saved += 1

    print(f"Saved {saved} preprocessed faces to {out}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess LFW faces")
    parser.add_argument("--raw", type=Path, default=None, help="Raw data directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    config = load_config()
    preprocess_dataset(args.raw, args.output, config["image_size"])


if __name__ == "__main__":
    main()
