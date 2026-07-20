from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.config import StoragePaths
from src.data.preprocess import FacePreprocessor, collect_image_paths


def preprocess_dataset() -> int:
    """Process all raw images and save aligned face arrays as .npy files."""
    paths = StoragePaths()
    paths.ensure_dirs()

    raw = paths.raw_data
    out = paths.processed_data
    out.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(raw)
    if not image_paths:
        print(f"No images found in {raw}")
        return 0

    saved = 0
    with FacePreprocessor() as preprocessor:
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
    preprocess_dataset()


if __name__ == "__main__":
    main()
