"""Dataset update and augmentation pipeline.

Core logic for growing the training set with new faces. Used by both the
``scripts/update_dataset.py`` CLI and the FastAPI ``POST /datasets/faces`` endpoint.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import StoragePaths
from src.data.preprocess import FacePreprocessor, collect_image_paths


def augment_image(image: np.ndarray) -> list[np.ndarray]:
    """Apply horizontal flip and brightness jitter augmentations."""
    augmented: list[np.ndarray] = [image]

    flipped = cv2.flip(image, 1)
    augmented.append(flipped)

    for factor in (0.85, 1.15):
        bright = np.clip(image.astype(np.float32) *
                         factor, 0, 255).astype(np.uint8)
        augmented.append(bright)

    return augmented


def add_faces_to_dataset(
    source_dir: Path,
    augment: bool = True,
    show_progress: bool = True,
) -> int:
    """
    Copy new face images into raw storage and preprocess them.

    This is the dataset update method: drop new images into a folder,
    run this function, and the training set grows automatically.
    """
    paths = StoragePaths()
    paths.ensure_dirs()

    new_images = collect_image_paths(source_dir)
    if not new_images:
        print(f"No images found in {source_dir}")
        return 0

    added = 0
    iterator = (
        tqdm(new_images, desc="Updating dataset") if show_progress else new_images
    )
    with FacePreprocessor() as preprocessor:
        for img_path in iterator:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            images_to_process = augment_image(image) if augment else [image]
            stem = img_path.stem

            for idx, variant in enumerate(images_to_process):
                suffix = f"_aug{idx}" if idx > 0 else ""
                raw_dest = paths.raw_data / "custom" / \
                    f"{stem}{suffix}{img_path.suffix}"
                raw_dest.parent.mkdir(parents=True, exist_ok=True)
                if idx == 0:
                    shutil.copy2(img_path, raw_dest)
                else:
                    cv2.imwrite(str(raw_dest), variant)

                face = preprocessor.process_image(variant)
                if face is None:
                    continue

                proc_dest = paths.processed_data / \
                    "custom" / f"{stem}{suffix}.npy"
                proc_dest.parent.mkdir(parents=True, exist_ok=True)
                np.save(proc_dest, face.astype(np.float32))
                added += 1

    print(f"Added {added} face samples to the dataset")
    return added
