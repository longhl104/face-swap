"""Dataset update and augmentation pipeline."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StoragePaths, load_config
from src.data.preprocess import FacePreprocessor, collect_image_paths


def augment_image(image: np.ndarray) -> list[np.ndarray]:
    """Apply horizontal flip and brightness jitter augmentations."""
    augmented: list[np.ndarray] = [image]

    flipped = cv2.flip(image, 1)
    augmented.append(flipped)

    for factor in (0.85, 1.15):
        bright = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        augmented.append(bright)

    return augmented


def add_faces_to_dataset(
    source_dir: Path,
    augment: bool = True,
    image_size: int = 128,
) -> int:
    """
    Copy new face images into raw storage and preprocess them.

    This is the dataset update method: drop new images into a folder,
    run this script, and the training set grows automatically.
    """
    paths = StoragePaths()
    paths.ensure_dirs()

    new_images = collect_image_paths(source_dir)
    if not new_images:
        print(f"No images found in {source_dir}")
        return 0

    added = 0
    with FacePreprocessor(image_size=image_size) as preprocessor:
        for img_path in tqdm(new_images, desc="Updating dataset"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            images_to_process = augment_image(image) if augment else [image]
            stem = img_path.stem

            for idx, variant in enumerate(images_to_process):
                suffix = f"_aug{idx}" if idx > 0 else ""
                raw_dest = paths.raw_data / "custom" / f"{stem}{suffix}{img_path.suffix}"
                raw_dest.parent.mkdir(parents=True, exist_ok=True)
                if idx == 0:
                    shutil.copy2(img_path, raw_dest)
                else:
                    cv2.imwrite(str(raw_dest), variant)

                face = preprocessor.process_image(variant)
                if face is None:
                    continue

                proc_dest = paths.processed_data / "custom" / f"{stem}{suffix}.npy"
                proc_dest.parent.mkdir(parents=True, exist_ok=True)
                np.save(proc_dest, face.astype(np.float32))
                added += 1

    print(f"Added {added} face samples to the dataset")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Update/augment the face dataset")
    parser.add_argument("source_dir", type=Path, help="Folder with new face images")
    parser.add_argument("--no-augment", action="store_true", help="Skip augmentation")
    args = parser.parse_args()

    config = load_config()
    add_faces_to_dataset(
        args.source_dir,
        augment=not args.no_augment,
        image_size=config["image_size"],
    )


if __name__ == "__main__":
    main()
