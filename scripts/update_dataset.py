"""CLI wrapper for the dataset update and augmentation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.update import add_faces_to_dataset


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
