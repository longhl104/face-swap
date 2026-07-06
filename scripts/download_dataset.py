"""Download the LFW dataset from Kaggle via kagglehub."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kagglehub

from src.config import StoragePaths, load_config


def download_lfw_dataset(dest: Path | None = None) -> Path:
    """Download LFW dataset and copy into project raw data directory."""
    paths = StoragePaths()
    paths.ensure_dirs()

    target = dest or paths.raw_data
    print("Downloading LFW dataset from Kaggle...")
    kaggle_path = Path(kagglehub.dataset_download("atulanandjha/lfwpeople"))
    print(f"Kaggle cache path: {kaggle_path}")

    # Copy dataset contents into project storage
    if target.exists() and any(target.iterdir()):
        print(f"Raw data directory already populated: {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    for item in kaggle_path.iterdir():
        dest_item = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)

    print(f"Dataset copied to: {target}")
    return target


def main() -> None:
    config = load_config()
    paths = StoragePaths(config)
    download_lfw_dataset(paths.raw_data)


if __name__ == "__main__":
    main()
