"""Download the LFW dataset from Kaggle via kagglehub."""

from __future__ import annotations
import kagglehub

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StoragePaths, load_config


def download_lfw_dataset(dest: Path | None = None) -> Path:
    paths = StoragePaths()
    paths.ensure_dirs()

    target = dest or paths.raw_data
    kaggle_path = Path(kagglehub.dataset_download("atulanandjha/lfwpeople"))

    # Copy dataset contents into project storage
    if target.exists() and any(target.iterdir()):
        return target

    target.mkdir(parents=True, exist_ok=True)
    for item in kaggle_path.iterdir():
        dest_item = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)

    return target


def main() -> None:
    config = load_config()
    paths = StoragePaths(config)
    download_lfw_dataset(paths.raw_data)


if __name__ == "__main__":
    main()
