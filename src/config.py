"""Configuration loader and storage path management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    path = config_path or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path to an absolute project path."""
    return PROJECT_ROOT / relative


class StoragePaths:
    """Centralized storage directory management."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_config()
        paths = cfg["paths"]
        self.raw_data = resolve_path(paths["raw_data"])
        self.processed_data = resolve_path(paths["processed_data"])
        self.uploads = resolve_path(paths["uploads"])
        self.model_weights = resolve_path(paths["model_weights"])
        self.training_output = resolve_path(paths["training_output"])
        self.inference_output = resolve_path(paths["inference_output"])

    def ensure_dirs(self) -> None:
        """Create all storage directories if they do not exist."""
        for directory in (
            self.raw_data,
            self.processed_data,
            self.uploads,
            self.model_weights,
            self.training_output,
            self.inference_output,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def best_model_path(self) -> Path:
        return self.model_weights / "best_model.pt"

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.model_weights / "latest_checkpoint.pt"
