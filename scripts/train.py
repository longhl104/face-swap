"""CLI entry point for model training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.training.trainer import train


def main() -> None:
    config = load_config()
    best_path = train(config)
    print(f"\nTraining complete. Best model: {best_path}")


if __name__ == "__main__":
    main()
