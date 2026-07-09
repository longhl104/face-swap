"""CLI entry point for image face swap inference."""

from __future__ import annotations
from src.inference.engine import FaceSwapEngine

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Face swap inference")
    parser.add_argument("--source", type=Path,
                        required=True, help="Source face image")
    parser.add_argument("--target", type=Path,
                        required=True, help="Target image")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path")
    parser.add_argument("--model", type=Path, default=None,
                        help="Model weights path")
    args = parser.parse_args()

    with FaceSwapEngine() as engine:
        result = engine.swap_from_paths(args.source, args.target, args.output)

    if result:
        print(f"Result saved to {result}")
    else:
        print("Face swap failed — could not detect faces in source or target.")
        sys.exit(1)


if __name__ == "__main__":
    main()
