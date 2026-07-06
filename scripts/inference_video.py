"""CLI entry point for video face swap inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import StoragePaths
from src.inference.engine import FaceSwapEngine
from src.inference.video import swap_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Video face swap inference")
    parser.add_argument("--source", type=Path, required=True, help="Source face image")
    parser.add_argument("--target", type=Path, required=True, help="Target video")
    parser.add_argument("--output", type=Path, default=None, help="Output video path")
    parser.add_argument("--model", type=Path, default=None, help="Model weights path")
    args = parser.parse_args()

    paths = StoragePaths()
    output = args.output or paths.inference_output / f"swap_{args.target.stem}.mp4"

    with FaceSwapEngine(model_path=args.model) as engine:
        result = swap_video(engine, args.source, args.target, output)

    print(f"Result saved to {result}")


if __name__ == "__main__":
    main()
