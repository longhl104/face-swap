from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.inference.video import swap_video
from src.inference.engine import FaceSwapEngine
from src.config import StoragePaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Video face swap inference")
    parser.add_argument("--source", type=Path,
                        required=True, help="Source face image")
    parser.add_argument("--target", type=Path,
                        required=True, help="Target video")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path")
    args = parser.parse_args()

    paths = StoragePaths()
    output = args.output or paths.inference_output / \
        f"swap_{args.target.stem}.mp4"

    with FaceSwapEngine() as engine:
        result = swap_video(engine, args.source, args.target, output)

    print(f"Result saved to {result}")


if __name__ == "__main__":
    main()
