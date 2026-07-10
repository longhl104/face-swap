"""Video face swapping."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2

from src.inference.engine import FaceSwapEngine


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _mux_audio(source_video: Path, silent_video: Path, output_video: Path) -> bool:
    """Copy audio from *source_video* onto the processed silent video."""
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        str(output_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Audio mux failed: {result.stderr.strip()}")
        return False
    return True


def swap_video(
    engine: FaceSwapEngine,
    source_path: Path,
    target_path: Path,
    output_path: Path,
    max_fps: int | None = None,
) -> Path:
    """Process a video frame-by-frame."""
    video_cfg = engine.config.get("video", {})
    fps_limit = max_fps or video_cfg.get("max_fps", 30)

    source_img = cv2.imread(str(source_path))
    if source_img is None:
        raise FileNotFoundError(f"Cannot read source image: {source_path}")

    cap = cv2.VideoCapture(str(target_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {target_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out_fps = min(src_fps, fps_limit)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    with tempfile.NamedTemporaryFile(
        suffix=".mp4", dir=output_path.parent, delete=False
    ) as tmp:
        silent_path = Path(tmp.name)

    writer = cv2.VideoWriter(str(silent_path), fourcc,
                             out_fps, (width, height))

    frame_skip = max(1, int(src_fps / out_fps))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        region = engine.preprocessor.detect_face(frame)
        if region is not None:
            source_region = engine.preprocessor.detect_face(source_img)
            if source_region is not None:
                source_face = engine.preprocessor.crop_and_align(
                    source_img, source_region)
                target_crop = engine.preprocessor.align_crop(frame, region)

                source_tensor = engine._to_tensor(source_face)
                target_tensor = engine._to_tensor(target_crop.face)

                swapped_tensor = engine.model.swap(
                    source_tensor, target_tensor)
                swapped_face = engine._from_tensor(swapped_tensor)

                from src.inference.blending import blend_face_into_image

                frame = blend_face_into_image(
                    frame, swapped_face, target_crop
                )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if not _mux_audio(target_path, silent_path, output_path):
        silent_path.replace(output_path)
        print(f"Video saved without audio to {output_path}")
    else:
        silent_path.unlink(missing_ok=True)
        print(f"Video saved to {output_path}")

    return output_path
