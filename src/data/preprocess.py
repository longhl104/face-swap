"""Face detection, alignment, and preprocessing utilities."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import PROJECT_ROOT

YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection_yunet.onnx"


@dataclass
class FaceRegion:
    """Detected face bounding box and optional landmarks."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    landmarks: np.ndarray | None = None


def _ensure_yunet_model() -> Path:
    YUNET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not YUNET_MODEL_PATH.exists():
        print(f"Downloading YuNet face detector to {YUNET_MODEL_PATH}...")
        urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
    return YUNET_MODEL_PATH


class FacePreprocessor:
    """Detect and align faces using OpenCV YuNet."""

    def __init__(self, image_size: int = 128) -> None:
        self.image_size = image_size
        model_path = _ensure_yunet_model()
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), 0.6, 0.3, 5000
        )

    def detect_face(self, image: np.ndarray) -> FaceRegion | None:
        """Return the largest detected face in the image."""
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            if h >= 64 and w >= 64:
                return FaceRegion(bbox=(0, 0, w, h))
            return None

        best = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        landmarks = best[4:14].reshape(5, 2).astype(np.float32)
        return FaceRegion(bbox=(x, y, fw, fh), landmarks=landmarks)

    def crop_and_align(self, image: np.ndarray, region: FaceRegion) -> np.ndarray:
        """Crop face region, pad to square, and resize."""
        x, y, w, h = region.bbox
        pad = int(max(w, h) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        crop = image[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        square = np.zeros((side, side, 3), dtype=np.uint8)
        dy = (side - ch) // 2
        dx = (side - cw) // 2
        square[dy : dy + ch, dx : dx + cw] = crop
        return cv2.resize(square, (self.image_size, self.image_size))

    def process_image(self, image: np.ndarray) -> np.ndarray | None:
        """Detect, crop, align, and return a normalized face tensor."""
        region = self.detect_face(image)
        if region is None:
            return None
        return self.crop_and_align(image, region)

    def close(self) -> None:
        pass

    def __enter__(self) -> FacePreprocessor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def collect_image_paths(root: Path) -> list[Path]:
    """Recursively collect image files under a directory."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        p for p in root.rglob("*") if p.suffix.lower() in extensions and p.is_file()
    )
