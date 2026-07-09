"""Face detection, alignment, and preprocessing utilities."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import PROJECT_ROOT, load_config

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


@dataclass
class AlignedCrop:
    """A square, resized face crop plus the geometry needed to reinsert it.

    ``face`` is the network-ready square crop. ``crop_rect`` is the exact
    (clamped) region in the original image the real face content came from,
    and ``pad`` is the reflection padding added to make the crop square.
    Together they let the swapped output be placed back with pixel-perfect
    alignment and no mirrored-edge (ghost ear) artifacts.
    """

    face: np.ndarray
    # x1, y1, x2, y2 in the original image
    crop_rect: tuple[int, int, int, int]
    # top, bottom, left, right reflection padding
    pad: tuple[int, int, int, int]
    side: int  # square side length (before resize to image_size)


def _ensure_yunet_model() -> Path:
    YUNET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not YUNET_MODEL_PATH.exists():
        print(f"Downloading YuNet face detector to {YUNET_MODEL_PATH}...")
        urllib.request.urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
    return YUNET_MODEL_PATH


class FacePreprocessor:
    """Detect and align faces using OpenCV YuNet."""

    def __init__(self) -> None:
        self.image_size = load_config()["image_size"]
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

    def align_crop(self, image: np.ndarray, region: FaceRegion) -> AlignedCrop:
        """Crop face region, pad to square, resize, and record insertion geometry."""
        x, y, w, h = region.bbox
        pad = int(max(w, h) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        crop = image[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        side = max(ch, cw)

        # Pad to square without introducing black borders (black borders become
        # visible as "rounded black blobs" after blending).
        top = (side - ch) // 2
        bottom = side - ch - top
        left = (side - cw) // 2
        right = side - cw - left
        square = cv2.copyMakeBorder(
            crop,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        face = cv2.resize(
            square, (self.image_size,
                     self.image_size), interpolation=cv2.INTER_LINEAR
        )
        return AlignedCrop(
            face=face,
            crop_rect=(x1, y1, x2, y2),
            pad=(top, bottom, left, right),
            side=side,
        )

    def crop_and_align(self, image: np.ndarray, region: FaceRegion) -> np.ndarray:
        """Crop face region, pad to square, and resize (network-ready face only)."""
        return self.align_crop(image, region).face

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
