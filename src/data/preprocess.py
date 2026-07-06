"""Face detection, alignment, and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceRegion:
    """Detected face bounding box and optional landmarks."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    landmarks: np.ndarray | None = None


class FacePreprocessor:
    """Detect and align faces using MediaPipe."""

    def __init__(self, image_size: int = 128) -> None:
        self.image_size = image_size
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def detect_face(self, image: np.ndarray) -> FaceRegion | None:
        """Return the largest detected face in the image."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        if not results.detections:
            return None

        h, w = image.shape[:2]
        best = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width
            * d.location_data.relative_bounding_box.height,
        )
        box = best.location_data.relative_bounding_box
        x = max(0, int(box.xmin * w))
        y = max(0, int(box.ymin * h))
        bw = min(int(box.width * w), w - x)
        bh = min(int(box.height * h), h - y)
        return FaceRegion(bbox=(x, y, bw, bh))

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
        face = self.crop_and_align(image, region)
        return face

    def close(self) -> None:
        self._detector.close()

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
