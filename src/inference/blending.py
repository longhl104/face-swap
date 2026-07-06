"""Color blending and seamless face reinsertion."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.preprocess import FaceRegion


def create_feather_mask(size: int, kernel: int = 15) -> np.ndarray:
    """Create a soft elliptical mask for seamless blending."""
    mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, size // 2)
    axes = (size // 2 - 2, size // 2 - 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return mask


def match_color(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Adjust source color statistics to match target region."""
    result = source.astype(np.float32)
    for c in range(3):
        src_mean, src_std = source[:, :, c].mean(), source[:, :, c].std() + 1e-6
        tgt_mean, tgt_std = target[:, :, c].mean(), target[:, :, c].std() + 1e-6
        result[:, :, c] = (result[:, :, c] - src_mean) * (tgt_std / src_std) + tgt_mean
    return np.clip(result, 0, 255).astype(np.uint8)


def blend_face_into_image(
    original: np.ndarray,
    swapped_face: np.ndarray,
    region: FaceRegion,
    blend_ratio: float = 0.85,
    feather_kernel: int = 15,
) -> np.ndarray:
    """Reinsert the swapped face back into the original image with feathering."""
    x, y, w, h = region.bbox
    pad = int(max(w, h) * 0.2)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(original.shape[1], x + w + pad)
    y2 = min(original.shape[0], y + h + pad)

    target_region = original[y1:y2, x1:x2]
    resized_face = cv2.resize(swapped_face, (x2 - x1, y2 - y1))

    color_matched = match_color(resized_face, target_region)
    mask = create_feather_mask(resized_face.shape[0], feather_kernel)
    if mask.shape[0] != color_matched.shape[0] or mask.shape[1] != color_matched.shape[1]:
        mask = cv2.resize(mask, (color_matched.shape[1], color_matched.shape[0]))

    mask_3d = mask[:, :, np.newaxis] * blend_ratio
    blended = (
        color_matched.astype(np.float32) * mask_3d
        + target_region.astype(np.float32) * (1 - mask_3d)
    )

    result = original.copy()
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result
