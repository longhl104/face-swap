"""Color blending and seamless face reinsertion."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.preprocess import FaceRegion


def create_feather_mask(height: int, width: int, kernel: int = 15) -> np.ndarray:
    """Create a soft elliptical mask for seamless blending."""
    mask = np.zeros((height, width), dtype=np.float32)
    center = (width // 2, height // 2)
    axes = (max(1, width // 2 - 2), max(1, height // 2 - 2))
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
    x1 = x - pad
    y1 = y - pad
    x2 = x + w + pad
    y2 = y + h + pad

    # Make reinsertion region square to avoid horizontally squashing the
    # square generator output when resizing back into a rectangular bbox.
    rw = x2 - x1
    rh = y2 - y1
    side = max(rw, rh)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x1 = cx - side // 2
    x2 = x1 + side
    y1 = cy - side // 2
    y2 = y1 + side

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(original.shape[1], x2)
    y2 = min(original.shape[0], y2)

    target_region = original[y1:y2, x1:x2]
    resized_face = cv2.resize(
        swapped_face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR
    )

    color_matched = match_color(resized_face, target_region)
    rh, rw = color_matched.shape[:2]
    mask = create_feather_mask(rh, rw, feather_kernel)

    # Safety: if the swapped face has near-black pixels (from padding),
    # downweight them so they don't show as side blobs.
    dark = (color_matched.mean(axis=2) < 8).astype(np.float32)
    if dark.mean() > 0.001:
        mask = mask * (1.0 - 0.95 * dark)

    mask_3d = mask[:, :, np.newaxis] * blend_ratio
    blended = (
        color_matched.astype(np.float32) * mask_3d
        + target_region.astype(np.float32) * (1 - mask_3d)
    )

    result = original.copy()
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result
