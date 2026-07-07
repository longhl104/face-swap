"""Color blending and seamless face reinsertion."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.preprocess import AlignedCrop
from src.inference.face_mask import get_face_parser


def _apply_mask_scale(mask: np.ndarray, scale: float) -> np.ndarray:
    """Shrink mask toward its center. scale=1.0 unchanged, 0.88 ≈ 12% smaller."""
    if scale >= 0.999:
        return mask
    h, w = mask.shape
    erosion = max(1, int(min(h, w) * (1.0 - scale) * 0.4))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion * 2 + 1, erosion * 2 + 1)
    )
    return cv2.erode(mask, kernel, iterations=1)


def create_face_mask(
    target_region: np.ndarray,
    feather_kernel: int = 15,
    mask_scale: float = 1.0,
) -> np.ndarray:
    """Build a face mask using the face-parsing ONNX model."""
    mask = get_face_parser().face_mask(target_region, feather_kernel)
    return _apply_mask_scale(mask, mask_scale)


def match_color(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match target skin chroma (Lab a/b) inside the face mask only."""
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    m = (mask > 0.1).astype(np.float32)
    if m.sum() < 1:
        m = np.ones_like(mask)

    out = src_lab.copy()
    for c in (1, 2):
        src_vals = src_lab[:, :, c][m > 0]
        tgt_vals = tgt_lab[:, :, c][m > 0]
        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        tgt_mean, tgt_std = tgt_vals.mean(), tgt_vals.std() + 1e-6
        out[:, :, c] = (out[:, :, c] - src_mean) * (tgt_std / src_std) + tgt_mean

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def blend_face_into_image(
    original: np.ndarray,
    swapped_face: np.ndarray,
    crop: AlignedCrop,
    blend_ratio: float = 0.85,
    feather_kernel: int = 15,
    mask_scale: float = 1.0,
) -> np.ndarray:
    """Reinsert the swapped face into the exact region it was cropped from.

    The swapped output is mapped back onto ``crop.crop_rect`` using the same
    geometry ``align_crop`` produced, and the reflection padding added to make
    the crop square is stripped off first. This keeps the face aligned and
    avoids the mirrored-edge "ghost" (nested face) artifact.
    """
    x1, y1, x2, y2 = crop.crop_rect
    top, bottom, left, right = crop.pad
    ch, cw = y2 - y1, x2 - x1

    # Undo resize back to the padded square, then drop the reflection border so
    # only the real face content remains.
    square = cv2.resize(
        swapped_face, (crop.side, crop.side), interpolation=cv2.INTER_LINEAR
    )
    face = square[top : top + ch, left : left + cw]

    target_region = original[y1:y2, x1:x2]
    mask = create_face_mask(target_region, feather_kernel, mask_scale)
    color_matched = match_color(face, target_region, mask)

    mask_3d = mask[:, :, np.newaxis] * blend_ratio
    blended = (
        color_matched.astype(np.float32) * mask_3d
        + target_region.astype(np.float32) * (1 - mask_3d)
    )

    result = original.copy()
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result
