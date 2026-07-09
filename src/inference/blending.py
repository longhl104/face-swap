"""Face reinsertion using the crop geometry from preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.preprocess import AlignedCrop


def _soft_ellipse_mask(
    height: int,
    width: int,
    inset: float = 0.04,
    blur_sigma: float = 0.06,
) -> np.ndarray:
    """Soft elliptical mask in [0, 1] to avoid visible rectangular paste edges."""
    cy, cx = height / 2.0, width / 2.0
    ry = max(1.0, height * (0.5 - inset))
    rx = max(1.0, width * (0.5 - inset))

    yy, xx = np.ogrid[:height, :width]
    dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    mask = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
    mask = mask * mask

    k = int(max(3, round(min(height, width) * blur_sigma)) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


def blend_face_into_image(
    original: np.ndarray,
    swapped_face: np.ndarray,
    crop: AlignedCrop,
    feather: bool = True,
) -> np.ndarray:
    """Reinsert the swapped face into the exact region it was cropped from.

    The swapped output is mapped back onto ``crop.crop_rect`` using the same
    geometry ``align_crop`` produced, and the reflection padding added to make
    the crop square is stripped off first. This keeps the face aligned and
    avoids the mirrored-edge "ghost" (nested face) artifact.

    When ``feather`` is True, an elliptical alpha mask softens the crop edges so
    the rectangular paste boundary is not visible in the final image.
    """
    x1, y1, x2, y2 = crop.crop_rect
    top, bottom, left, right = crop.pad
    ch, cw = y2 - y1, x2 - x1

    square = cv2.resize(
        swapped_face, (crop.side, crop.side), interpolation=cv2.INTER_LINEAR
    )
    face = square[top : top + ch, left : left + cw]

    result = original.copy()
    region = result[y1:y2, x1:x2]

    if not feather or ch < 4 or cw < 4:
        result[y1:y2, x1:x2] = face
        return result

    mask = _soft_ellipse_mask(ch, cw)[..., np.newaxis]
    blended = (
        region.astype(np.float32) * (1.0 - mask)
        + face.astype(np.float32) * mask
    )
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result
