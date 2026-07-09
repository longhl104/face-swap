"""Face reinsertion using the crop geometry from preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.preprocess import AlignedCrop


def blend_face_into_image(
    original: np.ndarray,
    swapped_face: np.ndarray,
    crop: AlignedCrop,
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

    result = original.copy()
    result[y1:y2, x1:x2] = face
    return result
