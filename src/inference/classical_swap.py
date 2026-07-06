"""Landmark-based face warping for visible identity transfer."""

from __future__ import annotations

import cv2
import numpy as np

from src.data.preprocess import FaceRegion
from src.inference.blending import match_color


def _estimate_transform(
    source_lm: np.ndarray, target_lm: np.ndarray
) -> np.ndarray | None:
    src = source_lm.astype(np.float32).reshape(-1, 1, 2)
    dst = target_lm.astype(np.float32).reshape(-1, 1, 2)
    transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return transform


def _face_mask(image_shape: tuple[int, ...], region: FaceRegion) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    if region.landmarks is not None and len(region.landmarks) >= 3:
        x, y, bw, bh = region.bbox
        lm = region.landmarks.copy()
        forehead = lm[2].copy()
        forehead[1] -= bh * 0.35
        chin = lm[2].copy()
        chin[1] += bh * 0.25
        cheek_l = lm[0].copy()
        cheek_l[0] -= bw * 0.15
        cheek_r = lm[1].copy()
        cheek_r[0] += bw * 0.15
        expanded = np.vstack([lm, forehead, chin, cheek_l, cheek_r])
        hull = cv2.convexHull(expanded.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
    else:
        x, y, bw, bh = region.bbox
        pad = int(max(bw, bh) * 0.15)
        center = (x + bw // 2, y + bh // 2)
        axes = (bw // 2 + pad, bh // 2 + pad)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    return cv2.GaussianBlur(mask, (21, 21), 0)


def warp_and_blend(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_region: FaceRegion,
    target_region: FaceRegion,
) -> np.ndarray | None:
    """Warp source face onto target using YuNet landmarks and soft alpha blending."""
    if source_region.landmarks is None or target_region.landmarks is None:
        return None

    transform = _estimate_transform(source_region.landmarks, target_region.landmarks)
    if transform is None:
        return None

    h, w = target_image.shape[:2]
    warped = cv2.warpAffine(
        source_image,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    mask = _face_mask(target_image.shape, target_region)
    mask_3d = mask[:, :, np.newaxis]

    x, y, bw, bh = target_region.bbox
    pad = int(max(bw, bh) * 0.2)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x2 > x1 and y2 > y1:
        warped[y1:y2, x1:x2] = match_color(
            warped[y1:y2, x1:x2], target_image[y1:y2, x1:x2]
        )

    blended = warped.astype(np.float32) * mask_3d + target_image.astype(np.float32) * (
        1.0 - mask_3d
    )
    return blended.astype(np.uint8)
