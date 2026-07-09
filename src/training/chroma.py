"""Lab color losses for target skin-tone transfer during training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _linearize_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def bgr_to_lab(x: torch.Tensor) -> torch.Tensor:
    """Convert BCHW BGR tensors in [-1, 1] to normalized Lab (L,a,b in [0, 1])."""
    x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    b, g, r = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    rgb = torch.cat([_linearize_srgb(r), _linearize_srgb(g), _linearize_srgb(b)], dim=1)

    m = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.einsum("bchw,oc->bohw", rgb, m)
    xyz = xyz.clamp(min=0.0)
    xyz = xyz / xyz.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = torch.where(
        xyz > eps,
        torch.pow(xyz.clamp(min=eps), 1.0 / 3.0),
        (kappa * xyz + 16.0) / 116.0,
    )

    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])

    lab = torch.cat([(L + 16.0) / 116.0, (a + 128.0) / 255.0, (b + 128.0) / 255.0], dim=1)
    return lab.clamp(0.0, 1.0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average values inside a soft mask (mask is B1HW or broadcastable)."""
    denom = mask.mean().clamp_min(1e-6)
    return (mask * values).mean() / denom


def _soft_ellipse(
    h: int,
    w: int,
    cy: float,
    cx: float,
    ry: float,
    rx: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy = torch.linspace(0, 1, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    xx = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, w).expand(h, w)
    dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    return torch.exp(-dist.clamp(min=0.0) * 3.0)


def feature_region_mask(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Soft mask over eyes, mouth, and chin on aligned face crops."""
    regions = [
        _soft_ellipse(height, width, 0.38, 0.35, 0.09, 0.13, device, dtype),
        _soft_ellipse(height, width, 0.38, 0.65, 0.09, 0.13, device, dtype),
        _soft_ellipse(height, width, 0.66, 0.50, 0.07, 0.16, device, dtype),
        _soft_ellipse(height, width, 0.82, 0.50, 0.12, 0.20, device, dtype),
    ]
    mask = torch.stack(regions).amax(dim=0)
    return mask.view(1, 1, height, width).expand(batch_size, 1, height, width)


class ChromaLoss(nn.Module):
    """L1 on Lab a/b channels — match target hue."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out_lab = bgr_to_lab(output)
        tgt_lab = bgr_to_lab(target)
        loss = F.l1_loss(out_lab[:, 1:3], tgt_lab[:, 1:3])
        if not torch.isfinite(loss):
            return output.new_tensor(0.0)
        return loss


class LuminanceLoss(nn.Module):
    """L1 on Lab L channel — prevents washed-out / whitened skin."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out_lab = bgr_to_lab(output)
        tgt_lab = bgr_to_lab(target)
        loss = F.l1_loss(out_lab[:, 0:1], tgt_lab[:, 0:1])
        if not torch.isfinite(loss):
            return output.new_tensor(0.0)
        return loss


class FeatureAppearanceLoss(nn.Module):
    """Full Lab match on eyes, mouth, and chin — fixes localized whitening."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out_lab = bgr_to_lab(output)
        tgt_lab = bgr_to_lab(target)
        b, _, h, w = output.shape
        mask = feature_region_mask(b, h, w, output.device, output.dtype)
        diff = (out_lab - tgt_lab).abs()
        loss = _masked_mean(diff, mask)
        if not torch.isfinite(loss):
            return output.new_tensor(0.0)
        return loss
