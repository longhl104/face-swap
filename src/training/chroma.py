"""Chroma-only color loss — match target skin tone without changing luminance."""

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


class ChromaLoss(nn.Module):
    """L1 on Lab a/b channels — transfer target skin hue, keep source structure."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out_lab = bgr_to_lab(output)
        tgt_lab = bgr_to_lab(target)
        loss = F.l1_loss(out_lab[:, 1:3], tgt_lab[:, 1:3])
        if not torch.isfinite(loss):
            return output.new_tensor(0.0)
        return loss
