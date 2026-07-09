"""VGG perceptual loss for sharper neural face generation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19


class PerceptualLoss(nn.Module):
    """Frozen VGG19 feature loss — improves visual realism of generator output."""

    def __init__(self) -> None:
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].eval() # type: ignore
        for param in vgg.parameters():
            param.requires_grad = False
        self.features = vgg
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Input in [-1, 1] -> ImageNet normalized
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std # type: ignore

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out_feat = self.features(self._normalize(output))
        tgt_feat = self.features(self._normalize(target))
        return nn.functional.l1_loss(out_feat, tgt_feat)
