"""Pretrained FaceNet identity extractor (frozen feature extractor)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

FACENET_EMBEDDING_DIM = 512


class FaceNetIdentityExtractor(nn.Module):
    """
    Frozen pretrained FaceNet (InceptionResnetV1) for identity embeddings.
    Uses VGGFace2 weights via facenet-pytorch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = InceptionResnetV1(pretrained="vggface2").eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: BCHW in [-1, 1] at project image_size (e.g. 128x128)
        x = F.interpolate(
            x,
            size=(160, 160),
            mode="bilinear",
            align_corners=False)

        x = (x + 1) / 127.5  # maps [-1, 1] -> [0, 255]
        x = fixed_image_standardization(x)

        embeddings = self.model(x)  # (B, 512)
        return F.normalize(embeddings, p=2, dim=1)
