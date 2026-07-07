"""Pretrained FaceNet identity extractor (frozen feature extractor)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

FACENET_EMBEDDING_DIM = 512


class FaceNetIdentityExtractor(nn.Module):
    """
    Frozen pretrained FaceNet (InceptionResnetV1) for identity embeddings.

    Uses VGGFace2 weights via facenet-pytorch. Not trained as part of this
    project — only the generator learns to use these fixed embeddings.
    """

    def __init__(self, pretrained: str = "vggface2") -> None:
        super().__init__()
        self.model = InceptionResnetV1(pretrained=pretrained).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: BCHW in [-1, 1] at project image_size (e.g. 128x128)
        x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        # FaceNet standardization: (pixel - 127.5) / 128
        x = (x + 1.0) * 127.5
        x = (x - 127.5) / 128.0
        embedding = self.model(x)
        return F.normalize(embedding, p=2, dim=1)
