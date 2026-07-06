"""Identity embedding extractor (FaceNet-style encoder)."""

from __future__ import annotations

import torch
import torch.nn as nn


class IdentityExtractor(nn.Module):
    """
    Convolutional encoder that maps a face image to a fixed-size
    identity embedding vector (ArcFace/FaceNet-style).
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x).view(x.size(0), -1)
        embedding = self.fc(features)
        return nn.functional.normalize(embedding, p=2, dim=1)
