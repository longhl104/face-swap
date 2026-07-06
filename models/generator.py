"""Generative face swap model (encoder-decoder with identity conditioning)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FaceSwapGenerator(nn.Module):
    """
    U-Net style generator that takes a target face (pose/expression)
    and a source identity embedding to produce a swapped face.
    """

    def __init__(self, identity_dim: int = 256) -> None:
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 7, 1, 3), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(inplace=True))

        # Identity conditioning
        self.identity_proj = nn.Linear(identity_dim, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )

        # Decoder
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(128, 3, 7, 1, 3), nn.Tanh())

    def forward(self, target_face: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(target_face)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Inject identity into bottleneck
        id_map = self.identity_proj(identity).unsqueeze(-1).unsqueeze(-1)
        id_map = id_map.expand(-1, -1, e4.size(2), e4.size(3))
        bottleneck = self.bottleneck(e4 + id_map)

        d4 = self.dec4(torch.cat([bottleneck, e3], dim=1))
        d3 = self.dec3(torch.cat([d4, e2], dim=1))
        d2 = self.dec2(torch.cat([d3, e1], dim=1))
        return self.dec1(torch.cat([d2, e1], dim=1))


class Discriminator(nn.Module):
    """PatchGAN discriminator for adversarial training."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
