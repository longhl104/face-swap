"""Combined face swap model wrapping identity extractor and generator."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.generator import Discriminator, FaceSwapGenerator
from models.identity_extractor import IdentityExtractor


class FaceSwapModel(nn.Module):
    """End-to-end face swap model for training and inference."""

    def __init__(self, identity_dim: int = 256) -> None:
        super().__init__()
        self.identity_extractor = IdentityExtractor(embedding_dim=identity_dim)
        self.generator = FaceSwapGenerator(identity_dim=identity_dim)
        self.discriminator = Discriminator()

    def extract_identity(self, source_face: torch.Tensor) -> torch.Tensor:
        return self.identity_extractor(source_face)

    def swap(
        self, source_face: torch.Tensor, target_face: torch.Tensor
    ) -> torch.Tensor:
        identity = self.extract_identity(source_face)
        return self.generator(target_face, identity)

    def forward(
        self, source_face: torch.Tensor, target_face: torch.Tensor
    ) -> torch.Tensor:
        return self.swap(source_face, target_face)
