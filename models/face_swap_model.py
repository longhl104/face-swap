"""Combined face swap model wrapping identity extractor and generator."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.generator import Discriminator, FaceSwapGenerator
from models.identity_extractor import FACENET_EMBEDDING_DIM, FaceNetIdentityExtractor


class FaceSwapModel(nn.Module):
    """End-to-end face swap model for training and inference."""

    def __init__(self, identity_dim: int | None = None, facenet_pretrained: str = "vggface2") -> None:
        super().__init__()
        self.identity_extractor = FaceNetIdentityExtractor(pretrained=facenet_pretrained)
        embed_dim = identity_dim or FACENET_EMBEDDING_DIM
        if embed_dim != FACENET_EMBEDDING_DIM:
            raise ValueError(
                f"FaceNet embedding dim is {FACENET_EMBEDDING_DIM}; got {embed_dim}"
            )
        self.generator = FaceSwapGenerator(identity_dim=embed_dim)
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

    def trainable_parameters(self):
        """Parameters updated during training (generator only)."""
        return self.generator.parameters()

    def save_trainable(self, path) -> None:
        """Save only trainable weights (generator). FaceNet stays pretrained."""
        torch.save({"generator": self.generator.state_dict()}, path)

    def load_trainable(self, path, map_location=None) -> None:
        """Load trainable weights, with fallback for legacy full checkpoints."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Expected dict checkpoint, got {type(checkpoint)}")

        if "generator" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator"])
            return

        gen_state = {
            k.removeprefix("generator."): v
            for k, v in checkpoint.items()
            if k.startswith("generator.")
        }
        if gen_state:
            self.generator.load_state_dict(gen_state)
            return

        raise ValueError(
            f"No generator weights found in {path}. Retrain with the FaceNet pipeline."
        )
