from __future__ import annotations

import torch
import torch.nn as nn

from models.generator import Discriminator, FaceSwapGenerator
from models.identity_extractor import FACENET_EMBEDDING_DIM, FaceNetIdentityExtractor


class FaceSwapModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity_extractor = FaceNetIdentityExtractor()
        self.generator = FaceSwapGenerator(identity_dim=FACENET_EMBEDDING_DIM)
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

    def save_trainable(self, path) -> None:
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
            },
            path,
        )

    def load_trainable(self, path, map_location=None) -> None:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Expected dict checkpoint, got {type(checkpoint)}")

        if "generator" in checkpoint:
            state = checkpoint["generator"]
            try:
                self.generator.load_state_dict(state)
            except RuntimeError:
                result = self.generator.load_state_dict(state, strict=False)
                missing = len(result.missing_keys)
                if missing:
                    print(
                        f"Warning: {missing} generator layers not loaded — "
                        "architecture changed. Retrain from scratch."
                    )
            if "discriminator" in checkpoint:
                self.discriminator.load_state_dict(
                    checkpoint["discriminator"], strict=False)
            return

        gen_state = {
            k.removeprefix("generator."): v
            for k, v in checkpoint.items()
            if k.startswith("generator.")
        }
        if gen_state:
            self.generator.load_state_dict(gen_state, strict=False)
            return

        raise ValueError(
            f"No generator weights found in {path}. Run scripts/train.py first."
        )
