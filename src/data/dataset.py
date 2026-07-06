"""PyTorch Dataset for preprocessed face tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

EMBEDDING_SUFFIX = "_facenet_emb"


class FaceDataset(Dataset):
    """Load preprocessed .npy face arrays for training."""

    def __init__(self, data_dir: Path, image_size: int = 128) -> None:
        self.data_dir = data_dir
        self.image_size = image_size
        self.samples = sorted(
            p
            for p in data_dir.rglob("*.npy")
            if not p.stem.endswith(EMBEDDING_SUFFIX)
        )
        self.use_embeddings = bool(self.samples) and self._embedding_path(
            self.samples[0]
        ).exists()

    def _embedding_path(self, face_path: Path) -> Path:
        return face_path.with_name(f"{face_path.stem}{EMBEDDING_SUFFIX}.npy")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        face = np.load(self.samples[idx])
        tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1.0

        if self.use_embeddings:
            emb = np.load(self._embedding_path(self.samples[idx]))
            return tensor, torch.from_numpy(emb).float()

        return tensor


def collate_faces(batch: list) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Collate faces with optional precomputed FaceNet embeddings."""
    if isinstance(batch[0], tuple):
        faces, embs = zip(*batch)
        return torch.stack(faces), torch.stack(embs)
    return torch.stack(batch)
