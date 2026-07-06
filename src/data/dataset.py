"""PyTorch Dataset for preprocessed face tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """Load preprocessed .npy face arrays for training."""

    def __init__(self, data_dir: Path, image_size: int = 128) -> None:
        self.data_dir = data_dir
        self.image_size = image_size
        self.samples = sorted(data_dir.rglob("*.npy"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        face = np.load(self.samples[idx])
        # Normalize to [-1, 1]
        tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor
