"""Precompute frozen FaceNet embeddings for all preprocessed faces."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.face_swap_model import FaceSwapModel
from src.config import StoragePaths, load_config
from src.data.dataset import EMBEDDING_SUFFIX, FaceDataset


@torch.no_grad()
def precompute_embeddings(
    data_dir: Path | None = None,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> int:
    """Compute and cache FaceNet embeddings next to each .npy face file."""
    paths = StoragePaths()
    data_dir = data_dir or paths.processed_data
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FaceDataset(data_dir)
    if len(dataset) == 0:
        print(f"No preprocessed faces found in {data_dir}")
        return 0

    model = FaceSwapModel().to(device)
    model.eval()

    saved = 0
    batch_faces: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    def flush() -> None:
        nonlocal saved
        if not batch_faces:
            return
        faces = torch.stack(batch_faces).to(device)
        embeddings = model.extract_identity(faces).cpu().numpy()
        for path, emb in zip(batch_paths, embeddings):
            emb_path = path.with_name(f"{path.stem}{EMBEDDING_SUFFIX}.npy")
            np.save(emb_path, emb.astype(np.float32))
            saved += 1
        batch_faces.clear()
        batch_paths.clear()

    for face_path in tqdm(dataset.samples, desc="Precomputing FaceNet embeddings"):
        emb_path = face_path.with_name(f"{face_path.stem}{EMBEDDING_SUFFIX}.npy")
        if emb_path.exists():
            continue

        face = np.load(face_path)
        tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 127.5 - 1.0
        batch_faces.append(tensor)
        batch_paths.append(face_path)

        if len(batch_faces) >= batch_size:
            flush()

    flush()
    print(f"Saved {saved} new embeddings to {data_dir}")
    return saved


def main() -> None:
    config = load_config()
    paths = StoragePaths(config)
    precompute_embeddings(paths.processed_data, batch_size=config["training"]["batch_size"])


if __name__ == "__main__":
    main()
