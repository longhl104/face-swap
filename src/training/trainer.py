"""Training loop for the face swap model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.face_swap_model import FaceSwapModel
from src.config import StoragePaths, load_config
from src.data.dataset import FaceDataset
from src.training.metrics import MetricsTracker


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=1).mean().item()


def train_epoch(
    model: FaceSwapModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    id_weight: float,
    recon_weight: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss = id_total = recon_total = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        # Self-reconstruction: source == target for unsupervised training
        source = batch
        target = batch

        identity = model.extract_identity(source)
        output = model.generator(target, identity)

        id_loss = F.mse_loss(model.extract_identity(output), identity)
        recon_loss = F.l1_loss(output, target)
        loss = id_weight * id_loss + recon_weight * recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        id_total += id_loss.item()
        recon_total += recon_loss.item()

    n = len(loader)
    return total_loss / n, id_total / n, recon_total / n


@torch.no_grad()
def validate(
    model: FaceSwapModel,
    loader: DataLoader,
    device: torch.device,
    id_weight: float,
    recon_weight: float,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = id_total = recon_total = 0.0
    similarities: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        identity = model.extract_identity(batch)
        output = model.generator(batch, identity)

        id_loss = F.mse_loss(model.extract_identity(output), identity)
        recon_loss = F.l1_loss(output, batch)
        loss = id_weight * id_loss + recon_weight * recon_loss

        total_loss += loss.item()
        id_total += id_loss.item()
        recon_total += recon_loss.item()
        similarities.append(cosine_similarity(model.extract_identity(output), identity))

    n = len(loader)
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    return total_loss / n, id_total / n, recon_total / n, avg_sim


def train(config: dict | None = None) -> Path:
    """Run full training loop and return path to best model weights."""
    cfg = config or load_config()
    train_cfg = cfg["training"]
    paths = StoragePaths(cfg)
    paths.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = FaceDataset(paths.processed_data, cfg["image_size"])
    if len(dataset) == 0:
        raise RuntimeError(
            "No preprocessed data found. Run scripts/download_dataset.py "
            "and scripts/preprocess_dataset.py first."
        )

    val_size = max(1, int(len(dataset) * train_cfg["val_split"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    model = FaceSwapModel(identity_dim=train_cfg["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    tracker = MetricsTracker()

    best_val_loss = float("inf")
    best_path = paths.best_model_path

    for epoch in range(1, train_cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{train_cfg['epochs']}")
        tr_loss, tr_id, tr_recon = train_epoch(
            model, train_loader, optimizer, device,
            train_cfg["identity_weight"], train_cfg["reconstruction_weight"],
        )
        va_loss, va_id, va_recon, id_acc = validate(
            model, val_loader, device,
            train_cfg["identity_weight"], train_cfg["reconstruction_weight"],
        )
        tracker.record(epoch, tr_loss, va_loss, tr_id, va_id, tr_recon, va_recon, id_acc)
        print(
            f"  Train loss: {tr_loss:.4f} | Val loss: {va_loss:.4f} "
            f"| Identity accuracy: {id_acc:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"  Saved best model to {best_path}")

        if epoch % train_cfg["checkpoint_every"] == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                paths.latest_checkpoint_path,
            )

    tracker.plot(paths.training_output)
    return best_path
