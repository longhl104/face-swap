"""Training loop for the face swap model."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.face_swap_model import FaceSwapModel
from src.config import StoragePaths, load_config
from src.data.dataset import FaceDataset, collate_faces
from src.training.metrics import MetricsTracker


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=1).mean().item()


def _shuffled_pairs(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create cross-identity source/target pairs within a batch."""
    batch_size = batch.size(0)
    if batch_size < 2:
        return batch, batch

    device = batch.device
    while True:
        perm = torch.randperm(batch_size, device=device)
        if not (perm == torch.arange(batch_size, device=device)).any():
            return batch, batch[perm]


def _loader_workers(requested: int) -> int:
    """Use 0 workers on Windows to avoid DataLoader hangs."""
    if sys.platform == "win32":
        return 0
    return requested


def train_epoch(
    model: FaceSwapModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    id_weight: float,
    recon_weight: float,
) -> tuple[float, float, float]:
    model.generator.train()
    total_loss = id_total = recon_total = 0.0

    progress = tqdm(loader, desc="Training", leave=False, mininterval=1.0)
    for step, batch in enumerate(progress, start=1):
        batch_start = time.perf_counter()

        embeddings = None
        if isinstance(batch, (list, tuple)):
            batch, embeddings = batch
        batch = batch.to(device)
        source, target = _shuffled_pairs(batch)

        if embeddings is not None:
            identity = embeddings.to(device)
        else:
            with torch.no_grad():
                identity = model.extract_identity(source)

        output = model.generator(target, identity)

        id_loss = 1.0 - F.cosine_similarity(
            model.extract_identity(output), identity, dim=1
        ).mean()
        recon_loss = F.l1_loss(output, target)
        loss = id_weight * id_loss + recon_weight * recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        id_total += id_loss.item()
        recon_total += recon_loss.item()

        if step == 1:
            elapsed = time.perf_counter() - batch_start
            progress.set_postfix({"batch_s": f"{elapsed:.1f}"})

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
    model.generator.eval()
    total_loss = id_total = recon_total = 0.0
    similarities: list[float] = []

    for batch in loader:
        embeddings = None
        if isinstance(batch, (list, tuple)):
            batch, embeddings = batch
        batch = batch.to(device)
        source, target = _shuffled_pairs(batch)

        identity = embeddings.to(device) if embeddings is not None else model.extract_identity(source)
        output = model.generator(target, identity)

        id_loss = 1.0 - F.cosine_similarity(
            model.extract_identity(output), identity, dim=1
        ).mean()
        recon_loss = F.l1_loss(output, target)
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

    if not dataset.use_embeddings:
        print(
            "No cached FaceNet embeddings found. Training will be slow on CPU.\n"
            "Run: python scripts/precompute_embeddings.py"
        )

    val_size = max(1, int(len(dataset) * train_cfg["val_split"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    workers = _loader_workers(train_cfg["num_workers"])
    loader_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=workers,
        collate_fn=collate_faces,
        pin_memory=device.type == "cuda",
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    facenet_pretrained = train_cfg.get("facenet_pretrained", "vggface2")
    model = FaceSwapModel(facenet_pretrained=facenet_pretrained).to(device)

    optimizer = torch.optim.Adam(
        model.trainable_parameters(), lr=train_cfg["learning_rate"]
    )
    print(
        f"Identity extractor: frozen pretrained FaceNet ({facenet_pretrained}), "
        f"embedding dim {train_cfg['latent_dim']}"
    )
    if workers != train_cfg["num_workers"]:
        print(f"DataLoader workers: {workers} (Windows-safe default)")

    tracker = MetricsTracker(paths.training_output)
    resumed = len(tracker.history)
    if resumed:
        print(f"Resumed metrics history from {resumed} prior epoch(s)")

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
        tracker.persist(paths.training_output)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            model.save_trainable(best_path)
            print(f"  Saved best generator weights to {best_path}")

        if epoch % train_cfg["checkpoint_every"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "generator": model.generator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                paths.latest_checkpoint_path,
            )

    return best_path
