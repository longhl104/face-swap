"""Training loop for the face swap model."""

from __future__ import annotations
from src.training.perceptual import PerceptualLoss
from src.training.metrics import MetricsTracker
from src.training.chroma import ChromaLoss
from src.data.dataset import FaceDataset, collate_faces
from src.config import StoragePaths, load_config
from models.face_swap_model import FaceSwapModel

import sys
import time
from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


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


def train_epoch(
    model: FaceSwapModel,
    loader: DataLoader,
    gen_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer | None,
    perceptual: PerceptualLoss,
    chroma: ChromaLoss,
    device: torch.device,
    id_weight: float,
    recon_weight: float,
    chroma_weight: float,
    perceptual_weight: float,
    adv_weight: float,
    identity_every: int,
    perceptual_every: int,
    adversarial_every: int,
    grad_clip: float,
) -> tuple[float, float, float, float, float, float]:
    model.generator.train()
    model.discriminator.train()

    total_loss = id_total = recon_total = chroma_total = perc_total = adv_total = 0.0

    progress = tqdm(loader, desc="Training", leave=False, mininterval=1.0)
    identity_every = max(1, int(identity_every))
    perceptual_every = max(1, int(perceptual_every))
    adversarial_every = max(1, int(adversarial_every))

    for step, batch in enumerate(progress, start=1):
        batch_start = time.perf_counter()

        batch = batch.to(device)
        source, target = _shuffled_pairs(batch)

        with torch.no_grad():
            identity = model.extract_identity(source)

        # --- Train discriminator (optional / throttled) ---
        train_disc_this_step = (
            disc_optimizer is not None
            and (step % adversarial_every == 0)
            and adv_weight > 0.0
        )
        if train_disc_this_step:
            with torch.no_grad():
                fake = model.generator(target, identity)
            disc_real = model.discriminator(target)
            disc_fake = model.discriminator(fake.detach())
            disc_loss = (
                F.relu(1.0 - disc_real).mean()
                + F.relu(1.0 + disc_fake).mean()
            )
            disc_optimizer.zero_grad()
            disc_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.discriminator.parameters(), grad_clip
                )
            disc_optimizer.step()

        # --- Train generator ---
        output = model.generator(target, identity)

        # Heavy: FaceNet forward on generated output. Throttle in speed mode.
        compute_id = id_weight > 0.0 and (step % identity_every == 0)
        if compute_id:
            id_loss = 1.0 - F.cosine_similarity(
                model.extract_identity(output), identity, dim=1
            ).mean()
        else:
            id_loss = output.new_tensor(0.0)
        recon_loss = F.l1_loss(output, target)
        chroma_loss = (
            chroma(output, target) if chroma_weight > 0.0 else output.new_tensor(0.0)
        )
        # Heavy: VGG forward. Throttle in speed mode.
        compute_perc = perceptual_weight > 0.0 and (
            step % perceptual_every == 0)
        perc_loss = perceptual(
            output, target) if compute_perc else output.new_tensor(0.0)

        gen_loss = (
            id_weight * id_loss
            + recon_weight * recon_loss
            + chroma_weight * chroma_loss
            + perceptual_weight * perc_loss
        )

        if adv_weight > 0.0:
            disc_out = model.discriminator(output)
            adv_loss = -disc_out.mean()
            gen_loss = gen_loss + adv_weight * adv_loss
            if torch.isfinite(adv_loss):
                adv_total += adv_loss.item()

        if not torch.isfinite(gen_loss):
            print(
                f"  Warning: non-finite loss at step {step}, skipping batch.")
            continue

        gen_optimizer.zero_grad()
        gen_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.generator.parameters(), grad_clip)
        gen_optimizer.step()

        total_loss += gen_loss.item()
        id_total += id_loss.item()
        recon_total += recon_loss.item()
        chroma_total += chroma_loss.item()
        perc_total += perc_loss.item()

        if step == 1:
            elapsed = time.perf_counter() - batch_start
            progress.set_postfix({"batch_s": f"{elapsed:.1f}"})

    n = len(loader)
    return (
        total_loss / n,
        id_total / n,
        recon_total / n,
        chroma_total / n,
        perc_total / n,
        adv_total / n,
    )


@torch.no_grad()
def validate(
    model: FaceSwapModel,
    loader: DataLoader,
    perceptual: PerceptualLoss,
    chroma: ChromaLoss,
    device: torch.device,
    id_weight: float,
    recon_weight: float,
    chroma_weight: float,
    perceptual_weight: float,
) -> tuple[float, float, float, float, float, float]:
    model.generator.eval()
    total_loss = id_total = recon_total = chroma_total = perc_total = 0.0
    similarities: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        source, target = _shuffled_pairs(batch)

        identity = model.extract_identity(source)
        output = model.generator(target, identity)

        id_loss = 1.0 - F.cosine_similarity(
            model.extract_identity(output), identity, dim=1
        ).mean()
        recon_loss = F.l1_loss(output, target)
        chroma_loss = chroma(output, target)
        perc_loss = perceptual(output, target)
        loss = (
            id_weight * id_loss
            + recon_weight * recon_loss
            + chroma_weight * chroma_loss
            + perceptual_weight * perc_loss
        )

        total_loss += loss.item()
        id_total += id_loss.item()
        recon_total += recon_loss.item()
        chroma_total += chroma_loss.item()
        perc_total += perc_loss.item()
        similarities.append(cosine_similarity(
            model.extract_identity(output), identity))

    n = len(loader)
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    return total_loss / n, id_total / n, recon_total / n, chroma_total / n, perc_total / n, avg_sim


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
        dataset, [train_size,
                  val_size], generator=torch.Generator().manual_seed(42)
    )

    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": 0,
        "collate_fn": collate_faces,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(train_ds, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = FaceSwapModel().to(device)
    perceptual = PerceptualLoss().to(device)
    chroma = ChromaLoss().to(device)

    gen_optimizer = torch.optim.Adam(
        model.generator.parameters(), lr=train_cfg["learning_rate"], betas=(0.5, 0.999)
    )
    disc_optimizer = (
        torch.optim.Adam(
            model.discriminator.parameters(),
            lr=train_cfg["learning_rate"],
            betas=(0.5, 0.999),
        )
    )

    tracker = MetricsTracker(paths.training_output)
    resumed = len(tracker.history)
    if resumed:
        print(f"Resumed metrics history from {resumed} prior epoch(s)")

    best_path = paths.best_model_path

    # Seed the best-loss threshold from prior history so a fresh/early epoch
    # can't overwrite an already-good best_model.pt.
    best_val_loss = float("inf")
    finite_val_losses = [
        m.val_loss for m in tracker.history if math.isfinite(m.val_loss)]
    if finite_val_losses:
        best_val_loss = min(finite_val_losses)

    # Resume model + optimizer state so reruns continue training instead of
    # restarting from random weights.
    start_epoch = 0
    resume_enabled = bool(train_cfg.get("resume", True))
    ckpt_path = paths.latest_checkpoint_path
    if resume_enabled and ckpt_path.exists():
        checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False)
        model.generator.load_state_dict(checkpoint["generator"])
        if "gen_optimizer" in checkpoint:
            gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        if disc_optimizer is not None:
            if "discriminator" in checkpoint:
                model.discriminator.load_state_dict(
                    checkpoint["discriminator"])
            if "disc_optimizer" in checkpoint:
                disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        print(
            f"Resumed model + optimizer from {ckpt_path.name} "
            f"(completed epoch {start_epoch}, best val loss {best_val_loss:.4f})"
        )
    elif resume_enabled and best_path.exists():
        model.load_trainable(best_path, map_location=device)
        start_epoch = max((m.epoch for m in tracker.history), default=0)
        print(
            f"Resumed generator weights from {best_path.name} "
            f"(no optimizer state; best val loss {best_val_loss:.4f})"
        )

    id_w = train_cfg["identity_weight"]
    recon_w = train_cfg["reconstruction_weight"]
    chroma_w = train_cfg.get("chroma_weight", 4.0)
    perc_w = train_cfg.get("perceptual_weight", 1.0)
    adv_w = train_cfg.get("adversarial_weight", 0.5)
    identity_every = int(train_cfg.get("identity_every", 1) or 1)
    perceptual_every = int(train_cfg.get("perceptual_every", 1) or 1)
    adversarial_every = int(train_cfg.get("adversarial_every", 1) or 1)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    total_epochs = train_cfg["epochs"]
    if start_epoch >= total_epochs:
        print(
            f"Requested {total_epochs} epochs but {start_epoch} already completed; "
            "nothing to train. Increase training.epochs to continue."
        )
        return best_path

    for epoch in range(start_epoch + 1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        tr_loss, tr_id, tr_recon, tr_chroma, tr_perc, tr_adv = train_epoch(
            model, train_loader, gen_optimizer, disc_optimizer, perceptual, chroma, device,
            id_w, recon_w, chroma_w, perc_w, adv_w,
            identity_every, perceptual_every, adversarial_every, grad_clip,
        )
        va_loss, va_id, va_recon, va_chroma, _, id_acc = validate(
            model, val_loader, perceptual, chroma, device, id_w, recon_w, chroma_w, perc_w,
        )
        tracker.record(
            epoch,
            tr_loss,
            va_loss,
            tr_id,
            va_id,
            tr_recon,
            va_recon,
            id_acc)
        print(
            f"  Train loss: {tr_loss:.4f} | Val loss: {va_loss:.4f} "
            f"| Identity accuracy: {id_acc:.4f}"
        )
        print(f"  Chroma: {tr_chroma:.4f} (val {va_chroma:.4f})")
        print(f"  Perceptual: {tr_perc:.4f} | Adversarial: {tr_adv:.4f}")
        tracker.persist(paths.training_output)

        if va_loss < best_val_loss and math.isfinite(va_loss):
            best_val_loss = va_loss
            model.save_trainable(best_path)
            print(f"  Saved best generator weights to {best_path}")

        if epoch % train_cfg["checkpoint_every"] == 0:
            ckpt = {
                "epoch": epoch,
                "generator": model.generator.state_dict(),
                "gen_optimizer": gen_optimizer.state_dict(),
            }
            if disc_optimizer is not None:
                ckpt["discriminator"] = model.discriminator.state_dict()
                ckpt["disc_optimizer"] = disc_optimizer.state_dict()
            torch.save(ckpt, paths.latest_checkpoint_path)

    return best_path
