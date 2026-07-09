"""Training loop for the face swap model."""

from __future__ import annotations
from src.training.perceptual import PerceptualLoss
from src.training.metrics import MetricsTracker
from src.training.chroma import ChromaLoss, FeatureAppearanceLoss, LuminanceLoss
from src.data.dataset import FaceDataset, collate_faces
from src.config import StoragePaths, load_config
from models.face_swap_model import FaceSwapModel

import sys
import time
from pathlib import Path
import math
import contextlib

import torch
import torch.nn as nn
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


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted L1 normalized by mask coverage (mask is B1HW or BCHW)."""
    denom = mask.mean().clamp_min(1e-6)
    return (mask * (pred - target).abs()).mean() / denom


def _sobel_grad(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradients (dx, dy) per channel. Input B3HW -> output B6HW."""
    kx = x.new_tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    ky = x.new_tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    x_ = x.view(-1, 1, x.shape[-2], x.shape[-1])
    dx = F.conv2d(x_, kx, padding=1)
    dy = F.conv2d(x_, ky, padding=1)
    g = torch.cat([dx, dy], dim=1)
    return g.view(x.shape[0], -1, x.shape[-2], x.shape[-1])


def _border_mask(batch: torch.Tensor, border_px: int) -> torch.Tensor:
    """Return B1HW mask with 1 on border ring, 0 in interior."""
    b, _, h, w = batch.shape
    m = batch.new_zeros((b, 1, h, w))
    bp = int(max(0, border_px))
    if bp <= 0:
        return m
    m[:, :, :bp, :] = 1.0
    m[:, :, -bp:, :] = 1.0
    m[:, :, :, :bp] = 1.0
    m[:, :, :, -bp:] = 1.0
    return m


class _TVLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        return dx + dy


def train_epoch(
    model: FaceSwapModel,
    loader: DataLoader,
    gen_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer | None,
    perceptual: PerceptualLoss,
    chroma: ChromaLoss,
    luminance: LuminanceLoss,
    feature_appearance: FeatureAppearanceLoss,
    device: torch.device,
    id_weight: float,
    recon_weight: float,
    chroma_weight: float,
    luminance_weight: float,
    feature_appearance_weight: float,
    perceptual_weight: float,
    adv_weight: float,
    grad_clip: float,
    border_weight: float,
    border_grad_weight: float,
    border_px: int,
    tv_weight: float,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float, float, float, float, float, float, float]:
    model.generator.train()
    model.discriminator.train()

    total_loss = id_total = recon_total = chroma_total = lum_total = feat_total = perc_total = adv_total = 0.0
    tv = _TVLoss()

    progress = tqdm(loader, desc="Training", leave=False, mininterval=1.0)

    for step, batch in enumerate(progress, start=1):
        batch_start = time.perf_counter()

        batch = batch.to(device)
        source, target = _shuffled_pairs(batch)

        with torch.no_grad():
            identity = model.extract_identity(source)

        # --- Train discriminator (optional / throttled) ---
        train_disc_this_step = (
            disc_optimizer is not None
            and adv_weight > 0.0
        )
        if train_disc_this_step:
            assert disc_optimizer is not None
            disc_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = model.generator(target, identity)
            autocast_ok = device.type == "cuda"
            with torch.amp.autocast(device_type="cuda", enabled=autocast_ok):
                disc_real = model.discriminator(target)
                disc_fake = model.discriminator(fake.detach())
                disc_loss = (
                    F.relu(1.0 - disc_real).mean()
                    + F.relu(1.0 + disc_fake).mean()
                )
            if scaler is not None and autocast_ok:
                scaler.scale(disc_loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(disc_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.discriminator.parameters(), grad_clip
                    )
                scaler.step(disc_optimizer)
            else:
                disc_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.discriminator.parameters(), grad_clip
                    )
                disc_optimizer.step()

        # --- Train generator ---
        gen_optimizer.zero_grad(set_to_none=True)
        autocast_ok = device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", enabled=autocast_ok):
            output = model.generator(target, identity)

        if id_weight > 0.0:
            # FaceNet weights are FP32; keep identity extraction in FP32 even when AMP is on.
            out_for_id = output.float()
            no_autocast = (
                torch.amp.autocast(device_type="cuda", enabled=False)
                if autocast_ok
                else contextlib.nullcontext()
            )
            with no_autocast:
                id_out = model.extract_identity(out_for_id)
            id_loss = 1.0 - F.cosine_similarity(id_out, identity, dim=1).mean()
        else:
            id_loss = output.new_tensor(0.0)

        recon_loss = F.l1_loss(output, target)
        chroma_loss = chroma(output, target) if chroma_weight > 0.0 else output.new_tensor(0.0)
        lum_loss = luminance(output, target) if luminance_weight > 0.0 else output.new_tensor(0.0)
        feat_loss = (
            feature_appearance(output, target)
            if feature_appearance_weight > 0.0
            else output.new_tensor(0.0)
        )
        perc_loss = perceptual(output, target) if perceptual_weight > 0.0 else output.new_tensor(0.0)

        if border_weight > 0.0 and border_px > 0:
            bm = _border_mask(output, border_px)
            border_loss = _masked_l1(output, target, bm)
        else:
            border_loss = output.new_tensor(0.0)

        if border_grad_weight > 0.0 and border_px > 0:
            bm = _border_mask(output, border_px)
            go = _sobel_grad(output)
            gt = _sobel_grad(target)
            bmg = bm.expand(-1, go.shape[1], -1, -1)
            border_grad_loss = _masked_l1(go, gt, bmg)
        else:
            border_grad_loss = output.new_tensor(0.0)

        tv_loss = tv(output) if tv_weight > 0.0 else output.new_tensor(0.0)

        gen_loss = (
            id_weight * id_loss
            + recon_weight * recon_loss
            + chroma_weight * chroma_loss
            + luminance_weight * lum_loss
            + feature_appearance_weight * feat_loss
            + perceptual_weight * perc_loss
            + border_weight * border_loss
            + border_grad_weight * border_grad_loss
            + tv_weight * tv_loss
        )

        if adv_weight > 0.0:
            with torch.amp.autocast(device_type="cuda", enabled=autocast_ok):
                disc_out = model.discriminator(output)
                adv_loss = -disc_out.mean()
                gen_loss = gen_loss + adv_weight * adv_loss
            if torch.isfinite(adv_loss):
                adv_total += float(adv_loss.detach().item())

        if not torch.isfinite(gen_loss):
            print(
                f"  Warning: non-finite loss at step {step}, skipping batch.")
            continue

        if scaler is not None and autocast_ok:
            scaler.scale(gen_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(gen_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.generator.parameters(), grad_clip
                )
            scaler.step(gen_optimizer)
            scaler.update()
        else:
            gen_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.generator.parameters(), grad_clip
                )
            gen_optimizer.step()

        total_loss += float(gen_loss.detach().item())
        id_total += float(id_loss.detach().item())
        recon_total += float(recon_loss.detach().item())
        chroma_total += float(chroma_loss.detach().item())
        lum_total += float(lum_loss.detach().item())
        feat_total += float(feat_loss.detach().item())
        perc_total += float(perc_loss.detach().item())

        if step == 1:
            elapsed = time.perf_counter() - batch_start
            progress.set_postfix({"batch_s": f"{elapsed:.1f}"})

    n = len(loader)
    return (
        total_loss / n,
        id_total / n,
        recon_total / n,
        chroma_total / n,
        lum_total / n,
        feat_total / n,
        perc_total / n,
        adv_total / n,
    )


@torch.no_grad()
def validate(
    model: FaceSwapModel,
    loader: DataLoader,
    perceptual: PerceptualLoss,
    chroma: ChromaLoss,
    luminance: LuminanceLoss,
    feature_appearance: FeatureAppearanceLoss,
    device: torch.device,
    id_weight: float,
    recon_weight: float,
    chroma_weight: float,
    luminance_weight: float,
    feature_appearance_weight: float,
    perceptual_weight: float,
    border_weight: float,
    border_grad_weight: float,
    border_px: int,
    tv_weight: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    model.generator.eval()
    total_loss = id_total = recon_total = chroma_total = lum_total = feat_total = perc_total = 0.0
    similarities: list[float] = []
    tv = _TVLoss()

    for batch in loader:
        batch = batch.to(device)
        source, target = _shuffled_pairs(batch)

        identity = model.extract_identity(source)
        output = model.generator(target, identity)

        if id_weight > 0.0:
            id_out = model.extract_identity(output.float())
            id_loss = 1.0 - F.cosine_similarity(id_out, identity, dim=1).mean()
        else:
            id_loss = output.new_tensor(0.0)
        recon_loss = F.l1_loss(output, target)
        chroma_loss = chroma(output, target) if chroma_weight > 0.0 else output.new_tensor(0.0)
        lum_loss = luminance(output, target) if luminance_weight > 0.0 else output.new_tensor(0.0)
        feat_loss = (
            feature_appearance(output, target)
            if feature_appearance_weight > 0.0
            else output.new_tensor(0.0)
        )
        perc_loss = perceptual(output, target) if perceptual_weight > 0.0 else output.new_tensor(0.0)

        if border_weight > 0.0 and border_px > 0:
            bm = _border_mask(output, border_px)
            border_loss = _masked_l1(output, target, bm)
        else:
            border_loss = output.new_tensor(0.0)

        if border_grad_weight > 0.0 and border_px > 0:
            bm = _border_mask(output, border_px)
            go = _sobel_grad(output)
            gt = _sobel_grad(target)
            bmg = bm.expand(-1, go.shape[1], -1, -1)
            border_grad_loss = _masked_l1(go, gt, bmg)
        else:
            border_grad_loss = output.new_tensor(0.0)

        tv_loss = tv(output) if tv_weight > 0.0 else output.new_tensor(0.0)
        loss = (
            id_weight * id_loss
            + recon_weight * recon_loss
            + chroma_weight * chroma_loss
            + luminance_weight * lum_loss
            + feature_appearance_weight * feat_loss
            + perceptual_weight * perc_loss
            + border_weight * border_loss
            + border_grad_weight * border_grad_loss
            + tv_weight * tv_loss
        )

        total_loss += loss.item()
        id_total += id_loss.item()
        recon_total += recon_loss.item()
        chroma_total += chroma_loss.item()
        lum_total += lum_loss.item()
        feat_total += feat_loss.item()
        perc_total += perc_loss.item()
        similarities.append(cosine_similarity(
            model.extract_identity(output), identity))

    n = len(loader)
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    return (
        total_loss / n,
        id_total / n,
        recon_total / n,
        chroma_total / n,
        lum_total / n,
        feat_total / n,
        perc_total / n,
        avg_sim,
    )


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
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "collate_fn": collate_faces,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(train_ds, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = FaceSwapModel().to(device)
    perceptual = PerceptualLoss().to(device)
    chroma = ChromaLoss().to(device)
    luminance = LuminanceLoss().to(device)
    feature_appearance = FeatureAppearanceLoss().to(device)

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
    chroma_w = train_cfg["chroma_weight"]
    lum_w = float(train_cfg.get("luminance_weight", 4.0))
    feat_w = float(train_cfg.get("feature_appearance_weight", 8.0))
    perc_w = train_cfg["perceptual_weight"]
    adv_w = train_cfg["adversarial_weight"]
    grad_clip = float(train_cfg["grad_clip"])
    border_w = float(train_cfg.get("border_weight", 4.0))
    border_grad_w = float(train_cfg.get("border_grad_weight", 1.0))
    border_px = int(train_cfg.get("border_px", 18))
    tv_w = float(train_cfg.get("tv_weight", 0.0))

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler: torch.amp.GradScaler | None = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_epochs = train_cfg["epochs"]
    if start_epoch >= total_epochs:
        print(
            f"Requested {total_epochs} epochs but {start_epoch} already completed; "
            "nothing to train. Increase training.epochs to continue."
        )
        return best_path

    for epoch in range(start_epoch + 1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")
        tr_loss, tr_id, tr_recon, tr_chroma, tr_lum, tr_feat, tr_perc, tr_adv = train_epoch(
            model,
            train_loader,
            gen_optimizer,
            disc_optimizer,
            perceptual,
            chroma,
            luminance,
            feature_appearance,
            device,
            id_w,
            recon_w,
            chroma_w,
            lum_w,
            feat_w,
            perc_w,
            adv_w,
            grad_clip,
            border_w,
            border_grad_w,
            border_px,
            tv_w,
            scaler,
        )
        va_loss, va_id, va_recon, va_chroma, va_lum, va_feat, _, id_acc = validate(
            model,
            val_loader,
            perceptual,
            chroma,
            luminance,
            feature_appearance,
            device,
            id_w,
            recon_w,
            chroma_w,
            lum_w,
            feat_w,
            perc_w,
            border_w,
            border_grad_w,
            border_px,
            tv_w,
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
        print(f"  Luminance: {tr_lum:.4f} (val {va_lum:.4f})")
        print(f"  Feature appearance: {tr_feat:.4f} (val {va_feat:.4f})")
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
