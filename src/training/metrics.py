"""Training metrics tracking and accuracy graph generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


class MetricsTracker:
    """Collect training/validation metrics and plot accuracy graphs."""

    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_id_losses: list[float] = []
        self.val_id_losses: list[float] = []
        self.train_recon_losses: list[float] = []
        self.val_recon_losses: list[float] = []
        self.identity_accuracies: list[float] = []

    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_id: float,
        val_id: float,
        train_recon: float,
        val_recon: float,
        id_accuracy: float,
    ) -> None:
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_id_losses.append(train_id)
        self.val_id_losses.append(val_id)
        self.train_recon_losses.append(train_recon)
        self.val_recon_losses.append(val_recon)
        self.identity_accuracies.append(id_accuracy)

    def plot(self, output_dir: Path) -> None:
        """Generate and save loss and accuracy graphs."""
        output_dir.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(self.train_losses) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, self.train_losses, label="Train")
        axes[0].plot(epochs, self.val_losses, label="Val")
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.train_recon_losses, label="Train Recon")
        axes[1].plot(epochs, self.val_recon_losses, label="Val Recon")
        axes[1].plot(epochs, self.train_id_losses, label="Train Identity")
        axes[1].plot(epochs, self.val_id_losses, label="Val Identity")
        axes[1].set_title("Component Losses")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.identity_accuracies, color="green")
        axes[2].set_title("Identity Preservation Accuracy")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Cosine Similarity")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "training_metrics.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Metrics graph saved to {plot_path}")
