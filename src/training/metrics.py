"""Training metrics tracking and accuracy graph generation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

METRICS_JSON = "training_metrics.json"
METRICS_CSV = "training_metrics.csv"
METRICS_PLOT = "training_metrics.png"

CSV_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_id_loss",
    "val_id_loss",
    "train_recon_loss",
    "val_recon_loss",
    "identity_accuracy",
    "recorded_at",
]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_id_loss: float
    val_id_loss: float
    train_recon_loss: float
    val_recon_loss: float
    identity_accuracy: float
    recorded_at: str = ""

    def __post_init__(self) -> None:
        if not self.recorded_at:
            self.recorded_at = datetime.now(timezone.utc).isoformat()


class MetricsTracker:
    """Collect training/validation metrics, persist after every epoch, and plot."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self.history: list[EpochMetrics] = []
        if output_dir is not None:
            self.load(output_dir)

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
    ) -> EpochMetrics:
        entry = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_id_loss=train_id,
            val_id_loss=val_id,
            train_recon_loss=train_recon,
            val_recon_loss=val_recon,
            identity_accuracy=id_accuracy,
        )
        self.history = [m for m in self.history if m.epoch != epoch]
        self.history.append(entry)
        self.history.sort(key=lambda m: m.epoch)
        return entry

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        """Write aggregated metrics to JSON and CSV."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        json_path = output_dir / METRICS_JSON
        csv_path = output_dir / METRICS_CSV

        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "epochs": [asdict(m) for m in self.history],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for m in self.history:
                writer.writerow(asdict(m))

        return json_path, csv_path

    def load(self, output_dir: Path) -> int:
        """Load prior metrics so resumed training continues the history."""
        json_path = output_dir / METRICS_JSON
        if not json_path.exists():
            return 0

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.history = [EpochMetrics(**row) for row in payload.get("epochs", [])]
        self.output_dir = output_dir
        return len(self.history)

    def plot(self, output_dir: Path | None = None) -> Path | None:
        """Generate and save loss and accuracy graphs."""
        if not self.history:
            return None

        out = output_dir or self.output_dir
        if out is None:
            return None

        out.mkdir(parents=True, exist_ok=True)
        epochs = [m.epoch for m in self.history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, [m.train_loss for m in self.history], label="Train")
        axes[0].plot(epochs, [m.val_loss for m in self.history], label="Val")
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, [m.train_recon_loss for m in self.history], label="Train Recon")
        axes[1].plot(epochs, [m.val_recon_loss for m in self.history], label="Val Recon")
        axes[1].plot(epochs, [m.train_id_loss for m in self.history], label="Train Identity")
        axes[1].plot(epochs, [m.val_id_loss for m in self.history], label="Val Identity")
        axes[1].set_title("Component Losses")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, [m.identity_accuracy for m in self.history], color="green")
        axes[2].set_title("Identity Preservation Accuracy")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Cosine Similarity")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out / METRICS_PLOT
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path

    def persist(self, output_dir: Path) -> None:
        """Save JSON/CSV and refresh plot after each epoch."""
        json_path, csv_path = self.save(output_dir)
        plot_path = self.plot(output_dir)
        print(f"  Metrics saved: {json_path.name}, {csv_path.name}", end="")
        if plot_path:
            print(f", {plot_path.name}")
        else:
            print()
