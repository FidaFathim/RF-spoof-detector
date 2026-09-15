"""Train the baseline CNN. Thin script — call from a notebook or CLI.

Usage (CLI):
    python -m src.baseline.train --config configs/default.yaml --split data/splits/split_seed0.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Subset

from ..preprocessing.normalize import iq_to_tensor, unit_energy_normalize
from ..preprocessing.session_split import apply_split, load_split
from ..preprocessing.wisig_loader import load_wisig, truncate_or_pad
from ..utils.config import load_config, snapshot_config
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from .cnn_model import RFFingerprintCNN

logger = get_logger(__name__)


class IQDataset(Dataset):
    def __init__(self, df, label_encoder: LabelEncoder, iq_length: int):
        self.iq = df["iq"].tolist()
        self.labels = label_encoder.transform(df["tx_id"])
        self.iq_length = iq_length

    def __len__(self) -> int:
        return len(self.iq)

    def __getitem__(self, idx: int):
        sig = truncate_or_pad(self.iq[idx], self.iq_length)
        sig = unit_energy_normalize(sig)
        x = iq_to_tensor(sig)
        return torch.from_numpy(x), int(self.labels[idx])


def train_baseline(config: dict, split_path: str, out_dir: str) -> RFFingerprintCNN:
    set_seed(config["seed"])
    data_cfg = config["data"]
    model_cfg = config["baseline_cnn"]

    df = load_wisig(data_cfg["wisig_root"], subset=data_cfg["wisig_subset"], equalized=data_cfg["equalized"])
    split = load_split(split_path)
    parts = apply_split(df, split)

    label_encoder = LabelEncoder().fit(df["tx_id"])
    num_classes = len(label_encoder.classes_)
    logger.info(f"{num_classes} devices, train/val/test sizes: "
                f"{len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])}")

    train_ds = IQDataset(parts["train"], label_encoder, data_cfg["iq_length"])
    val_ds = IQDataset(parts["val"], label_encoder, data_cfg["iq_length"])

    train_loader = DataLoader(train_ds, batch_size=model_cfg["batch_size"], shuffle=True)

    # Per-epoch validation on a fixed RANDOM subset (never a positional slice: the
    # DataFrame is grouped by transmitter, so `range(N)` would be a single device).
    max_val = model_cfg.get("max_val_samples")
    if max_val and max_val < len(val_ds):
        rng = np.random.default_rng(config["seed"])
        val_idx = rng.choice(len(val_ds), size=max_val, replace=False).tolist()
        val_ds = Subset(val_ds, val_idx)
    val_loader = DataLoader(val_ds, batch_size=model_cfg["batch_size"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RFFingerprintCNN(
        num_classes=num_classes,
        in_channels=model_cfg["in_channels"],
        conv_channels=model_cfg["conv_channels"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"], weight_decay=model_cfg["weight_decay"])
    # Cosine decay: held-out-session accuracy oscillated 60-94% between epochs at a
    # constant LR in the week-1 runs; annealing lets the later epochs settle.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_cfg["epochs"])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(model_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)
        scheduler.step()

        val_acc = evaluate_accuracy(model, val_loader, device)
        logger.info(f"epoch {epoch+1}/{model_cfg['epochs']}  train_loss={train_loss:.4f}  "
                    f"val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    np.save(out_dir / "label_classes.npy", label_encoder.classes_)
    snapshot_config(config, out_dir)
    logger.info(f"best val_acc={best_val_acc:.4f}, checkpoint saved to {out_dir / 'best_model.pt'}")
    return model


@torch.no_grad()
def evaluate_accuracy(model: RFFingerprintCNN, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", default="results/baseline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_baseline(cfg, args.split, args.out)
