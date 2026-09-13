"""Run all enabled attacks against a trained baseline CNN and record raw attack success
rate (before any defense) — this reproduces the "known weakness," it's not the novel result.

Usage:
    python -m src.attacks.run_attacks --config configs/default.yaml \
        --checkpoint results/baseline/best_model.pt --split data/splits/split_seed0.json \
        --out results/attacks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from ..baseline.cnn_model import RFFingerprintCNN
from ..baseline.train import IQDataset
from ..preprocessing.session_split import apply_split, load_split
from ..preprocessing.wisig_loader import load_wisig
from ..utils.config import load_config
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from .fgsm import fgsm_targeted
from .pgd import pgd_targeted
from .uap import apply_uap, train_uap

logger = get_logger(__name__)


@torch.no_grad()
def attack_success_rate(model, x_adv: torch.Tensor, target_label: int, device: str) -> float:
    model.eval()
    logits = model(x_adv.to(device))
    preds = logits.argmax(dim=1).cpu().numpy()
    return float(np.mean(preds == target_label))


def run(config: dict, checkpoint_path: str, split_path: str, out_dir: str) -> None:
    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = config["data"]
    attack_cfg = config["attacks"]

    df = load_wisig(data_cfg["wisig_root"])
    split = load_split(split_path)
    parts = apply_split(df, split)

    label_encoder = LabelEncoder().fit(df["tx_id"])
    num_classes = len(label_encoder.classes_)

    model = RFFingerprintCNN(
        num_classes=num_classes,
        in_channels=config["baseline_cnn"]["in_channels"],
        conv_channels=config["baseline_cnn"]["conv_channels"],
        kernel_size=config["baseline_cnn"]["kernel_size"],
        dropout=config["baseline_cnn"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_ds = IQDataset(parts["test"], label_encoder, data_cfg["iq_length"])
    test_loader = DataLoader(test_ds, batch_size=config["baseline_cnn"]["batch_size"], shuffle=False)

    # Pick an arbitrary but fixed target label (the "impersonated" device) for the targeted
    # attacks — the attacker wants any signal to be misclassified as this specific device.
    target_label = 0
    results = {}

    for x, y in test_loader:
        x = x.to(device)
        non_target_mask = (y != target_label)
        x_victim = x[non_target_mask]
        if x_victim.numel() == 0:
            continue
        target = torch.full((x_victim.size(0),), target_label, dtype=torch.long, device=device)

        x_fgsm = fgsm_targeted(model, x_victim, target, attack_cfg["fgsm"]["epsilon"])
        x_pgd = pgd_targeted(
            model, x_victim, target,
            attack_cfg["pgd"]["epsilon"], attack_cfg["pgd"]["alpha"], attack_cfg["pgd"]["steps"],
        )

        results.setdefault("fgsm", []).append(attack_success_rate(model, x_fgsm, target_label, device))
        results.setdefault("pgd", []).append(attack_success_rate(model, x_pgd, target_label, device))
        break  # one batch is enough for a smoke test; loop over all batches for the real run

    # UAP is trained once on a batch of non-target clean signals, then applied broadly.
    all_x = torch.cat([x for x, _ in test_loader], dim=0).to(device)
    all_y = torch.cat([y for _, y in test_loader], dim=0)
    victims_x = all_x[all_y != target_label]
    delta = train_uap(model, victims_x, target_label, attack_cfg["uap"]["epsilon"], attack_cfg["uap"]["max_iters"])
    x_uap = apply_uap(victims_x, delta)
    results["uap"] = [attack_success_rate(model, x_uap, target_label, device)]

    summary = {name: float(np.mean(vals)) for name, vals in results.items()}
    logger.info(f"Attack success rates (targeted, direct-injection): {summary}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "attack_success_rates.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", default="results/attacks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, args.checkpoint, args.split, args.out)
