"""Attack-only report: run FGSM / PGD / UAP against the trained baseline CNN and record the
raw (undefended) attack success rate per attack. This reproduces the "known weakness"; it
is not the novel result.

This is a thin wrapper around src.evaluation.run_full_eval, which already generates the
same attacks (chunked, UAP trained on validation victims only) as part of the full table --
so the attack numbers here are guaranteed identical to the ones in the final report rather
than being a second, slightly different implementation.

Usage:
    python -m src.attacks.run_attacks --config configs/default.yaml \
        --checkpoint results/baseline/best_model.pt --split data/splits/split_seed0.json \
        --features results/features/features.csv --gate-dir results/consistency \
        --out results/attacks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.run_full_eval import run as run_full_eval
from ..utils.config import load_config
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def run(config: dict, checkpoint: str, split: str, features: str, gate_dir: str, out_dir: str,
        max_test_samples: int | None = None) -> dict[str, float]:
    run_full_eval(config, split, checkpoint, features, gate_dir, out_dir, max_test_samples)
    with open(Path(out_dir) / "diagnostics.json") as f:
        diag = json.load(f)
    summary = {name: v["cnn_attack_success"] for name, v in diag["attacks"].items()}
    with open(Path(out_dir) / "attack_success_rates.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Undefended CNN attack success (targeted, direct-injection): {summary}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="results/baseline/best_model.pt")
    p.add_argument("--split", required=True)
    p.add_argument("--features", default="results/features/features.csv")
    p.add_argument("--gate-dir", default="results/consistency")
    p.add_argument("--out", default="results/attacks")
    p.add_argument("--max-test-samples", type=int, default=None)
    a = p.parse_args()
    run(load_config(a.config), a.checkpoint, a.split, a.features, a.gate_dir, a.out, a.max_test_samples)
