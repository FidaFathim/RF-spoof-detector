"""Run the whole pipeline for one seed: split -> train CNN -> extract features -> train gate
-> full evaluation. Each stage is the same canonical script the notebooks call, so results
are identical whether you run this or the notebooks one by one.

Usage:
    python -m src.run_pipeline --config configs/default.yaml --seed 0
    python -m src.run_pipeline --config configs/default.yaml --seed 0 --skip-train   # reuse checkpoint
    python -m src.run_pipeline --config configs/default.yaml --all-seeds              # seeds_for_variation

Outputs land in results/<stage>_seed<N>/ ; the final table is results/final_seed<N>/results_table.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .baseline.train import train_baseline
from .consistency.train_consistency import run as train_gate
from .evaluation.report import run_variation_across_seeds
from .evaluation.run_full_eval import run as full_eval
from .features.extract_all import run as extract_features
from .preprocessing.session_split import make_session_split, save_split
from .preprocessing.wisig_loader import load_wisig
from .utils.config import load_config
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_seed(config: dict, seed: int, skip_train: bool = False, max_test_samples: int | None = None):
    config = dict(config)
    config["seed"] = seed
    data_cfg = config["data"]

    split_path = Path(data_cfg["splits_dir"]) / f"split_seed{seed}.json"
    if not split_path.exists():
        df = load_wisig(data_cfg["wisig_root"], subset=data_cfg["wisig_subset"], equalized=data_cfg["equalized"])
        save_split(make_session_split(df, seed=seed), split_path, seed)
        logger.info(f"wrote {split_path}")

    baseline_dir = f"results/baseline_seed{seed}"
    checkpoint = Path(baseline_dir) / "best_model.pt"
    if skip_train and checkpoint.exists():
        logger.info(f"--skip-train: reusing {checkpoint}")
    else:
        train_baseline(config, str(split_path), baseline_dir)

    features_csv = f"results/features_seed{seed}/features.csv"
    if not Path(features_csv).exists():
        extract_features(config, str(split_path), features_csv)
    else:
        logger.info(f"reusing {features_csv}")

    gate_dir = f"results/consistency_seed{seed}"
    train_gate(config, features_csv, gate_dir)

    return full_eval(config, str(split_path), str(checkpoint), features_csv, gate_dir,
                     f"results/final_seed{seed}", max_test_samples)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all-seeds", action="store_true", help="run every seed in config.seeds_for_variation")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--max-test-samples", type=int, default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    seeds = cfg["seeds_for_variation"] if a.all_seeds else [cfg["seed"] if a.seed is None else a.seed]

    tables = [run_seed(cfg, s, a.skip_train, a.max_test_samples) for s in seeds]
    if len(tables) > 1:
        summary = run_variation_across_seeds(tables)
        out = Path("results/final_all_seeds")
        out.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out / "results_mean_std.csv", index=False)
        with open(out / "results_mean_std.md", "w") as f:
            f.write(summary.to_markdown(index=False))
        logger.info(f"mean/std across seeds {seeds} written to {out}")
