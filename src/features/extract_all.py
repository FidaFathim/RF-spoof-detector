"""Extract CFO / I-Q imbalance / phase noise for every signal in every split and write one
CSV (with `split`, `tx_id`, `rx_id`, `day`, `session_id` columns) for the consistency gate.

Usage:
    python -m src.features.extract_all --config configs/default.yaml \
        --split data/splits/split_seed0.json --out results/features/features.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..preprocessing.session_split import apply_split, load_split
from ..preprocessing.wisig_loader import load_wisig
from ..utils.config import load_config
from ..utils.logging_utils import get_logger
from .pipeline import FEATURE_COLUMNS, extract_features

logger = get_logger(__name__)


def run(config: dict, split_path: str, out_path: str) -> pd.DataFrame:
    data_cfg = config["data"]
    cfo_cfg = config["features"]["cfo"]

    df = load_wisig(data_cfg["wisig_root"], subset=data_cfg["wisig_subset"], equalized=data_cfg["equalized"])
    parts = apply_split(df, load_split(split_path))

    frames = []
    for part_name, part_df in parts.items():
        logger.info(f"extracting {part_name}: {len(part_df)} signals")
        feats = extract_features(
            part_df,
            data_cfg["sample_rate_hz"],
            repeat_len=cfo_cfg["repeat_len"],
            num_repeats=cfo_cfg["num_repeats"],
        )
        feats["split"] = part_name
        frames.append(feats.drop(columns=["iq"]))

    out = pd.concat(frames, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"wrote {len(out)} rows x {len(FEATURE_COLUMNS)} features to {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", default="results/features/features.csv")
    args = parser.parse_args()
    run(load_config(args.config), args.split, args.out)
