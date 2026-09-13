"""Train the consistency gate on legitimate TRAIN-split physical features, then calibrate
its threshold on the legitimate VAL-split features. Thin script — call from a notebook or CLI.

Usage:
    python -m src.consistency.train_consistency --config configs/default.yaml \
        --features results/features/features.csv --split data/splits/split_seed0.json \
        --out results/consistency
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from ..features.pipeline import FEATURE_COLUMNS
from ..features.session_normalize import add_session_relative_features, relative_columns
from ..utils.config import load_config, snapshot_config
from ..utils.logging_utils import get_logger
from .calibrate import select_threshold_for_fpr
from .gate import ConsistencyGate

logger = get_logger(__name__)


def run(config: dict, features_csv: str, out_dir: str) -> None:
    gate_cfg = config["consistency_gate"]
    fusion_cfg = config["fusion"]
    base_cols = gate_cfg.get("feature_set", FEATURE_COLUMNS)

    df = pd.read_csv(features_csv)
    # `df` must already carry a `split` column ("train"/"val"/"test") from the feature
    # extraction step — see notebooks/03_feature_extraction.ipynb.
    if gate_cfg.get("use_session_relative", True):
        df = add_session_relative_features(df, base_cols)
        feature_cols = relative_columns(base_cols)
    else:
        feature_cols = base_cols

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    gate = ConsistencyGate(
        method=gate_cfg["method"],
        mode=gate_cfg["mode"],
        method_kwargs=gate_cfg.get(gate_cfg["method"], {}),
    )
    gate.fit(train_df[feature_cols].values, train_df["tx_id"].values)

    # Calibrate on legitimate val signals, scored under their OWN true tx_id (a stand-in
    # for "the CNN correctly claimed this device" — at full-pipeline eval time you'd use
    # the CNN's predicted label instead; see src/evaluation/combine.py).
    val_scores = gate.score(val_df[feature_cols].values, val_df["tx_id"].values)
    threshold = select_threshold_for_fpr(val_scores, fusion_cfg["target_val_false_positive_rate"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gate, out_dir / "consistency_gate.joblib")
    with open(out_dir / "threshold.json", "w") as f:
        json.dump({"threshold": threshold, "target_fpr": fusion_cfg["target_val_false_positive_rate"]}, f, indent=2)
    snapshot_config(config, out_dir)

    logger.info(f"Gate trained ({gate_cfg['method']}/{gate_cfg['mode']}); "
                f"threshold={threshold:.4f} for target val FPR={fusion_cfg['target_val_false_positive_rate']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", default="results/consistency")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, args.features, args.out)
