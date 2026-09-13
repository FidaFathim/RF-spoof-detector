"""Session-aware train/val/test split.

Splitting individual packets at random leaks channel/receiver/session state between
splits (packets from the same (day, receiver) session share near-identical channel
conditions). This module splits whole sessions instead, and writes the resulting
assignment to a small JSON file that every downstream script (baseline training,
feature extraction, gate calibration, evaluation) should load rather than re-splitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_session_split(
    df: pd.DataFrame,
    seed: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict[str, list[str]]:
    """Split unique `session_id`s (not rows) into train/val/test.

    Returns a dict {"train": [...session_ids...], "val": [...], "test": [...]}.
    """
    sessions = sorted(df["session_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)

    n = len(sessions)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    split = {
        "train": sessions[:n_train],
        "val": sessions[n_train : n_train + n_val],
        "test": sessions[n_train + n_val :],
    }
    if not split["test"]:
        raise ValueError(
            f"Only {n} sessions total — not enough to hold out a test split. "
            "Check that wisig_loader produced multiple (day, rx) combinations."
        )
    return split


def save_split(split: dict[str, list[str]], out_path: str | Path, seed: int) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"seed": seed, **split}, f, indent=2)


def load_split(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r") as f:
        return json.load(f)


def apply_split(df: pd.DataFrame, split: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    return {
        part: df[df["session_id"].isin(sessions)].reset_index(drop=True)
        for part, sessions in split.items()
        if part in ("train", "val", "test")
    }
