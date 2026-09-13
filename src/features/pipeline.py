"""Batch feature extraction: run cfo/iq_imbalance/phase_noise over a whole DataFrame of
WiSig records and return a feature table ready for the consistency gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cfo import estimate_cfo
from .iq_imbalance import estimate_iq_imbalance
from .phase_noise import estimate_phase_noise

FEATURE_COLUMNS = ["cfo_hz", "iq_amp_imbalance_db", "iq_phase_imbalance_rad", "phase_noise_rad"]


def extract_features_row(
    iq: np.ndarray,
    sample_rate_hz: float,
    repeat_len: int = 16,
    num_repeats: int = 10,
) -> dict[str, float]:
    cfo_hz = estimate_cfo(iq, sample_rate_hz, repeat_len, num_repeats)
    amp_db, phase_rad = estimate_iq_imbalance(iq)
    pn_rad = estimate_phase_noise(iq, cfo_hz, sample_rate_hz, repeat_len, num_repeats)
    return {
        "cfo_hz": cfo_hz,
        "iq_amp_imbalance_db": amp_db,
        "iq_phase_imbalance_rad": phase_rad,
        "phase_noise_rad": pn_rad,
    }


def extract_features(
    df: pd.DataFrame,
    sample_rate_hz: float,
    repeat_len: int = 16,
    num_repeats: int = 10,
    show_progress: bool = True,
) -> pd.DataFrame:
    """df must have an `iq` column of complex ndarrays. Returns df with FEATURE_COLUMNS added;
    rows where extraction fails (e.g. too-short capture) are dropped and reported."""
    records = []
    dropped = 0
    iterator = df["iq"]
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="extracting physical features")

    for iq in iterator:
        try:
            records.append(extract_features_row(iq, sample_rate_hz, repeat_len, num_repeats))
        except ValueError:
            records.append({c: np.nan for c in FEATURE_COLUMNS})
            dropped += 1

    feat_df = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    if dropped:
        print(f"[extract_features] dropped/NaN'd {dropped}/{len(df)} rows (signal too short for preamble config)")

    return out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
