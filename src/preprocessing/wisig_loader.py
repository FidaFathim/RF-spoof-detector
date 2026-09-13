"""Load WiSig captures into a flat DataFrame of (iq_samples, tx_id, rx_id, day, capture_idx).

IMPORTANT: run `inspect_wisig.py` on your actual download first. The key names below
(`tx_id`/`rx_id`/day-as-top-level-dict) reflect the commonly documented WiSig pickle layout
(nested dict: day -> tx -> rx -> array of captures), but dataset releases have shipped
slightly different nestings across subsets. If `inspect_wisig.py` shows a different shape,
fix `_load_single_pickle` below — the rest of the pipeline only depends on the DataFrame
schema returned by `load_wisig`, not on this function's internals.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class WiSigRecord:
    iq: np.ndarray  # complex64, shape (num_samples,)
    tx_id: str
    rx_id: str
    day: str
    capture_idx: int


def _load_single_pickle(path: Path) -> list[WiSigRecord]:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    records: list[WiSigRecord] = []
    day = path.stem  # fallback: use filename as the day/session identifier

    # Expected layout: obj[tx_id][rx_id] -> ndarray of shape (num_captures, num_samples), complex
    if not isinstance(obj, dict):
        raise TypeError(
            f"{path}: expected a top-level dict (tx_id -> rx_id -> array), got {type(obj)}. "
            "Re-check with inspect_wisig.py and update _load_single_pickle."
        )

    for tx_id, rx_dict in obj.items():
        if not isinstance(rx_dict, dict):
            raise TypeError(
                f"{path}: expected obj[{tx_id!r}] to be a dict of rx_id -> array, got {type(rx_dict)}."
            )
        for rx_id, captures in rx_dict.items():
            captures = np.asarray(captures)
            if captures.ndim == 1:
                captures = captures[None, :]  # single capture, add batch dim
            for i in range(captures.shape[0]):
                records.append(
                    WiSigRecord(
                        iq=captures[i].astype(np.complex64),
                        tx_id=str(tx_id),
                        rx_id=str(rx_id),
                        day=day,
                        capture_idx=i,
                    )
                )
    return records


def load_wisig(root: str | Path, pattern: str = "*.pkl") -> pd.DataFrame:
    """Return a DataFrame with columns: iq (complex64 ndarray), tx_id, rx_id, day, capture_idx, session_id.

    `session_id` = f"{day}__{rx_id}" — this is the unit that `session_split.py` splits on.
    """
    root = Path(root)
    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} under {root} — check data/README.md")

    all_records: list[WiSigRecord] = []
    for f in files:
        all_records.extend(_load_single_pickle(f))

    df = pd.DataFrame(
        {
            "iq": [r.iq for r in all_records],
            "tx_id": [r.tx_id for r in all_records],
            "rx_id": [r.rx_id for r in all_records],
            "day": [r.day for r in all_records],
            "capture_idx": [r.capture_idx for r in all_records],
        }
    )
    df["session_id"] = df["day"] + "__" + df["rx_id"]
    return df


def truncate_or_pad(iq: np.ndarray, length: int) -> np.ndarray:
    """Fix every example to the same length so it can be batched into the CNN."""
    if len(iq) >= length:
        return iq[:length]
    out = np.zeros(length, dtype=iq.dtype)
    out[: len(iq)] = iq
    return out
