"""Load WiSig captures into a flat DataFrame of (iq, tx_id, rx_id, day, equalized, capture_idx).

Confirmed against the dataset authors' own loading code (WiSig-dataset/wisig-examples,
`data_utilities.py`, fetched 2026-09-14) — this is not guesswork.

Two on-disk formats, both handled here:

1. **Compact subsets** (ManySig / ManyTx / ManyRx / SingleDay — start here, these are
   1-2.5 GB each and download as a single .pkl). Structure:

       {
         "tx_list": [tx_name, ...],             # length n_tx
         "rx_list": [rx_name, ...],             # length n_rx
         "capture_date_list": [date_str, ...],  # length n_day
         "equalized_list": [0] or [0, 1],       # 0 = raw, 1 = channel-equalized
         "data": nested list [tx_i][rx_i][day_i][eq_i] -> ndarray (n_sig, 256, 2),
       }

   The trailing (..., 2) axis is [I, Q] (real, imag) — NOT a complex dtype. Empty
   tx/rx/day/eq combinations are `np.zeros((0, 256, 2))`, not missing keys.

2. **Full/Raw WiSig per-node files** (`dataset_<date>_node<rx>.pkl`, used if you're working
   from the multi-day/multi-receiver raw release instead of a compact subset):

       {"node_list": [tx_name, ...], "data": [ndarray (n_sig, 256, 2) per tx_i]}

   One file per (day, receiver); the day/receiver come from the filename, not the pickle.

IMPORTANT: the WiSig "signals" here are already-extracted 256-sample preambles ("Id
Signals"), not full packets — `configs/default.yaml`'s `data.iq_length: 256` matches this
directly. Whether the 256 samples are exactly the 802.11 L-STF (160 samples) plus a margin,
and at what sample rate, isn't stated in the loading code itself — confirm against the
WiSig paper (Hanna, Karunaratne, Cabric, IEEE Access 2022) or `inspect_wisig.py`'s printed
shapes before trusting `configs/default.yaml`'s `sample_rate_hz` / `features.cfo.*` values.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

COMPACT_SUBSETS = ("ManySig", "ManyTx", "ManyRx", "SingleDay")

_NODE_FILE_RE = re.compile(r"dataset_(?P<date>[^_]+)_node(?P<rx>.+)\.pkl$")


def _channels_to_complex(arr: np.ndarray) -> np.ndarray:
    """(..., 256, 2) real [I, Q] -> (..., 256) complex64."""
    return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64)


def _load_compact_pickle(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    required = {"tx_list", "rx_list", "capture_date_list", "equalized_list", "data"}
    missing = required - set(dataset.keys())
    if missing:
        raise KeyError(
            f"{path}: expected compact-WiSig keys {sorted(required)}, missing {sorted(missing)}. "
            "Run inspect_wisig.py and compare against this file's actual keys."
        )

    rows = []
    for tx_i, tx_id in enumerate(dataset["tx_list"]):
        for rx_i, rx_id in enumerate(dataset["rx_list"]):
            for day_i, day in enumerate(dataset["capture_date_list"]):
                for eq_i, eq_flag in enumerate(dataset["equalized_list"]):
                    arr = np.asarray(dataset["data"][tx_i][rx_i][day_i][eq_i])
                    if arr.size == 0 or arr.shape[0] == 0:
                        continue
                    iq_all = _channels_to_complex(arr)
                    for k in range(iq_all.shape[0]):
                        rows.append(
                            {
                                "iq": iq_all[k],
                                "tx_id": str(tx_id),
                                "rx_id": str(rx_id),
                                "day": str(day),
                                "equalized": int(eq_flag),
                                "capture_idx": k,
                            }
                        )
    return pd.DataFrame(rows)


def _load_raw_node_pickle(path: Path) -> pd.DataFrame:
    match = _NODE_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"{path.name} doesn't match the expected 'dataset_<date>_node<rx>.pkl' pattern")
    day, rx_id = match.group("date"), match.group("rx")

    with open(path, "rb") as f:
        tdataset = pickle.load(f)
    if not {"node_list", "data"}.issubset(tdataset.keys()):
        raise KeyError(f"{path}: expected keys 'node_list'/'data', got {list(tdataset.keys())}")

    rows = []
    for tx_i, tx_id in enumerate(tdataset["node_list"]):
        arr = np.asarray(tdataset["data"][tx_i])
        if arr.size == 0 or arr.shape[0] == 0:
            continue
        iq_all = _channels_to_complex(arr)
        for k in range(iq_all.shape[0]):
            rows.append(
                {
                    "iq": iq_all[k],
                    "tx_id": str(tx_id),
                    "rx_id": str(rx_id),
                    "day": day,
                    "equalized": 0,  # raw per-node files are the non-equalized capture
                    "capture_idx": k,
                }
            )
    return pd.DataFrame(rows)


def load_wisig(
    root: str | Path,
    subset: str | None = "ManySig",
    equalized: int = 0,
) -> pd.DataFrame:
    """Load a WiSig compact subset (default) or, if `subset` is None, every
    `dataset_<date>_node<rx>.pkl` file under `root` (Full/Raw WiSig layout).

    Returns columns: iq (complex64), tx_id, rx_id, day, equalized, capture_idx, session_id.
    `session_id` = f"{day}__{rx_id}" — the unit `session_split.py` splits on.
    Filters to `equalized == equalized` (default 0 = raw, non-channel-equalized) since RF
    hardware impairments are what we want to keep; channel equalization may distort CFO/
    I-Q-imbalance/phase-noise estimates — verify this empirically in week 1/2
    (notebook 03's device-vs-session clustering check) rather than assuming it.
    """
    root = Path(root)

    if subset is not None:
        if subset not in COMPACT_SUBSETS:
            raise ValueError(f"Unknown compact subset {subset!r}, expected one of {COMPACT_SUBSETS}")
        path = root / f"{subset}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - download it from the WiSig page, see data/README.md")
        df = _load_compact_pickle(path)
    else:
        files = sorted(root.rglob("dataset_*_node*.pkl"))
        if not files:
            raise FileNotFoundError(f"No 'dataset_<date>_node<rx>.pkl' files under {root}")
        df = pd.concat([_load_raw_node_pickle(f) for f in files], ignore_index=True)

    df = df[df["equalized"] == equalized].reset_index(drop=True)
    df["session_id"] = df["day"] + "__" + df["rx_id"]
    return df


def truncate_or_pad(iq: np.ndarray, length: int) -> np.ndarray:
    """Fix every example to the same length so it can be batched into the CNN."""
    if len(iq) >= length:
        return iq[:length]
    out = np.zeros(length, dtype=iq.dtype)
    out[: len(iq)] = iq
    return out
