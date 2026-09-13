"""Session-relative feature normalization.

Empirical finding (see PAPER_NOTES.md "Cross-session drift"): raw CFO conflates the
transmitter's oscillator offset with the RECEIVER's own oscillator offset (CFO is measured
as tx_LO - rx_LO) and drifts measurably day-to-day even for a fixed (tx, rx) pair. A
consistency gate trained on some (day, receiver) sessions and evaluated on held-out sessions
of the same device sees the receiver/day component as part of the "signature" and false-
rejects legitimate signals from a session it wasn't trained on — this is exactly the
"the model may learn environment or receiver behavior instead of transmitter behavior"
limitation flagged in the project guide, confirmed empirically (test-set false rejection
rate ~84% before this fix, on real WiSig data).

Fix: subtract each session's own median feature value (computed across all devices captured
in that session) from every signal in that session, before fitting/scoring the gate. This
cancels the shared per-session receiver-LO/channel component and leaves (approximately) only
the tx-relative-to-its-peers signature, which is what should actually be device-specific.

Deployment caveat (be upfront about this in the writeup): this requires multiple devices'
signals from the SAME session to compute a baseline. That's a fair assumption for THIS
offline research evaluation (WiSig sessions contain many devices), but a real single-signal
authentication deployment would need a different calibration strategy (e.g. a receiver
self-calibration baseline built from its own history, or a per-receiver reference
transmission) — flag this as a limitation, not a solved deployment problem.
"""
from __future__ import annotations

import pandas as pd


def add_session_relative_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return `df` with one new `<col>_rel` column per `feature_cols` entry: the value
    minus that session's median for that column (session = df['session_id']).
    """
    if "session_id" not in df.columns:
        raise KeyError("add_session_relative_features requires a 'session_id' column")

    out = df.copy()
    session_medians = df.groupby("session_id")[feature_cols].transform("median")
    for col in feature_cols:
        out[f"{col}_rel"] = df[col] - session_medians[col]
    return out


def relative_columns(feature_cols: list[str]) -> list[str]:
    return [f"{c}_rel" for c in feature_cols]
