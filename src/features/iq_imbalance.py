"""I/Q amplitude and phase imbalance via a moment-based estimator.

Model (widely-linear IQ-imbalance model): the ideal baseband signal s(t) is circularly
symmetric (E[s^2] ~ 0, i.e. I and Q are independent and equal-power) before the transmitter's
mixer/DAC impairments distort it. A gain mismatch g (amplitude imbalance) and phase mismatch
phi (phase imbalance) between the I and Q mixer paths produce:

    I_out = I_in
    Q_out = g * (Q_in * cos(phi) + I_in * sin(phi))

For a segment where I_in, Q_in are approximately independent, zero-mean, and equal power
(true for a random-data payload segment, and approximately true for many preambles), the
imbalance parameters can be recovered from second-order moments alone:

    amp_imbalance_db   = 20 * log10( std(Q) / std(I) )
    phase_imbalance_rad = arcsin( E[I*Q] / (std(I) * std(Q)) )

This is the standard "moment/correlation" IQ-imbalance estimator (the same family used for
image-rejection-ratio measurement); it is an approximation (it assumes the underlying
symbols are proper-complex), which is why we run it on a stable preamble segment by default
rather than arbitrary payload — flip `window` to "full" in configs/default.yaml if payload
segments turn out to give more stable per-device estimates once you inspect real data.
"""
from __future__ import annotations

import numpy as np


def estimate_iq_imbalance(iq: np.ndarray) -> tuple[float, float]:
    """Return (amplitude_imbalance_db, phase_imbalance_rad)."""
    i = iq.real.astype(np.float64)
    q = iq.imag.astype(np.float64)

    i = i - i.mean()
    q = q - q.mean()

    std_i = i.std()
    std_q = q.std()
    if std_i == 0 or std_q == 0:
        return 0.0, 0.0

    amp_imbalance_db = 20.0 * np.log10(std_q / std_i)

    corr = np.mean(i * q) / (std_i * std_q)
    corr = np.clip(corr, -1.0, 1.0)  # guard against tiny numerical overshoot before arcsin
    phase_imbalance_rad = float(np.arcsin(corr))

    return float(amp_imbalance_db), phase_imbalance_rad
