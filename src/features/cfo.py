"""Carrier Frequency Offset (CFO) estimation via Moose's algorithm.

Standard, textbook estimator (Moose, 1994) using a preamble made of repeated identical
segments — e.g. the 802.11a/g/n L-STF, which is 10 repetitions of a 16-sample sequence
(160 samples total at 20 MHz). Given two copies of the same transmitted segment D samples
apart, any phase rotation between them must come from the CFO (plus noise):

    cfo_hz = angle( sum_n conj(x[n]) * x[n+D] ) / (2*pi*D*Ts)

This is pure signal processing — no learned parameters, no neural network. Confirm
`preamble_len`/`repeat_len`/`num_repeats` against your actual WiSig capture format;
defaults assume the standard 802.11 L-STF layout.
"""
from __future__ import annotations

import numpy as np


def estimate_cfo(
    iq: np.ndarray,
    sample_rate_hz: float,
    repeat_len: int = 16,
    num_repeats: int = 10,
) -> float:
    """Estimate CFO in Hz from a signal whose first `repeat_len * num_repeats` samples
    are `num_repeats` back-to-back repetitions of the same underlying sequence.
    """
    preamble_len = repeat_len * num_repeats
    if len(iq) < preamble_len:
        raise ValueError(
            f"Signal too short ({len(iq)} samples) for {num_repeats} repeats of length "
            f"{repeat_len} (needs {preamble_len}) - check preamble_len against your actual capture format."
        )

    preamble = iq[:preamble_len]
    segments = preamble.reshape(num_repeats, repeat_len)

    # Average phase rotation between consecutive repeats (D = repeat_len samples apart).
    correlations = np.sum(np.conj(segments[:-1]) * segments[1:], axis=1)
    mean_corr = np.sum(correlations)  # coherent combining across all repeat pairs

    phase = np.angle(mean_corr)
    cfo_hz = phase / (2 * np.pi * repeat_len / sample_rate_hz)
    return float(cfo_hz)


def compensate_cfo(iq: np.ndarray, cfo_hz: float, sample_rate_hz: float) -> np.ndarray:
    """Remove the estimated CFO's linear phase ramp — used by phase_noise.py to isolate
    the residual (non-linear, non-deterministic) phase jitter."""
    n = np.arange(len(iq))
    correction = np.exp(-1j * 2 * np.pi * cfo_hz * n / sample_rate_hz)
    return iq * correction
