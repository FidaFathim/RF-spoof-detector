"""Phase noise proxy: residual phase jitter after removing the deterministic CFO ramp.

Oscillator phase noise is commonly modeled as a Wiener (random-walk) process superimposed
on the deterministic carrier. After compensating for the estimated CFO (a linear phase
ramp) across the repeated preamble segments used in cfo.py, any remaining phase variation
between repeats that isn't explained by a constant phase offset is attributed to phase
noise. We report the standard deviation of that residual as a single scalar summary —
simple and testable, matching the project guide's "start simple" instruction.
"""
from __future__ import annotations

import numpy as np

from .cfo import compensate_cfo


def estimate_phase_noise(
    iq: np.ndarray,
    cfo_hz: float,
    sample_rate_hz: float,
    repeat_len: int = 16,
    num_repeats: int = 10,
) -> float:
    """Return the phase-noise std (radians) estimated from the CFO-compensated preamble."""
    preamble_len = repeat_len * num_repeats
    compensated = compensate_cfo(iq[:preamble_len], cfo_hz, sample_rate_hz)
    segments = compensated.reshape(num_repeats, repeat_len)

    # Phase of each repeat relative to the first repeat (ideally ~0 after CFO compensation).
    ref = segments[0]
    phases = []
    for k in range(1, num_repeats):
        rel = np.angle(np.sum(np.conj(ref) * segments[k]))
        phases.append(rel)
    phases = np.array(phases)

    # Remove the mean (constant) phase offset; std of what's left is the phase-noise proxy.
    residual = phases - phases.mean()
    return float(np.std(residual))
