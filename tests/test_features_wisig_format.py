"""Regression test for the WiSig preamble format (IMPROVEMENTS.md #1).

WiSig receivers captured at 25 Msps (WiSig paper, Sec. III). The 802.11 L-STF repeats every
0.8 us, i.e. every 20 samples at 25 Msps (16 samples would be the 20 Msps value), and lasts
10 repeats = 200 samples; the remaining 56 of the 256 stored samples are not part of the repeat.

An earlier version of configs/default.yaml assumed 20 Msps / 16-sample repeats / 160-sample
preamble. Real data showed the repeating pattern at a 20-sample period (similarity ~0.99 at
shifts 20, 40, 60, 80 vs 0.08 at 16). These tests keep that from regressing.
"""
import numpy as np
import pytest

from src.features.cfo import estimate_cfo

FS = 25_000_000
REPEAT_LEN = 20
NUM_REPEATS = 10
SIGNAL_LEN = 256


def make_wisig_like_signal(rng: np.random.Generator, cfo_hz: float) -> np.ndarray:
    """256 samples: a 20-sample pattern repeated 10 times (200 samples) followed by 56 samples
    of unrelated content, with a known CFO applied to the whole signal."""
    base = rng.standard_normal(REPEAT_LEN) + 1j * rng.standard_normal(REPEAT_LEN)
    base = base / np.sqrt(np.mean(np.abs(base) ** 2))
    tail = rng.standard_normal(SIGNAL_LEN - REPEAT_LEN * NUM_REPEATS) \
        + 1j * rng.standard_normal(SIGNAL_LEN - REPEAT_LEN * NUM_REPEATS)
    sig = np.concatenate([np.tile(base, NUM_REPEATS), tail])
    n = np.arange(SIGNAL_LEN)
    return sig * np.exp(1j * 2 * np.pi * cfo_hz * n / FS)


@pytest.mark.parametrize("cfo_hz", [-25_000.0, -5_000.0, 12_000.0])
def test_cfo_recovered_with_wisig_parameters(cfo_hz):
    rng = np.random.default_rng(7)
    est = estimate_cfo(make_wisig_like_signal(rng, cfo_hz), FS, REPEAT_LEN, NUM_REPEATS)
    assert est == pytest.approx(cfo_hz, abs=300.0)


def test_old_20msps_16sample_parameters_do_not_recover_cfo():
    """With the old (wrong) parameters the estimate must be clearly off for most signals --
    this documents WHY the correction mattered, and fails loudly if someone reverts it."""
    cfo_hz = -25_000.0
    errors = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        est = estimate_cfo(make_wisig_like_signal(rng, cfo_hz), 20_000_000, 16, 10)
        errors.append(abs(est - cfo_hz))
    assert np.median(errors) > 5_000.0
