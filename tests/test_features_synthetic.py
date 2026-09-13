"""Ground-truth tests: inject a KNOWN CFO / I-Q imbalance / phase noise into a synthetic
signal, then check the estimators in src/features/ recover it within tolerance.

This is Step 3 of the project guide ("First test them on synthetic signals where the
correct impairment is known") and should pass before the estimators are ever trusted on
real WiSig data.
"""
import numpy as np
import pytest

from src.features.cfo import estimate_cfo
from src.features.iq_imbalance import estimate_iq_imbalance
from src.features.phase_noise import estimate_phase_noise

SAMPLE_RATE_HZ = 20_000_000
REPEAT_LEN = 16
NUM_REPEATS = 10


def make_clean_preamble(rng: np.random.Generator) -> np.ndarray:
    """A repeated-segment preamble like the 802.11 L-STF: `num_repeats` copies of a random
    length-`repeat_len` complex sequence, unit power."""
    base = rng.standard_normal(REPEAT_LEN) + 1j * rng.standard_normal(REPEAT_LEN)
    base = base / np.sqrt(np.mean(np.abs(base) ** 2))
    return np.tile(base, NUM_REPEATS)


def apply_cfo(iq: np.ndarray, cfo_hz: float) -> np.ndarray:
    n = np.arange(len(iq))
    return iq * np.exp(1j * 2 * np.pi * cfo_hz * n / SAMPLE_RATE_HZ)


def apply_iq_imbalance(iq: np.ndarray, gain_db: float, phase_rad: float) -> np.ndarray:
    g = 10 ** (gain_db / 20.0)
    i, q = iq.real, iq.imag
    q_out = g * (q * np.cos(phase_rad) + i * np.sin(phase_rad))
    return i + 1j * q_out


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_cfo_recovers_known_offset(rng):
    true_cfo_hz = 5_000.0  # 5 kHz, a realistic-scale LO mismatch at 20 MSps
    clean = make_clean_preamble(rng)
    signal = apply_cfo(clean, true_cfo_hz)

    est = estimate_cfo(signal, SAMPLE_RATE_HZ, REPEAT_LEN, NUM_REPEATS)
    assert est == pytest.approx(true_cfo_hz, rel=0.02)


def test_cfo_zero_when_no_offset(rng):
    clean = make_clean_preamble(rng)
    est = estimate_cfo(clean, SAMPLE_RATE_HZ, REPEAT_LEN, NUM_REPEATS)
    assert est == pytest.approx(0.0, abs=50.0)  # within 50 Hz of zero


@pytest.mark.parametrize("gain_db,phase_deg", [(1.5, 3.0), (-2.0, -5.0), (0.5, 0.0)])
def test_iq_imbalance_recovers_known_values(rng, gain_db, phase_deg):
    phase_rad = np.deg2rad(phase_deg)
    # Use a longer, i.i.d.-like segment for a stable moment estimate (the moment estimator
    # assumes near-independent, equal-power I/Q — a short 160-sample preamble is noisy for
    # this particular check, hence the larger N here vs. the CFO/phase-noise tests).
    n = 20_000
    i = rng.standard_normal(n)
    q = rng.standard_normal(n)
    clean = i + 1j * q

    imbalanced = apply_iq_imbalance(clean, gain_db, phase_rad)
    est_amp_db, est_phase_rad = estimate_iq_imbalance(imbalanced)

    assert est_amp_db == pytest.approx(gain_db, abs=0.3)
    assert est_phase_rad == pytest.approx(phase_rad, abs=0.05)


def test_phase_noise_higher_for_noisier_oscillator(rng):
    """We don't have an independent ground-truth phase-noise estimator to check an exact
    value against (unlike CFO/IQ-imbalance, which have closed-form injected values) — so
    the meaningful, checkable property is monotonicity: injecting more per-repeat phase
    jitter should increase the estimated phase-noise proxy."""
    clean = make_clean_preamble(rng)
    cfo_hz = 2_000.0
    base = apply_cfo(clean, cfo_hz)

    def with_jitter(std_rad: float) -> np.ndarray:
        segments = base.reshape(NUM_REPEATS, REPEAT_LEN).copy()
        jitter = rng.normal(0, std_rad, size=NUM_REPEATS)
        for k in range(NUM_REPEATS):
            segments[k] *= np.exp(1j * jitter[k])
        return segments.reshape(-1)

    low_jitter_signal = with_jitter(0.01)
    high_jitter_signal = with_jitter(0.2)

    pn_low = estimate_phase_noise(low_jitter_signal, cfo_hz, SAMPLE_RATE_HZ, REPEAT_LEN, NUM_REPEATS)
    pn_high = estimate_phase_noise(high_jitter_signal, cfo_hz, SAMPLE_RATE_HZ, REPEAT_LEN, NUM_REPEATS)

    assert pn_high > pn_low


def test_cfo_raises_on_too_short_signal(rng):
    with pytest.raises(ValueError):
        estimate_cfo(np.zeros(10, dtype=complex), SAMPLE_RATE_HZ, REPEAT_LEN, NUM_REPEATS)
