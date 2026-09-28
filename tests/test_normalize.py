"""Ground-truth tests for src/preprocessing/normalize.py — pure NumPy, no pandas/torch
needed, so these run in any environment (including one where compiled pandas/torch
extensions are blocked by a local Application Control policy)."""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.normalize import (
    apply_standard_scaler,
    fit_standard_scaler,
    iq_to_tensor,
    unit_energy_normalize,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_unit_energy_normalize_sets_mean_power_to_one(rng):
    iq = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)) * 7.3
    out = unit_energy_normalize(iq)
    assert np.mean(np.abs(out) ** 2) == pytest.approx(1.0, rel=1e-6)


def test_unit_energy_normalize_preserves_relative_phase(rng):
    iq = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    out = unit_energy_normalize(iq)
    np.testing.assert_allclose(np.angle(out), np.angle(iq), atol=1e-6)


def test_unit_energy_normalize_handles_all_zero_signal():
    iq = np.zeros(10, dtype=complex)
    out = unit_energy_normalize(iq)
    np.testing.assert_array_equal(out, iq)


def test_iq_to_tensor_shape_and_dtype(rng):
    iq = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)
    tensor = iq_to_tensor(iq)
    assert tensor.shape == (2, 256)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0], iq.real, atol=1e-6)
    np.testing.assert_allclose(tensor[1], iq.imag, atol=1e-6)


def test_standard_scaler_train_stats_are_zero_mean_unit_std(rng):
    features = rng.normal(loc=[10.0, -3.0, 0.5], scale=[2.0, 0.1, 5.0], size=(5000, 3))
    mean, std = fit_standard_scaler(features)
    scaled = apply_standard_scaler(features, mean, std)
    np.testing.assert_allclose(scaled.mean(axis=0), [0.0, 0.0, 0.0], atol=0.05)
    np.testing.assert_allclose(scaled.std(axis=0), [1.0, 1.0, 1.0], atol=0.05)


def test_standard_scaler_applies_train_stats_to_new_data_without_refitting():
    train = np.array([[0.0], [2.0], [4.0]])
    mean, std = fit_standard_scaler(train)  # mean=2, std=sqrt(8/3)
    val = np.array([[2.0], [10.0]])
    scaled = apply_standard_scaler(val, mean, std)
    assert scaled[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert scaled[1, 0] > 0


def test_standard_scaler_guards_against_zero_variance_column():
    features = np.column_stack([np.full(100, 5.0), np.arange(100, dtype=float)])
    mean, std = fit_standard_scaler(features)
    assert std[0] == 1.0  # would otherwise divide by zero
    scaled = apply_standard_scaler(features, mean, std)
    assert np.all(np.isfinite(scaled))
