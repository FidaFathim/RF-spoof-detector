"""Signal-level normalization shared by the CNN input pipeline and the feature extractors."""
from __future__ import annotations

import numpy as np


def unit_energy_normalize(iq: np.ndarray) -> np.ndarray:
    """Scale so mean sample power is 1. Removes gain differences that are receiver/AGC
    artifacts, not transmitter fingerprint — keeps the CNN from learning received power
    as a shortcut feature."""
    power = np.mean(np.abs(iq) ** 2)
    if power == 0:
        return iq
    return iq / np.sqrt(power)


def iq_to_tensor(iq: np.ndarray) -> np.ndarray:
    """complex64 (N,) -> float32 (2, N) as [I; Q] channels for the CNN."""
    return np.stack([iq.real, iq.imag], axis=0).astype(np.float32)


def fit_standard_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) computed on TRAIN features only. Apply the same mean/std to
    val/test — never refit on val/test, that's target leakage for the consistency gate's
    threshold selection."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_standard_scaler(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std
