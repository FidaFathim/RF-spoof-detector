"""Select the consistency-gate accept/reject threshold on the VALIDATION split only.

Never tune this on the test split — the project guide is explicit about this, and doing
so would invalidate the reported detection-rate/false-positive-rate numbers.
"""
from __future__ import annotations

import numpy as np


def select_threshold_for_fpr(val_scores_legit: np.ndarray, target_fpr: float) -> float:
    """Given consistency scores for LEGITIMATE validation signals, return the score
    threshold that would reject exactly `target_fpr` fraction of them.

    Accept rule at eval time: score >= threshold.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    return float(np.quantile(val_scores_legit, target_fpr))


def false_positive_rate(scores_legit: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores_legit < threshold))


def detection_rate(scores_attack: np.ndarray, threshold: float) -> float:
    """Fraction of attack samples the gate correctly rejects."""
    return float(np.mean(scores_attack < threshold))
