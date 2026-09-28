"""Ground-truth tests for src/consistency/calibrate.py — threshold selection is the one
number in the whole pipeline that must never be tuned on test/attack data, so its
arithmetic needs to be right. Pure NumPy, no pandas/torch."""
from __future__ import annotations

import numpy as np
import pytest

from src.consistency.calibrate import detection_rate, false_positive_rate, select_threshold_for_fpr


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_threshold_rejects_approximately_target_fraction(rng):
    scores = rng.normal(size=20_000)
    threshold = select_threshold_for_fpr(scores, target_fpr=0.05)
    achieved_fpr = false_positive_rate(scores, threshold)
    assert achieved_fpr == pytest.approx(0.05, abs=0.01)


@pytest.mark.parametrize("bad_fpr", [0.0, 1.0, -0.1, 1.5])
def test_invalid_target_fpr_raises(bad_fpr):
    with pytest.raises(ValueError):
        select_threshold_for_fpr(np.array([1.0, 2.0, 3.0]), bad_fpr)


def test_false_positive_rate_definition():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # threshold=3 -> scores < 3 are rejected (2 out of 5)
    assert false_positive_rate(scores, threshold=3.0) == pytest.approx(0.4)


def test_detection_rate_definition():
    attack_scores = np.array([-5.0, -1.0, 0.5, 10.0])
    # threshold=0 -> scores < 0 are correctly rejected (2 out of 4)
    assert detection_rate(attack_scores, threshold=0.0) == pytest.approx(0.5)


def test_higher_target_fpr_gives_higher_threshold(rng):
    scores = rng.normal(size=5000)
    thr_strict = select_threshold_for_fpr(scores, 0.01)
    thr_loose = select_threshold_for_fpr(scores, 0.20)
    assert thr_loose > thr_strict


def test_perfectly_separated_attack_scores_are_fully_detected():
    legit_scores = np.random.default_rng(1).normal(loc=10.0, scale=0.1, size=1000)
    attack_scores = np.random.default_rng(2).normal(loc=-10.0, scale=0.1, size=1000)
    threshold = select_threshold_for_fpr(legit_scores, target_fpr=0.05)
    assert detection_rate(attack_scores, threshold) == pytest.approx(1.0)
    assert false_positive_rate(legit_scores, threshold) == pytest.approx(0.05, abs=0.01)
