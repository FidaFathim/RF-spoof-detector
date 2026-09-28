"""Ground-truth tests for src/evaluation/metrics.py — the metric definitions every
results table in results/ is built from. Getting the direction of any one of these
wrong (e.g. detection rate vs. false-rejection rate) would silently invalidate every
number downstream, so each one is checked against a hand-computed expectation."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    cnn_attack_success_rate,
    combined_detection_rate,
    false_rejection_rate,
    precision_recall,
    roc_auc,
)


def test_cnn_attack_success_rate_counts_fraction_matching_target():
    preds = np.array([0, 0, 1, 2, 0])
    assert cnn_attack_success_rate(preds, target_label=0) == pytest.approx(3 / 5)


def test_combined_detection_rate_is_fraction_of_attacks_rejected():
    # final_accept over ATTACK samples: True = system wrongly accepted the forgery.
    final_accept = np.array([True, False, False, False])
    assert combined_detection_rate(final_accept) == pytest.approx(3 / 4)


def test_false_rejection_rate_is_fraction_of_legit_wrongly_rejected():
    final_accept = np.array([True, True, True, False])
    assert false_rejection_rate(final_accept) == pytest.approx(1 / 4)


def test_precision_recall_treats_reject_as_positive_class():
    # 3 attacks, 2 correctly rejected (accept=False) -> recall = 2/3
    final_accept_attack = np.array([False, False, True])
    # 4 legit, 1 wrongly rejected -> that's a false positive for "reject"
    final_accept_legit = np.array([True, True, True, False])
    out = precision_recall(final_accept_attack, final_accept_legit)
    assert out["recall"] == pytest.approx(2 / 3)
    tp = 2
    fp = 1
    assert out["precision"] == pytest.approx(tp / (tp + fp))


def test_precision_recall_handles_no_positives_without_dividing_by_zero():
    out = precision_recall(np.array([True, True]), np.array([True, True]))
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0


def test_roc_auc_is_perfect_when_scores_fully_separate_classes():
    scores_legit = np.array([5.0, 6.0, 7.0, 8.0])  # high = "accept-worthy" = correctly normal
    scores_attack = np.array([1.0, 2.0, 3.0, 4.0])  # low = flagged as attack-like
    auc, fpr, tpr = roc_auc(scores_legit, scores_attack)
    assert auc == pytest.approx(1.0)
    assert fpr[0] == 0.0 and tpr[-1] == 1.0


def test_roc_auc_is_chance_when_scores_carry_no_signal(rng=np.random.default_rng(3)):
    scores_legit = rng.normal(size=2000)
    scores_attack = rng.normal(size=2000)  # same distribution -> no separability
    auc, _, _ = roc_auc(scores_legit, scores_attack)
    assert auc == pytest.approx(0.5, abs=0.05)
