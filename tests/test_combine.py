"""Ground-truth tests for src/evaluation/combine.py — the AND-rule fusion and the 3
ablation baselines. These are what turn "the gate has good ROC-AUC" into "the combined
system detects impersonation," so the boolean wiring is worth checking directly."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.combine import (
    FeaturesAsInputClassifier,
    and_rule_accept,
    calibrate_confidence_threshold,
    confidence_threshold_accept,
)


def test_and_rule_requires_both_cnn_match_and_gate_pass():
    cnn_pred = np.array(["A", "A", "B", "A"])
    claimed = np.array(["A", "A", "A", "A"])
    gate_scores = np.array([10.0, -10.0, 10.0, 10.0])
    threshold = 0.0
    accept = and_rule_accept(cnn_pred, claimed, gate_scores, threshold)
    # row0: cnn matches + gate passes -> accept
    # row1: cnn matches but gate fails -> reject
    # row2: cnn doesn't match (regardless of gate) -> reject
    # row3: cnn matches + gate passes -> accept
    np.testing.assert_array_equal(accept, [True, False, False, True])


def test_and_rule_is_never_more_permissive_than_cnn_alone():
    rng = np.random.default_rng(5)
    n = 500
    cnn_pred = rng.integers(0, 3, size=n)
    claimed = rng.integers(0, 3, size=n)
    gate_scores = rng.normal(size=n)
    accept = and_rule_accept(cnn_pred, claimed, gate_scores, gate_threshold=0.0)
    cnn_alone = cnn_pred == claimed
    assert np.all(accept <= cnn_alone)  # AND-rule can only reject more, never fewer


def test_calibrate_confidence_threshold_matches_calibrate_module_semantics():
    confidences = np.linspace(0.0, 1.0, 1000)
    threshold = calibrate_confidence_threshold(confidences, target_fpr=0.1)
    rejected_fraction = np.mean(confidences < threshold)
    assert rejected_fraction == pytest.approx(0.1, abs=0.01)


def test_confidence_threshold_accept_is_a_simple_comparison():
    confidences = np.array([0.1, 0.5, 0.9])
    accept = confidence_threshold_accept(confidences, threshold=0.5)
    np.testing.assert_array_equal(accept, [False, True, True])


def test_features_as_input_classifier_separates_well_clustered_devices():
    rng = np.random.default_rng(11)
    # 3 devices, each with a distinct, well-separated 4D feature-space cluster.
    centers = {"dev0": [0, 0, 0, 0], "dev1": [10, 0, 0, 0], "dev2": [0, 10, 0, 0]}
    X, y = [], []
    for name, center in centers.items():
        X.append(rng.normal(loc=center, scale=0.5, size=(200, 4)))
        y.extend([name] * 200)
    X = np.concatenate(X)
    y = np.array(y)

    clf = FeaturesAsInputClassifier(random_state=0).fit(X, y)
    pred, conf = clf.predict_and_confidence(X)

    accuracy = np.mean(pred == y)
    assert accuracy > 0.95
    assert np.all((conf >= 0) & (conf <= 1))
