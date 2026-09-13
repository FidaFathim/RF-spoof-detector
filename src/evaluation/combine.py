"""Fusion logic: the proposed AND-rule, plus the 3 ablation baselines the project guide
requires before "the separate gate helped" is a supportable claim:

  1. cnn_confidence_only         — threshold the CNN's own softmax confidence, no physical
                                    features at all.
  2. single_feature_gate         — ConsistencyGate restricted to one feature column at a time.
  3. features_as_classifier_input — physical features fed into an ORDINARY supervised
                                    classifier (not a one-class gate) as extra input, to show
                                    that "using the features" isn't the same as "a separate
                                    consistency gate."
  4. separate_consistency_gate   — the actual proposal: AND-rule below.

Keep all 4 in every results table — a defense that only beats (1) but not (2)/(3) is a much
weaker claim than one that beats all three.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier


def and_rule_accept(
    cnn_predicted_label: np.ndarray,
    claimed_label: np.ndarray,
    gate_scores: np.ndarray,
    gate_threshold: float,
) -> np.ndarray:
    """The proposed fusion: accept iff the CNN's prediction matches the claimed identity
    AND the consistency gate (scored under that claimed identity) clears the threshold."""
    cnn_accepts = cnn_predicted_label == claimed_label
    gate_accepts = gate_scores >= gate_threshold
    return cnn_accepts & gate_accepts


def calibrate_confidence_threshold(confidences_legit: np.ndarray, target_fpr: float) -> float:
    """Shared by the CNN-confidence-only and features-as-input ablations: pick the
    confidence threshold that rejects exactly `target_fpr` of legitimate validation
    samples. Mirrors src/consistency/calibrate.py's threshold selection so all baselines
    are held to the same false-positive-rate operating point."""
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    return float(np.quantile(confidences_legit, target_fpr))


def confidence_threshold_accept(confidences: np.ndarray, threshold: float) -> np.ndarray:
    return confidences >= threshold


class FeaturesAsInputClassifier:
    """Ablation (3): an ORDINARY supervised classifier trained to predict device identity
    directly from the physical features (CFO, I/Q imbalance, phase noise) — i.e. exactly
    what the project guide means by "all three features as ordinary classifier inputs."
    Accept/reject uses this classifier's own confidence threshold, calibrated the same way
    as the CNN-confidence-only baseline. This is deliberately a small, simple model (an MLP
    over a 4-dim input) — the point of the ablation is the *architecture* (features as
    ordinary classifier input vs. a separate gate), not classifier capacity.
    """

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (32,), random_state: int = 0):
        self._clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=500
        )
        self._classes: np.ndarray | None = None

    def fit(self, features: np.ndarray, tx_ids: np.ndarray) -> "FeaturesAsInputClassifier":
        self._clf.fit(features, tx_ids)
        self._classes = self._clf.classes_
        return self

    def predict_and_confidence(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proba = self._clf.predict_proba(features)
        pred_idx = proba.argmax(axis=1)
        pred_label = self._classes[pred_idx]
        confidence = proba[np.arange(len(features)), pred_idx]
        return pred_label, confidence
