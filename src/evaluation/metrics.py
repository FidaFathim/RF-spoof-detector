"""Metrics used throughout results/ — kept in one place so every table uses the same
definitions. Matches the metric set the project guide asks for."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def cnn_attack_success_rate(cnn_pred_labels: np.ndarray, target_label: int) -> float:
    """Fraction of attack samples the undefended CNN misclassifies as the impersonated
    target device. This is the "known weakness" baseline number, not the novel result."""
    return float(np.mean(cnn_pred_labels == target_label))


def combined_detection_rate(final_accept: np.ndarray) -> float:
    """`final_accept` is a boolean array over ATTACK samples (True = system wrongly
    accepted the forged signal). Detection rate = fraction correctly rejected."""
    return float(np.mean(~final_accept))


def false_rejection_rate(final_accept: np.ndarray) -> float:
    """`final_accept` is a boolean array over LEGITIMATE samples. FRR = fraction wrongly
    rejected."""
    return float(np.mean(~final_accept))


def precision_recall(final_accept_attack: np.ndarray, final_accept_legit: np.ndarray) -> dict[str, float]:
    """Treat 'reject' as the positive class (the thing we want the system to do to attacks)."""
    tp = np.sum(~final_accept_attack)  # attacks correctly rejected
    fn = np.sum(final_accept_attack)  # attacks wrongly accepted
    fp = np.sum(~final_accept_legit)  # legit wrongly rejected
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": float(precision), "recall": float(recall)}


def roc_auc(scores_legit: np.ndarray, scores_attack: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """scores: higher = more "accept-worthy" (matches ConsistencyGate.score's convention).
    Returns (auc, fpr_array, tpr_array) where "positive" = attack-detected-as-attack."""
    y_true = np.concatenate([np.zeros_like(scores_legit), np.ones_like(scores_attack)])
    y_score = -np.concatenate([scores_legit, scores_attack])  # flip: higher = more "attack-like"
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc), fpr, tpr
