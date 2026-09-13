"""Evaluate a trained baseline CNN: accuracy, confusion matrix, per-device accuracy."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .cnn_model import RFFingerprintCNN


@torch.no_grad()
def predict(model: RFFingerprintCNN, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, max_softmax_confidence) over the whole loader."""
    model.eval()
    all_true, all_pred, all_conf = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        all_true.append(y.numpy())
        all_pred.append(pred.cpu().numpy())
        all_conf.append(conf.cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_conf)


def per_device_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, float]:
    out = {}
    for idx, name in enumerate(class_names):
        mask = y_true == idx
        if mask.sum() == 0:
            continue
        out[name] = float((y_pred[mask] == idx).mean())
    return out


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
