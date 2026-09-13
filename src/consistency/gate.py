"""The physical-consistency gate — the actual novelty.

Trained ONLY on legitimate (never attacked) signals' physical features (CFO, I/Q amplitude
imbalance, I/Q phase imbalance, phase noise). At test time it scores whether a feature
vector is "physically plausible," conditioned on the device the CNN claims produced it.

Two modes (keep both — they're an ablation, not a decision to make once and discard):
  - "per_device": one one-class model per tx_id, trained on that device's own legitimate
    features. Tests "could THIS specific claimed device have produced this exact physical
    signature?" — the framing in the project draft ("physically plausible for real hardware
    to have produced" the claimed device's transmission).
  - "global": one one-class model over the pooled legitimate features of ALL devices. Tests
    the weaker claim "could ANY real transceiver have produced this?" — useful because pure
    gradient-based attacks (FGSM/PGD/UAP) perturb raw IQ samples with no notion of hardware
    physics at all, so they may fall off the general real-hardware manifold even without
    being compared to a specific claimed device.

Three interchangeable one-class methods, all fit on standardized features:
  - "gaussian": Mahalanobis distance to the fitted mean/covariance (simplest, matches the
    project guide's "start with a Gaussian distance model" instruction).
  - "one_class_svm": sklearn OneClassSVM.
  - "isolation_forest": sklearn IsolationForest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from ..preprocessing.normalize import apply_standard_scaler, fit_standard_scaler


class GaussianOneClass:
    """Mahalanobis-distance one-class model. score_samples: higher = more normal."""

    def __init__(self):
        self._cov = EmpiricalCovariance()

    def fit(self, X: np.ndarray) -> "GaussianOneClass":
        self._cov.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # Mahalanobis distance is a "how abnormal" score; negate so higher = more normal,
        # consistent with sklearn's OneClassSVM/IsolationForest score_samples convention.
        return -self._cov.mahalanobis(X)


_METHODS = {
    "gaussian": lambda cfg: GaussianOneClass(),
    "one_class_svm": lambda cfg: OneClassSVM(
        nu=cfg.get("nu", 0.05), kernel=cfg.get("kernel", "rbf"), gamma=cfg.get("gamma", "scale")
    ),
    "isolation_forest": lambda cfg: IsolationForest(
        n_estimators=cfg.get("n_estimators", 200),
        contamination=cfg.get("contamination", 0.05),
        random_state=0,
    ),
}


@dataclass
class ConsistencyGate:
    method: str = "gaussian"
    mode: str = "per_device"  # "per_device" | "global"
    method_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self._models: dict[str, object] = {}  # key: tx_id, or "__global__" for global mode
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None

    def fit(self, features: np.ndarray, tx_ids: np.ndarray) -> "ConsistencyGate":
        """features: (N, D) legitimate-only feature matrix. tx_ids: (N,) device labels.
        Standardization is fit on this (training) data only — never refit at eval time."""
        self._scaler_mean, self._scaler_std = fit_standard_scaler(features)
        scaled = apply_standard_scaler(features, self._scaler_mean, self._scaler_std)

        if self.mode == "global":
            model = _METHODS[self.method](self.method_kwargs)
            model.fit(scaled)
            self._models["__global__"] = model
        elif self.mode == "per_device":
            for tx_id in np.unique(tx_ids):
                mask = tx_ids == tx_id
                model = _METHODS[self.method](self.method_kwargs)
                model.fit(scaled[mask])
                self._models[str(tx_id)] = model
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")
        return self

    def score(self, features: np.ndarray, claimed_tx_ids: np.ndarray) -> np.ndarray:
        """Return a per-sample consistency score; higher = more physically plausible.
        In per_device mode, claimed_tx_ids selects which device's model scores each row
        (this should be the CNN's *predicted* label at eval time, not the true label)."""
        scaled = apply_standard_scaler(features, self._scaler_mean, self._scaler_std)
        scores = np.empty(len(features), dtype=np.float64)

        if self.mode == "global":
            scores[:] = self._models["__global__"].score_samples(scaled)
            return scores

        for tx_id in np.unique(claimed_tx_ids):
            mask = claimed_tx_ids == tx_id
            model = self._models.get(str(tx_id))
            if model is None:
                # Claimed device never seen in training — can't vouch for it; treat as
                # maximally inconsistent rather than silently skipping.
                scores[mask] = -np.inf
                continue
            scores[mask] = model.score_samples(scaled[mask])
        return scores
