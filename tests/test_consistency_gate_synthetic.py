"""Ground-truth tests for src/consistency/gate.py using a SYNTHETIC physical-feature model
with a KNOWN joint structure — the same "first prove it on synthetic signals where the
correct answer is known" standard the project guide applies to the feature extractors
(tests/test_features_synthetic.py), applied one level up to the gate itself.

Model: each device has its own mean in 4D physical-feature space (cfo_hz-like,
iq_amp_imbalance-like, iq_phase_imbalance-like, phase_noise-like) and, critically, its own
JOINT relationship between the first two features (real hardware mixer impairments produce
correlated amplitude/phase imbalance — see src/features/iq_imbalance.py's docstring). This
is what a "physical-consistency gate" is supposed to be checking that a per-feature/marginal
check cannot: is the *combination* plausible, not just each value in isolation.

The key experiment (`test_joint_gate_catches_decorrelated_attack_better_than_marginal_gate`)
constructs an attacker that matches each feature's own mean/std under the claimed device
(so a single-feature gate sees nothing wrong) but samples the features independently,
destroying the cross-feature correlation real hardware has. This is a synthetic stand-in
for "a gradient-based attack optimizes against the CNN's decision boundary and has no
reason to also respect the joint physical relationship between hardware impairments" — the
core hypothesis in PAPER_NOTES.md. Passing this test does not prove the hypothesis on real
WiSig data/real attacks (that still requires the Colab GPU pipeline) — it proves the gate
implementation actually behaves the way the design claims it should, on a case built to
exercise exactly that mechanism.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.consistency.calibrate import detection_rate, false_positive_rate, select_threshold_for_fpr
from src.consistency.gate import ConsistencyGate, GaussianOneClass

DEVICE_MEANS = {
    "dev_a": np.array([0.0, 0.0, 0.0, 0.0]),
    "dev_b": np.array([5.0, 5.0, 0.0, 0.0]),
    "dev_c": np.array([-5.0, 3.0, 2.0, -1.0]),
}
JOINT_CORR = 0.85  # correlation between feature 0 and feature 1 for genuine hardware
TARGET_FPR = 0.05


def _device_covariance() -> np.ndarray:
    cov = np.eye(4)
    cov[0, 1] = cov[1, 0] = JOINT_CORR
    return cov


def _legit_samples(rng: np.random.Generator, n_per_device: int):
    cov = _device_covariance()
    X, tx = [], []
    for name, mean in DEVICE_MEANS.items():
        X.append(rng.multivariate_normal(mean, cov, size=n_per_device))
        tx += [name] * n_per_device
    return np.concatenate(X), np.array(tx)


def _decorrelated_attack_samples(rng: np.random.Generator, claimed_device: str, n: int):
    """Matches the claimed device's per-feature mean and marginal std (both are 1.0 here,
    since the covariance diagonal is 1 regardless of JOINT_CORR) but samples each feature
    INDEPENDENTLY — a single-feature check can't tell this apart from the real thing."""
    mean = DEVICE_MEANS[claimed_device]
    return rng.normal(loc=mean, scale=1.0, size=(n, 4))


def _crude_attack_samples(rng: np.random.Generator, claimed_device: str, n: int, offset: float = 8.0):
    """A much cruder attack that gets the claimed device's region roughly right but with an
    obviously wrong offset -- every gate, including single-feature ones, should catch this."""
    mean = DEVICE_MEANS[claimed_device] + offset
    return rng.normal(loc=mean, scale=1.0, size=(n, 4))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_gate_calibrates_to_target_false_positive_rate_on_held_out_legit_data(rng):
    train_X, train_tx = _legit_samples(rng, n_per_device=2000)
    val_X, val_tx = _legit_samples(rng, n_per_device=2000)

    gate = ConsistencyGate(method="gaussian", mode="per_device").fit(train_X, train_tx)
    val_scores = gate.score(val_X, val_tx)
    threshold = select_threshold_for_fpr(val_scores, TARGET_FPR)

    assert false_positive_rate(val_scores, threshold) == pytest.approx(TARGET_FPR, abs=0.01)


def test_joint_gate_catches_decorrelated_attack_better_than_marginal_gate(rng):
    train_X, train_tx = _legit_samples(rng, n_per_device=3000)
    val_X, val_tx = _legit_samples(rng, n_per_device=3000)

    joint_gate = ConsistencyGate(method="gaussian", mode="per_device").fit(train_X, train_tx)
    joint_threshold = select_threshold_for_fpr(joint_gate.score(val_X, val_tx), TARGET_FPR)

    # One independent single-feature gate per column, same calibration recipe.
    single_gates = []
    for col in range(4):
        g = ConsistencyGate(method="gaussian", mode="per_device").fit(train_X[:, [col]], train_tx)
        thr = select_threshold_for_fpr(g.score(val_X[:, [col]], val_tx), TARGET_FPR)
        single_gates.append((g, thr))

    attack_X, attack_claimed = [], []
    for name in DEVICE_MEANS:
        attack_X.append(_decorrelated_attack_samples(rng, name, n=1000))
        attack_claimed += [name] * 1000
    attack_X = np.concatenate(attack_X)
    attack_claimed = np.array(attack_claimed)

    joint_detection = detection_rate(joint_gate.score(attack_X, attack_claimed), joint_threshold)
    single_feature_detections = [
        detection_rate(g.score(attack_X[:, [col]], attack_claimed), thr)
        for col, (g, thr) in enumerate(single_gates)
    ]
    best_single = max(single_feature_detections)

    # The joint gate should substantially outperform EVERY single-feature gate at catching
    # an attack specifically engineered to look right on every marginal but violate the
    # joint (cross-feature) relationship -- this is the entire premise of the project.
    assert joint_detection > best_single + 0.2
    assert joint_detection > 0.25


def test_all_gates_catch_a_crude_offset_attack(rng):
    """Sanity check: an attack that's wrong in an obvious way should be caught by the joint
    gate AND single-feature gates alike -- the decorrelated attack above is deliberately the
    hard case, not the only case."""
    train_X, train_tx = _legit_samples(rng, n_per_device=2000)
    val_X, val_tx = _legit_samples(rng, n_per_device=2000)
    gate = ConsistencyGate(method="gaussian", mode="per_device").fit(train_X, train_tx)
    threshold = select_threshold_for_fpr(gate.score(val_X, val_tx), TARGET_FPR)

    attack_X = _crude_attack_samples(rng, "dev_a", n=500)
    claimed = np.array(["dev_a"] * 500)
    assert detection_rate(gate.score(attack_X, claimed), threshold) > 0.9


def test_per_device_gate_rejects_claim_for_unseen_device(rng):
    train_X, train_tx = _legit_samples(rng, n_per_device=200)
    gate = ConsistencyGate(method="gaussian", mode="per_device").fit(train_X, train_tx)

    unseen_claim = np.array(["dev_z"])
    score = gate.score(np.zeros((1, 4)), unseen_claim)
    assert score[0] == -np.inf


def test_global_mode_fits_one_model_over_all_devices(rng):
    train_X, train_tx = _legit_samples(rng, n_per_device=500)
    gate = ConsistencyGate(method="gaussian", mode="global").fit(train_X, train_tx)

    # claimed_tx_ids is ignored in global mode -- any label should score identically.
    x = train_X[:5]
    s1 = gate.score(x, np.array(["dev_a"] * 5))
    s2 = gate.score(x, np.array(["dev_b"] * 5))
    np.testing.assert_array_equal(s1, s2)


@pytest.mark.parametrize("method", ["gaussian", "one_class_svm", "isolation_forest"])
def test_every_method_scores_far_outliers_lower_than_inliers(rng, method):
    kwargs = {"one_class_svm": {"nu": 0.05}, "isolation_forest": {"n_estimators": 50}}.get(method, {})
    train_X, train_tx = _legit_samples(rng, n_per_device=500)
    gate = ConsistencyGate(method=method, mode="global", method_kwargs=kwargs).fit(train_X, train_tx)

    inlier = DEVICE_MEANS["dev_a"].reshape(1, -1)
    far_outlier = (DEVICE_MEANS["dev_a"] + 100.0).reshape(1, -1)
    claimed = np.array(["dev_a"])

    inlier_score = gate.score(inlier, claimed)[0]
    outlier_score = gate.score(far_outlier, claimed)[0]
    assert inlier_score > outlier_score


def test_gaussian_one_class_score_samples_matches_negative_mahalanobis(rng):
    X = rng.normal(size=(500, 3))
    model = GaussianOneClass().fit(X)
    point = np.array([[3.0, 3.0, 3.0]])
    manual = -model._cov.mahalanobis(point)
    np.testing.assert_allclose(model.score_samples(point), manual)
