"""Full evaluation: clean false-rejection + attack detection for every ablation baseline.

Produces the exact results table from the project guide (one row per attack x ablation):
    experiment | ablation | cnn_attack_success | combined_detection_rate | false_rejection_rate

Inputs (all produced by earlier canonical scripts):
    - trained CNN checkpoint            (src.baseline.train)
    - features CSV with `split` column  (src.features.extract_all)
    - trained gate + threshold          (src.consistency.train_consistency)

Threat model: targeted, direct-injection, full-access. The attacker perturbs the exact
tensor the CNN consumes and claims to be `target_label`. UAP is trained on VALIDATION
victims and applied to TEST victims (never tuned on test).

Usage:
    python -m src.evaluation.run_full_eval --config configs/default.yaml \
        --split data/splits/split_seed0.json --checkpoint results/baseline/best_model.pt \
        --features results/features/features.csv --gate-dir results/consistency \
        --out results/final_seed0 [--max-test-samples 6000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from ..attacks.fgsm import fgsm_targeted
from ..attacks.pgd import pgd_targeted
from ..attacks.uap import apply_uap, train_uap
from ..baseline.cnn_model import RFFingerprintCNN
from ..consistency.calibrate import select_threshold_for_fpr
from ..consistency.gate import ConsistencyGate
from ..features.pipeline import FEATURE_COLUMNS, extract_features_row
from ..features.session_normalize import add_session_relative_features, relative_columns
from ..preprocessing.normalize import iq_to_tensor, unit_energy_normalize
from ..preprocessing.session_split import apply_split, load_split
from ..preprocessing.wisig_loader import load_wisig, truncate_or_pad
from ..utils.config import load_config, snapshot_config
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed
from .combine import (
    FeaturesAsInputClassifier,
    and_rule_accept,
    calibrate_confidence_threshold,
    confidence_threshold_accept,
)
from .metrics import cnn_attack_success_rate, combined_detection_rate, false_rejection_rate
from .report import add_result_row, build_report, save_report

logger = get_logger(__name__)


def _to_tensor(iq_list, iq_length: int, device: str) -> torch.Tensor:
    xs = [iq_to_tensor(unit_energy_normalize(truncate_or_pad(iq, iq_length))) for iq in iq_list]
    return torch.from_numpy(np.stack(xs)).to(device)


@torch.no_grad()
def _cnn_predict(model, x: torch.Tensor, batch_size: int = 512):
    model.eval()
    preds, confs = [], []
    for i in range(0, x.size(0), batch_size):
        probs = torch.softmax(model(x[i : i + batch_size]), dim=1)
        c, p = probs.max(dim=1)
        preds.append(p.cpu().numpy())
        confs.append(c.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def _raw_features_from_tensor(x: torch.Tensor, sample_rate_hz: float, repeat_len: int, num_repeats: int) -> np.ndarray:
    x = x.detach().cpu().numpy()
    rows = []
    for i in range(x.shape[0]):
        iq = x[i, 0] + 1j * x[i, 1]
        f = extract_features_row(iq, sample_rate_hz, repeat_len, num_repeats)
        rows.append([f[c] for c in FEATURE_COLUMNS])
    return np.asarray(rows)


def run(config: dict, split_path: str, checkpoint: str, features_csv: str, gate_dir: str,
        out_dir: str, max_test_samples: int | None = None) -> pd.DataFrame:
    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg, cnn_cfg, atk_cfg, gate_cfg, fusion_cfg = (
        config["data"], config["baseline_cnn"], config["attacks"], config["consistency_gate"], config["fusion"]
    )
    sr, rl, nr = data_cfg["sample_rate_hz"], config["features"]["cfo"]["repeat_len"], config["features"]["cfo"]["num_repeats"]
    target_fpr = fusion_cfg["target_val_false_positive_rate"]
    target_label_id = 0

    # ---------------- data ----------------
    df = load_wisig(data_cfg["wisig_root"], subset=data_cfg["wisig_subset"], equalized=data_cfg["equalized"])
    parts = apply_split(df, load_split(split_path))
    label_encoder = LabelEncoder().fit(df["tx_id"])
    classes = label_encoder.classes_
    target_name = classes[target_label_id]

    test_df = parts["test"]
    if max_test_samples and max_test_samples < len(test_df):
        test_df = test_df.sample(max_test_samples, random_state=config["seed"])
    test_df = test_df.reset_index(drop=True)
    val_df = parts["val"]
    val_sub = val_df.sample(min(6000, len(val_df)), random_state=config["seed"]).reset_index(drop=True)

    # ---------------- model ----------------
    model = RFFingerprintCNN(
        num_classes=len(classes), in_channels=cnn_cfg["in_channels"], conv_channels=cnn_cfg["conv_channels"],
        kernel_size=cnn_cfg["kernel_size"], dropout=cnn_cfg["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # ---------------- features (precomputed, legitimate only) ----------------
    feats = pd.read_csv(features_csv)
    use_rel = gate_cfg.get("use_session_relative", True)
    feats = add_session_relative_features(feats, FEATURE_COLUMNS)
    feat_cols = relative_columns(FEATURE_COLUMNS) if use_rel else FEATURE_COLUMNS
    feats_train = feats[feats["split"] == "train"]
    feats_val = feats[feats["split"] == "val"]
    feats_test = feats[feats["split"] == "test"]
    # Session medians from LEGITIMATE test signals -- the baseline an attacker's injected
    # signal is measured against in its own session.
    session_median = feats_test.groupby("session_id")[FEATURE_COLUMNS].median()

    def rel_features_for(x: torch.Tensor, sessions: np.ndarray) -> np.ndarray:
        raw = _raw_features_from_tensor(x, sr, rl, nr)
        if not use_rel:
            return raw
        med = session_median.loc[sessions].values
        return raw - med

    # ---------------- gate + ablation models ----------------
    gate: ConsistencyGate = joblib.load(Path(gate_dir) / "consistency_gate.joblib")
    with open(Path(gate_dir) / "threshold.json") as f:
        gate_thr = json.load(f)["threshold"]

    single_gates = {}
    for col in feat_cols:
        g = ConsistencyGate(method=gate_cfg["method"], mode=gate_cfg["mode"], method_kwargs=gate_cfg.get(gate_cfg["method"], {}))
        g.fit(feats_train[[col]].values, feats_train["tx_id"].values)
        thr = select_threshold_for_fpr(g.score(feats_val[[col]].values, feats_val["tx_id"].values), target_fpr)
        single_gates[col] = (g, thr)

    feat_clf = FeaturesAsInputClassifier(random_state=config["seed"]).fit(feats_train[feat_cols].values, feats_train["tx_id"].values)
    _, val_conf_fc = feat_clf.predict_and_confidence(feats_val[feat_cols].values)
    feat_clf_thr = calibrate_confidence_threshold(val_conf_fc, target_fpr)

    # CNN confidence threshold calibrated on VAL clean signals.
    x_val = _to_tensor(val_sub["iq"].tolist(), data_cfg["iq_length"], device)
    _, val_conf_cnn = _cnn_predict(model, x_val)
    cnn_conf_thr = calibrate_confidence_threshold(val_conf_cnn, target_fpr)

    # ---------------- clean test signals ----------------
    x_clean = _to_tensor(test_df["iq"].tolist(), data_cfg["iq_length"], device)
    y_true = label_encoder.transform(test_df["tx_id"])
    y_true_names = classes[y_true]
    sessions_clean = test_df["session_id"].values
    pred_clean, conf_clean = _cnn_predict(model, x_clean)
    pred_clean_names = classes[pred_clean]
    cnn_test_acc = float(np.mean(pred_clean == y_true))
    logger.info(f"CNN clean test accuracy (n={len(test_df)}): {cnn_test_acc:.4f}")

    rel_clean = rel_features_for(x_clean, sessions_clean)

    def decisions(x_adv, pred, conf, claimed_names, sessions):
        """Accept/reject arrays for every ablation. `claimed_names` is the identity being
        claimed (true label for clean signals, the impersonated target for attacks)."""
        claimed_ids = label_encoder.transform(claimed_names)
        rel = rel_features_for(x_adv, sessions) if x_adv is not x_clean else rel_clean
        out = {}
        out["cnn_confidence_only"] = confidence_threshold_accept(conf, cnn_conf_thr) & (pred == claimed_ids)
        for col, (g, thr) in single_gates.items():
            ci = feat_cols.index(col)
            out[f"single_feature_gate[{col}]"] = and_rule_accept(pred, claimed_ids, g.score(rel[:, [ci]], claimed_names), thr)
        fc_pred, fc_conf = feat_clf.predict_and_confidence(rel)
        out["features_as_classifier_input"] = (pred == claimed_ids) & (fc_pred == claimed_names) & (fc_conf >= feat_clf_thr)
        out["separate_consistency_gate"] = and_rule_accept(pred, claimed_ids, gate.score(rel, claimed_names), gate_thr)
        return out

    clean_acc = decisions(x_clean, pred_clean, conf_clean, y_true_names, sessions_clean)
    frr = {k: false_rejection_rate(v) for k, v in clean_acc.items()}

    # Validity check from the project guide: are false rejections concentrated in particular
    # sessions/receivers (gate learning environment) or spread evenly?
    rej = ~clean_acc["separate_consistency_gate"]
    frr_by_session = pd.Series(rej).groupby(sessions_clean).mean().sort_values(ascending=False)
    frr_by_rx = pd.Series(rej).groupby(test_df["rx_id"].values).mean().sort_values(ascending=False)

    rows: list[dict] = []
    for abl, v in frr.items():
        add_result_row(rows, "clean", abl, None, None, v)

    # ---------------- attacks ----------------
    victim_mask = y_true != target_label_id
    x_victim = x_clean[victim_mask]
    sessions_victim = sessions_clean[victim_mask]
    n_v = x_victim.size(0)
    target_t = torch.full((n_v,), target_label_id, dtype=torch.long, device=device)
    claimed_attack = np.full(n_v, target_name)

    # UAP trained on VAL victims only.
    val_victims = val_sub[val_sub["tx_id"] != target_name]
    val_victims = val_victims.sample(min(2048, len(val_victims)), random_state=config["seed"])
    x_val_victims = _to_tensor(val_victims["iq"].tolist(), data_cfg["iq_length"], device)
    delta = train_uap(model, x_val_victims, target_label_id, atk_cfg["uap"]["epsilon"], atk_cfg["uap"]["max_iters"])

    def chunked(fn, chunk: int = 1024) -> torch.Tensor:
        """Gradient-based attacks keep activations for the whole batch; run in chunks."""
        outs = []
        for i in range(0, n_v, chunk):
            outs.append(fn(x_victim[i : i + chunk], target_t[i : i + chunk]).detach())
        return torch.cat(outs, dim=0)

    attacks = {
        "fgsm": lambda: chunked(lambda x, t: fgsm_targeted(model, x, t, atk_cfg["fgsm"]["epsilon"])),
        "pgd": lambda: chunked(lambda x, t: pgd_targeted(
            model, x, t, atk_cfg["pgd"]["epsilon"], atk_cfg["pgd"]["alpha"], atk_cfg["pgd"]["steps"])),
        "uap": lambda: apply_uap(x_victim, delta),
    }
    # Epsilon sweep: the configured epsilons can be too weak to fool the CNN at all, which makes
    # every defense look ~100% effective. Reporting the whole curve (not a single hand-picked
    # point) shows where the attack starts to work and what the gate catches from there.
    steps = atk_cfg["pgd"]["steps"]
    for eps in atk_cfg.get("epsilon_sweep", []):
        delta_e = train_uap(model, x_val_victims, target_label_id, eps, atk_cfg["uap"]["max_iters"])
        attacks[f"fgsm@{eps}"] = lambda e=eps: chunked(lambda x, t: fgsm_targeted(model, x, t, e))
        attacks[f"pgd@{eps}"] = lambda e=eps: chunked(lambda x, t: pgd_targeted(model, x, t, e, e / 5, steps))
        attacks[f"uap@{eps}"] = lambda d=delta_e: apply_uap(x_victim, d)

    attack_summary = {}
    for name, make in attacks.items():
        x_adv = make().detach()
        pred_adv, conf_adv = _cnn_predict(model, x_adv)
        success = cnn_attack_success_rate(pred_adv, target_label_id)
        acc = decisions(x_adv, pred_adv, conf_adv, claimed_attack, sessions_victim)
        det = {k: combined_detection_rate(v) for k, v in acc.items()}
        # Detection among attacks that actually fooled the CNN: the only number that isolates
        # what the extra check adds, since a failed attack is rejected by the CNN alone.
        fooled = pred_adv == target_label_id
        n_fooled = int(fooled.sum())
        det_fooled = {k: (combined_detection_rate(v[fooled]) if n_fooled else None) for k, v in acc.items()}
        attack_summary[name] = {"cnn_attack_success": success, "n_fooled": n_fooled,
                                "combined_detection_rate": det, "detection_given_fooled": det_fooled}
        for abl, d in det.items():
            add_result_row(rows, name, abl, success, d, frr[abl], det_fooled[abl], n_fooled)
        gate_f = det_fooled["separate_consistency_gate"]
        conf_f = det_fooled["cnn_confidence_only"]
        logger.info(f"{name}: CNN attack success={success:.3f} (n_fooled={n_fooled})  "
                    f"detection(gate)={det['separate_consistency_gate']:.3f}  "
                    f"detection(cnn-conf-only)={det['cnn_confidence_only']:.3f}  "
                    f"| given fooled: gate={gate_f if gate_f is None else round(gate_f, 3)} "
                    f"cnn-conf-only={conf_f if conf_f is None else round(conf_f, 3)}")

    # ---------------- report ----------------
    report = build_report(rows)
    out = Path(out_dir)
    save_report(report, out)
    with open(out / "diagnostics.json", "w") as f:
        json.dump(
            {
                "cnn_clean_test_accuracy": cnn_test_acc,
                "n_test": int(len(test_df)),
                "n_victims": int(n_v),
                "target_label": target_name,
                "gate_threshold": gate_thr,
                "cnn_confidence_threshold": cnn_conf_thr,
                "false_rejection_rate": frr,
                "gate_frr_by_session": frr_by_session.to_dict(),
                "gate_frr_by_receiver": frr_by_rx.to_dict(),
                "attacks": attack_summary,
            },
            f, indent=2, default=float,
        )
    snapshot_config(config, out)
    logger.info(f"results written to {out}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", required=True)
    p.add_argument("--checkpoint", default="results/baseline/best_model.pt")
    p.add_argument("--features", default="results/features/features.csv")
    p.add_argument("--gate-dir", default="results/consistency")
    p.add_argument("--out", required=True)
    p.add_argument("--max-test-samples", type=int, default=None)
    a = p.parse_args()
    run(load_config(a.config), a.split, a.checkpoint, a.features, a.gate_dir, a.out, a.max_test_samples)
