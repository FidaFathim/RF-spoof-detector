"""Assemble the exact results table shape from the project guide:

    Experiment | CNN attack success | Combined detection rate | False rejection of clean signals

with one row per attack (clean, fgsm, pgd, uap, gan, hwe) and one such table per ablation
baseline (cnn_confidence_only / single_feature_gate / features_as_classifier_input /
separate_consistency_gate) — a defense is only interesting if its row beats the other
baselines' rows at a comparable false-rejection rate.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULT_COLUMNS = ["experiment", "ablation", "cnn_attack_success", "combined_detection_rate",
                  "false_rejection_rate", "detection_given_fooled", "n_fooled"]
_METRIC_COLUMNS = ["cnn_attack_success", "combined_detection_rate", "false_rejection_rate",
                   "detection_given_fooled", "n_fooled"]
_EXPERIMENT_ORDER = ["clean", "fgsm", "pgd", "uap", "gan", "hwe"]


def add_result_row(
    rows: list[dict],
    experiment: str,
    ablation: str,
    cnn_attack_success: float | None,
    combined_detection_rate: float | None,
    false_rejection_rate: float,
    detection_given_fooled: float | None = None,
    n_fooled: int | None = None,
) -> None:
    """Mutates `rows` in place — call once per (attack, ablation) combination while running
    the week-4 evaluation loop, then pass `rows` to `build_report`. `experiment` is an attack
    name, optionally with an epsilon suffix such as "pgd@0.1"."""
    rows.append(
        {
            "experiment": experiment,
            "ablation": ablation,
            "cnn_attack_success": cnn_attack_success,
            "combined_detection_rate": combined_detection_rate,
            "false_rejection_rate": false_rejection_rate,
            "detection_given_fooled": detection_given_fooled,
            "n_fooled": n_fooled,
        }
    )


def _experiment_sort_key(name: str) -> tuple[int, float]:
    base, _, eps = name.partition("@")
    order = _EXPERIMENT_ORDER.index(base) if base in _EXPERIMENT_ORDER else len(_EXPERIMENT_ORDER)
    return order, float(eps) if eps else 0.0


def build_report(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    keys = df["experiment"].map(_experiment_sort_key)
    df["_order"] = [k[0] for k in keys]
    df["_eps"] = [k[1] for k in keys]
    df = df.sort_values(["ablation", "_order", "_eps"]).drop(columns=["_order", "_eps"])
    return df.reset_index(drop=True)


def save_report(df: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results_table.csv", index=False)
    with open(out_dir / "results_table.md", "w") as f:
        f.write(df.to_markdown(index=False))


def run_variation_across_seeds(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-seed result tables (one per random seed, ≥3 per the project guide) into
    mean ± std per (experiment, ablation) cell."""
    combined = pd.concat(dfs, keys=range(len(dfs)), names=["seed"]).reset_index(level=0)
    combined[_METRIC_COLUMNS] = combined[_METRIC_COLUMNS].astype(float)
    grouped = combined.groupby(["experiment", "ablation"], observed=True, sort=False)[_METRIC_COLUMNS]
    summary = grouped.agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()
