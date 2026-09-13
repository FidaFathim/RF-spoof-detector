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

RESULT_COLUMNS = ["experiment", "ablation", "cnn_attack_success", "combined_detection_rate", "false_rejection_rate"]


def add_result_row(
    rows: list[dict],
    experiment: str,
    ablation: str,
    cnn_attack_success: float | None,
    combined_detection_rate: float | None,
    false_rejection_rate: float,
) -> None:
    """Mutates `rows` in place — call once per (attack, ablation) combination while running
    the week-4 evaluation loop, then pass `rows` to `build_report`."""
    rows.append(
        {
            "experiment": experiment,
            "ablation": ablation,
            "cnn_attack_success": cnn_attack_success,
            "combined_detection_rate": combined_detection_rate,
            "false_rejection_rate": false_rejection_rate,
        }
    )


def build_report(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    experiment_order = ["clean", "fgsm", "pgd", "uap", "gan", "hwe"]
    df["experiment"] = pd.Categorical(df["experiment"], categories=experiment_order, ordered=True)
    return df.sort_values(["ablation", "experiment"]).reset_index(drop=True)


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
    grouped = combined.groupby(["experiment", "ablation"], observed=True)[
        ["cnn_attack_success", "combined_detection_rate", "false_rejection_rate"]
    ]
    summary = grouped.agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()
