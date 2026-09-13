"""Load a YAML experiment config and keep a resolved copy next to results.

The project guide requires every reported number to be traceable to a fixed experiment
configuration. `load_config` returns a plain dict; `snapshot_config` writes the resolved
config (plus package versions) alongside a results run so it can be audited later.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        )
        return sorted(out.stdout.splitlines())
    except Exception:
        return []


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def snapshot_config(config: dict[str, Any], out_dir: str | Path) -> Path:
    """Write config.json + environment.json into out_dir. Call this once per run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "pip_freeze": _pip_freeze(),
    }
    with open(out_dir / "environment.json", "w") as f:
        json.dump(env, f, indent=2)

    return out_dir
