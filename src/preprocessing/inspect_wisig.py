"""Run this FIRST, before trusting wisig_loader.py.

WiSig ships as pickle files whose exact nesting (dict keys for day/tx/rx, array shapes,
dtypes) has varied slightly across the dataset's released subsets ("ManySig", "ShortSig",
"ALLSIG"). Rather than guess, this script recursively walks whatever you downloaded and
prints its structure so you can fix wisig_loader.py's key names in 15 minutes instead of
debugging silently-wrong data later.

Usage:
    python src/preprocessing/inspect_wisig.py --root data/raw/wisig
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def describe(obj, prefix: str = "", max_depth: int = 4, depth: int = 0) -> None:
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{prefix}: ... (max depth reached)")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}{prefix}: dict with {len(keys)} keys, e.g. {keys[:5]}")
        if keys:
            describe(obj[keys[0]], prefix=f"[{keys[0]!r}]", max_depth=max_depth, depth=depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{prefix}: {type(obj).__name__} of length {len(obj)}")
        if len(obj) > 0:
            describe(obj[0], prefix="[0]", max_depth=max_depth, depth=depth + 1)
    elif isinstance(obj, np.ndarray):
        print(f"{indent}{prefix}: ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{indent}{prefix}: {type(obj).__name__} = {str(obj)[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing extracted WiSig files")
    parser.add_argument("--max-files", type=int, default=3, help="How many files to sample")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist — extract WiSig there first (see data/README.md)")

    all_files = sorted(root.rglob("*"))
    files = [f for f in all_files if f.is_file()]
    print(f"Found {len(files)} files under {root}")
    for f in files[:20]:
        print(" ", f.relative_to(root))
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")

    pkl_files = [f for f in files if f.suffix in (".pkl", ".pickle")]
    if not pkl_files:
        print("\nNo .pkl/.pickle files found — check the extraction, or update this script's "
              "file-extension filter if WiSig shipped a different format (.npz, .h5, etc.).")
        return

    for f in pkl_files[: args.max_files]:
        print(f"\n=== {f.relative_to(root)} ===")
        with open(f, "rb") as fh:
            obj = pickle.load(fh)
        describe(obj)


if __name__ == "__main__":
    main()
