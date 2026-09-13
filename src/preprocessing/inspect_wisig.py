"""Run this FIRST, before trusting wisig_loader.py's parsing.

wisig_loader.py is written against the confirmed structure from the dataset authors' own
`data_utilities.py` (WiSig-dataset/wisig-examples on GitHub): compact subsets (ManySig/
ManyTx/ManyRx/SingleDay) are a single pickle with keys tx_list/rx_list/capture_date_list/
equalized_list/data, where data[tx_i][rx_i][day_i][eq_i] is an (n_sig, 256, 2) real array
([I, Q] in the last axis). This script checks your actual download matches that before you
run anything downstream — dataset releases occasionally drift from documented code.

Usage:
    python src/preprocessing/inspect_wisig.py --root data/raw/wisig
    python src/preprocessing/inspect_wisig.py --root data/raw/wisig --subset ManyTx
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

COMPACT_SUBSETS = ("ManySig", "ManyTx", "ManyRx", "SingleDay")


def inspect_compact(path: Path) -> None:
    print(f"=== {path.name} (expected: compact WiSig subset) ===")
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    print("top-level keys:", list(dataset.keys()))
    expected = {"tx_list", "rx_list", "capture_date_list", "equalized_list", "data"}
    missing = expected - set(dataset.keys())
    if missing:
        print(f"  !! MISSING expected keys: {missing} — wisig_loader.py's _load_compact_pickle will need updating.")
        return

    print(f"  tx_list ({len(dataset['tx_list'])}):", dataset["tx_list"][:5], "...")
    print(f"  rx_list ({len(dataset['rx_list'])}):", dataset["rx_list"][:5], "...")
    print(f"  capture_date_list ({len(dataset['capture_date_list'])}):", dataset["capture_date_list"])
    print(f"  equalized_list:", dataset["equalized_list"])

    arr = np.asarray(dataset["data"][0][0][0][0])
    print(f"  data[0][0][0][0] shape={arr.shape} dtype={arr.dtype}")
    if arr.ndim == 3 and arr.shape[-1] == 2:
        print("  matches expected (n_sig, 256, 2) [I, Q] layout.")
    else:
        print("  !! UNEXPECTED shape — update _channels_to_complex/_load_compact_pickle in wisig_loader.py.")

    # Find first non-empty leaf to report a realistic n_sig.
    for tx_i in range(len(dataset["tx_list"])):
        for rx_i in range(len(dataset["rx_list"])):
            leaf = np.asarray(dataset["data"][tx_i][rx_i][0][0])
            if leaf.shape[0] > 0:
                print(f"  first non-empty leaf: tx_i={tx_i} rx_i={rx_i}, n_sig={leaf.shape[0]}")
                break
        else:
            continue
        break


def inspect_generic(path: Path) -> None:
    """Fallback for anything that isn't a recognized compact-subset filename — e.g. raw
    per-node files, or a differently-named release. Recursively prints structure."""
    print(f"=== {path.name} (generic dump) ===")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    _describe(obj)


def _describe(obj, prefix: str = "", depth: int = 0, max_depth: int = 4) -> None:
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{prefix}: ... (max depth reached)")
        return
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}{prefix}: dict with {len(keys)} keys, e.g. {keys[:5]}")
        if keys:
            _describe(obj[keys[0]], prefix=f"[{keys[0]!r}]", depth=depth + 1, max_depth=max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{prefix}: {type(obj).__name__} of length {len(obj)}")
        if len(obj) > 0:
            _describe(obj[0], prefix="[0]", depth=depth + 1, max_depth=max_depth)
    elif isinstance(obj, np.ndarray):
        print(f"{indent}{prefix}: ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{indent}{prefix}: {type(obj).__name__} = {str(obj)[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing extracted WiSig files")
    parser.add_argument("--subset", default=None, help="One of ManySig/ManyTx/ManyRx/SingleDay, if known")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist — extract WiSig there first (see data/README.md)")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"Found {len(files)} files under {root}")
    for f in files[:20]:
        print(" ", f.relative_to(root))
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more")
    print()

    if args.subset:
        candidates = [root / f"{args.subset}.pkl"]
    else:
        candidates = [f for f in files if f.stem in COMPACT_SUBSETS]

    if candidates:
        for c in candidates:
            if c.exists():
                inspect_compact(c)
            else:
                print(f"{c} not found")
        return

    pkl_files = [f for f in files if f.suffix in (".pkl", ".pickle")]
    if not pkl_files:
        print("No .pkl files found and no recognized compact-subset filename — check the extraction.")
        return
    for f in pkl_files[:3]:
        inspect_generic(f)


if __name__ == "__main__":
    main()
