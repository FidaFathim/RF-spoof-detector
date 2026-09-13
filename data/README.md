# Data notes

**Never commit the raw dataset or processed arrays** — `.gitignore` already excludes
`data/raw/`, `data/processed/`, and common array/tensor extensions.

## Getting WiSig

WiSig (Hanna, Karunaratne, Cabric, IEEE Access 2022) is a large multi-day, multi-receiver
WiFi (802.11) capture dataset collected on the POWDER testbed: 174 transmitters, up to 41
receivers, several capture days, released as Python pickle files (there are "ManySig" /
"ShortSig"/"ALLSIG" subset variants with different tx/rx/day combinations — pick the subset
that matches how much compute you have on Colab; start with a smaller subset for week 1).

1. Download it (IEEE DataPort / the authors' release link) and extract into `data/raw/wisig/`.
2. **Before writing any loader-dependent code**, run:
   ```bash
   python src/preprocessing/inspect_wisig.py --root data/raw/wisig
   ```
   This prints the actual top-level structure (dict keys / array shapes / dtypes) of what you
   downloaded. The exact key names for day/session/tx/rx in `src/preprocessing/wisig_loader.py`
   are written against the commonly-documented WiSig pickle layout, but **must be confirmed
   against your actual files** — dataset releases occasionally change key names between
   versions. This is a 15-minute task, do it first, in week 1.
3. Update `configs/default.yaml` → `data.wisig_root`, and fix up `wisig_loader.py` if
   `inspect_wisig.py` shows a mismatch.

## Session-aware split — read this before splitting anything

WiSig captures are grouped by **day** and **receiver**. Packets captured in the same
session share the same channel state, receiver front-end, and approximate temperature —
splitting individual packets randomly across train/val/test leaks that shared session state
and will make both the baseline CNN and the consistency gate look better than they really are.

`src/preprocessing/session_split.py` splits by **(day, receiver)** session identifiers, not by
packet. Run it once, record the seed, and commit the resulting split files under `data/splits/`
so every subsequent run (baseline training, feature extraction, gate calibration, evaluation)
uses the exact same train/val/test session assignment.

## Directory layout after setup

```
data/
├── README.md          (this file)
├── raw/                (gitignored — your WiSig extraction)
│   └── wisig/
├── processed/          (gitignored — cached feature/tensor arrays)
└── splits/             (committed — small JSON/CSV files, not the data itself)
    ├── split_seed0.json
    └── ...
```
