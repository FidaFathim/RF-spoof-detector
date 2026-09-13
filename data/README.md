# Data notes

**Never commit the raw dataset or processed arrays** — `.gitignore` already excludes
`data/raw/`, `data/processed/`, and common array/tensor extensions.

## Getting WiSig

- **Dataset page (official, UCLA CORES lab):** https://cores.ee.ucla.edu/downloads/datasets/wisig/
- **Loading code + examples:** https://github.com/WiSig-dataset/wisig-examples
- **Raw-capture processing code:** https://github.com/WiSig-dataset/wisig-process-raw
- **Paper:** S. Hanna, S. Karunaratne, D. Cabric, "WiSig: A Large-Scale WiFi Signal Dataset
  for Receiver and Channel Agnostic RF Fingerprinting," IEEE Access, vol. 10, pp. 22808-22818, 2022.

174 transmitters, 41 receivers, ~10M packets across 4 capture days (~1 month span).

### Which download to grab

| Subset | Tx | Rx | Signals/pair | Days | Size | Use for |
|---|---|---|---|---|---|---|
| **ManySig** | 6 | 12 | 1,000 | 4 | 1.4 GB | **start here** — enough signals/device to train + attack + gate |
| ManyTx | 150 | 18 | 50 | 4 | 2.5 GB | scaling to many devices, fewer signals each |
| ManyRx | 10 | 32 | 200 | 4 | 1.2 GB | receiver-generalization studies |
| SingleDay | 28 | 10 | 800 | 1 | 1 GB | no cross-day split available — skip unless space-constrained |
| Full WiSig | 174 | 41 | — | 4 | 42 GB zipped / 77 GB unzipped | only if the compact subsets prove insufficient |
| Raw WiSig | 174 | 41 | full packets | 4 | 1.4 TB | not needed for this project — the compact subsets already extract the 256-sample preambles |

All are `.pkl` files distributed via Google Drive links on the dataset page. Get **ManySig**
first — it has enough signals per device for the baseline CNN, and 4 capture days gives
you real session diversity for the session-aware split.

### Confirmed pickle structure (from the authors' own `data_utilities.py`)

```
{
  "tx_list": [...],              # transmitter names, index = tx_i
  "rx_list": [...],              # receiver names, index = rx_i
  "capture_date_list": [...],    # capture days, index = day_i
  "equalized_list": [0] or [0, 1],  # 0 = raw, 1 = channel-equalized
  "data": [tx_i][rx_i][day_i][eq_i] -> ndarray shape (n_sig, 256, 2),  # [I, Q] real/imag
}
```

`src/preprocessing/wisig_loader.py` already parses this. Still run the diagnostic first —
dataset releases sometimes drift from documented loading code:

```bash
python src/preprocessing/inspect_wisig.py --root data/raw/wisig --subset ManySig
```

If it reports a mismatch, fix `_load_compact_pickle` in `wisig_loader.py` — the rest of the
pipeline only depends on the DataFrame schema `load_wisig()` returns.

### One open question — verify before trusting Hz-scale features

The 256 samples per signal are already-extracted preambles ("Id Signals"), not full
packets, and the exact sample rate isn't stated in the loading code. `configs/default.yaml`
assumes 20 MSps (the standard 802.11 capture rate used throughout this literature) — confirm
against the WiSig paper's experimental setup section before trusting the CFO (Hz) and
phase-noise (rad) values `src/features/` produces on real data. This doesn't affect
`iq_length: 256`, which is confirmed correct.

### Raw vs. equalized — CONFIRMED, don't revert this without re-testing

`configs/default.yaml` uses `equalized: 1` for everything (CNN **and** physical features).
This was empirically tested on the real ManySig download, not assumed — see
`PAPER_NOTES.md`'s "Week 1 empirical findings" section for the full numbers, summarized here:

- Training the baseline CNN on **raw** (`equalized: 0`) signal with a proper session-aware
  split gives **16.7% test accuracy on held-out sessions — exactly random chance for 6
  classes.** The CNN overfits to receiver/channel state instead of learning device identity.
- The same CNN trained on **equalized** (`equalized: 1`) signal gets **100% test accuracy**
  on the identical held-out sessions.
- Raw CFO/I-Q-imbalance/phase-noise are *more* per-device-discriminative than the equalized
  versions of the same features (equalization partially removes the hardware artifacts we
  want) — but the CNN and the consistency gate must use the SAME signal representation,
  because the attacker only ever manipulates the one representation the classifier actually
  consumes. Scoring an attack against a raw-domain gate while the CNN sees equalized input
  would silently compare two different signal spaces and inflate detection numbers. So both
  branches use equalized signal, accepting a somewhat less crisp physical fingerprint as the
  honest cost of a methodologically sound comparison.

## Session-aware split — read this before splitting anything

WiSig captures are grouped by **day** and **receiver**. Packets captured in the same
session share the same channel state, receiver front-end, and approximate temperature —
splitting individual packets randomly across train/val/test leaks that shared session state
and will make both the baseline CNN and the consistency gate look better than they really are.

`src/preprocessing/session_split.py` splits by **(day, receiver)** session identifiers
(`session_id` = `f"{day}__{rx_id}"`), not by packet. Run it once, record the seed, and
commit the resulting split files under `data/splits/` so every subsequent run (baseline
training, feature extraction, gate calibration, evaluation) uses the exact same train/val/
test session assignment.

Note: **ManySig only has 4 capture days.** A (day, receiver) session split with that few
days per split is coarse — you may end up needing to split primarily by receiver within a
shared day, or pool days differently than a naive 60/20/20 session shuffle would. Inspect
`make_session_split`'s output session counts per split before committing to it; if any
split ends up with too few sessions to be representative, adjust `train_frac`/`val_frac` in
`session_split.py`'s call, or switch to ManyRx (32 receivers) for more split granularity.

## Directory layout after setup

```
data/
├── README.md          (this file)
├── raw/                (gitignored — your WiSig extraction)
│   └── wisig/
│       └── ManySig.pkl
├── processed/          (gitignored — cached feature/tensor arrays)
└── splits/             (committed — small JSON files, not the data itself)
    ├── split_seed0.json
    └── ...
```
