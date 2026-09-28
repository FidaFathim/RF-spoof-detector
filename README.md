# RF Spoofing Detection via Physical-Consistency Gating

A CNN identifies a WiFi/IoT transmitter from its raw-IQ "radio accent" (RF fingerprint).
Published attacks (FGSM, PGD, UAP, GAN-based impersonation) fool that CNN into accepting a
forged signal as a trusted device more than 90% of the time. This project adds a **second,
independent check** that asks a different question than the CNN: *given the hardware physics
of real radios, could this exact combination of impairments actually have come from one
transceiver?*

A signal is accepted only if **both** checks pass:

```
accept(x)  =  CNN(x) says "claimed device"   AND   ConsistencyGate(x | claimed device) says "physically plausible"
```

## Novelty statement (keep this narrow — see `PAPER_NOTES.md`)

RF fingerprinting, CNN classifiers, adversarial attacks on them, and CFO/I-Q-imbalance/phase-noise
as hardware features are **all pre-existing, published work**. We are not claiming any of that.
The one hypothesis we are actually testing:

> A **separate** forgery-detection model — trained only on legitimate signals, never on attacks —
> that learns the *joint physical relationship* between CFO, I/Q imbalance, and phase noise can
> catch adversarial examples that already fooled the CNN, because gradient-based attacks optimize
> against the CNN's decision boundary and have no reason to also respect real-hardware physics.

This is a research hypothesis, not a proven result. Treat "novel" as "promising but unverified until
the literature review + experiments in `PAPER_NOTES.md` / `results/` say otherwise."

## Status (2026-09-28)

All pipeline code (loader, split, CNN, attacks, physical features, consistency gate,
fusion + ablations, reporting) is implemented and wired end-to-end — see `PLAN.md` for the
week-by-week build log. What's actually been *run* so far:

- **Verified locally, 44/44 tests green (`pytest tests/`):** the physical-feature
  estimators against known-injected CFO/I-Q-imbalance/phase-noise ground truth, and — new —
  the consistency gate, calibration, fusion (AND-rule + all 3 ablations), and metrics
  modules against synthetic data with a known joint structure. One test in particular,
  `test_joint_gate_catches_decorrelated_attack_better_than_marginal_gate`, constructs an
  attack that matches every feature's marginal mean/std but breaks the cross-feature
  correlation real hardware has, and confirms the joint gate catches it while every
  single-feature gate doesn't — i.e. it verifies the gate *implementation* behaves the way
  the design in `PAPER_NOTES.md` claims it should. It is **not** a substitute for running the
  real attacks against the real trained CNN on real WiSig data.
- **Blocked on this machine, not a code issue:** this dev environment has a Windows
  Application Control policy that blocks loading compiled extensions for both `torch` *and*
  `pandas` (confirmed via direct DLL-load errors, not a missing-package issue) — so nothing
  that touches a DataFrame or the CNN can execute here. Every module in `src/` that has no
  pandas/torch dependency (`consistency/`, most of `evaluation/`, `preprocessing/normalize.py`)
  was confirmed importable and correct in isolation; everything else needs to run where that
  restriction doesn't apply (Colab, or any machine without this policy) — see Quickstart below.
- **Not yet run anywhere:** the actual CNN training, attacks, gate calibration on real
  features, and full evaluation table — i.e. no real detection-rate/false-positive-rate
  numbers exist yet. Run the one-command pipeline below to produce them.

Read `THREAT_MODEL.md` before trusting any future results table — it states exactly which
attacker is assumed for each attack, what's out of scope (over-the-air transmission, replay,
training-time poisoning), and what a high detection rate does and doesn't prove.

## Repository layout

```
rf-spoof-detection/
├── data/                  # WiSig notes, split files (not the raw dataset — see data/README.md)
├── configs/               # experiment configs (YAML)
├── src/
│   ├── preprocessing/     # WiSig loading, session-aware split, normalization
│   ├── features/          # CFO, I/Q imbalance, phase noise — pure signal processing, no ML
│   ├── baseline/          # the CNN device classifier (known work, reproduced)
│   ├── attacks/           # FGSM, PGD, UAP, GAN-impersonation adapter
│   ├── consistency/       # the physical-consistency gate (the actual novelty)
│   ├── evaluation/        # metrics, AND-rule fusion, ablation baselines, result tables
│   └── utils/             # seeding, config loading, logging
├── notebooks/             # thin Colab notebooks that call into src/ (see notebooks/00_colab_setup.ipynb)
├── tests/                 # synthetic-signal ground-truth tests for the feature extractors
└── results/               # frozen metrics tables / figures, one subfolder per run
```

## Quickstart (Google Colab)

1. Zip this folder (or push it to a private GitHub repo) and get it into your Google Drive.
2. Open `notebooks/00_colab_setup.ipynb` in Colab, mount Drive, and run the setup cell —
   it installs `requirements.txt` and puts `src/` on `sys.path`.
3. Put `ManySig.pkl` at `data/raw/wisig/ManySig.pkl` (see `data/README.md` for the link).
4. Either follow the notebooks in order (`01` → `06`, each a thin wrapper around `src/`), or
   run the whole thing in one cell:
   ```bash
   python -m src.run_pipeline --config configs/default.yaml --seed 0
   python -m src.run_pipeline --config configs/default.yaml --all-seeds   # 3 seeds, mean +/- std
   ```
   Final table: `results/final_seed0/results_table.md` (+ `diagnostics.json` with the
   per-receiver false-rejection validity check).
5. Read `data/README.md` before touching WiSig — the exact split rule (session-aware,
   never packet-random) is what makes the results valid.

## Local dev (no GPU, no dataset needed)

Everything under `tests/` is pure NumPy/SciPy/scikit-learn — no GPU, no WiSig download, no
`torch`/`pandas` needed. If you install Python locally:

```bash
pip install -r requirements.txt
pytest tests/
```

44 tests, in two groups:

- `test_features_synthetic.py` — generates synthetic IQ signals with a **known** CFO, I/Q
  imbalance, and phase-noise value injected, then checks the estimators recover it. Step 3
  of `PLAN.md`; should be the first thing that passes before touching real WiSig data.
- `test_consistency_gate_synthetic.py`, `test_calibrate.py`, `test_combine.py`,
  `test_metrics.py`, `test_normalize.py` — exercise the consistency gate, threshold
  calibration, AND-rule fusion + all 3 ablations, and the evaluation metrics against
  synthetic feature distributions with a known joint structure. These are what let you
  trust the gate's *logic* independently of ever running it on real WiSig data.

If `torch` or `pandas` fails to import with a DLL-load error rather than an
"ImportError: No module named" error, that's a local machine policy (Windows Application
Control / WDAC / Smart App Control blocking unsigned compiled extensions), not a broken
install — reinstalling won't fix it. Everything that needs those two packages (the WiSig
loader, the CNN, the attacks, the full pipeline) has to run somewhere without that
restriction; Colab is the path this repo is set up for.

## Key documents

- [`PLAN.md`](PLAN.md) — the 4-week, 2-person build plan and what "done" looks like each week.
- [`PAPER_NOTES.md`](PAPER_NOTES.md) — literature grounding, the honest novelty-gap analysis, and the rules that keep the experiment valid.
- [`THREAT_MODEL.md`](THREAT_MODEL.md) — attacker capabilities per attack, what's explicitly out of scope, and what a passing result does/doesn't prove.
- [`data/README.md`](data/README.md) — WiSig download/layout notes and the session-aware split rule.

## References

- S. Hanna, S. Karunaratne, D. Cabric, "WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting," IEEE Access, 2022.
- S. Karunaratne, E. Krijestorac, D. Cabric, "Penetrating RF Fingerprinting-based Authentication with a Generative Adversarial Attack," arXiv:2011.01538.
- H. Albousayri, B. Hamdaoui, "Replicating the Signature: Unsupervised Targeted Impersonation Attack on RF Fingerprinting," arXiv:2607.05549, Jul 2026. (HWE/HWE+ — stretch-goal attack)
- L. Yang, S. Camtepe, Y. Gao, V. Liu, D. Jayalath, "Robustness and Security Enhancement of RF Fingerprint Identification in Time-Varying Channels," arXiv:2410.07591, Oct 2024. (closest prior defense — single-feature PA-nonlinearity quotient + softmax posterior-difference countermeasure; **not** the multi-feature joint-consistency gate we test here)
