# Build Plan — 4 weeks, 2 people

Compressed from the original 8-week / 2-person schedule in `RF_Fingerprinting_Experimental_Project_Guide`.
Both partners work together in week 1 (shared foundation) and week 4 (shared integration/writeup);
weeks 2-3 split into a "baseline + attacks" track and a "physical consistency" track that run in parallel.

Legend: **A** = baseline/attacks partner, **B** = physical-consistency partner.

## Week 1 — Shared foundation

- [x] Download WiSig (ManySig), run `src/preprocessing/inspect_wisig.py` on it, confirm the
      on-disk structure matches `src/preprocessing/wisig_loader.py`. *(Done 2026-09-14:
      288,000 signals, 6 tx x 12 rx x 4 days, loader verified against the authors' own code.)*
- [x] Build the session-aware train/val/test split (`src/preprocessing/session_split.py`).
      **Never split by random packet** — same-session packets share channel/receiver state and
      leak information across the split (this is called out explicitly in the project guide's
      "rules that protect validity").
- [x] Record the split seed and commit the split files under `data/splits/`.
      *(`split_seed0.json`: 29/10/9 sessions; seeds 1 and 2 are generated on demand by
      `python -m src.run_pipeline --all-seeds`.)*
- [x] A: CNN forward pass + training loop verified on real WiSig tensors.
      **Open:** cross-session generalization. 20k-sample / 5-epoch runs gave chance-level
      test accuracy; the full 174k train split for 15 epochs reached 71-94% but oscillated
      (see `PAPER_NOTES.md` #1/#5). `train.py` now has cosine LR decay + best-val
      checkpointing — first job on Colab is to run it and confirm a stable >=90% held-out
      test accuracy before anything downstream is trusted.
- [x] B: `tests/test_features_synthetic.py` green (7/7). Real-data sanity check done: with
      the receiver held fixed, CFO separates the 6 devices cleanly (means hundreds of kHz
      apart, within-device std single-digit kHz). Session-relative correction added
      (`src/features/session_normalize.py`) — required, see `PAPER_NOTES.md` #2/#3.

**Week 1 exit criteria:** loader confirmed against real WiSig files, split files committed,
synthetic feature tests green. **All met.** Carry-over into week 2: lock in a CNN that
generalizes across sessions (see above).

**Fast path for week 2:** the whole pipeline is now one command —
`python -m src.run_pipeline --config configs/default.yaml --seed 0` — producing
`results/final_seed0/results_table.md`. Run it on Colab GPU first thing; every number in
that table is the real thing (attacks on the test split, UAP trained on validation, gate
never sees attacks).

## Week 2 — Parallel tracks

**A — baseline classifier**
- [ ] Train the CNN to convergence on the train split; validate on the val split.
- [ ] Save checkpoints, confusion matrix, per-device accuracy to `results/baseline/`.
- [ ] This step is *reproducing known work* — don't over-invest in architecture search.
      **Blocked on this machine** (2026-09-28): this dev environment's Windows Application
      Control policy blocks loading `torch`'s compiled extension (confirmed DLL-load error,
      not a missing package) — training cannot execute here. Run on Colab
      (`notebooks/02_baseline_cnn.ipynb` or `python -m src.run_pipeline`) instead.

**B — physical features**
- [ ] Run the feature pipeline (`src/features/pipeline.py`) over all legitimate train/val/test
      signals; save raw + normalized CFO / I-Q-imbalance / phase-noise per sample.
      **Blocked on this machine for the same reason as above, but for `pandas`** (also
      confirmed via DLL-load error): `src/features/extract_all.py` needs a WiSig DataFrame,
      which needs pandas. Run on Colab, or any machine without this policy.
- [ ] Plot per-device and per-session distributions. Sanity check: features should cluster by
      device; if they cluster by *session/receiver* instead, flag it now (this is the "rules
      that protect validity" limitation check — do it early, not at the end). Not yet run —
      depends on the feature-extraction step above.
- [x] Start training the consistency gate (`src/consistency/gate.py`) — start with the Gaussian/
      Mahalanobis baseline, only add OC-SVM / isolation forest / autoencoder if the simple one
      is insufficient. Train **only on legitimate training-split signals**. *(Done 2026-09-28,
      to the extent possible without pandas/real features: the gate class itself — fit/score,
      all 3 one-class methods, both per_device/global modes, threshold calibration — is fully
      implemented and verified against synthetic feature data with a known joint structure;
      see `tests/test_consistency_gate_synthetic.py`. Training it on the REAL feature CSV still
      needs the pandas-dependent step above run somewhere without the local block.)*

**Week 2 exit criteria:** baseline CNN trained + evaluated; feature distributions sanity-checked;
first consistency-gate checkpoint trained. **Not yet met** — blocked on running the
pandas/torch-dependent stages somewhere without this machine's Application Control policy
(see `README.md` "Status"). The gate/calibration/fusion *logic* is implemented and unit-tested
(44/44 local tests green); no real numbers exist yet.

## Week 3 — Attacks + gate calibration

**A — attacks**
- [ ] Implement FGSM, PGD, UAP against the trained baseline CNN (`src/attacks/`).
- [ ] Measure and record raw CNN attack success rate per attack type — this reproduces the
      "known weakness," it is not the novel result.
- [ ] Adapt the public GAN-based impersonation attack (Karunaratne et al. 2020) if the
      reference implementation is reproducible in the time available; otherwise document why
      and drop to the stretch list.

**B — gate calibration**
- [ ] Finish consistency-gate training (per-device and global variants — keep both, they're
      an ablation, not a choice to make now).
- [ ] Select the accept/reject threshold on the **validation** split only (never on test).
- [ ] Build the 3 required ablation baselines from the project guide: (1) CNN-confidence
      threshold alone, (2) single physical feature alone, (3) all three features fed as
      ordinary extra classifier inputs. These exist to prove the *separate gate* is doing
      something the simpler alternatives don't.

**Week 3 exit criteria:** attack success rates recorded against the undefended baseline;
consistency gate + all 3 ablation baselines calibrated on validation data.

## Week 4 — Integration, evaluation, writeup (shared)

- [ ] Wire the AND-rule fusion (`src/evaluation/combine.py`): accept only if CNN says claimed
      device AND consistency gate passes.
- [ ] Run the full evaluation matrix from the project guide (clean / FGSM / PGD / UAP / GAN ×
      {CNN attack success, combined detection rate, false-rejection rate}) — `src/evaluation/report.py`
      produces this table directly.
- [ ] Run at least 3 random seeds; report mean ± spread, not just the best run.
- [ ] Stretch goal only if ahead of schedule: HWE/HWE+ (arXiv:2607.05549) — the hardest attack,
      since it explicitly tries to replicate real hardware impairments. No public code exists
      for it as of this writing; budget this as "attempt a from-scratch reimplementation of the
      impairment-estimation stage only" rather than the full OTA pipeline.
- [ ] Write up: results tables/figures, the honest novelty-gap discussion from `PAPER_NOTES.md`,
      and the limitation check (did the gate reject a different receiver/channel instead of an
      actual spoof?).

**Week 4 exit criteria:** full results table with detection rate + false-positive rate per
attack type, 3-seed variation reported, writeup + slides drafted.

## Do NOT do (things that would invalidate the results)

- Don't split packets randomly across train/val/test — split by session/day.
- Don't train the consistency gate on any attack sample, ever.
- Don't tune the gate threshold or attack strength on the test split.
- Don't claim novelty beyond "a separate physical-consistency gate, trained clean-only,
  tested against adversarial impersonation" — the CNN, the attacks, and the individual
  features are all prior art.
- Don't target 100% detection / 0% false positives as a result — report the tradeoff curve.
