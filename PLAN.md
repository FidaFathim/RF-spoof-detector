# Build Plan — 4 weeks, 2 people

Compressed from the original 8-week / 2-person schedule in `RF_Fingerprinting_Experimental_Project_Guide`.
Both partners work together in week 1 (shared foundation) and week 4 (shared integration/writeup);
weeks 2-3 split into a "baseline + attacks" track and a "physical consistency" track that run in parallel.

Legend: **A** = baseline/attacks partner, **B** = physical-consistency partner.

## Week 1 — Shared foundation

- [ ] Download WiSig, run `src/preprocessing/inspect_wisig.py` on it, confirm the on-disk
      structure matches (or fix) `src/preprocessing/wisig_loader.py`.
- [ ] Build the session-aware train/val/test split (`src/preprocessing/session_split.py`).
      **Never split by random packet** — same-session packets share channel/receiver state and
      leak information across the split (this is called out explicitly in the project guide's
      "rules that protect validity").
- [ ] Record the split seed and commit the split files under `data/splits/`.
- [ ] A: sketch the CNN architecture (`src/baseline/cnn_model.py`), get one forward pass running
      on a batch of real WiSig IQ tensors (shape check only, not trained yet).
- [ ] B: run `tests/test_features_synthetic.py`, confirm all three feature estimators
      (`src/features/cfo.py`, `iq_imbalance.py`, `phase_noise.py`) recover known synthetic
      ground truth before touching real signals.

**Week 1 exit criteria:** loader confirmed against real WiSig files, split files committed,
synthetic feature tests green.

## Week 2 — Parallel tracks

**A — baseline classifier**
- [ ] Train the CNN to convergence on the train split; validate on the val split.
- [ ] Save checkpoints, confusion matrix, per-device accuracy to `results/baseline/`.
- [ ] This step is *reproducing known work* — don't over-invest in architecture search.

**B — physical features**
- [ ] Run the feature pipeline (`src/features/pipeline.py`) over all legitimate train/val/test
      signals; save raw + normalized CFO / I-Q-imbalance / phase-noise per sample.
- [ ] Plot per-device and per-session distributions. Sanity check: features should cluster by
      device; if they cluster by *session/receiver* instead, flag it now (this is the "rules
      that protect validity" limitation check — do it early, not at the end).
- [ ] Start training the consistency gate (`src/consistency/gate.py`) — start with the Gaussian/
      Mahalanobis baseline, only add OC-SVM / isolation forest / autoencoder if the simple one
      is insufficient. Train **only on legitimate training-split signals**.

**Week 2 exit criteria:** baseline CNN trained + evaluated; feature distributions sanity-checked;
first consistency-gate checkpoint trained.

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
