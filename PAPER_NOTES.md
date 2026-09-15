# Literature grounding & honest novelty gap

Source docs this is distilled from: `RFF_Whole_Process.pdf`, `RFF_Team_Explanation_Draft.pdf`,
`RF_Fingerprinting_Experimental_Project_Guide.pdf`, arXiv:2607.05549, arXiv:2410.07591.
Keep this file updated as the weekly Scholar/arXiv alerts (mentioned in the team draft) turn
up anything new — the novelty claim below is provisional, not final.

## What is definitely NOT ours (prior art, use as-is)

| Component | Status | Key reference |
|---|---|---|
| RF fingerprinting via CNN on raw IQ | well established | general RFFI literature |
| CFO / I-Q imbalance / phase noise as hardware features | well established | classical RFFI feature literature |
| FGSM / PGD / UAP against RFFI classifiers | active research area, published | general adversarial-ML-on-RFFI papers |
| GAN-based impersonation attack | published, code available | Karunaratne, Krijestorac, Cabric, arXiv:2011.01538 (2020) |
| Hardware-impairment-replicating impersonation (HWE/HWE+) | published Jul 2026, no public code | Albousayri & Hamdaoui, arXiv:2607.05549 |
| Multiple physical features used *together as classifier input* | published, not new | e.g. general RFFI-with-features literature |

## Closest existing defenses (read these before writing the novelty section)

1. **PA-nonlinearity quotient + transfer learning** (Yang et al., arXiv:2410.07591, 2024).
   Uses *one* physical feature (power-amplifier nonlinearity, estimated via STFT ratio of a
   high-power/low-power preamble pair) to build a channel-robust fingerprint, and separately
   proposes a **keyless countermeasure**: train two classifiers (a fine-tuned transfer-learning
   model and a from-scratch deep model) on the same enrollment data, and flag impersonation
   when the softmax posterior-probability difference between them flips sign at the claimed
   class (Eq. 8–10 in the paper). This is the closest published defense we found.
   **How we differ:** their check is a statistical disagreement between two *classifiers*
   trained on the same input; it does not reason about physical relationships between multiple
   hardware impairments at all. Ours is a physics-based gate over CFO/I-Q-imbalance/phase-noise
   jointly, structurally separate from any classifier.

2. **HWE/HWE+ impersonation** (Albousayri & Hamdaoui, arXiv:2607.05549, Jul 2026) is the
   *toughest test case* for our own idea, not a defense: it explicitly estimates a target
   device's impairments (CFO, phase offset, I/Q amplitude/phase imbalance, I/Q DC offset,
   distorted BT product, max-frequency-deviation offset) via unsupervised learning and
   synthesizes a signal that reproduces them. If our consistency gate is only checking
   "is this physically *plausible in general*," HWE is specifically designed to pass that check.
   This is why it's a stretch goal, not a core requirement — a defense that beats FGSM/PGD/UAP/GAN
   but not HWE is still a legitimate, reportable result; just don't oversell it.

## The gap we are actually testing (research hypothesis, not yet proven)

> A model trained **only on legitimate signals** that learns the joint relationship between
> CFO, I/Q imbalance, and phase noise, used as a **separate gate after** the ordinary CNN
> classifier (not as extra classifier input features), increases adversarial-impersonation
> detection at a similar false-positive rate compared to: (a) CNN confidence thresholding alone,
> (b) any single physical feature alone, (c) feeding the same three features into the classifier
> as ordinary inputs.

Per the project guide: describe this as a **research hypothesis**, not a proven novel
contribution, until the systematic literature review (see checklist below) and the
week-4 experiments both support it.

## Systematic literature review checklist (do before finalizing any novelty claim in the writeup)

Search each of these on IEEE Xplore, ACM DL, Google Scholar, arXiv, and Scopus (or your
institution's equivalent), dedupe, and manually check full text — not just titles/abstracts:

- "RF fingerprinting adversarial attack"
- "RF spoofing detection hardware impairments"
- "CFO I/Q imbalance phase noise authentication"
- "one-class RF fingerprinting"
- "physics-informed RF fingerprinting"
- "hardware-aware RF impersonation"
- "feature consistency RF fingerprinting"

For each hit, log: title, year, dataset, signal type, features used, attack model tested,
defense type, **whether the defense is structurally separate from the classifier**, and
**whether it models relationships between features** (vs. using them independently). Only
after this table exists should the final paper state an exact "N prior papers found" count
with a search date.

## Week 1 empirical findings (on real ManySig data, 2026-09-14)

These aren't hypothetical concerns anymore — they were reproduced end-to-end against the
actual downloaded `ManySig.pkl` (6 devices, 12 receivers, 4 capture days, session-aware
split via `data/splits/split_seed0.json`).

1. **The baseline CNN overfits to receiver/channel, not device — for BOTH raw and
   equalized signal.** A baseline CNN trained on `equalized: 0` (raw) IQ hit 100%
   *validation* accuracy but only **16.7% test accuracy on held-out (day, receiver) sessions
   — indistinguishable from random guessing on 6 classes.** This is the textbook "RFFI
   classifier learns the channel, not the transmitter" failure this exact literature exists
   to address (it's the whole premise of WiSig's title, and of Yang et al. 2410.07591's
   channel-robust PA-nonlinearity feature).
   An initial test suggested WiSig's channel-equalized variant (`equalized: 1`) fixed this
   (100% test accuracy) — **that result was wrong**, caused by an evaluation bug: the test
   subset used was the first N rows of the test split, which (because `wisig_loader.py`
   iterates transmitter-outer) happened to be *entirely one device*, a degenerate
   single-class evaluation. Evaluating on a proper random sample or the full test set shows
   equalized-signal test accuracy is also **16.4-16.7% — still random chance.** Equalization
   does not fix this on its own with the current architecture/training budget (5-15 epochs,
   up to the full 174k train signals tried so far). Lesson for the team: always evaluate on
   a randomly-shuffled or fully-enumerated subset, never a positional slice of a DataFrame
   assembled by grouped iteration — this exact class of bug is easy to reintroduce.
   **This is an open problem for weeks 1-2 of the actual project, not a solved one** — see
   the "Baseline CNN generalization is unresolved" note below for candidate fixes to try.

2. **Raw CFO conflates transmitter and receiver oscillator offset, and drifts across days.**
   CFO is measured as `tx_LO - rx_LO`, so holding the receiver fixed shows clean per-device
   separation (six devices, means from -223 kHz to +416 kHz, within-device std single-digit
   kHz) — but holding the *device* fixed and varying the receiver shows the same swing
   (-243 kHz to +423 kHz), and holding both device AND receiver fixed but varying capture
   *day* still shows tens-of-kHz drift. A per-device one-class gate trained on some sessions
   and evaluated on session-aware held-out sessions of the *same* device initially
   false-rejected **84% of legitimate signals** — almost entirely explained by the CNN
   failure in finding #1 above (`1 - 0.167 ≈ 0.83`), not the gate itself. Isolating the gate
   from the CNN showed its own generalization gap was real but much smaller: ~9% false
   rejection against a 5% calibration target.

3. **Fix: session-relative features.** Subtracting each session's own median feature value
   (`src/features/session_normalize.py`) — computed across all devices captured in that
   exact session — before fitting/scoring the gate cancels most of the shared receiver/day
   component. Test false-rejection rate dropped from ~9% to ~7.5-8% (raw signal) / ~9.1% to
   ~8.2% (equalized signal). This is a real, if partial, improvement — not a complete fix.
   **Deployment caveat:** this requires multiple devices' signals from the same session to
   compute a baseline, which is fine for this offline research evaluation (WiSig sessions
   have many devices) but is not itself a deployment-ready calibration strategy — say this
   plainly in the writeup rather than presenting it as solved.

4. **Design decision that still holds regardless of #1's correction:** use ONE consistent
   signal representation for *both* the CNN and the physical-feature extraction (currently
   equalized, since raw already failed and equalized is at least no worse) rather than mixing
   domains. The alternative (CNN on equalized, gate on raw) would let the gate "detect"
   attacks partly because attack signals live in a different signal-processing domain than
   the gate was calibrated on — not because the attack is physically implausible. That would
   be a methodological confound, not a result. If a future fix changes which domain the CNN
   uses, update the gate's domain to match.

5. **Baseline CNN cross-session generalization is an open problem, not solved.** Candidate
   fixes to try in week 1-2 before treating this as blocking:
   - Transfer learning: train a base model on train-session data, then fine-tune briefly on
     a small amount of target-session data — this is literally Yang et al. 2410.07591's
     approach to the same problem, already in our reference list.
   - A frequency-domain / spectrogram input representation instead of raw time-domain IQ —
     also the representation Yang et al. found most accurate.
   - More model capacity or regularization/augmentation across sessions (channel simulation,
     mixup across receivers) so the model can't shortcut on a single session's channel state.
   - Sanity-check with a random (not session-aware) split first, to confirm the architecture
     *can* reach high accuracy at all before concluding a fix targets the right failure mode.
   Until one of these closes the gap, the whole downstream pipeline (attacks, consistency
   gate, combined detection) is being evaluated on a classifier that doesn't yet solve its
   own baseline task on held-out sessions — say so explicitly in any results reported before
   this is fixed, rather than quietly using in-distribution (non-session-aware) numbers.

## Rules that keep the experiment valid (do not skip these)

- Never train the consistency gate on attack samples.
- Session-aware split only; record + publish the split seed and split files.
- Tune attack strength and gate thresholds on validation data, never on test.
- Run ≥3 random seeds, report variation, not just the best run.
- Always compare against the 3 ablation baselines (CNN-confidence-only, single-feature-only,
  features-as-classifier-input) — without these, "the gate helped" is not a supported claim.
- Explicitly check whether the gate's false positives correlate with receiver/channel/session
  identity rather than being uniformly distributed across legitimate signals — if it does, the
  gate may be learning environment, not hardware, and that must be reported as a limitation.
- Log code version, package versions, and hardware/config for every run that produces a
  reported number.
