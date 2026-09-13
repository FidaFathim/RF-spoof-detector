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
