# Threat model

This project defends an RF-fingerprint-based transmitter authentication system: a receiver
observes a signal claiming to come from device `D`, and must decide whether to trust that
claim. This document states, in one place, exactly who the attacker is assumed to be, what
they can and can't do, and what a passing/failing result actually means — the kind of
statement a network-security review expects and a pure ML paper often skips.

## Asset being protected

Device identity / authentication decisions in an RF-fingerprinting system: "is this signal
really from the enrolled transmitter it claims to be from?" A false accept lets an attacker
impersonate a trusted device (spoofing); a false reject denies service to a legitimate one.

## System under attack

```
signal x, claimed identity D
        |
        v
  +-----------+        +----------------------+
  |  CNN(x)   |         | ConsistencyGate(x|D) |
  | -> D' ?   |         | -> physically        |
  +-----------+         |    plausible for D?  |
        |               +----------------------+
        |                         |
        +----------+  +-----------+
                   v  v
             accept iff D' == D  AND  gate passes
```

`src/evaluation/combine.py:and_rule_accept` implements the fusion rule. Either check alone
can be fooled; the hypothesis under test is that a joint physical-consistency gate isn't
fooled by the *same* perturbation that fools the CNN (see `PAPER_NOTES.md` for the precise,
still-unproven research hypothesis).

## Attacker model (per attack — these are NOT all the same attacker)

| Attack | Access assumed | Goal | Constraint | Where |
|---|---|---|---|---|
| FGSM | White-box: full CNN weights/gradients, direct-injection (attacker's tensor is what the CNN consumes, no channel/RF re-transmission modeled) | Targeted impersonation of one enrolled device | L∞ perturbation ≤ `epsilon` | `src/attacks/fgsm.py` |
| PGD | Same as FGSM, iterative | Same | L∞ ball, `steps` iterations | `src/attacks/pgd.py` |
| UAP | White-box, but perturbation is trained ONCE on a batch and reused with no per-sample compute | Same, at scale (single perturbation broadcast to many transmissions) | Single perturbation, L∞ ≤ `epsilon` | `src/attacks/uap.py` |
| GAN impersonation (stretch, not yet wired — `src/attacks/gan_impersonation.py`) | Access to a generator trained against the classifier (Karunaratne et al. 2020); more realistic than raw gradient perturbation | Same | Whatever the reference generator produces | integration stub |
| HWE/HWE+ (stretch, not implemented) | Estimates a target's real hardware impairments and synthesizes a signal reproducing them — the attack explicitly designed to also respect physical plausibility | Same, and specifically to beat a consistency-style gate | No implementation exists locally; documented as the hardest open case | `PAPER_NOTES.md` |

**Direct-injection assumption, stated plainly:** every implemented attack perturbs the exact
tensor the CNN consumes. None of them model actually transmitting a forged RF signal through
real hardware and a physical channel to reach the receiver's ADC — that's a materially
harder attack (the perturbation has to survive real hardware nonlinearity, filtering, and
noise) and is out of scope here. A defense that beats direct-injection attacks is a weaker,
but still meaningful and separately reportable, claim than one that survives over-the-air
adversarial transmission. Say this explicitly in any write-up — don't imply OTA robustness
was tested.

## Explicitly out of scope

- Physical / over-the-air adversarial transmission (RF hardware limits on realizable
  perturbations, channel effects on the attack itself).
- Attacks on the receiver's demodulation/preamble-detection front end (this project assumes
  a clean 256-sample "Id Signal" preamble is already extracted, per WiSig's format).
- Denial-of-service against the authentication pipeline itself (e.g. flooding, resource
  exhaustion) — this is a spoofing/impersonation study, not an availability study.
- Attacks on the training pipeline (data poisoning of the CNN's or the gate's training set).
- Replay attacks (retransmitting a captured legitimate signal verbatim) — a physically
  plausible *and* CNN-correct replay would pass both checks by construction; that's a
  separate problem (freshness/liveness), not what this gate is designed to catch.

## What "the gate caught it" does and doesn't prove

- The gate is trained **only** on legitimate signals, never on any attack sample
  (`PAPER_NOTES.md` "rules that keep the experiment valid") — so a high detection rate is
  evidence of genuine physical implausibility, not overfitting to known attack patterns.
- A high detection rate against FGSM/PGD/UAP does **not** imply robustness against an
  attacker who specifically targets the gate (e.g. HWE-style impairment replication, or a
  future adaptive attack that optimizes against both the CNN and a *known* gate
  simultaneously — this project's threat model does not include an attacker with white-box
  access to the gate itself, only to the CNN). State this limitation in any results writeup.
- `src/consistency/session_normalize` style features require multiple devices' signals from
  the same session to compute a baseline — a real deployment would need its own calibration
  strategy; treat the reported false-rejection rate as a research measurement, not a
  deployment SLA.

## Validity checks this project treats as required, not optional

- Session-aware train/val/test split (never split individual packets at random) —
  `src/preprocessing/session_split.py`.
- Attack strength and gate/threshold calibration tuned on validation data only, never test.
- Explicit check for whether the gate's false rejections correlate with receiver/session
  identity rather than being spread evenly across legitimate signals (would indicate the
  gate learned environment, not hardware) — `src/evaluation/run_full_eval.py`'s
  `gate_frr_by_session` / `gate_frr_by_receiver` diagnostics.
- ≥3 random seeds reported as mean ± spread, not a single best run —
  `python -m src.run_pipeline --all-seeds`.
