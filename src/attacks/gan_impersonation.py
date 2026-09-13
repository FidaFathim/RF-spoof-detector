"""Adapter stub for the GAN-based impersonation attack (Karunaratne, Krijestorac, Cabric,
"Penetrating RF Fingerprinting-based Authentication with a Generative Adversarial Attack",
arXiv:2011.01538, 2020). Public code exists for this attack (per the team's draft plan);
this file is NOT a from-scratch reimplementation — it's the seam where you plug that
reference implementation in.

Integration steps (do this in week 3):
1. Clone the authors' reference implementation (check the paper for the current repo link;
   arXiv preprints don't always keep a stable code link, so verify it's still up before
   committing to it).
2. Adapt their generator's input to your WiSig-derived clean signals (their pipeline may
   assume a specific signal representation/length — reconcile with `data.iq_length` in
   configs/default.yaml).
3. Implement `generate_gan_impersonation` below to call their trained generator and return
   forged signals in the same (N, C, L) tensor convention used by fgsm.py/pgd.py/uap.py, so
   `run_attacks.py` and the evaluation pipeline don't need attack-specific branching.
4. Set `attacks.gan_impersonation.enabled: true` in configs/default.yaml once wired up.

If the reference implementation is not reproducible in the time available, document that
explicitly in PAPER_NOTES.md / the final writeup and drop this to a stretch item — don't
silently skip it.
"""
from __future__ import annotations

import torch


def generate_gan_impersonation(clean_signals: torch.Tensor, target_label: int) -> torch.Tensor:
    raise NotImplementedError(
        "Plug in the adapted Karunaratne et al. (2020) generator here — see module docstring."
    )
