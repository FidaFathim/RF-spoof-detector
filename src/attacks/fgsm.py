"""FGSM (Goodfellow et al. 2015), targeted: craft x' = x + eps*sign(grad) that makes the
victim CNN classify x' as `target_label` (the impersonated device). Direct-injection /
full-access threat model (raw samples fed straight to the classifier) — the simplest,
best-documented baseline attack in the RFFI-adversarial literature.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def fgsm_targeted(
    model: nn.Module,
    x: torch.Tensor,
    target_label: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """x: (batch, C, N) float32, requires no grad on input to start. Returns adversarial x'."""
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = nn.functional.cross_entropy(logits, target_label)
    # Targeted attack: descend the loss toward the target label -> subtract the gradient sign.
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv - epsilon * grad.sign()

    return x_adv.detach()
