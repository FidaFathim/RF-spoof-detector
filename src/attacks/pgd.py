"""PGD (Madry et al. 2018), targeted: iterated FGSM with projection back into an
epsilon-ball around the original signal after each step. Stronger than single-step FGSM;
still direct-injection / full-access, still the "known weakness" baseline, not our novelty.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pgd_targeted(
    model: nn.Module,
    x: torch.Tensor,
    target_label: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    model.eval()
    x_orig = x.clone().detach()
    x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, target_label)
        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() - alpha * grad.sign()
        perturbation = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = (x_orig + perturbation).detach()

    return x_adv
