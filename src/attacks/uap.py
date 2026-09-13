"""Universal Adversarial Perturbation (Moosavi-Dezfooli et al. 2017 style), targeted and
input-independent: find ONE perturbation vector that, added to many different clean
signals, makes the CNN classify them as `target_label`. Optimized once over a batch, then
reused at test time with no per-sample computation — this is what "input-independent" in
the project draft refers to.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def train_uap(
    model: nn.Module,
    x_batch: torch.Tensor,
    target_label: int,
    epsilon: float,
    max_iters: int,
    lr: float = 0.01,
) -> torch.Tensor:
    """x_batch: (N, C, L) clean signals from *any* devices (not necessarily the target).
    Returns a single perturbation tensor of shape (1, C, L)."""
    model.eval()
    device = x_batch.device
    delta = torch.zeros((1, *x_batch.shape[1:]), device=device, requires_grad=True)
    target = torch.full((x_batch.size(0),), target_label, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(max_iters):
        optimizer.zero_grad()
        x_adv = x_batch + delta
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)

    return delta.detach()


def apply_uap(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return x + delta
