"""Baseline RF-fingerprinting CNN — reproduces known work (e.g. the ORACLE/Sankhe-style
1D-CNN-on-raw-IQ family used throughout the RFFI literature). This is the "known work"
branch: not the novelty, just a realistic, attackable device classifier to build on.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RFFingerprintCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 2,
        conv_channels: list[int] = (64, 128, 256),
        kernel_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in conv_channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c_in, c_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c_in // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_channels, num_samples) float32 -> logits (batch, num_classes)."""
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
