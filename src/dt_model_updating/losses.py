"""Loss functions used by model update experiments."""

from __future__ import annotations

import torch
from torch import nn


class BinaryDetectionLoss(nn.Module):
    """Numerically stable binary anomaly-detection loss."""

    def __init__(self, positive_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("positive_weight", torch.tensor([positive_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1), targets.float().reshape(-1), pos_weight=self.positive_weight
        )


class ScaledDefectLoss(nn.Module):
    """Mean squared error after fixed physical-range target scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([0.9, 4.0], dtype=torch.float32))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean(((predictions - targets) / self.scale) ** 2)
