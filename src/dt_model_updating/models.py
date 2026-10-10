"""Neural models for defect regression and algorithm-model updating."""

from __future__ import annotations

import torch
from torch import nn


class SignalBackbone(nn.Module):
    """Three-layer one-dimensional convolutional feature extractor."""

    output_features = 256

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        return self.network(inputs).flatten(1)


class AlgorithmClassifier(nn.Module):
    """Baseline anomaly detector with a reusable convolutional backbone."""

    def __init__(self, backbone: SignalBackbone | None = None) -> None:
        super().__init__()
        self.backbone = backbone or SignalBackbone()
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(inputs)).reshape(-1)


class AdapterClassifier(nn.Module):
    """Frozen-feature classifier with a 256-to-64 adaptation layer."""

    def __init__(self, backbone: SignalBackbone | None = None, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = backbone or SignalBackbone()
        self.adapter = nn.Sequential(nn.Linear(256, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.adapter(self.backbone(inputs))).reshape(-1)


class DefectParameterCNN(nn.Module):
    """CNN regressor for outer-ring defect depth and width."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(1)
        return self.regressor(self.feature_network(features).flatten(1))


def copy_backbone_weights(source: AlgorithmClassifier, target: nn.Module) -> None:
    """Copy learned convolutional weights into another compatible model."""
    target.backbone.load_state_dict(source.backbone.state_dict())
