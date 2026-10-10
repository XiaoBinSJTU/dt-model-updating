"""Deterministic random-state and neutral device-selection helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch without imposing unavailable kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(policy: str = "auto") -> tuple[torch.device, str]:
    """Select an accelerator first in auto mode and return a neutral label."""
    normalized = policy.lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device policy: {policy}")
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        label = "GPU-first auto selection" if normalized == "auto" else "GPU"
        return torch.device("cuda"), label
    if normalized == "cuda":
        raise RuntimeError("GPU execution was requested but is unavailable")
    label = "CPU fallback" if normalized == "auto" else "CPU"
    return torch.device("cpu"), label


def release_device(device: torch.device) -> None:
    """Release unused accelerator cache after a validation unit."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
