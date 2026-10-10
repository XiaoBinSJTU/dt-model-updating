"""Numerically stable preprocessing utilities."""

from __future__ import annotations

import numpy as np


def standardize_from_training(
    training: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Standardize arrays using only training-set scalar statistics."""
    mean = float(np.mean(training))
    standard_deviation = float(np.std(training))
    scale = standard_deviation if standard_deviation > 1e-8 else 1.0
    arrays = (training,) + others
    return tuple(((array - mean) / scale).astype(np.float32) for array in arrays)


def minmax_rows(values: np.ndarray) -> np.ndarray:
    """Scale each row independently into the unit interval."""
    values = np.asarray(values, dtype=np.float64)
    minimum = values.min(axis=1, keepdims=True)
    span = values.max(axis=1, keepdims=True) - minimum
    return ((values - minimum) / np.maximum(span, 1e-12)).astype(np.float64)


def stratified_indices(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split indices while preserving every represented class."""
    generator = np.random.default_rng(seed)
    first: list[int] = []
    second: list[int] = []
    labels = np.asarray(labels).reshape(-1)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        cut = max(1, min(len(indices) - 1, int(round(len(indices) * fraction))))
        first.extend(indices[:cut].tolist())
        second.extend(indices[cut:].tolist())
    return np.asarray(sorted(first)), np.asarray(sorted(second))
