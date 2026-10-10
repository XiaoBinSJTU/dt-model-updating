"""Reduced mechanism-response mapping and bounded trust-region calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class CalibrationResult:
    """Compact result returned by mechanism-parameter calibration."""

    damping: float
    stiffness: float
    objective: float
    evaluations: int


def mechanism_response_summary(parameters: np.ndarray) -> np.ndarray:
    """Map damping, stiffness, defect depth, and width to response descriptors."""
    damping, stiffness, depth, width = np.asarray(parameters, dtype=np.float64)
    natural_frequency = np.sqrt(max(stiffness, 1.0) / 12.0) / (2.0 * np.pi)
    damping_ratio = damping / (2.0 * np.sqrt(max(stiffness * 12.0, 1.0)))
    return np.asarray(
        [
            natural_frequency,
            damping_ratio,
            depth * (1.0 + 0.08 * width),
            width * np.sqrt(max(depth, 1e-4)),
            natural_frequency * (1.0 - 0.12 * depth),
            damping_ratio * (1.0 + 0.05 * width),
            depth / max(width, 1e-4),
            np.log1p(stiffness) / max(np.log1p(damping), 1e-4),
        ],
        dtype=np.float64,
    )


def trust_region_calibrate(
    target_summary: np.ndarray,
    depth: float,
    width: float,
    initial: tuple[float, float] = (3376.0, 5.24e4),
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((2500.0, 4500.0), (3.0e4, 6.5e4)),
    max_evaluations: int = 80,
) -> CalibrationResult:
    """Estimate damping and stiffness with bounded trust-region least squares."""
    target = np.asarray(target_summary, dtype=np.float64).reshape(-1)
    if target.shape != (8,):
        raise ValueError("target_summary must contain eight response descriptors")
    scale = np.maximum(np.abs(target), 1e-6)

    def residual(pair: np.ndarray) -> np.ndarray:
        prediction = mechanism_response_summary(np.asarray([pair[0], pair[1], depth, width]))
        return (prediction - target) / scale

    result = optimize.least_squares(
        residual,
        np.asarray(initial, dtype=np.float64),
        bounds=(
            np.asarray([bounds[0][0], bounds[1][0]], dtype=np.float64),
            np.asarray([bounds[0][1], bounds[1][1]], dtype=np.float64),
        ),
        method="trf",
        max_nfev=max_evaluations,
    )
    objective = float(np.mean(residual(result.x) ** 2))
    return CalibrationResult(float(result.x[0]), float(result.x[1]), objective, int(result.nfev))
