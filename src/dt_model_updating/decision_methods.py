"""Global and local degradation-aware update decision methods."""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.mixture import GaussianMixture

from .features import fner
from .preprocessing import minmax_rows


def ewma_statistics(values: np.ndarray, smoothing: float, limit_coefficient: float, reference_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate EWMA statistics and one-sided dynamic upper control limits."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reference = values[:reference_count]
    mean = float(reference.mean())
    standard_deviation = float(reference.std(ddof=1))
    statistic = np.empty_like(values)
    limits = np.empty_like(values)
    statistic[0] = mean
    for index, value in enumerate(values):
        if index > 0:
            statistic[index] = smoothing * value + (1.0 - smoothing) * statistic[index - 1]
        multiplier = math.sqrt(smoothing / (2.0 - smoothing) * (1.0 - (1.0 - smoothing) ** (2 * (index + 1))))
        limits[index] = mean + limit_coefficient * standard_deviation * multiplier
    return statistic, limits


def _first_consecutive(mask: np.ndarray, start: int, consecutive: int) -> int:
    run = 0
    for index in range(start, len(mask)):
        run = run + 1 if bool(mask[index]) else 0
        if run >= consecutive:
            return index - consecutive + 1
    return len(mask)


def _average_run_length_objective(params: np.ndarray, reference: np.ndarray, seed: int) -> float:
    smoothing, limit = float(params[0]), float(params[1])
    generator = np.random.default_rng(seed)
    mean = float(reference.mean())
    standard_deviation = max(float(reference.std(ddof=1)), 1e-6)
    delays: list[float] = []
    false_alarms = 0
    for _ in range(18):
        sequence = generator.normal(mean, standard_deviation, 120)
        change = 60
        sequence[change:] += generator.uniform(0.55, 0.9) * standard_deviation
        statistic, limits = ewma_statistics(sequence, smoothing, limit, 40)
        detected = _first_consecutive(statistic > limits, 40, 3)
        if detected < change:
            false_alarms += 1
        delays.append(float(max(detected - change, 0) if detected < len(sequence) else 60))
    return float(np.mean(delays) + 12.0 * false_alarms)


def bayesian_optimize_ewma(reference: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Select EWMA parameters through bounded Gaussian-process optimization."""
    smoothing_values = np.linspace(0.08, 0.55, 13)
    limit_values = np.linspace(2.0, 3.5, 13)
    candidates = np.asarray([(smoothing, limit) for smoothing in smoothing_values for limit in limit_values])
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(candidates), size=8, replace=False).tolist()
    observed_x = candidates[selected].tolist()
    observed_y = [_average_run_length_objective(candidates[index], reference, seed + index) for index in selected]
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.15, 0.4], nu=2.5) + WhiteKernel(1e-5)
    for iteration in range(12):
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed + iteration, n_restarts_optimizer=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(np.asarray(observed_x), np.asarray(observed_y))
        mean, standard_deviation = model.predict(candidates, return_std=True)
        acquisition = mean - 1.5 * standard_deviation
        acquisition[np.asarray(selected, dtype=int)] = np.inf
        next_index = int(np.argmin(acquisition))
        selected.append(next_index)
        observed_x.append(candidates[next_index].tolist())
        observed_y.append(_average_run_length_objective(candidates[next_index], reference, seed + next_index))
    best = int(np.argmin(observed_y))
    return float(observed_x[best][0]), float(observed_x[best][1]), float(observed_y[best])


def _fit_gmm(features: np.ndarray, seed: int) -> GaussianMixture:
    best_model: GaussianMixture | None = None
    best_bic = float("inf")
    maximum_components = min(3, max(1, len(features) // 8))
    for components in range(1, maximum_components + 1):
        model = GaussianMixture(n_components=components, covariance_type="full", reg_covar=1e-5, random_state=seed)
        model.fit(features)
        bic = float(model.bic(features))
        if bic < best_bic:
            best_bic = bic
            best_model = model
    if best_model is None:
        raise RuntimeError("GMM fitting failed")
    return best_model


def gmm_distribution_validation(before: np.ndarray, after: np.ndarray, seed: int) -> dict[str, float | bool]:
    """Validate feature drift using fitted GMMs, symmetric divergence, and a KS test."""
    combined = np.vstack([before, after])
    components = min(combined.shape[1], max(1, min(5, combined.shape[0] - 1)))
    projected = PCA(n_components=components, random_state=seed).fit_transform(combined)
    first = projected[: len(before)]
    second = projected[len(before) :]
    first_model = _fit_gmm(first, seed)
    second_model = _fit_gmm(second, seed + 1)
    first_samples, _ = first_model.sample(256)
    second_samples, _ = second_model.sample(256)
    kl_first = np.mean(first_model.score_samples(first_samples) - second_model.score_samples(first_samples))
    kl_second = np.mean(second_model.score_samples(second_samples) - first_model.score_samples(second_samples))
    divergence = float(max(0.0, 0.5 * (kl_first + kl_second)))
    _, p_value = stats.ks_2samp(first[:, 0], second[:, 0])
    return {"kl_divergence": divergence, "p_value": float(p_value), "validated": bool(divergence > 0.03 and p_value < 0.1)}


def proposed_global_decision(
    errors: np.ndarray,
    features: np.ndarray,
    reference_count: int,
    window: int,
    consecutive: int,
    seed: int,
    require_gmm: bool = True,
    optimize_parameters: bool = True,
) -> dict[str, float | int | bool]:
    """Run Bayes-EWMA detection with optional GMM inverse validation."""
    if optimize_parameters:
        smoothing, limit, arl_objective = bayesian_optimize_ewma(errors[:reference_count], seed)
    else:
        smoothing, limit, arl_objective = 0.25, 3.0, float("nan")
    statistic, limits = ewma_statistics(errors, smoothing, limit, reference_count)
    crossing = statistic > limits
    candidates: list[int] = []
    search_start = min(reference_count + window, len(errors))
    while search_start < len(errors):
        candidate = _first_consecutive(crossing, search_start, consecutive)
        if candidate >= len(errors):
            break
        candidates.append(candidate)
        search_start = candidate + consecutive
    selected = len(errors)
    validation = {"kl_divergence": 0.0, "p_value": 1.0, "validated": not require_gmm}
    for candidate in candidates:
        if not require_gmm:
            selected = candidate
            break
        validation_width = max(8, window // 4)
        left = max(reference_count // 2, candidate - validation_width)
        right = min(len(features), candidate + validation_width)
        if candidate - left < 8 or right - candidate < 8:
            continue
        validation = gmm_distribution_validation(features[left:candidate], features[candidate:right], seed + candidate)
        if validation["validated"]:
            selected = candidate
            break
    return {
        "detection_index": int(selected),
        "smoothing": float(smoothing),
        "limit_coefficient": float(limit),
        "arl_objective": float(arl_objective),
        "kl_divergence": float(validation["kl_divergence"]),
        "p_value": float(validation["p_value"]),
        "validated": bool(validation["validated"]),
    }


def grey_relational_series(
    simulation_signals: np.ndarray,
    measured_signals: np.ndarray,
    sampling_frequency: float,
    fault_frequency: float,
    adaptive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed or FNER-adaptive grey relational degrees."""
    simulation = minmax_rows(simulation_signals)
    measured = minmax_rows(measured_signals)
    differences = np.abs(simulation - measured)
    global_min = float(differences.min())
    global_max = max(float(differences.max()), 1e-12)
    coefficients = np.empty(len(differences), dtype=np.float64)
    xis = np.empty(len(differences), dtype=np.float64)
    for index, row in enumerate(differences):
        if adaptive:
            ratio = fner(measured_signals[index], sampling_frequency, fault_frequency)
            xi = 0.45 + 0.17 * np.tanh(np.log1p(max(ratio, 0.0)) / 4.0)
        else:
            xi = 0.5
        xis[index] = xi
        coefficients[index] = float(np.mean((global_min + xi * global_max) / (row + xi * global_max)))
    return coefficients, xis


def detect_local_degradation(relational_degree: np.ndarray, reference_count: int, consecutive: int = 3) -> int:
    """Detect a sustained local-fit decrease using a healthy-reference control limit."""
    reference = np.asarray(relational_degree[:reference_count], dtype=np.float64)
    threshold = min(0.8, float(reference.mean() - 3.0 * reference.std(ddof=1)))
    return _first_consecutive(np.asarray(relational_degree) < threshold, reference_count, consecutive)
