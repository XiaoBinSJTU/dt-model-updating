"""Signal features used by decision and parameter-update experiments."""

from __future__ import annotations

import numpy as np
import pywt
from scipy import stats


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-12))


def time_frequency_features(signal: np.ndarray, sampling_frequency: float) -> np.ndarray:
    """Extract paper-aligned time and frequency summary features."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    centered = values - values.mean()
    rms = float(np.sqrt(np.mean(values**2)))
    absolute_mean = float(np.mean(np.abs(values)))
    spectrum = np.abs(np.fft.rfft(centered)) + 1e-12
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sampling_frequency)
    probability = spectrum / spectrum.sum()
    centroid = float(np.sum(frequencies * probability))
    spectral_spread = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * probability)))
    spectral_entropy = float(-np.sum(probability * np.log(probability)) / np.log(len(probability)))
    spectral_rms = float(np.sqrt(np.sum((frequencies**2) * probability)))
    spectral_skew = float(np.sum(((frequencies - centroid) / max(spectral_spread, 1e-12)) ** 3 * probability))
    spectral_kurtosis = float(np.sum(((frequencies - centroid) / max(spectral_spread, 1e-12)) ** 4 * probability))
    features = np.array(
        [
            values.mean(),
            rms,
            values.var(),
            values.max(),
            values.min(),
            absolute_mean,
            float(stats.skew(values, bias=False)),
            float(stats.kurtosis(values, fisher=False, bias=False)),
            _safe_ratio(np.max(np.abs(values)), rms),
            _safe_ratio(rms, absolute_mean),
            centroid,
            spectral_rms,
            spectral_spread,
            spectral_entropy,
            spectral_skew,
            spectral_kurtosis,
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def batch_time_frequency_features(signals: np.ndarray, sampling_frequency: float) -> np.ndarray:
    """Extract time-frequency features for a signal matrix."""
    return np.stack([time_frequency_features(signal, sampling_frequency) for signal in signals])


def cwt_energy_features(signals: np.ndarray, wavelet: str = "morl") -> np.ndarray:
    """Extract multiscale energy, dominant scale, and entropy per window."""
    feature_rows: list[np.ndarray] = []
    scales = np.asarray([2, 4, 8, 16, 24, 32], dtype=np.float64)
    for signal in np.asarray(signals):
        coefficients, _ = pywt.cwt(signal, scales, wavelet)
        energy = np.mean(np.abs(coefficients) ** 2, axis=1)
        normalized = energy / max(float(energy.sum()), 1e-12)
        entropy = -float(np.sum(normalized * np.log(normalized + 1e-12)))
        dominant = float(scales[int(np.argmax(energy))]) / float(scales[-1])
        feature_rows.append(np.concatenate([np.log1p(energy), [dominant, entropy]]))
    return np.asarray(feature_rows, dtype=np.float32)


def fner(signal: np.ndarray, sampling_frequency: float, fault_frequency: float) -> float:
    """Compute a feature-to-noise energy ratio around fault harmonics."""
    values = np.asarray(signal, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sampling_frequency)
    mask = np.zeros_like(frequencies, dtype=bool)
    resolution = max(sampling_frequency / len(values), 1.0)
    for harmonic in (1.0, 2.0, 3.0):
        mask |= np.abs(frequencies - harmonic * fault_frequency) <= 2.5 * resolution
    feature_energy = float(spectrum[mask].sum())
    noise_energy = float(spectrum[~mask].mean() * max(int(mask.sum()), 1))
    return feature_energy / max(noise_energy, 1e-12)


def spectral_inverse_validation(
    signal: np.ndarray,
    sampling_frequency: float,
    fault_frequency: float,
) -> dict[str, float | bool]:
    """Run an autocorrelation, low-rank, and energy-operator validation chain."""
    values = np.asarray(signal, dtype=np.float64)
    values = values - values.mean()
    autocorrelation = np.correlate(values, values, mode="full")[len(values) - 1 :]
    autocorrelation /= max(float(np.max(np.abs(autocorrelation))), 1e-12)
    width = min(48, len(autocorrelation) // 3)
    hankel = np.lib.stride_tricks.sliding_window_view(autocorrelation, width)[:width]
    left, singular, right = np.linalg.svd(hankel, full_matrices=False)
    rank = max(1, int(np.searchsorted(np.cumsum(singular) / singular.sum(), 0.9) + 1))
    reconstructed = (left[:, :rank] * singular[:rank]) @ right[:rank]
    enhanced = reconstructed[0]
    teo = enhanced[1:-1] ** 2 - enhanced[:-2] * enhanced[2:]
    spectrum = np.abs(np.fft.rfft(teo, n=len(values)))
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sampling_frequency)
    resolution = sampling_frequency / len(values)
    ratios = []
    for harmonic in (1.0, 2.0, 3.0):
        target = harmonic * fault_frequency
        mask = np.abs(frequencies - target) <= 3.0 * resolution
        ratios.append(float(spectrum[mask].max() / max(np.median(spectrum), 1e-12)))
    return {
        "validated": bool(ratios[0] > 4.0 and max(ratios[1:]) > 1.2),
        "fundamental_ratio": ratios[0],
        "harmonic_ratio": max(ratios[1:]),
    }
