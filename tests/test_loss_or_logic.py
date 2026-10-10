"""Numerical tests for losses, decision logic, and mechanism calibration."""

import numpy as np
import torch

from dt_model_updating import (
    BinaryDetectionLoss,
    ScaledDefectLoss,
    mechanism_response_summary,
    proposed_global_decision,
    trust_region_calibrate,
)


def test_losses_are_finite_and_differentiable() -> None:
    logits = torch.tensor([-0.5, 0.8], requires_grad=True)
    binary_loss = BinaryDetectionLoss()(logits, torch.tensor([0.0, 1.0]))
    binary_loss.backward()
    assert torch.isfinite(binary_loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    predictions = torch.tensor([[0.3, 1.8]], requires_grad=True)
    regression_loss = ScaledDefectLoss()(predictions, torch.tensor([[0.4, 2.0]]))
    regression_loss.backward()
    assert torch.isfinite(regression_loss)
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()


def test_global_decision_runs_on_synthetic_drift() -> None:
    generator = np.random.default_rng(17)
    errors = np.abs(generator.normal(0.03, 0.005, 72))
    errors[44:] += 0.3
    features = generator.normal(0.0, 0.1, (72, 5))
    features[44:] += 2.5
    result = proposed_global_decision(
        errors,
        features,
        reference_count=28,
        window=12,
        consecutive=3,
        seed=17,
        require_gmm=True,
        optimize_parameters=False,
    )
    assert 28 <= result["detection_index"] <= len(errors)
    assert np.isfinite(result["kl_divergence"])


def test_trust_region_mechanism_calibration_reduces_objective() -> None:
    known = np.asarray([3725.0, 5.7e4, 0.5, 2.2])
    target = mechanism_response_summary(known)
    initial = np.asarray([3376.0, 5.24e4, known[2], known[3]])
    initial_scale = np.maximum(np.abs(target), 1e-6)
    initial_objective = float(np.mean(((mechanism_response_summary(initial) - target) / initial_scale) ** 2))
    result = trust_region_calibrate(target, depth=known[2], width=known[3])
    assert result.objective < initial_objective
    assert 2500.0 <= result.damping <= 4500.0
    assert 3.0e4 <= result.stiffness <= 6.5e4
