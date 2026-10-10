"""Run a small, genuine core-algorithm path on synthetic demonstration inputs."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*", category=UserWarning)

from dt_model_updating import (  # noqa: E402
    AlgorithmClassifier,
    BinaryDetectionLoss,
    DefectParameterCNN,
    ScaledDefectLoss,
    mechanism_response_summary,
    proposed_global_decision,
    trust_region_calibrate,
)
from dt_model_updating.device import release_device, select_device, set_global_seed  # noqa: E402


def main() -> None:
    """Validate forward, loss, backward, decision, and calibration paths."""
    set_global_seed(2026)
    device, device_label = select_device("auto")

    inputs = torch.randn(8, 128, device=device)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32, device=device)
    classifier = AlgorithmClassifier().to(device)
    logits = classifier(inputs)
    classification_loss = BinaryDetectionLoss().to(device)(logits, labels)
    classification_loss.backward()

    regressor = DefectParameterCNN().to(device)
    defect_targets = torch.tensor([[0.2, 1.5], [0.4, 2.0], [0.6, 2.5], [0.8, 3.0]], device=device)
    defect_predictions = regressor(inputs[:4])
    regression_loss = ScaledDefectLoss().to(device)(defect_predictions, defect_targets)
    regression_loss.backward()

    generator = np.random.default_rng(2026)
    errors = np.abs(generator.normal(0.04, 0.008, 80))
    errors[50:] += 0.25
    features = generator.normal(0.0, 0.15, (80, 6))
    features[50:] += 2.0
    decision = proposed_global_decision(
        errors,
        features,
        reference_count=32,
        window=12,
        consecutive=3,
        seed=2026,
        require_gmm=True,
        optimize_parameters=False,
    )

    known = np.asarray([3600.0, 5.5e4, 0.45, 2.4])
    target_summary = mechanism_response_summary(known)
    calibration = trust_region_calibrate(target_summary, depth=known[2], width=known[3])

    assert logits.shape == (8,) and torch.isfinite(logits).all()
    assert defect_predictions.shape == (4, 2) and torch.isfinite(defect_predictions).all()
    assert torch.isfinite(classification_loss) and torch.isfinite(regression_loss)
    assert 0 <= decision["detection_index"] <= len(errors)
    assert np.isfinite(calibration.objective)

    print("Synthetic inputs for demonstration only")
    print(f"Device mode: {device_label}")
    print("Classifier forward/loss/backward: passed")
    print("Defect-regression forward/loss/backward: passed")
    print(f"Decision chain: passed (detected index {decision['detection_index']})")
    print(f"Mechanism calibration: passed (objective {calibration.objective:.3e})")
    release_device(device)


if __name__ == "__main__":
    main()
