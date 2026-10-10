"""Public API for the digital-twin model-updating reference implementation."""

from .decision_methods import (
    bayesian_optimize_ewma,
    detect_local_degradation,
    ewma_statistics,
    gmm_distribution_validation,
    grey_relational_series,
    proposed_global_decision,
)
from .losses import BinaryDetectionLoss, ScaledDefectLoss
from .mechanism import mechanism_response_summary, trust_region_calibrate
from .models import AdapterClassifier, AlgorithmClassifier, DefectParameterCNN, SignalBackbone, copy_backbone_weights

__all__ = [
    "AdapterClassifier",
    "AlgorithmClassifier",
    "BinaryDetectionLoss",
    "DefectParameterCNN",
    "ScaledDefectLoss",
    "SignalBackbone",
    "bayesian_optimize_ewma",
    "copy_backbone_weights",
    "detect_local_degradation",
    "ewma_statistics",
    "gmm_distribution_validation",
    "grey_relational_series",
    "mechanism_response_summary",
    "proposed_global_decision",
    "trust_region_calibrate",
]
