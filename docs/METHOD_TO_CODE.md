# Method-to-Code Mapping

This document maps the paper's main computational stages to the compact public implementation. 

## Information Flow

```text
Monitoring residuals and public input arrays
    -> signal and feature operators
    -> global Bayes-EWMA candidate detection
    -> GMM inverse validation
    -> local adaptive grey relational analysis
    -> mechanism or algorithm update path
    -> bounded calibration or differentiable adapter output
```

## Global Decision Chain

- `decision_methods.ewma_statistics` computes the dynamic one-sided EWMA statistic and control limit.
- `decision_methods.bayesian_optimize_ewma` searches bounded smoothing and limit candidates with a Gaussian-process surrogate and an average-run-length objective.
- `decision_methods.gmm_distribution_validation` applies PCA, fits Gaussian-mixture models, estimates symmetric distribution divergence, and performs a two-sample KS test.
- `decision_methods.proposed_global_decision` connects candidate detection and optional inverse validation.

Direct dependencies: NumPy, SciPy, scikit-learn, `features.fner`, and `preprocessing.minmax_rows`.

## Local Decision Chain

- `features.fner` measures harmonic feature energy relative to background energy.
- `decision_methods.grey_relational_series` adjusts the grey relational distinguishing coefficient with FNER.
- `decision_methods.detect_local_degradation` applies a sustained reference-derived control limit.
- `features.spectral_inverse_validation` provides an autocorrelation, low-rank reconstruction, and energy-operator validation path for fault-frequency evidence.

## Mechanism Update

- `mechanism.mechanism_response_summary` maps damping, stiffness, defect depth, and defect width to eight response descriptors.
- `mechanism.trust_region_calibrate` estimates damping and stiffness with bounded trust-region least squares while depth and width remain observed inputs.

## Algorithm Update

- `models.SignalBackbone` extracts reusable one-dimensional convolutional features.
- `models.AlgorithmClassifier` provides the baseline anomaly-detection head.
- `models.AdapterClassifier` freezes the backbone by default and applies a compact adaptation layer and classification head.
- `models.copy_backbone_weights` transfers the learned convolutional representation.
- `models.DefectParameterCNN` estimates two defect geometry parameters.
- `losses.BinaryDetectionLoss` and `losses.ScaledDefectLoss` provide the differentiable training objectives.

