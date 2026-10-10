# Model Updating Approach for Digital Twin-Driven Industrial Equipment Monitoring

[Bin Xiao's GitHub profile](https://github.com/XiaoBinSJTU)

## Paper Information

**Bin Xiao, Liang Chen, Xiaofeng Zhou, Jingshu Zhong, Zixing Wang, Siqi Qiu, and Yu Zheng.**  
*Model updating approach for digital twin-driven industrial equipment monitoring.*  
Advanced Engineering Informatics, 74, 104678, 2026.  
[https://doi.org/10.1016/j.aei.2026.104678](https://doi.org/10.1016/j.aei.2026.104678)

## Graphical Abstract

![Graphical Abstract](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/04-dt-model-updating.svg)

## Research Highlights

This reference implementation exposes the degradation-aware decision chain and collaborative digital-twin model update in reusable modules. It combines Bayes-tuned EWMA monitoring, GMM inverse validation, adaptive grey relational reasoning, bounded mechanism calibration, and neural feature adaptation. A deterministic synthetic smoke test exercises genuine forward, loss, backward, decision, and optimization paths without releasing industrial observations or claiming paper-level experimental reproduction.

## Method Overview

### 1. Global degradation-aware update decision

Monitoring residuals are processed by an exponentially weighted moving average (EWMA). The smoothing and control-limit parameters can be selected with bounded Gaussian-process optimization over an average-run-length objective. A candidate change is then checked through projected Gaussian-mixture distributions and a two-sample test.

### 2. Local degradation identification

The local decision path computes grey relational degrees between simulated and measured windows. Its distinguishing coefficient adapts to the feature-to-noise energy ratio, allowing the relation score to respond to changing signal quality before sustained degradation is declared.

### 3. Mechanism-model calibration

A reduced response descriptor connects damping, stiffness, defect depth, and defect width. Bounded trust-region least squares estimates the mechanical parameters from a target descriptor while preserving a transparent numerical residual path.

### 4. Algorithm-model adaptation

One-dimensional convolutional networks support anomaly detection and defect-parameter regression. The update model can reuse a learned backbone, freeze its convolutional features, and train a compact adapter and decision head. The included losses preserve differentiable detection and scaled physical-parameter objectives.

## Method-to-Code Map

| Paper method element | Public implementation | Main entry point |
| --- | --- | --- |
| Bayes-EWMA monitoring and GMM inverse validation | `src/dt_model_updating/decision_methods.py` | `proposed_global_decision` |
| Adaptive degradation grey incidence analysis | `src/dt_model_updating/decision_methods.py` | `grey_relational_series` |
| Time, frequency, CWT, FNER, and inverse-validation features | `src/dt_model_updating/features.py` | `time_frequency_features`, `cwt_energy_features`, `spectral_inverse_validation` |
| Bounded mechanism update | `src/dt_model_updating/mechanism.py` | `trust_region_calibrate` |
| Baseline, adapter, and defect-regression networks | `src/dt_model_updating/models.py` | `AlgorithmClassifier`, `AdapterClassifier`, `DefectParameterCNN` |
| Detection and scaled defect losses | `src/dt_model_updating/losses.py` | `BinaryDetectionLoss`, `ScaledDefectLoss` |
| Input normalization and stratified partition utilities | `src/dt_model_updating/preprocessing.py` | `standardize_from_training`, `stratified_indices` |

See [docs/METHOD_TO_CODE.md](docs/METHOD_TO_CODE.md) for the detailed information flow.

## Quick Start

Python 3.10 or newer is recommended. From Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python examples\minimal_demo.py
python -m pytest -q
```

The demo uses deterministic, synthetic inputs for demonstration only. It selects a compatible GPU first when available and otherwise runs on CPU. Successful output confirms import, forward, loss, backward, decision, and calibration paths; it does not report an accuracy, AUC, F1 score, or an industrial validation result.

## Citation and Paper Link

Citation metadata is provided in [CITATION.cff](CITATION.cff). The published article is available through the [paper DOI](https://doi.org/10.1016/j.aei.2026.104678).

## Author

For related digital-twin research, visit [XiaoBinSJTU](https://github.com/XiaoBinSJTU).
