"""Forward-path tests for the public neural components."""

import torch

from dt_model_updating import AdapterClassifier, AlgorithmClassifier, DefectParameterCNN, copy_backbone_weights


def test_classifier_and_adapter_forward() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(5, 128)
    source = AlgorithmClassifier()
    target = AdapterClassifier()
    copy_backbone_weights(source, target)

    source_logits = source(inputs)
    target_logits = target(inputs)

    assert source_logits.shape == (5,)
    assert target_logits.shape == (5,)
    assert torch.isfinite(source_logits).all()
    assert torch.isfinite(target_logits).all()
    assert not any(parameter.requires_grad for parameter in target.backbone.parameters())


def test_defect_regressor_forward() -> None:
    outputs = DefectParameterCNN()(torch.randn(4, 128))
    assert outputs.shape == (4, 2)
    assert torch.isfinite(outputs).all()
