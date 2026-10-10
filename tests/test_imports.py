"""Import checks for the public package."""


def test_public_api_imports() -> None:
    from dt_model_updating import (
        AdapterClassifier,
        AlgorithmClassifier,
        BinaryDetectionLoss,
        DefectParameterCNN,
        proposed_global_decision,
        trust_region_calibrate,
    )

    assert all(
        item is not None
        for item in (
            AdapterClassifier,
            AlgorithmClassifier,
            BinaryDetectionLoss,
            DefectParameterCNN,
            proposed_global_decision,
            trust_region_calibrate,
        )
    )
