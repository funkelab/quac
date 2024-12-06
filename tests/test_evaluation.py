import pytest
from quac.evaluation import Evaluator
from funlib.learn.torch.models import Vgg2D
import numpy as np


@pytest.mark.skip("Skip test until we have usable data in CI/CD")
@pytest.mark.parametrize("input_shape", [(1, 128, 128)])
@pytest.mark.parametrize("num_thresholds", [50, 100])
def test_evaluator(input_shape, num_thresholds):
    """Checks that evaluator runs without errors"""
    x_real = np.random.rand(*input_shape)
    x_fake = np.random.rand(*input_shape)
    y_real = 0
    y_fake = 1
    attribution = np.random.rand(*input_shape)
    vmin = -1
    vmax = 1
    classifier = Vgg2D(
        (input_shape[1], input_shape[2]), output_classes=2, input_fmaps=input_shape[0]
    )

    predictions = {
        "original": np.random.rand(2),
        "counterfactual": np.random.rand(2),
    }

    evaluator = Evaluator(classifier, num_thresholds=num_thresholds)
    results = evaluator.evaluate(
        x_real,
        x_fake,
        y_real,
        y_fake,
        attribution,
        predictions,
        vmin,
        vmax,
    )
    assert "thresholds" in results
    assert len(results["thresholds"]) == num_thresholds
    assert "hybrids" in results
    assert results["hybrids"][0].shape == input_shape
    assert "mask_sizes" in results
    assert len(results["mask_sizes"]) == num_thresholds
