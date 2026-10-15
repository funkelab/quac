import numpy as np
import pytest
import torch

from quac.attribution import PairAttribution, normalize
from quac.evaluation import area_masks, evaluate, optimal_threshold_index


class MeanClassifier(torch.nn.Module):
    def forward(self, images):
        value = images.mean((1, 2, 3))
        return torch.stack([1 - value, value], dim=1)


def test_area_masks_rank_ties_and_endpoints():
    attribution = torch.tensor([[0.0, 0.9], [0.9, 0.1]])
    areas, cutoffs, masks = area_masks(attribution, 5, 0)
    np.testing.assert_allclose(areas, [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_array_equal(masks[1], [[0, 1], [0, 0]])
    assert masks[0].sum() == 0 and masks[-1].sum() == 4


def test_blur_preserves_area_measurement():
    attribution_map = torch.zeros(5, 5)
    areas, _, blurred = area_masks(attribution_map, 6, 1)
    assert areas[1] == 0.2
    assert np.any((blurred[1] > 0) & (blurred[1] < 1))
    assert blurred.min() >= 0 and blurred.max() <= 1


def test_optimum_uses_ratio_not_difference():
    assert optimal_threshold_index([0, 0.1, 0.5, 1], [0, 0.15, 0.6, 0.8]) == 1
    assert optimal_threshold_index([0, 0.25, 0.5], [0, 0.25, 0.5]) == 1
    assert optimal_threshold_index([0, 0.25, 0.5], [0, -0.2, -0.1]) == 2


def test_sweep_predictions_and_blending():
    classifier = MeanClassifier()
    source_image = torch.zeros(1, 4, 4)
    generated_image = torch.ones_like(source_image)
    source_probability = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[1].item()
    result = evaluate(
        classifier,
        source_image,
        generated_image,
        torch.arange(16).reshape(4, 4) / 15,
        1,
        2,
        source_probability,
        levels=5,
        sigma=0,
        batch_size=2,
    )
    np.testing.assert_allclose(result.counterfactuals.mean((1, 2, 3)), result.areas)
    np.testing.assert_allclose(result.logits[:, 1], result.areas)
    np.testing.assert_allclose(
        result.deltas, result.probabilities[:, 1] - source_probability
    )
    np.testing.assert_array_equal(result.counterfactuals[0], source_image)
    np.testing.assert_array_equal(result.counterfactuals[-1], generated_image)


@pytest.mark.parametrize(
    "method",
    [
        "DIntegratedGradients",
        "DInGrad",
        "VanillaIntegratedGradients",
        "residual",
        "random",
    ],
)
def test_attributions_are_finite(method):
    attribution_map = PairAttribution(MeanClassifier(), method, 4).attribute(
        torch.zeros(1, 4, 4), torch.ones(1, 4, 4), 0, 1
    )
    assert attribution_map.shape == (4, 4)
    assert torch.isfinite(attribution_map).all()
    assert attribution_map.min() >= 0 and attribution_map.max() <= 1


def test_constant_attribution():
    assert torch.equal(normalize(torch.ones(3, 4, 4)), torch.zeros(4, 4))


@pytest.mark.parametrize("method", ["DDeepLift", "VanillaDeepLift"])
def test_deeplift_eager_baselines(method):
    classifier = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(16, 2), torch.nn.ReLU()
    )
    attribution_map = PairAttribution(classifier, method).attribute(
        torch.zeros(1, 4, 4), torch.ones(1, 4, 4), 0, 1
    )
    assert torch.isfinite(attribution_map).all()
    assert attribution_map.shape == (4, 4)


def test_deeplift_torchscript_error_is_actionable():
    classifier = torch.jit.script(MeanClassifier())
    with pytest.raises(ValueError, match="eager"):
        PairAttribution(classifier, "DDeepLift")
