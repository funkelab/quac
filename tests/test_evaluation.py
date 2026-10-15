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
    importance = torch.zeros(5, 5)
    areas, _, blurred = area_masks(importance, 6, 1)
    assert areas[1] == 0.2
    assert np.any((blurred[1] > 0) & (blurred[1] < 1))
    assert blurred.min() >= 0 and blurred.max() <= 1


def test_optimum_uses_ratio_not_difference():
    assert optimal_threshold_index([0, 0.1, 0.5, 1], [0, 0.15, 0.6, 0.8]) == 1
    assert optimal_threshold_index([0, 0.25, 0.5], [0, 0.25, 0.5]) == 1
    assert optimal_threshold_index([0, 0.25, 0.5], [0, -0.2, -0.1]) == 2


def test_sweep_predictions_and_blending():
    classifier = MeanClassifier()
    original = torch.zeros(1, 4, 4)
    generated = torch.ones_like(original)
    original_p = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[1].item()
    result = evaluate(
        classifier,
        original,
        generated,
        torch.arange(16).reshape(4, 4) / 15,
        1,
        2,
        original_p,
        levels=5,
        sigma=0,
        batch_size=2,
    )
    np.testing.assert_allclose(result.images.mean((1, 2, 3)), result.areas)
    np.testing.assert_allclose(result.logits[:, 1], result.areas)
    np.testing.assert_allclose(result.deltas, result.probabilities[:, 1] - original_p)
    np.testing.assert_array_equal(result.images[0], original)
    np.testing.assert_array_equal(result.images[-1], generated)


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
    importance = PairAttribution(MeanClassifier(), method, 4).attribute(
        torch.zeros(1, 4, 4), torch.ones(1, 4, 4), 0, 1
    )
    assert importance.shape == (4, 4)
    assert torch.isfinite(importance).all()
    assert importance.min() >= 0 and importance.max() <= 1


def test_constant_attribution():
    assert torch.equal(normalize(torch.ones(3, 4, 4)), torch.zeros(4, 4))


@pytest.mark.parametrize("method", ["DDeepLift", "VanillaDeepLift"])
def test_deeplift_eager_baselines(method):
    classifier = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(16, 2), torch.nn.ReLU()
    )
    importance = PairAttribution(classifier, method).attribute(
        torch.zeros(1, 4, 4), torch.ones(1, 4, 4), 0, 1
    )
    assert torch.isfinite(importance).all()
    assert importance.shape == (4, 4)


def test_deeplift_torchscript_error_is_actionable():
    classifier = torch.jit.script(MeanClassifier())
    with pytest.raises(ValueError, match="eager"):
        PairAttribution(classifier, "DDeepLift")
