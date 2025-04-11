from quac.training.classification import ClassifierWrapper
import torch
import pytest


class DummyModel(torch.nn.Module):
    """
    Dummy model that expects a certain data range, and returns a boolean indicating if the data is in range.
    """

    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return (x >= self.min) & (x <= self.max)


@pytest.mark.parametrize(
    "min, max, scale, shift",
    [
        (0, 1, 1, 0),
        (-1, 1, 2, -1),
    ],
)
def test_wrapper_rand(min, max, scale, shift):
    model = DummyModel(min, max)
    classifier = ClassifierWrapper(model, scale=scale, shift=shift)
    tensor = torch.rand((10, 3))
    assert classifier(tensor).all()


@pytest.mark.parametrize(
    "min, max, scale, shift",
    [
        (0, 1, 0.5, 0.5),
        (-1, 1, 1, 0),
    ],
)
def test_wrapper_randn(min, max, scale, shift):
    model = DummyModel(min, max)
    classifier = ClassifierWrapper(model, scale=scale, shift=shift)
    tensor = torch.randn((10, 3))
    tensor = torch.clamp(tensor, -1, 1)
    assert classifier(tensor).all()
