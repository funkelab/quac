import pytest
import torch

from quac.classifier import predict


class SpyClassifier(torch.nn.Module):
    def forward(self, image):
        assert image.min() >= 0 and image.max() <= 1
        return image.mean((2, 3))[:, :2]


def test_classifier_receives_unit_range_without_rescaling():
    images = torch.rand(5, 3, 8, 9)
    logits, probabilities = predict(SpyClassifier(), images, 2, batch_size=2)
    torch.testing.assert_close(logits, images.mean((2, 3))[:, :2])
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(5))
    with pytest.raises(ValueError, match="logits"):
        predict(SpyClassifier(), images, 3)
