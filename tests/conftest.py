import pytest
import torch

from quac.config import Config
from quac.data import write_image
from quac.pipeline import explain


class MeanClassifier(torch.nn.Module):
    def forward(self, images):
        value = images.mean((1, 2, 3))
        return torch.stack([1 - value, value], dim=1)


class UnitTranslator:
    def translate(self, images, targets, reference=None):
        return targets[:, None, None, None].float().expand_as(images)


@pytest.fixture
def report_path(tmp_path):
    for label, name in enumerate(["a", "b"]):
        write_image(
            torch.full((1, 4, 4), float(label)), tmp_path / "images" / name / "one.tif"
        )
    config = Config.model_validate(
        dict(
            data={"source": str(tmp_path / "images")},
            explanation=dict(
                methods=["residual", "random"], samples=1, levels=5, sigma=0
            ),
            output={"directory": str(tmp_path / "output")},
        )
    )
    return explain(config, generator=UnitTranslator(), classifier=MeanClassifier())
