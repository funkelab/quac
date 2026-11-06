import pytest
import torch

from quac.config import Config
from quac.data import write_image
from quac.pipeline import explain


class MeanClassifier(torch.nn.Module):
    def forward(self, images):
        value = images.mean((1, 2, 3))
        return torch.stack([1 - value, value], dim=1)


class UnitGenerator:
    def generate(self, source_images, target_classes, reference_images=None):
        return target_classes[:, None, None, None].float().expand_as(source_images)


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
                methods=["residual", "random"],
                samples=1,
                levels=5,
                sigma=0,
                closing_size=1,
            ),
            output={"directory": str(tmp_path / "output")},
        )
    )
    return explain(config, generator=UnitGenerator(), classifier=MeanClassifier())


class ThreeClassClassifier(torch.nn.Module):
    def forward(self, images):
        value = images.mean((1, 2, 3))
        return torch.stack(
            [2 * (1 - value), 1 - 2 * (value - 0.5).abs(), 2 * value], dim=1
        )


class ThreeClassGenerator:
    def generate(self, source_images, target_classes, reference_images=None):
        return (target_classes[:, None, None, None].float() / 2).expand_as(
            source_images
        )


@pytest.fixture
def multiclass_report_path(tmp_path):
    for label, name in enumerate(["a", "b", "c"]):
        for index in range(2):
            image = torch.full((1, 4, 4), 0.1 + 0.4 * label + 0.02 * index)
            write_image(image, tmp_path / "images" / name / f"{index}.tif")
    config = Config(
        data={"source": tmp_path / "images"},
        explanation={
            "methods": ["residual", "random"],
            "samples": 1,
            "levels": 5,
            "sigma": 0,
            "closing_size": 1,
        },
        output={"directory": tmp_path / "output"},
    )
    return explain(
        config, generator=ThreeClassGenerator(), classifier=ThreeClassClassifier()
    )
