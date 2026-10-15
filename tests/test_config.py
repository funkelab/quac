from pathlib import Path

import pytest
from pydantic import ValidationError

from quac.config import Config, load_config


def test_minimal_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  source: images\n")
    config = load_config(path)
    assert config.data.source == tmp_path / "images"
    assert config.output.report == tmp_path / "quac-output/report.sqlite"
    assert config.classifier.checkpoint is None
    assert config.explanation.levels == 101
    assert "img_size" not in type(config.generator).model_fields


@pytest.mark.parametrize(
    "value",
    [
        {"data": {"source": ".", "scale": 2}},
        {"data": {"source": "."}, "explanation": {"levels": 1}},
        {"data": {"source": "."}, "training": {"lr": -1}},
    ],
)
def test_invalid_config(value):
    with pytest.raises(ValidationError):
        Config.model_validate(value)
