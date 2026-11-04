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
    assert config.data.validation is None and not config.wandb.enabled
    assert config.data.check_shapes and config.training.num_workers == 0


def test_validation_and_wandb_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  source: images\n  validation: held_out\n"
        "wandb:\n  enabled: true\n  mode: offline\n  tags: [demo]\n"
    )
    config = load_config(path)
    assert config.data.validation == tmp_path / "held_out"
    assert config.wandb.mode == "offline" and config.wandb.tags == ["demo"]


@pytest.mark.parametrize(
    "value",
    [
        {"data": {"source": ".", "scale": 2}},
        {"data": {"source": "."}, "explanation": {"levels": 1}},
        {"data": {"source": "."}, "training": {"lr": -1}},
        {"data": {"source": "."}, "wandb": {"mode": "cloud"}},
    ],
)
def test_invalid_config(value):
    with pytest.raises(ValidationError):
        Config.model_validate(value)
