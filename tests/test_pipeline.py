import builtins
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from quac.cli import main
from quac.config import Config
from quac.data import write_image
from quac.pipeline import explain, train
from quac.report import Report
from conftest import MeanClassifier, UnitTranslator


def test_explain_requires_explicit_classifier():
    with pytest.raises(ValueError, match="classifier.checkpoint"):
        explain(Config(data={"source": "missing"}))


def test_failed_run_records_error(tmp_path):
    for name in ["a", "b"]:
        write_image(torch.zeros(1, 4, 4), tmp_path / "images" / name / "one.tif")

    class BrokenAttribution:
        def attribute(self, *args):
            raise ValueError("bad attribution")

    config = Config(
        data={"source": tmp_path / "images"},
        explanation={"methods": ["broken"]},
        output={"directory": tmp_path / "output"},
    )
    with pytest.raises(ValueError, match="bad attribution"):
        explain(
            config,
            generator=UnitTranslator(),
            classifier=MeanClassifier(),
            attributions={"broken": BrokenAttribution()},
        )
    with Report(tmp_path / "output/report.sqlite") as report:
        assert report.status["status"] == "failed"
        assert "bad attribution" in report.status["error"]
        assert report.methods() == []


def test_overwrite_protection(report_path):
    config = Config(
        data={"source": "/nonexistent"},
        explanation={"methods": ["residual"]},
        output={"report": report_path},
    )
    with pytest.raises(FileExistsError, match="already exists"):
        explain(config, generator=UnitTranslator(), classifier=MeanClassifier())


def test_cli_config_errors_and_missing_extra(tmp_path, capsys, monkeypatch):
    missing = tmp_path / "missing.yaml"
    assert main(["train", "--config", str(missing)]) == 1
    path = tmp_path / "config.yaml"
    path.write_text("data:\n  source: images\n")
    assert main(["explain", "--config", str(path)]) == 1
    assert "classifier.checkpoint" in capsys.readouterr().err
    original_import = builtins.__import__

    def without_plotting(name, *args, **kwargs):
        if name == "plotting" or name.startswith("plotly"):
            raise ImportError("plot extra missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_plotting)
    assert main(["summarize", "--config", str(path)]) == 1
    assert "quac[plot]" in capsys.readouterr().err


def test_core_import_does_not_load_plotting():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import quac.pipeline, sys; assert 'plotly' not in sys.modules; assert 'quac.plotting' not in sys.modules; assert 'matplotlib' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_real_cli_end_to_end(tmp_path):
    torch.set_num_threads(1)
    for label, name in enumerate(["a", "b"]):
        write_image(
            torch.full((1, 13, 19), label * 0.7), tmp_path / "images" / name / "one.tif"
        )
    torch.jit.script(MeanClassifier()).save(str(tmp_path / "classifier.pt"))
    path = tmp_path / "config.yaml"
    path.write_text("""data:
  source: images
classifier:
  checkpoint: classifier.pt
generator:
  base_channels: 4
  max_channels: 8
  mapping_width: 8
  latent_dim: 2
  style_dim: 4
training:
  steps: 1
  batch_size: 1
explanation:
  methods: [residual]
  samples: 1
  levels: 5
""")
    assert main(["train", "-c", str(path)]) == 0
    assert main(["explain", "-c", str(path)]) == 0
    pytest.importorskip("plotly")
    assert main(["summarize", "-c", str(path)]) == 0
    assert (tmp_path / "quac-output/summary.html").exists()
    with Report(tmp_path / "quac-output/report.sqlite") as report:
        assert report.status["status"] == "complete"
        assert report.metadata["classifier"]["sha256"]
        assert len(report.top("residual")) == 2


def test_reference_sampling_records_reference_path(tmp_path):
    for label, name in enumerate(["a", "b"]):
        write_image(
            torch.full((1, 4, 4), float(label)), tmp_path / "images" / name / "one.tif"
        )

    class ReferenceTranslator:
        def translate(self, images, targets, reference=None):
            assert reference is not None
            assert reference.mean().item() == targets.item()
            return reference

    config = Config(
        data={"source": tmp_path / "images"},
        generator={"sampling": "reference"},
        explanation={"methods": ["residual"], "samples": 1, "levels": 3},
        output={"directory": tmp_path / "output"},
    )
    path = explain(config, generator=ReferenceTranslator(), classifier=MeanClassifier())
    with Report(path) as report:
        for row in report.top("residual"):
            assert report.pair(row["pair_id"])["reference_path"].endswith("one.tif")
