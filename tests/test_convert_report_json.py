import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from quac.report import Report

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_report_json.py"


@pytest.fixture
def json_report(tmp_path):
    source = np.zeros((4, 4), dtype=np.float32)
    generated = np.ones_like(source)
    mask = np.zeros_like(source)
    mask[:2] = 1
    for name, image in (
        ("source", source),
        ("generated", generated),
        ("counterfactual", mask),
    ):
        tifffile.imwrite(tmp_path / f"{name}.tif", image)
    np.save(tmp_path / "mask.npy", mask[None])
    np.save(tmp_path / "attribution.npy", mask[None])
    record = dict(
        query_path="source.tif",
        generated_path="generated.tif",
        counterfactual_path="counterfactual.tif",
        mask_path="mask.npy",
        attribution_path="attribution.npy",
        query_prediction=[0.9, 0.1],
        generated_prediction=[0.1, 0.9],
        counterfactual_prediction=[0.2, 0.8],
        source_class=0,
        target_class=1,
        score=0.55,
        normalized_mask_sizes=[1, 0.5, 0],
        score_changes=[0.8, 0.7, 0],
        optimal_threshold=0.1,
        method=None,
    )
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "name": "default_report",
                "metadata": {},
                "results": [record, {**record, "method": "random"}],
            }
        )
    )
    return path


def test_convert_json_preserves_curves_and_available_images(json_report):
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(json_report),
            "--method",
            "residual",
            "--classes",
            "a",
            "b",
        ],
        check=True,
        cwd=json_report.parent,
    )
    path = json_report.parent / "quac-output" / "report.sqlite"
    with Report(path) as report:
        assert report.status["status"] == "complete"
        assert report.metadata["classes"] == ["a", "b"]
        assert report.methods() == ["random", "residual"]
        row = report.top("residual")[0]
        assert report.connection.execute("SELECT count(*) FROM pair").fetchone()[0] == 1
        assert row["auc"] == 0.55
        sweep = report.thresholds(row["id"])
        assert len(sweep) == 1 and sweep[0]["area"] == 0.5
        assert np.isnan(sweep[0]["logits"]).all()
        np.testing.assert_allclose(sweep[0]["probabilities"], [0.2, 0.8])
        assert (path.parent / sweep[0]["mask_path"]).is_file()
        np.testing.assert_allclose(report.curve("residual")[1][::50], [0, 0.7, 0.8])
        assert report.example_curves(row["pair_id"])["random"]["areas"] == [0, 0.5, 1]
    from quac.plotting import summarize

    figure = summarize(path, 1)
    assert len(figure.layout.sliders[0].steps) == 1
    assert "nan" not in figure.layout.annotations[2].text
    with pytest.raises(FileExistsError):
        runpy.run_path(str(SCRIPT))["convert_report"](json_report)


def test_failed_conversion_does_not_publish_partial_report(json_report):
    (json_report.parent / "generated.tif").unlink()
    with pytest.raises(ValueError, match="Result 0"):
        runpy.run_path(str(SCRIPT))["convert_report"](json_report)
    output = json_report.parent / "quac-output"
    assert not (output / "report.sqlite").exists()
    assert not list(output.rglob("*.tif"))


@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (2.0, -1.0)])
def test_reconstructs_complete_sweep_without_saved_counterfactual(
    json_report, scale, shift
):
    import torch
    from conftest import MeanClassifier

    pytest.importorskip("cv2")
    model = MeanClassifier().eval()
    checkpoint = json_report.parent / "classifier.pt"
    torch.jit.trace(model, torch.zeros(1, 1, 4, 4)).save(str(checkpoint))
    source = np.zeros((1, 4, 4), dtype=np.float32)
    generated = np.ones_like(source)
    attribution = np.load(json_report.parent / "attribution.npy")[0]
    thresholds = np.arange(-1, 1, 2 / 5)
    masks = np.stack([attribution > cutoff for cutoff in thresholds]).astype(np.float32)
    images = source[None] * (1 - masks[:, None]) + generated[None] * masks[:, None]
    with torch.no_grad():
        pair_logits = model(torch.tensor(np.stack([source, generated])) * scale + shift)
        pair_probabilities = pair_logits.softmax(1).numpy()
        logits = model(torch.tensor(images) * scale + shift).numpy()
        probabilities = torch.tensor(logits).softmax(1).numpy()
    areas = masks.mean(axis=(1, 2))
    deltas = probabilities[:, 1] - pair_probabilities[0, 1]
    optimum = int(np.argmin(areas - deltas))
    data = json.loads(json_report.read_text())
    for record in data["results"]:
        record.update(
            query_prediction=pair_probabilities[0].tolist(),
            generated_prediction=pair_probabilities[1].tolist(),
            counterfactual_prediction=probabilities[optimum].tolist(),
            normalized_mask_sizes=areas.tolist(),
            score_changes=deltas.tolist(),
            optimal_threshold=float(thresholds[optimum]),
        )
    json_report.write_text(json.dumps(data))
    (json_report.parent / "counterfactual.tif").unlink()
    (json_report.parent / "mask.npy").unlink()
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(json_report),
            "--classifier-checkpoint",
            str(checkpoint),
            "--method",
            "residual",
            "--closing-size",
            "1",
            "--blur-size",
            "1",
            "--batch-size",
            "2",
            "--scale",
            str(scale),
            "--shift",
            str(shift),
        ],
        check=True,
    )
    output = json_report.parent / "quac-output" / "report.sqlite"
    with Report(output) as report:
        row = report.top("residual")[0]
        pair = report.pair(row["pair_id"])
        np.testing.assert_allclose(pair["source_logits"], pair_logits[0])
        np.testing.assert_allclose(pair["generated_logits"], pair_logits[1])
        sweep = report.thresholds(row["id"])
        assert len(sweep) == 5
        for item, index in zip(sweep, np.argsort(areas, kind="stable")):
            np.testing.assert_allclose(item["mask"], masks[index])
            np.testing.assert_allclose(item["counterfactual"], images[index])
            np.testing.assert_allclose(item["logits"], logits[index])
            np.testing.assert_allclose(item["probabilities"], probabilities[index])
        np.testing.assert_allclose(
            report.example_curves(row["pair_id"])["residual"]["deltas"],
            [item["delta"] for item in sweep],
        )
    from quac.plotting import summarize

    assert len(summarize(output, 1).layout.sliders[0].steps) == 5
    module = runpy.run_path(str(SCRIPT))
    bad_reconstructor = module["Reconstructor"](
        checkpoint, scale=scale, shift=shift + 1, closing_size=1, blur_size=1
    )
    with pytest.raises(ValueError, match="Classifier disagrees"):
        module["convert_report"](
            json_report,
            json_report.parent / "bad.sqlite",
            reconstructor=bad_reconstructor,
        )
    assert not (json_report.parent / "bad.sqlite").exists()
