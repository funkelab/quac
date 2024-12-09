from quac.report import Report
from quac.data import SampleWithAttribution
import torch
from pathlib import Path
import pytest
import matplotlib.pyplot as plt


@pytest.fixture
def result():
    result = {
        "thresholds": [0.1, 0.2, 0.3],
        "mask_sizes": [10, 20, 30],
        "score_change": [0.1, 0.2, 0.3],
        "hybrids": [1, 2, 3],
    }
    return result


@pytest.fixture
def inputs():
    inputs = SampleWithAttribution(
        image=torch.Tensor([1, 2, 3]),
        counterfactual=torch.Tensor([4, 5, 6]),
        attribution=torch.Tensor([7, 8, 9]),
        source_class_index=0,
        target_class_index=1,
        path=Path("sample_path"),
        counterfactual_path=Path("target_path"),
        source_class="duck",
        target_class="pigeon",
        attribution_path=Path("attribution_path"),
    )
    return inputs


@pytest.fixture
def predictions():
    predictions = {"original": [10, 11, 12], "counterfactual": [13, 14, 15]}
    return predictions


def test_report(inputs, predictions, result, tmpdir):
    """Make sure that the report class accumulates correctly."""
    report = Report("report_test")
    for _ in range(10):
        report.accumulate(inputs, predictions, result)

    assert len(report.thresholds) == 10
    assert len(report.normalized_mask_sizes) == 10
    assert len(report.score_changes) == 10

    # This should test the rest
    fig, ax = plt.subplots()
    report.plot_curve(ax)
    plt.close(fig)

    tmpdir = Path(tmpdir)
    report.store(tmpdir)
    assert (tmpdir / f"{report.name}.json").exists(), "The files in {} are: {}".format(
        tmpdir, list(tmpdir.iterdir())
    )
    # TODO Move this to a teardown?
    # remove the file
    (tmpdir / f"{report.name}.json").unlink()
    # Remove the directory
    tmpdir.rmdir()
