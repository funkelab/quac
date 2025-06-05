import json
from quac.explanation import Explanation
from quac.report import Report
from quac.data import SampleWithAttribution
import torch
from pathlib import Path
import pytest


@pytest.fixture
def result():
    result = {
        "counterfactual_path": Path("path/to/counterfactual"),
        "mask_path": Path("path/to/mask"),
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
        generated=torch.Tensor([4, 5, 6]),
        attribution=torch.Tensor([7, 8, 9]),
        source_class_index=0,
        target_class_index=1,
        path=Path("sample_path"),
        generated_path=Path("target_path"),
        source_class="duck",
        target_class="pigeon",
        attribution_path=Path("attribution_path"),
    )
    return inputs


@pytest.fixture
def predictions():
    predictions = {
        "original": [10, 11, 12],
        "counterfactual": [13, 14, 15],
        "generated": [16, 17, 18],
    }
    return predictions


def make_report(source_classes, target_classes, scores):
    """
    A simple report with 3 exp
    """
    report = Report("report_test")
    report.explanations = [
        Explanation(
            "query_path",
            "counterfactual_path",
            "mask_path",
            query_prediction=[0, 1],
            counterfactual_prediction=[1, 0],
            source_class=i,
            target_class=j,
            score=k,
        )
        for i, j, k in zip(source_classes, target_classes, scores)
    ]
    return report


def test_report(inputs, predictions, result, tmpdir):
    """Make sure that the report class accumulates correctly."""
    report = Report("report_test")
    for _ in range(10):
        report.accumulate(inputs, predictions, result)

    assert len(report) == 10

    tmpdir = Path(tmpdir)
    report.store(tmpdir)
    assert (tmpdir / f"{report.name}.json").exists(), "The files in {} are: {}".format(
        tmpdir, list(tmpdir.iterdir())
    )
    # Load the report back
    loaded_report = Report()
    loaded_report.load(tmpdir / f"{report.name}.json")
    # Check that the loaded report is the same as the original
    assert loaded_report.explanations[0] == report.explanations[0], (
        f"Loaded report: {loaded_report.explanations[0]} != {report.explanations[0]}"
    )


def test_bad_report_io(tmp_path):
    bad_json = {
        # Missing "thresholds" key
        "mask_sizes": [10, 20, 30],
        "score_change": [0.1, 0.2],  # Mismatched length
        "hybrids": [1, 2, 3],
        "extra_field": "unexpected",
        "unexpected_list": [1, 2, 3],
    }
    with open(tmp_path / "bad_report.json", "w") as f:
        json.dump(bad_json, f)
    with pytest.raises(KeyError):
        report = Report()
        report.load(tmp_path / "bad_report.json")


def test_report_filtering():
    """Check that filtering works correctly."""
    report = make_report(
        [0, 0, 1, 1, 2, 2],  # source classes
        [1, 2, 0, 2, 0, 1],  # target classes
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # scores
    )

    # filter by source class
    filtered_report = report.from_source(0)
    # check
    assert all(
        [explanation.source_class == 0 for explanation in filtered_report.explanations]
    ), "Filtering by source class failed"

    # filter by target class
    filtered_report = report.to_target(1)
    # check
    assert all(
        [explanation.target_class == 1 for explanation in filtered_report.explanations]
    ), "Filtering by target class failed"

    # threshold scores
    filtered_report = report.score_threshold(0.3)
    # check
    assert all(
        [explanation.score >= 0.3 for explanation in filtered_report.explanations]
    ), "Filtering by score threshold failed"

    # top n
    filtered_report = report.top_n(2)
    # check that the values are 0.5 and 0.6
    assert {explanation.score for explanation in filtered_report.explanations} == {
        0.5,
        0.6,
    }, "Filtering by top n failed"

    # Check that the original report is unchanged
    assert len(report.explanations) == 6, "Original report was modified"
