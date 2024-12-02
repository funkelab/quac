from quac.report import Report
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
    inputs = {
        "sample": [1, 2, 3],
        "target_sample": [4, 5, 6],
        "class_index": 0,
        "target_class_index": 1,
        "sample_path": "sample_path",
        "target_path": "target_path",
    }
    return inputs


@pytest.fixture
def predictions():
    predictions = {"original": [10, 11, 12], "counterfactual": [13, 14, 15]}
    return predictions


@pytest.fixture
def attribution():
    return [1, 2, 3]


def test_report(inputs, predictions, attribution, result):
    """Make sure that the report class accumulates correctly."""
    report = Report("report_test_dir_delete_me")
    for _ in range(10):
        report.accumulate(
            inputs,
            predictions,
            attribution,
            result,
            save_attribution=False,
            save_intermediates=False,
        )

    assert len(report.thresholds) == 10
    assert len(report.normalized_mask_sizes) == 10
    assert len(report.score_changes) == 10

    # This should test the rest
    fig, ax = plt.subplots()
    report.plot_curve(ax)

    report.store()
    assert (report.save_dir / "attribution.json").exists()
    # TODO Move this to a teardown?
    # remove the file
    (report.save_dir / "attribution.json").unlink()
    # Remove the directory
    report.save_dir.rmdir()
