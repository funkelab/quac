from pathlib import Path

import pytest

pytest.importorskip("plotly")
from quac.plotting import summarize, write_summary
from quac.report import Report


def test_summary_contains_direction_and_method_selectors(report_path):
    figure = summarize(report_path, top_k=5)
    assert len(figure.frames) == 4
    assert len(figure.layout.sliders) == 1
    slider = figure.layout.sliders[0]
    assert slider.steps[0].args[2] == (2, 6)
    assert len(slider.steps) == 5
    assert "annotations[2].text" in slider.steps[0].args[1]
    assert len(figure.layout.updatemenus) == 3
    assert [len(menu.buttons) for menu in figure.layout.updatemenus] == [2, 2, 1]
    assert len(figure.data) == 9
    assert figure.data[4].fill == "tonexty"


def test_example_curves_scores_and_mask_area_markers(multiclass_report_path):
    import numpy as np

    figure = summarize(multiclass_report_path, top_k=2)
    with Report(multiclass_report_path) as report:
        methods = report.methods()
        for frame in figure.frames:
            selection = frame.layout.meta
            rows = report.top(
                selection["method"],
                2,
                selection["source_class"],
                selection["target_class"],
            )
            for slot, row in enumerate(rows):
                for index, method in enumerate(methods):
                    explanation = report.connection.execute(
                        "SELECT id, auc FROM explanation WHERE pair_id=? AND method=?",
                        (row["pair_id"], method),
                    ).fetchone()
                    records = report.thresholds(explanation["id"])
                    trace = frame.data[11 + slot * len(methods) + index]
                    assert trace.name == method
                    assert trace.xaxis == f"x{slot * 4 + 4}"
                    np.testing.assert_allclose(trace.x, [r["area"] for r in records])
                    np.testing.assert_allclose(trace.y, [r["delta"] for r in records])
                    assert np.isclose(
                        np.trapezoid(trace.y, trace.x), explanation["auc"]
                    )
                    assert (
                        f"QuAC score: {explanation['auc']:.4f}" in trace.hovertemplate
                    )
                assert (
                    f"QuAC score: {row['auc']:.4f}"
                    in frame.layout.annotations[7 + slot].text
                )
                records = report.thresholds(row["id"])
                marker = frame.layout.shapes[slot * 4 + 3]
                assert marker.x0 == marker.x1 == records[row["optimal_index"]]["area"]
                assert marker.xref == f"x{slot * 4 + 4}"
                for record, step in zip(records, frame.layout.sliders[slot].steps):
                    assert step.args[1][f"shapes[{slot * 4 + 3}].x0"] == record["area"]
                    assert step.args[1][f"shapes[{slot * 4 + 3}].x1"] == record["area"]


def test_html_is_self_contained(report_path):
    # Remove the source image dataset; the report references copies under output/images.
    for image in report_path.parent.parent.glob("images/*/*.tif"):
        image.unlink()
    path = write_summary(report_path, report_path.with_suffix(".html"), top_k=1)
    html = path.read_text()
    assert "Plotly.newPlot" in html
    assert "Mask area" in html and "logit=" in html
    assert '<script src="' not in html
    assert len(html) > 1_000_000


def test_empty_report_and_invalid_k(tmp_path):
    path = tmp_path / "empty.sqlite"
    with Report.create(path, {}) as report:
        report.finish()
    with pytest.raises(ValueError, match="no completed"):
        summarize(path)
    with pytest.raises(ValueError, match="positive"):
        summarize(path, top_k=0)


def test_direction_without_explanations_is_empty(report_path):
    import sqlite3

    database = sqlite3.connect(report_path)
    row = database.execute(
        "SELECT e.id, p.source_class, p.target_class FROM explanation e "
        "JOIN pair p ON p.id=e.pair_id WHERE e.method='residual' ORDER BY e.id DESC LIMIT 1"
    ).fetchone()
    database.execute("DELETE FROM threshold WHERE explanation_id=?", (row[0],))
    database.execute("DELETE FROM explanation WHERE id=?", (row[0],))
    database.commit()
    database.close()
    figure = summarize(report_path, top_k=5)
    missing = next(
        frame
        for frame in figure.frames
        if frame.layout.meta
        == {"method": "residual", "source_class": row[1], "target_class": row[2]}
    )
    assert not any(slider.visible for slider in missing.layout.sliders)
    assert missing.layout.annotations[0].text == "No exemplar"
    assert "No explanations" in missing.layout.annotations[3].text
    assert not missing.data[5].x and not missing.data[5].y


def test_multiclass_summary_covers_every_direction(multiclass_report_path):
    import numpy as np

    figure = summarize(multiclass_report_path, top_k=2)
    expected = {
        (method, source, target)
        for method in ("random", "residual")
        for source in range(3)
        for target in range(3)
        if source != target
    }
    actual = {
        (
            frame.layout.meta["method"],
            frame.layout.meta["source_class"],
            frame.layout.meta["target_class"],
        )
        for frame in figure.frames
    }
    assert actual == expected
    with Report(multiclass_report_path) as report:
        for frame in figure.frames:
            selection = frame.layout.meta
            source, target = selection["source_class"], selection["target_class"]
            assert (
                f"{report.metadata['classes'][source]} → {report.metadata['classes'][target]}"
                in frame.layout.title.text
            )
            assert [button.label for button in frame.layout.updatemenus[2].buttons] == [
                name
                for index, name in enumerate(report.metadata["classes"])
                if index != source
            ]
            expected_curve = report.curve(selection["method"], source, target)[1]
            np.testing.assert_allclose(frame.data[8].y, expected_curve)
            assert len(frame.layout.sliders) == 2
