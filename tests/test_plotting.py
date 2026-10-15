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
    assert slider.steps[0].args[2] == (2,)
    assert len(slider.steps) == 5
    assert "annotations[2].text" in slider.steps[0].args[1]
    assert len(figure.layout.updatemenus) == 3
    assert [len(menu.buttons) for menu in figure.layout.updatemenus] == [2, 2, 1]
    assert len(figure.data) == 6
    assert figure.data[-2].fill == "tonexty"


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
    assert not missing.data[-1].x and not missing.data[-1].y


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
            np.testing.assert_allclose(frame.data[-1].y, expected_curve)
            assert len(frame.layout.sliders) == 2
