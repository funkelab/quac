from pathlib import Path

import pytest

pytest.importorskip("plotly")
from quac.plotting import summarize, write_summary
from quac.report import Report


def test_summary_contains_independent_sliders_and_method_frames(report_path):
    figure = summarize(report_path, top_k=5)
    assert len(figure.frames) == 2
    assert len(figure.layout.sliders) == 2
    first, second = figure.layout.sliders
    assert first.steps[0].args[2] == (2,)
    assert second.steps[0].args[2] == (5,)
    assert len(first.steps) == len(second.steps) == 5
    assert "annotations[2].text" in first.steps[0].args[1]
    assert "annotations[5].text" in second.steps[0].args[1]
    assert len(figure.layout.updatemenus[0].buttons) == 2
    assert len(figure.data) == 9  # Six images, percentile bounds, and median.
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


def test_methods_with_different_exemplar_counts(report_path):
    import sqlite3

    database = sqlite3.connect(report_path)
    row = database.execute(
        "SELECT id FROM explanation WHERE method='residual' ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]
    database.execute("DELETE FROM threshold WHERE explanation_id=?", (row,))
    database.execute("DELETE FROM explanation WHERE id=?", (row,))
    database.commit()
    database.close()
    figure = summarize(report_path, top_k=5)
    frames = {frame.name: frame for frame in figure.frames}
    assert len(frames["random"].layout.sliders) == 2
    assert sum(slider.visible for slider in frames["residual"].layout.sliders) == 1
    assert frames["residual"].layout.annotations[3].text == "No exemplar"
