import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("QUAC_BROWSER_TESTS") != "1",
    reason="Set QUAC_BROWSER_TESTS=1 and install Playwright/Chromium",
)


def test_independent_sliders_and_method_selection(multiclass_report_path):
    report_path = multiclass_report_path
    from playwright.sync_api import sync_playwright
    from quac.plotting import write_summary

    html = write_summary(report_path, report_path.with_suffix(".html"), top_k=2)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1400})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(html.as_uri())
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot')?._fullLayout?.sliders?.length === 2"
        )
        for label in ("Attribution method", "Source class", "Target class"):
            box = page.get_by_text(label, exact=True).bounding_box()
            assert box is not None and 0 < box["y"] < 150
        initial = page.evaluate("""() => {
            const plot = document.querySelector('.js-plotly-plot');
            return {first: JSON.stringify(plot.data[2].z), second: JSON.stringify(plot.data[5].z),
                    text: plot.layout.annotations[2].text,
                    marker: plot.layout.shapes[7].x0,
                    curves: JSON.stringify(plot.data.slice(11).map(trace => [trace.x, trace.y]))};
        }""")
        rail = page.locator(".slider-rail-touch-rect").first
        box = rail.bounding_box()
        page.mouse.click(box["x"] + 1, box["y"] + box["height"] / 2)
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.sliders[0].active === 0"
        )
        page.wait_for_function(
            """previous => {
            const plot = document.querySelector('.js-plotly-plot');
            return JSON.stringify(plot.data[2].z) !== previous.first &&
                   plot.layout.annotations[2].text.includes('area=0.000');
        }""",
            arg=initial,
        )
        after = page.evaluate("""() => {
            const plot = document.querySelector('.js-plotly-plot');
            return {first: JSON.stringify(plot.data[2].z), second: JSON.stringify(plot.data[5].z),
                    text: plot.layout.annotations[2].text};
        }""")
        assert after["first"] != initial["first"]
        assert after["second"] == initial["second"]
        assert after["text"] != initial["text"]
        assert "area=0.000" in after["text"]
        assert page.evaluate(
            """previous => {
            const plot = document.querySelector('.js-plotly-plot');
            return plot.layout.shapes[3].x0 === 0 && plot.layout.shapes[3].x1 === 0 &&
                   plot.layout.shapes[7].x0 === previous.marker &&
                   JSON.stringify(plot.data.slice(11).map(trace => [trace.x, trace.y])) === previous.curves;
        }""",
            initial,
        )
        # Switch methods using the actual dropdown; the new method resets its own sliders.
        page.locator(".updatemenu-header").nth(0).click()
        page.locator(".updatemenu-dropdown-button").filter(has_text="residual").click()
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.title.text.includes('residual')"
        )
        assert (
            page.evaluate(
                "document.querySelector('.js-plotly-plot').layout.sliders.length"
            )
            == 2
        )
        rail = page.locator(".slider-rail-touch-rect").nth(1)
        box = rail.bounding_box()
        page.mouse.click(box["x"] + 1, box["y"] + box["height"] / 2)
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.sliders[1].active === 0"
        )
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.annotations[5].text.includes('area=0.000')"
        )
        assert "area=0.000" in page.evaluate(
            "document.querySelector('.js-plotly-plot').layout.annotations[5].text"
        )
        page.screenshot(path=str(html.with_suffix(".png")), full_page=True)
        assert not errors
        browser.close()


def test_method_switch_hides_missing_exemplar_slider(multiclass_report_path):
    report_path = multiclass_report_path
    import sqlite3
    from playwright.sync_api import sync_playwright
    from quac.plotting import write_summary

    database = sqlite3.connect(report_path)
    row = database.execute(
        "SELECT id FROM explanation WHERE method='residual' AND pair_id IN (SELECT id FROM pair WHERE source_class=0 AND target_class=1) ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]
    database.execute("DELETE FROM threshold WHERE explanation_id=?", (row,))
    database.execute("DELETE FROM explanation WHERE id=?", (row,))
    database.commit()
    database.close()
    html = write_summary(report_path, report_path.with_suffix(".html"), top_k=2)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(html.as_uri())
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot')?._fullLayout?.sliders?.length === 2"
        )
        page.locator(".updatemenu-header").nth(0).click()
        page.locator(".updatemenu-dropdown-button").filter(has_text="residual").click()
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.title.text.includes('residual')"
        )
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.sliders.filter(slider => slider.visible).length === 1"
        )
        assert (
            page.evaluate(
                "document.querySelector('.js-plotly-plot').layout.annotations[3].text"
            )
            == "No exemplar"
        )
        page.locator(".updatemenu-header").nth(0).click()
        page.locator(".updatemenu-dropdown-button").filter(has_text="random").click()
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.sliders.filter(slider => slider.visible).length === 2"
        )
        browser.close()


def test_class_selectors_cover_all_directions(multiclass_report_path):
    import re
    import sqlite3
    from playwright.sync_api import sync_playwright
    from quac.plotting import write_summary
    from quac.report import Report

    database = sqlite3.connect(multiclass_report_path)
    ids = database.execute(
        "SELECT e.id FROM explanation e JOIN pair p ON p.id=e.pair_id "
        "WHERE e.method='residual' AND p.source_class=2 AND p.target_class=0"
    ).fetchall()
    for (explanation_id,) in ids:
        database.execute(
            "DELETE FROM threshold WHERE explanation_id=?", (explanation_id,)
        )
        database.execute("DELETE FROM explanation WHERE id=?", (explanation_id,))
    database.commit()
    database.close()
    expected = {}
    with Report(multiclass_report_path) as report:
        classes = report.metadata["classes"]
        for source, target in report.directions("random"):
            row = report.top("random", 1, source, target)[0]
            pair = report.pair(row["pair_id"])
            expected[source, target] = {
                "source_pixel": round(float(pair["source_image"][0, 0, 0]) * 255),
                "generated_pixel": round(float(pair["generated_image"][0, 0, 0]) * 255),
                "delta": float(report.curve("random", source, target)[1][-1]),
            }
    html = write_summary(
        multiclass_report_path, multiclass_report_path.with_suffix(".html"), 2
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1500})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(html.as_uri())
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot')?.layout?.meta?.method === 'random'"
        )

        def choose(menu, label):
            page.locator(".updatemenu-header").nth(menu).click()
            page.locator(".updatemenu-dropdown-button:visible").filter(
                has_text=re.compile(f"^{re.escape(label)}$")
            ).click()

        for (source, target), values in expected.items():
            choose(1, classes[source])
            page.wait_for_function(
                "source => document.querySelector('.js-plotly-plot').layout.meta.source_class === source",
                arg=source,
            )
            choose(2, classes[target])
            page.wait_for_function(
                """selection => {
                const plot = document.querySelector('.js-plotly-plot');
                return plot.layout.meta.source_class === selection.source &&
                       plot.layout.meta.target_class === selection.target &&
                       plot.data[0].z[0][0][0] === selection.source_pixel &&
                       plot.data[1].z[0][0][0] === selection.generated_pixel &&
                       Math.abs(plot.data[8].y[100] - selection.delta) < 1e-8;
            }""",
                arg={"source": source, "target": target, **values},
            )
            assert (
                page.evaluate(
                    "document.querySelector('.js-plotly-plot').layout.sliders.filter(slider => slider.visible).length"
                )
                == 2
            )
        choose(0, "residual")
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.meta.method === 'residual'"
        )
        assert page.evaluate(
            "document.querySelector('.js-plotly-plot').layout.meta"
        ) == {"method": "residual", "source_class": 2, "target_class": 1}
        choose(1, "b")
        page.wait_for_function("""() => {
            const selection = document.querySelector('.js-plotly-plot').layout.meta;
            return selection.source_class === 1 && selection.target_class === 2;
        }""")
        choose(1, "c")
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.meta.source_class === 2"
        )
        choose(2, "a")
        page.wait_for_function("""() => {
            const plot = document.querySelector('.js-plotly-plot');
            return plot.layout.meta.target_class === 0 && plot.data[8].y.length === 0 &&
                   plot.layout.sliders.every(slider => !slider.visible) &&
                   plot.layout.shapes[3].visible === false &&
                   plot.layout.annotations[0].text === 'No exemplar';
        }""")
        choose(2, "b")
        page.wait_for_function("""() => {
            const plot = document.querySelector('.js-plotly-plot');
            return plot.layout.meta.target_class === 1 && plot.data[8].y.length === 101 &&
                   plot.layout.sliders.every(slider => slider.visible) &&
                   plot.layout.shapes[3].visible === true &&
                   plot.layout.annotations[0].text.includes('c → b');
        }""")
        page.wait_for_function("""() => {
            const layout = document.querySelector('.js-plotly-plot')._fullLayout;
            const x = layout.xaxis, y = layout.yaxis;
            const xScale = Math.abs(x._length / (x.range[1] - x.range[0]));
            const yScale = Math.abs(y._length / (y.range[1] - y.range[0]));
            return Math.abs(xScale - yScale) < 0.01;
        }""")
        page.screenshot(path=str(html.with_suffix(".png")), full_page=True)
        assert not errors
        browser.close()
