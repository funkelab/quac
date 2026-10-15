"""Optional real-browser checks: QUAC_BROWSER_TESTS=1 pytest tests/test_browser.py."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("QUAC_BROWSER_TESTS") != "1",
    reason="Set QUAC_BROWSER_TESTS=1 and install Playwright/Chromium",
)


def test_independent_sliders_and_method_selection(report_path):
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
        initial = page.evaluate("""() => {
            const plot = document.querySelector('.js-plotly-plot');
            return {first: JSON.stringify(plot.data[2].z), second: JSON.stringify(plot.data[5].z),
                    text: plot.layout.annotations[2].text};
        }""")
        rail = page.locator(".slider-rail-touch-rect").first
        box = rail.bounding_box()
        page.mouse.click(box["x"] + 1, box["y"] + box["height"] / 2)
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.sliders[0].active === 0"
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
        # Switch methods using the actual dropdown; the new method resets its own sliders.
        page.locator(".updatemenu-header").click()
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


def test_method_switch_hides_missing_exemplar_slider(report_path):
    import sqlite3
    from playwright.sync_api import sync_playwright
    from quac.plotting import write_summary

    database = sqlite3.connect(report_path)
    row = database.execute(
        "SELECT id FROM explanation WHERE method='residual' ORDER BY id DESC LIMIT 1"
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
        page.locator(".updatemenu-header").click()
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
        page.locator(".updatemenu-header").click()
        page.locator(".updatemenu-dropdown-button").filter(has_text="random").click()
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.sliders.filter(slider => slider.visible).length === 2"
        )
        browser.close()
