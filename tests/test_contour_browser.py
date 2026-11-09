import base64
from io import BytesIO
import os

import numpy as np
from PIL import Image
import pytest
from scipy.ndimage import gaussian_filter, label

from quac.contours import raster_contour

pytestmark = pytest.mark.skipif(
    os.environ.get("QUAC_BROWSER_TESTS") != "1",
    reason="Set QUAC_BROWSER_TESTS=1 and install Playwright/Chromium",
)


def _pixels(source):
    return np.asarray(Image.open(BytesIO(base64.b64decode(source.split(",", 1)[1]))))


def test_contour_controls_follow_thresholds_and_selections(multiclass_report_path):
    from playwright.sync_api import sync_playwright
    from quac.plotting import write_summary

    html = write_summary(
        multiclass_report_path, multiclass_report_path.with_suffix(".html"), 2
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1600})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(html.as_uri())
        color = page.get_by_label("Contour color", exact=True)
        width = page.get_by_label("Contour width (%)", exact=True)
        smoothness = page.get_by_label("Contour smoothness", exact=True)
        color.wait_for()
        assert page.locator("input[type=color]").count() == 1
        assert color.input_value() == "#ff1493"
        assert width.input_value() == "1" and smoothness.input_value() == "0"
        snapshot = page.evaluate("""() => {
            const plot = document.querySelector('.js-plotly-plot');
            return {image: plot.data[2].z, prediction: plot.layout.annotations[2].text,
                    curve: plot.data[8].y, mask: plot.data[9].meta.quac_contour.mask,
                    secondMask: plot.data[10].meta.quac_contour.mask, contour: plot.data[9].source};
        }""")
        np.testing.assert_array_equal(
            _pixels(snapshot["contour"]), raster_contour(snapshot["mask"])
        )

        def wait_style(style):
            page.wait_for_function(
                """style => document.querySelector('.js-plotly-plot').data
                .filter(trace => trace.meta?.quac_contour)
                .every(trace => trace.meta.quac_contour.style === style)""",
                arg=style,
            )

        color.evaluate(
            "element => { element.value = '#00ff00'; element.dispatchEvent(new Event('input', {bubbles: true})); }"
        )
        wait_style("#00ff00:1:0")
        contour = page.evaluate(
            "document.querySelector('.js-plotly-plot').data[9].source"
        )
        np.testing.assert_array_equal(
            _pixels(contour), raster_contour(snapshot["mask"], "#00ff00")
        )
        after = page.evaluate("""() => {
            const plot = document.querySelector('.js-plotly-plot');
            return {image: plot.data[2].z, prediction: plot.layout.annotations[2].text, curve: plot.data[8].y};
        }""")
        assert after == {key: snapshot[key] for key in ("image", "prediction", "curve")}
        width.fill("0")
        wait_style("#00ff00:0:0")
        np.testing.assert_array_equal(
            _pixels(
                page.evaluate(
                    "document.querySelector('.js-plotly-plot').data[9].source"
                )
            ),
            raster_contour(snapshot["mask"], "#00ff00", 0),
        )
        rail = page.locator(".slider-rail-touch-rect").first.bounding_box()
        page.mouse.click(rail["x"] + rail["width"] - 1, rail["y"] + rail["height"] / 2)
        page.wait_for_function("""() => {
            const contour = document.querySelector('.js-plotly-plot').data[9].meta.quac_contour;
            return contour.mask.every(row => row.every(value => value === 1)) && contour.style === '#00ff00:0:0';
        }""")
        assert (
            np.count_nonzero(
                _pixels(
                    page.evaluate(
                        "document.querySelector('.js-plotly-plot').data[9].source"
                    )
                )[..., 3]
            )
            == 12
        )
        assert (
            page.evaluate(
                "document.querySelector('.js-plotly-plot').data[10].meta.quac_contour.mask"
            )
            == snapshot["secondMask"]
        )
        width.fill("50")
        wait_style("#00ff00:50:0")
        assert (
            np.count_nonzero(
                _pixels(
                    page.evaluate(
                        "document.querySelector('.js-plotly-plot').data[9].source"
                    )
                )[..., 3]
            )
            == 16
        )
        smoothness.focus()
        page.keyboard.press("End")
        wait_style("#00ff00:50:3")
        assert page.get_by_text("4× antialiasing", exact=True).is_visible()
        page.locator(".updatemenu-header").nth(0).click()
        page.locator(".updatemenu-dropdown-button:visible").filter(
            has_text="residual"
        ).click()
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.meta.method === 'residual'"
        )
        wait_style("#00ff00:50:3")
        page.locator(".updatemenu-header").nth(1).click()
        page.locator(".updatemenu-dropdown-button:visible").filter(has_text="c").click()
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.meta.source_class === 2"
        )
        wait_style("#00ff00:50:3")
        page.locator(".updatemenu-header").nth(0).click()
        page.locator(".updatemenu-dropdown-button:visible").filter(
            has_text="residual"
        ).click()
        wait_style("#00ff00:50:3")
        assert (
            color.input_value() == "#00ff00"
            and width.input_value() == "50"
            and smoothness.input_value() == "3"
        )
        page.screenshot(path=str(html.with_suffix(".png")), full_page=True)
        assert not errors
        browser.close()


def test_smoothed_contours_use_mask_values_and_at_most_four_times_resolution(
    report_path,
):
    from playwright.sync_api import sync_playwright
    from quac.plotting import write_summary

    html = write_summary(report_path, report_path.with_suffix(".html"), 1)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(html.as_uri())
        page.get_by_label("Contour color", exact=True).wait_for()

        def render(mask, smoothness):
            return np.array(
                page.evaluate(
                    """settings => QuacContours.render(settings.mask, '#ff1493', 0, settings.smoothness)""",
                    {"mask": mask.tolist(), "smoothness": smoothness},
                )
            )

        y, x = np.mgrid[:40, :48]
        circle = gaussian_filter(
            (((y - 19.3) ** 2 + (x - 23.7) ** 2) < 12**2).astype(float), sigma=2
        )
        for mask in (circle, np.eye(12), np.zeros((8, 10)), np.ones((8, 10))):
            np.testing.assert_array_equal(
                render(mask, 0), raster_contour(mask, width_percent=0)
            )
        mask_a = np.tile([0, 0, 0.4, 0.9, 0.9, 0.9, 0, 0], (16, 1))
        mask_b = np.tile([0, 0, 0.1, 0.6, 0.6, 0.6, 0, 0], (16, 1))
        np.testing.assert_array_equal(render(mask_a, 0), render(mask_b, 0))
        smooth_a, smooth_b = render(mask_a, 3), render(mask_b, 3)
        assert smooth_a[8, 2, 3] > smooth_a[8, 3, 3]
        assert smooth_b[8, 2, 3] < smooth_b[8, 3, 3]
        smooth = render(circle, 3)
        assert np.any((smooth[..., 3] > 0) & (smooth[..., 3] < 255))
        assert label(smooth[..., 3] > 0, structure=np.ones((3, 3)))[1] == 1
        channels = np.stack([circle, np.roll(circle, 8, axis=1), np.zeros_like(circle)])
        np.testing.assert_array_equal(
            render(channels, 0), raster_contour(channels, "#ff1493")
        )
        np.testing.assert_array_equal(
            render(channels, 3),
            np.maximum.reduce([render(channel, 3) for channel in channels]),
        )
        sizes = page.evaluate(
            """mask => {
            const create = document.createElement.bind(document), canvases = [];
            document.createElement = (tag, ...args) => {
                const element = create(tag, ...args);
                if (tag === 'canvas') canvases.push(element);
                return element;
            };
            try { QuacContours.render(mask, '#ff1493', 0, 3); }
            finally { document.createElement = create; }
            return canvases.map(canvas => [canvas.width, canvas.height]);
        }""",
            circle.tolist(),
        )
        assert sizes == [[48 * 4, 40 * 4]]
        browser.close()


def test_rgb_colors_follow_channels_and_persist_across_selections(rgb_report_path):
    from playwright.sync_api import sync_playwright
    from quac.plotting import write_summary

    html = write_summary(rgb_report_path, rgb_report_path.with_suffix(".html"), 1)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1850, "height": 1200})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(html.as_uri())
        controls = [
            page.get_by_label(f"{channel} channel color", exact=True)
            for channel in ("Red", "Green", "Blue")
        ]
        controls[0].wait_for()
        assert page.locator("input[type=color]").count() == 3
        assert [control.input_value() for control in controls] == [
            "#ff0000",
            "#00ff00",
            "#0000ff",
        ]
        snapshot = page.evaluate("""() => {
            const plot = document.querySelector('.js-plotly-plot');
            return {mask: plot.data[6].meta.quac_contour.mask, contour: plot.data[6].source, image: plot.data[2].z};
        }""")
        np.testing.assert_array_equal(
            _pixels(snapshot["contour"]), raster_contour(snapshot["mask"])
        )
        palette = ["#ffff00", "#00ffff", "#ff00ff"]
        for control, color in zip(controls, palette):
            control.evaluate(
                "(element, color) => { element.value = color; element.dispatchEvent(new Event('input', {bubbles:true})); }",
                color,
            )

        def verify():
            page.wait_for_function(
                """style => document.querySelector('.js-plotly-plot').data[6].meta.quac_contour.style === style""",
                arg=",".join(palette) + ":1:0",
            )
            state = page.evaluate("""() => {
                const trace = document.querySelector('.js-plotly-plot').data[6];
                return {mask: trace.meta.quac_contour.mask, contour: trace.source};
            }""")
            np.testing.assert_array_equal(
                _pixels(state["contour"]), raster_contour(state["mask"], palette)
            )

        verify()
        assert (
            page.evaluate("document.querySelector('.js-plotly-plot').data[2].z")
            == snapshot["image"]
        )
        rail = page.locator(".slider-rail-touch-rect").first.bounding_box()
        page.mouse.click(rail["x"] + rail["width"] - 1, rail["y"] + rail["height"] / 2)
        page.wait_for_function(
            "document.querySelector('.js-plotly-plot').layout.sliders[0].active === 4"
        )
        verify()
        for menu, label in ((0, "residual"), (1, "c"), (2, "a")):
            page.locator(".updatemenu-header").nth(menu).click()
            page.locator(".updatemenu-dropdown-button:visible").get_by_text(
                label, exact=True
            ).click()
            page.wait_for_function(
                """selection => {
                const plot = document.querySelector('.js-plotly-plot');
                const menu = plot.layout.updatemenus[selection.menu];
                return menu.buttons[menu.active].label === selection.label;
            }""",
                arg={"menu": menu, "label": label},
            )
            verify()
        page.get_by_label("Contour smoothness", exact=True).focus()
        page.keyboard.press("End")
        page.wait_for_function(
            """style => document.querySelector('.js-plotly-plot').data[6].meta.quac_contour.style === style""",
            arg=",".join(palette) + ":1:3",
        )
        state = page.evaluate(
            """palette => {
            const trace = document.querySelector('.js-plotly-plot').data[6];
            return {contour: trace.source, pixels: QuacContours.render(trace.meta.quac_contour.mask, palette, 1, 3)};
        }""",
            palette,
        )
        np.testing.assert_array_equal(_pixels(state["contour"]), state["pixels"])
        assert [control.input_value() for control in controls] == palette
        assert not errors
        browser.close()
