from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative

from .contours import contour_metadata, contour_source
from .report import Report


def _display(image):
    array = np.moveaxis(image, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.rint(array * 255).clip(0, 255).astype(np.uint8).tolist()


def _prediction(probabilities, logits, classes):
    return "<br>".join(
        f"{name}: p={probability:.4f}, logit={logit:.4f}"
        for name, probability, logit in zip(classes, probabilities, logits)
    )


def _threshold_text(record, classes):
    return (
        f"<b>Counterfactual</b><br>area={record['area']:.3f}, cutoff={record['cutoff']:.3f}, Δ={record['delta']:.4f}<br>"
        + _prediction(record["probabilities"], record["logits"], classes)
    )


def _image_axes(figure, shape, rows):
    height, width = shape[-2:]
    plot_width = figure.layout.width - figure.layout.margin.l - figure.layout.margin.r
    plot_height = figure.layout.height - figure.layout.margin.t - figure.layout.margin.b
    for index in (row * 4 + column for row in range(rows) for column in (1, 2, 3)):
        suffix = str(index) if index > 1 else ""
        xaxis, yaxis = figure.layout[f"xaxis{suffix}"], figure.layout[f"yaxis{suffix}"]
        x0, x1 = xaxis.domain
        y0, y1 = yaxis.domain
        scale = min((x1 - x0) * plot_width / width, (y1 - y0) * plot_height / height)
        half_width, half_height = width * scale / (2 * plot_width), height * scale / (
            2 * plot_height
        )
        xaxis.update(
            domain=[(x0 + x1) / 2 - half_width, (x0 + x1) / 2 + half_width],
            range=[-0.5, width - 0.5],
            autorange=False,
            fixedrange=True,
            constrain="range",
        )
        yaxis.update(
            domain=[(y0 + y1) / 2 - half_height, (y0 + y1) / 2 + half_height],
            range=[height - 0.5, -0.5],
            autorange=False,
            fixedrange=True,
            scaleanchor=False,
            constrain="range",
        )


def _direction_figure(report, method, k, source_class, target_class):
    rows = report.top(method, k, source_class, target_class)
    classes = report.metadata["classes"]
    methods = report.methods()
    specs = [[{}, {}, {}, {}] for _ in range(k)] + [[{"colspan": 4}, None, None, None]]
    titles = []
    for index in range(k):
        titles.extend(
            ["Source image", "Generated image", "Counterfactual", None]
            if index < len(rows)
            else ["No exemplar", "No exemplar", "No exemplar", None]
        )
    direction = f"{classes[source_class]} → {classes[target_class]}"
    titles.append(
        f"{method} · {direction}: median and 25–75% quantiles"
        if rows
        else "No explanations for this selection"
    )
    figure = make_subplots(
        rows=k + 1,
        cols=4,
        column_widths=[1, 1, 1, 1.25],
        horizontal_spacing=0.045,
        specs=specs,
        subplot_titles=titles,
        vertical_spacing=0.45 / (k + 1),
    )
    sliders, overlays, example_curves, curve_titles = [], [], [], []
    for slot in range(k):
        if slot < len(rows):
            row = rows[slot]
            pair = report.pair(row["pair_id"])
            example_curves.append(report.example_curves(row["pair_id"]))
            curve_titles.append(f"<b>QuAC score: {row['auc']:.4f}</b> · {method}")
            thresholds = report.thresholds(row["id"])
            optimum = row["optimal_index"]
            record = thresholds[optimum]
            contour_sources = [contour_source(item["mask"]) for item in thresholds]
            contour_masks = [contour_metadata(item["mask"]) for item in thresholds]
            overlay = go.Image(
                source=contour_sources[optimum],
                meta=contour_masks[optimum],
                hoverinfo="skip",
            )
            images = [
                pair["source_image"],
                pair["generated_image"],
                record["counterfactual"],
            ]
            texts = [
                f"<b>Source image · {classes[pair['source_class']]} → {classes[pair['target_class']]}</b><br>"
                + _prediction(
                    pair["source_probabilities"], pair["source_logits"], classes
                ),
                "<b>Generated image</b><br>"
                + _prediction(
                    pair["generated_probabilities"], pair["generated_logits"], classes
                ),
                _threshold_text(record, classes),
            ]
        else:
            example_curves.append({})
            curve_titles.append("No exemplar")
            images = [np.zeros(report.metadata["shape"], dtype=np.float32)] * 3
            texts = ["No exemplar"] * 3
            overlay = go.Image(
                source=contour_source(np.zeros(report.metadata["shape"][-2:])),
                meta={"quac_contour": None},
                hoverinfo="skip",
            )
        overlays.append(overlay)
        for column, (image, text) in enumerate(zip(images, texts), 1):
            figure.add_trace(
                go.Image(z=_display(image), hoverinfo="skip"), row=slot + 1, col=column
            )
            figure.add_shape(
                type="rect",
                x0=-0.5,
                y0=-0.5,
                x1=image.shape[-1] - 0.5,
                y1=image.shape[-2] - 0.5,
                line={"color": "#d1d5db", "width": 1},
                row=slot + 1,
                col=column,
            )
            figure.layout.annotations[slot * 3 + column - 1].text = text
            figure.update_xaxes(visible=False, row=slot + 1, col=column)
            figure.update_yaxes(visible=False, row=slot + 1, col=column)
        if slot < len(rows):
            axis = figure.layout[f"yaxis{slot * 4 + 3}"]
            marker_key = f"shapes[{slot * 4 + 3}]"
            figure.add_vline(
                x=record["area"],
                visible=True,
                line_color="#374151",
                line_width=2,
                row=slot + 1,
                col=4,
                exclude_empty_subplots=False,
            )
            annotation_key = f"annotations[{slot * 3 + 2}].text"
            steps = [
                dict(
                    label=f"{record['area']:.2f}",
                    method="update",
                    args=[
                        {
                            "z": [_display(record["counterfactual"]), None],
                            "source": [None, contour_sources[index]],
                            "meta": [{}, contour_masks[index]],
                        },
                        {
                            annotation_key: _threshold_text(record, classes),
                            f"{marker_key}.x0": record["area"],
                            f"{marker_key}.x1": record["area"],
                        },
                        [slot * 3 + 2, k * 3 + 3 + slot],
                    ],
                )
                for index, record in enumerate(thresholds)
            ]
            sliders.append(
                dict(
                    visible=True,
                    active=optimum,
                    steps=steps,
                    x=figure.layout[f"xaxis{slot * 4 + 3}"].domain[0],
                    len=(
                        figure.layout[f"xaxis{slot * 4 + 3}"].domain[1]
                        - figure.layout[f"xaxis{slot * 4 + 3}"].domain[0]
                    ),
                    y=axis.domain[0] - 0.01,
                    yanchor="top",
                    pad={"t": 0},
                    currentvalue={"prefix": "Mask area: "},
                    ticklen=0,
                )
            )
        else:
            figure.add_vline(
                x=0,
                visible=False,
                row=slot + 1,
                col=4,
                exclude_empty_subplots=False,
            )
            sliders.append(dict(visible=False, steps=[]))
    grid, median, p25, p75 = (
        report.curve(method, source_class, target_class)
        if rows
        else (np.array([]),) * 4
    )
    for y, name, fill in [
        (p25, "25%", None),
        (p75, "25–75%", "tonexty"),
        (median, "Median", None),
    ]:
        figure.add_trace(
            go.Scatter(
                x=grid.tolist(),
                y=y.tolist(),
                name=name,
                mode="lines",
                fill=fill,
                line={"width": 2 if name == "Median" else 0, "color": "#326ebe"},
                fillcolor="rgba(50,110,190,0.2)",
                showlegend=False,
            ),
            row=k + 1,
            col=1,
        )
    for slot, overlay in enumerate(overlays):
        figure.add_trace(overlay, row=slot + 1, col=3)
    for slot, curves in enumerate(example_curves):
        for index, name in enumerate(methods):
            curve = curves.get(name)
            figure.add_trace(
                go.Scatter(
                    x=curve["areas"] if curve else [],
                    y=curve["deltas"] if curve else [],
                    name=name,
                    legendgroup=name,
                    showlegend=slot == 0,
                    mode="lines",
                    line={
                        "color": qualitative.Plotly[index % len(qualitative.Plotly)],
                        "width": 3 if name == method else 1.5,
                    },
                    hovertemplate=(
                        (
                            f"{name}<br>QuAC score: {curve['score']:.4f}<br>"
                            "Mask area: %{x:.3f}<br>Classifier change: %{y:.4f}<extra></extra>"
                        )
                        if curve
                        else None
                    ),
                ),
                row=slot + 1,
                col=4,
            )
        figure.update_xaxes(title_text="Mask area", range=[0, 1], row=slot + 1, col=4)
        figure.update_yaxes(title_text="Classifier change", row=slot + 1, col=4)
        subplot = figure.get_subplot(slot + 1, 4)
        figure.add_annotation(
            text=curve_titles[slot],
            x=sum(subplot.xaxis.domain) / 2,
            y=subplot.yaxis.domain[1],
            xref="paper",
            yref="paper",
            yshift=8,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
        )
    figure.update_xaxes(title_text="Binary mask area", row=k + 1, col=1)
    figure.update_yaxes(title_text="Delta", row=k + 1, col=1)
    figure.update_layout(
        sliders=sliders,
        height=520 * k + 400,
        width=1800,
        legend={
            "orientation": "h",
            "x": 0,
            "yanchor": "bottom",
            "y": 1 + 65 / (520 * k + 100),
        },
        margin={"l": 80, "r": 80, "t": 220, "b": 80, "autoexpand": False},
        title={"text": f"QuAC — {method} — {direction}", "x": 0.5, "y": 0.98},
        template="plotly_white",
        meta={
            "method": method,
            "source_class": source_class,
            "target_class": target_class,
        },
    )
    _image_axes(figure, report.metadata["shape"], k)
    for slot, slider in enumerate(figure.layout.sliders):
        if slider.visible:
            slider.y = figure.layout[f"yaxis{slot * 4 + 3}"].domain[0] - 0.01
    figure.update_annotations(font={"size": 12})
    return figure


def _frame_name(method_index, source_class, target_class):
    return f"{method_index}:{source_class}:{target_class}"


def _button(label, method_index, source_class, target_class):
    return dict(
        label=label,
        method="animate",
        args=[
            [_frame_name(method_index, source_class, target_class)],
            {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            },
        ],
    )


def _selectors(methods, classes, method_index, source_class, target_class):
    targets = [index for index in range(len(classes)) if index != source_class]
    return [
        dict(
            active=method_index,
            x=0,
            buttons=[
                _button(method, index, source_class, target_class)
                for index, method in enumerate(methods)
            ],
        ),
        dict(
            active=source_class,
            x=0.36,
            buttons=[
                _button(
                    name,
                    method_index,
                    index,
                    source_class if index == target_class else target_class,
                )
                for index, name in enumerate(classes)
            ],
        ),
        dict(
            active=targets.index(target_class),
            x=0.70,
            buttons=[
                _button(classes[index], method_index, source_class, index)
                for index in targets
            ],
        ),
    ]


def summarize(report_path: str | Path, top_k: int = 5) -> go.Figure:
    """Return an interactive figure with method and source/target class selectors."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    with Report(report_path) as report:
        methods = report.methods()
        if not methods:
            raise ValueError("Report has no completed explanations")
        classes = report.metadata["classes"]
        directions = {method: report.directions(method) for method in methods}
        if not any(directions.values()):
            raise ValueError("Report has no explanations between different classes")
        k = max(
            len(report.top(method, top_k, source, target))
            for method in methods
            for source, target in directions[method]
        )
        initial_method = next(
            index for index, method in enumerate(methods) if directions[method]
        )
        initial_source, initial_target = directions[methods[initial_method]][0]
        views = {}
        for method_index, method in enumerate(methods):
            for source_class in range(len(classes)):
                for target_class in range(len(classes)):
                    if source_class == target_class:
                        continue
                    view = _direction_figure(
                        report, method, k, source_class, target_class
                    )
                    menus = _selectors(
                        methods, classes, method_index, source_class, target_class
                    )
                    plot_height = (
                        view.layout.height - view.layout.margin.t - view.layout.margin.b
                    )
                    menu_y = 1 + (view.layout.margin.t - 90) / plot_height
                    label_y = 1 + (view.layout.margin.t - 70) / plot_height
                    for menu in menus:
                        menu.update(
                            direction="down", y=menu_y, yanchor="top", xanchor="left"
                        )
                    view.update_layout(updatemenus=menus)
                    for label, x in (
                        ("Attribution method", 0),
                        ("Source class", 0.36),
                        ("Target class", 0.70),
                    ):
                        view.add_annotation(
                            text=label,
                            x=x,
                            y=label_y,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            xanchor="left",
                            yanchor="bottom",
                        )
                    views[_frame_name(method_index, source_class, target_class)] = view
    figure = views[_frame_name(initial_method, initial_source, initial_target)]
    figure.frames = [
        go.Frame(name=name, data=list(view.data), layout=view.layout)
        for name, view in views.items()
    ]
    return figure


def write_summary(
    report_path: str | Path, output_path: str | Path, top_k: int = 5
) -> Path:
    """Write a portable HTML summary with bundled JavaScript and all threshold data."""
    figure = summarize(report_path, top_k)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("contours.js").read_text(encoding="utf-8")
    script += '\nQuacContours.attach(document.getElementById("{plot_id}"));'
    figure.write_html(path, include_plotlyjs=True, auto_play=False, post_script=script)
    return path
