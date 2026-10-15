from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def _method_figure(report, method, k):
    rows = report.top(method, k)
    classes = report.metadata["classes"]
    specs = [[{}, {}, {}] for _ in range(k)] + [[{"colspan": 3}, None, None]]
    titles = []
    for index in range(k):
        titles.extend(
            ["Source image", "Generated image", "Counterfactual"]
            if index < len(rows)
            else ["No exemplar", "No exemplar", "No exemplar"]
        )
    titles.append(f"{method}: median and 25–75% quantiles")
    figure = make_subplots(
        rows=k + 1,
        cols=3,
        specs=specs,
        subplot_titles=titles,
        vertical_spacing=0.45 / (k + 1),
    )
    sliders = []
    for slot in range(k):
        if slot < len(rows):
            row = rows[slot]
            pair = report.pair(row["pair_id"])
            thresholds = report.thresholds(row["id"])
            optimum = row["optimal_index"]
            record = thresholds[optimum]
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
            images = [np.zeros((3, 1, 1))] * 3
            texts = ["No exemplar"] * 3
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
            axis = figure.layout[f"yaxis{slot * 3 + 3}"]
            annotation_key = f"annotations[{slot * 3 + 2}].text"
            steps = [
                dict(
                    label=f"{record['area']:.2f}",
                    method="update",
                    args=[
                        {"z": [_display(record["counterfactual"])]},
                        {annotation_key: _threshold_text(record, classes)},
                        [slot * 3 + 2],
                    ],
                )
                for record in thresholds
            ]
            sliders.append(
                dict(
                    visible=True,
                    active=optimum,
                    steps=steps,
                    x=0.70,
                    len=0.30,
                    y=axis.domain[0] - 0.01,
                    yanchor="top",
                    pad={"t": 0},
                    currentvalue={"prefix": "Mask area: "},
                    ticklen=0,
                )
            )
        else:
            sliders.append(dict(visible=False, steps=[]))
    grid, median, p25, p75 = report.curve(method)
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
    figure.update_xaxes(title_text="Binary mask area", row=k + 1, col=1)
    figure.update_yaxes(title_text="Target probability change", row=k + 1, col=1)
    figure.update_layout(
        sliders=sliders,
        height=520 * k + 400,
        width=1400,
        margin={"t": 180, "b": 80},
        title={"text": f"QuAC — {method}", "x": 0.5, "y": 0.98},
        template="plotly_white",
    )
    figure.update_annotations(font={"size": 12})
    return figure


def summarize(report_path: str | Path, top_k: int = 5) -> go.Figure:
    """Return one interactive figure with independent exemplar sliders and method selection."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    with Report(report_path) as report:
        methods = report.methods()
        if not methods:
            raise ValueError("Report has no completed explanations")
        k = min(top_k, max(len(report.top(method, top_k)) for method in methods))
        figures = [_method_figure(report, method, k) for method in methods]
    figure = figures[0]
    figure.frames = [
        go.Frame(name=method, data=list(item.data), layout=item.layout)
        for method, item in zip(methods, figures)
    ]
    buttons = [
        dict(
            label=method,
            method="animate",
            args=[
                [method],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        )
        for method in methods
    ]
    figure.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.13)]
    )
    return figure


def write_summary(
    report_path: str | Path, output_path: str | Path, top_k: int = 5
) -> Path:
    """Write a portable HTML summary with bundled JavaScript and all threshold data."""
    figure = summarize(report_path, top_k)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, auto_play=False)
    return path
