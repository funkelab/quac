# Inspect reports

Install `quac[plot]`, then run:

```sh
quac summarize --config config.yaml
```

Open `quac-output/summary.html`. Each exemplar's slider independently selects a
stored mask area and updates its image, delta, logits, and probabilities. Select
the attribution method, source class, and target class from the menus. Both the
exemplars and the median curve with 25th–75th percentile shading use the selected
direction. The target menu excludes the source class; choosing the current target
as the source swaps direction. Selections without stored explanations show an
empty result. The figure is self-contained and works offline.

Set `summary.top_k` to change the number of exemplars (default five). Python users
can call `quac.plotting.summarize(path, top_k=5)` to obtain a Plotly figure, or
`write_summary(path, output_path, top_k=5)` to write HTML. Summary generation reads
image files referenced by the database under `quac-output/images`. Keep that
directory with the report. The source image dataset and models are unnecessary;
no counterfactuals or predictions are recomputed. Once exported, the HTML needs
no external files.

```python
from quac.report import Report
with Report("quac-output/report.sqlite") as report:
    best = report.top("DIntegratedGradients", k=5, source_class=0, target_class=1)
    sweep = report.thresholds(best[0]["id"])
    pair = report.pair(best[0]["pair_id"])
    area, median, q25, q75 = report.curve("DIntegratedGradients", source_class=0, target_class=1)
```
