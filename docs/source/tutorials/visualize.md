# Inspect reports

Install `quac[plot]`, then run:

```sh
quac summarize --config config.yaml
```

Open `quac-output/summary.html`. Each exemplar's slider independently selects a
stored mask area and updates its image, delta, logits, and probabilities. A menu
changes attribution methods and refreshes the exemplars and median curve with
25th–75th percentile shading. The figure is self-contained and works offline.

Set `summary.top_k` to change the number of exemplars (default five). Python users
can call `quac.plotting.summarize(path, top_k=5)` to obtain a Plotly figure, or
`write_summary(path, output_path, top_k=5)` to write HTML. Summary generation reads
image files referenced by the database under `quac-output/images`. Keep that
directory with the report. The original dataset and models are unnecessary;
no counterfactuals or predictions are recomputed. Once exported, the HTML needs
no external files.

```python
from quac.report import Report
with Report("quac-output/report.sqlite") as report:
    best = report.top("DIntegratedGradients", k=5, source=0, target=1)
    sweep = report.thresholds(best[0]["id"])
    pair = report.pair(best[0]["pair_id"])
```
