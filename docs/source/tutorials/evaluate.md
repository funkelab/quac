# Evaluate counterfactuals

```sh
quac explain --config config.yaml
```

Pixels are ranked by importance and selected at equidistant areas, including empty
and full masks. Stable row-major order resolves equal importance. Gaussian blur
uses `explanation.sigma` (default 1); `explanation.levels` defaults to 101.
Counterfactuals blend generated content into real images under these soft masks.

Delta is the change in target probability. The optimum maximizes delta divided by
positive binary area, with ties favoring smaller masks. Exemplars are ranked by
optimal delta. Curve AUC is stored separately.

`report.sqlite` stores every image, attribution, threshold mask, counterfactual,
logit, probability, delta, and run metadata. Completed method sweeps are committed
atomically. Interrupted/failed runs retain their completed records and failure
status. Existing reports require explicit `output.overwrite: true` to replace.
