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

`report.sqlite` stores logits, probabilities, deltas, metadata, and relative paths
to image artifacts. Originals, generated images, attribution maps, masks, and
counterfactuals are saved as lossless TIFF files under `quac-output/images`.
Completed method sweeps are committed atomically; failed writes remove newly
created image files. Interrupted/failed runs retain their completed records and failure
status. Existing reports require explicit `output.overwrite: true` to replace.

Keep the database and its image directory together when moving a report. Schema
version 2 replaces image blobs with paths; regenerate earlier reports by rerunning
`quac explain` with `output.overwrite: true`.
