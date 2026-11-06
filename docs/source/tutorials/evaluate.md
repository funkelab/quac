# Evaluate counterfactuals

```sh
quac explain --config config.yaml
```

Attribution channels are preserved. Channel pixels are ranked together by attribution
magnitude and selected at evenly spaced fractions, including empty and full masks.
Stable channel-first, row-major order resolves ties. Each channel's binary mask
undergoes morphological closing with an elliptical kernel controlled by
`explanation.closing_size` (default 10; use 1 to disable closing). Mask area is the
fraction of selected channel pixels after closing and before blur, so the resulting
areas need not be evenly spaced.

Gaussian blur acts independently within each channel, using `explanation.sigma`
(default 1); `explanation.levels` defaults to 101. Each mask blends the corresponding
channel of the generated image into the source image.

Delta is the change in target probability. The optimum maximizes delta divided by
positive binary area, with ties favoring smaller masks. Exemplars are ranked by
optimal delta. Curve AUC is stored separately.

`report.sqlite` stores logits, probabilities, deltas, metadata, and relative paths
to image artifacts. Source images, generated images, attribution maps, masks, and
counterfactuals are saved as lossless TIFF files under `quac-output/images`.
Completed method sweeps are committed atomically; failed writes remove newly
created image files. Interrupted/failed runs retain their completed records and failure
status. Existing reports require explicit `output.overwrite: true` to replace.

Keep the database and its image directory together when moving a report.
