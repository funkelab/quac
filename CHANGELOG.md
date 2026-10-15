# Refactor implementation notes

## 1. Core components and StarGAN backend

- Added strict monolithic configuration with inferred dataset metadata.
- Established float32 CHW images in [0,1], with signed scaling and spatial padding confined to StarGAN adapters.
- Separated classifier inference, pair attribution, StarGAN losses, optimization, EMA, and checkpoints.
- Preserved compatible legacy EMA weight loading; native checkpoints also store optimizer and RNG state for resume.
- Validation: 27 focused tests passed, including grayscale/RGB I/O, attributions, numerical evaluation, real StarGAN training, checkpoint reload, and resume.

## 2. Counterfactual evaluation and SQLite reports

- Replaced intensity-spaced masks and the previous area-minus-delta objective with ranked, equidistant binary areas and delta/area optimization.
- Store all images, masks, attribution maps, exact logits/probabilities, deltas, model identities, and run status in a versioned database.
- Added SQL queries for methods, source/target filtering, exemplar ranking, and summary curves; database readers use read-only connections.
- Added component-injectable orchestration and removed the superseded filesystem pipeline and JSON report implementation.
- Validation: 17 evaluation/report/summary tests passed, including full sweep round trips, SQL filtering, overwrite protection, transaction rollback, independent slider definitions, and source-independent HTML generation.

## 3. CLI, optional HTML plotting, and integration checks

- Added installed `quac train`, `quac explain`, and `quac summarize` entry points sharing one configuration.
- Added an offline Plotly figure with independent exemplar sliders, full prediction captions, method selection, and median/IQR curves.
- Browser verification caught Plotly retaining a slider when the next method has fewer exemplars; absent controls are now explicitly hidden.
- Training imports no plotting libraries. DeepLift provides a clear eager-classifier requirement; integration tests cover all retained attribution families.
- Trimmed unused dependencies and added `quac[plot]`. Kept the existing unpinned Torch dependency; corrected `<=3.12` to `>=3.10,<3.13` so Python 3.12 patch releases can install.
- Native checkpoints now preserve CPU and CUDA RNG states. Invalid generated images are rejected before classifier inference.
- Validation: 45 standard tests pass; two optional Chromium tests pass, exercising real slider clicks and method changes with unequal exemplar counts. Editable installation and wheel building succeed.
