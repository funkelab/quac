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
