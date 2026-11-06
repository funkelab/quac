# Quantitative Attribution with Counterfactuals

QuAC explains image classifiers by generating images conditioned on a target
class, computing attribution maps for source/generated image pairs, and blending
progressively larger regions of each generated image into its source image to
form counterfactuals. A SQLite report stores scores, predictions, and paths to
image files. Image artifacts live under `quac-output/images`.

Install with Python 3.10–3.12:

```sh
pip install .
# Include the optional interactive HTML summaries:
pip install '.[plot]'
```

Arrange grayscale or RGB images in one directory per class. All images must have
the same dimensions and channel count. Class indices follow sorted directory
names. Floating images must already be finite and in `[0,1]`; integer images are
scaled by their dtype maximum. Classifiers receive float32 images in `[0,1]`.

A source-only configuration is sufficient for generator training:

```yaml
data:
  source: images
```

To explain predictions, additionally supply your classifier, a TorchScript model
accepting `[N,C,H,W]` float32 images in `[0,1]` and returning `[N,num_classes]`
logits in sorted class order:

```yaml
data:
  source: images
classifier:
  checkpoint: classifier.pt
```

Run the three stages using the same configuration:

```sh
quac train --config config.yaml
quac explain --config config.yaml
quac summarize --config config.yaml
```

Paths are relative to the config file. The default outputs are
`quac-output/checkpoints/latest.pt`, `quac-output/report.sqlite`, and
`quac-output/summary.html`. Open the HTML in a browser: each exemplar has its own
mask-area slider. Select the attribution method, source class, and target class
to inspect each direction; exemplars and curves follow the same selection.
Each example shows its QuAC score (area under its mask-area versus classifier-change
curve) and a plot to the right of the counterfactual comparing all attribution
methods for that image pair. A vertical line follows the selected mask area.
JavaScript and image records are bundled for offline use. `summarize` needs `quac[plot]`; training and
explanation do not import the plotting module. Existing reports require an
explicit `output.overwrite: true` to replace them.

Each counterfactual shows a bright pink contour around the region where the stored
blurred blending mask exceeds 0.5. For RGB images, the contours of all three
channel masks are overlaid. Shared controls at the top set its color, width,
and smoothness. Width defaults to 1% of the smaller image dimension, with a minimum
of one image pixel. The default contour uses native pixels; smoothing follows the
blurred mask's 0.5 level crossing with 2×, 3×, or 4× antialiasing. Settings apply
across exemplars, mask areas, attribution methods, and class directions.

## Self-contained example

```sh
python examples/two_classes/generate.py
quac train -c examples/two_classes/config.yaml
quac explain -c examples/two_classes/config.yaml
quac summarize -c examples/two_classes/config.yaml
```

The setup creates small synthetic images differing in a localized bright marker
and exports a classifier that detects that detail. The example configuration uses
small networks and a short training run to demonstrate the whole workflow.
Generating meaningful images requires longer training; inspect the generated
images before interpreting explanations.

## Python API and report queries

```python
from quac import train, explain
from quac.config import load_config
from quac.report import Report

config = load_config("config.yaml")
train(config)
path = explain(config)
with Report(path) as report:
    methods = report.methods()
    best = report.top(methods[0], k=5, source_class=0, target_class=1)
    sweep = report.thresholds(best[0]["id"])
    pair = report.pair(best[0]["pair_id"])
    area, median, q25, q75 = report.curve(methods[0], source_class=0, target_class=1)

# Requires quac[plot]:
from quac.plotting import summarize, write_summary
figure = summarize(path, top_k=5)
write_summary(path, "summary.html", top_k=5)
```

SQLite stores relative paths to lossless TIFF files for source images, generated
images, attribution maps, threshold masks, and counterfactuals. These files live under
`quac-output/images`; exact logits and probabilities, labels, optimal selections,
curve AUCs, resolved configuration, model hashes, and run status stay in SQLite.
The report API loads the referenced files when returning images. Keep the database
and its image directory together when moving or sharing a report. Source images
in the input dataset and model checkpoints can be moved or deleted. Database
readers are read-only, and exported HTML summaries remain standalone.

For every pair, channel pixels are ranked together by attribution magnitude and
selected at evenly spaced fractions, with stable channel-first, row-major ordering
for ties. Each channel mask undergoes elliptical morphological closing
(`explanation.closing_size`, default 10; 1 disables closing), followed by Gaussian
smoothing within that channel. Mask area is measured after closing and before blur,
as a fraction of all channel pixels. Delta is the change in target-class probability.
The optimal mask maximizes delta divided by positive binary area, with ties
favoring smaller areas. Exemplars are ranked by their optimal delta. Curves show
median delta and the 25th–75th percentile distribution across explanations.

## Configuration and extension

Optional sections are `generator`, `training`, `explanation`, `output`, and
`summary`; `device` defaults to `auto` (CUDA when available, otherwise CPU) and
`seed` to 0. Unknown settings are errors. The validated defaults live in
`quac.config.Config`; use `Config.model_json_schema()` for the complete settings
and constraints.

Common overrides:

```yaml
generator:
  kind: stargan
  sampling: latent  # latent vector sampling; or reference image sampling
  # checkpoint: checkpoints/latest.pt
training:
  steps: 100000
  batch_size: 4
explanation:
  methods: [DIntegratedGradients, residual]
  samples: 3
  levels: 101
  sigma: 1.0
  closing_size: 10
summary:
  top_k: 5
```

Available attribution methods are `DIntegratedGradients`, `DDeepLift`, `DInGrad`,
`VanillaIntegratedGradients`, `VanillaDeepLift`, `residual`, and `random`.
DeepLift requires a compatible eager PyTorch classifier; inject one through the
Python API. The default integrated gradients method works with the CLI's
TorchScript contract.

`quac.interfaces` defines the image generator and attribution contracts. Python callers
can inject `generator`, `classifier`, and a dictionary of `attributions` into
`explain`. Custom attribution names must match `explanation.methods`. Generators
receive/return unit-range NCHW images, and attribution components receive CHW
image pairs and return spatial HW or channel CHW attribution maps. Register additional
CLI generator factories in `quac.generators.build_generator`; training routines
belong to the corresponding backend. Class count, channel count, and spatial
size are inferred from the dataset. StarGAN pads internally to a supported
square size and crops generated images to the source image dimensions.

Checkpoints contain trainable weights, EMA weights, optimizer states,
architecture metadata, and CPU/CUDA RNG states. Set `generator.checkpoint` to resume
training, with `training.steps` specifying the final step. It also selects the
checkpoint for explanation.

## Development

```sh
pip install -e '.[dev,plot]'
pytest tests
black --check src tests examples
```

Optional browser checks require Playwright and Chromium:

```sh
pip install playwright
playwright install chromium
QUAC_BROWSER_TESTS=1 pytest tests/test_browser.py
```

The StarGAN implementation is adapted from NAVER's StarGAN v2; see [NOTICE](NOTICE).
Project documentation is at
https://funkelab.github.io/quac/.
