# Quantitative Attribution with Counterfactuals

QuAC explains how an image classifiers discriminates between different classes.

To do so, it generates counterfactual images by making minimal changes to a
real image of one class, such that it gets classified as another class.

## Installation

```sh
pip install '.[plot]'
```
or just
```
pip install .
```
if you don't need the HTML plot summary.

## Usage

Organize your images such that each class is in one directory. Class
indices will be assigned to the sorted names of those directories (e.g.,
classes `A`, `B`, and `C` will have indices 0, 1, and 2).

All images must have the same dimensions and channel count.

A single YAML configuration file is used to configure QuAC. It should contain
at least the path to the directory containing the class directories, and the
classifier under investigation:

```yaml
data:
  source: path/to/directory
classifier:
  checkpoint: classifier.pt
```

The classifier should be TorchScript model accepting `[N,C,H,W]` float32 images
in `[0,1]` and returning `[N,num_classes]` logits. Importantly, the outputs of
the classifier must follow the same order as the sorted image directories.

QuAC runs in three steps: First, it trains a conversion network to translate
images between different classes. Second, it converts each image into each
other class, and runs attribution methods on those paired images to find the
most important changes and create counterfactuals from them. Finally (and only
if installed with the `[plot]` dependency group), a visual summary of the
highest scoring counterfactuals can be produced:

```sh
quac train
quac explain
quac summarize
```
(This assumes that a `config.yaml` exists in the same working directory, pass
`--config <file>` to use a different one.)

Once done, you will find an interactive summary in `quac-output/summary.html`.

## Self-contained example

```sh
python examples/two_classes/generate.py
quac train -c examples/two_classes/config.yaml
quac explain -c examples/two_classes/config.yaml
quac summarize -c examples/two_classes/config.yaml
```

The setup creates small synthetic images differing in a localized bright marker
and exports a classifier that detects that detail. The example configuration
uses small networks and a short training run to demonstrate the whole workflow.
Generating good counterfactuals requires longer training (around 5000
iterations).

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
readers are read-only.

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

The StarGAN implementation is adapted from NAVER's StarGAN v2; see
[NOTICE](NOTICE). Project documentation is at https://funkelab.github.io/quac/.
