# RGB example

From the repository root, after installing `quac[plot]`:

```sh
python examples/rgb_classes/generate.py
quac train -c examples/rgb_classes/config.yaml
quac explain -c examples/rgb_classes/config.yaml
quac summarize -c examples/rgb_classes/config.yaml
```

Open `examples/rgb_classes/quac-output/summary.html` in a browser. Select the
source and target classes to inspect all six directions between `blue`, `green`,
and `red`.

The setup script creates four 32×32 RGB TIFF images per class and exports a
TorchScript classifier. Each image contains a noisy gray disk and one marker:

| Class | Marker position | Image channel |
|---|---|---|
| blue | Upper right | Blue (2) |
| green | Upper left | Green (1) |
| red | Lower center | Red (0) |

Classifier outputs follow the sorted class order: `blue`, `green`, `red`. Its
logits depend on the corresponding channel in each marker region. Images with
the same filename share the background across classes. Use `--count` to change
the number of images per class or `--directory` to choose their destination.

The evaluation uses a 3×3 elliptical closing kernel and Gaussian blur with sigma
1 independently in each channel. The summary overlays red, green, and blue channel
contours with independent color selectors, and compares integrated-gradients and residual attribution curves for each example.

The small generator and 50 training steps provide a quick workflow demonstration.
Increase `training.steps` for better generated images, and inspect them before
interpreting explanations. To resume training, set `generator.checkpoint` to
`quac-output/checkpoints/latest.pt` and increase the final step count. To repeat
explanation, set `output.overwrite: true` or choose another output directory.
