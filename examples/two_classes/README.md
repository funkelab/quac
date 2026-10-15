# Two-class example

From the repository root, after installing `quac[plot]`:

```sh
python examples/two_classes/generate.py
quac train -c examples/two_classes/config.yaml
quac explain -c examples/two_classes/config.yaml
quac summarize -c examples/two_classes/config.yaml
```

Open `examples/two_classes/quac-output/summary.html` in a browser. The generated
`marked` and `plain` classes share a noisy disk-shaped object; only `marked` has a
small bright upper-right square. Classifier outputs follow that sorted class
order. The setup script exports its TorchScript classifier along with the images.

The config deliberately uses small networks and 50 training steps. This is a
workflow demonstration, not a convergence guarantee. Increase `training.steps`
to improve the generated images. To resume, set `generator.checkpoint` to
`quac-output/checkpoints/latest.pt` and increase the final step count. Repeating
explanation requires `output.overwrite: true` or a different output directory.

Only `data.source` is required to train. `classifier.checkpoint` is additionally
required to explain; all remaining example settings make the demonstration
smaller or show multiple attribution methods.
