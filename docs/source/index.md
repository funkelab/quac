.. quac documentation master file

Quantitative Attributions with Counterfactuals
==============================================

```{attention}
  Check out the new pre-print on biorxiv: [Quantitative Attributions with Counterfactuals](https://doi.org/10.1101/2024.11.26.625505)!
```

```{image} assets/overview.png
  :width: 800
  :align: center
```

QuAC is a tool for understanding the sometimes subtle differences between classes of images, by questioning a classifier's decisions.
The method generates counterfactual images: making slight changes in an image to change its clasification.
The counterfactuals in QuAC have localized changes, and are quantitatively scored to determine how well they describe the classifier's decision.

Get started with [installation](install), then check out the [tutorials](tutorials) to run each step of the QuAC pipeline on an example dataset.

Unsure whether QuAC is right for your dataset? Check out the [examples](examples) to see how it has been used already!


```{toctree}
  :maxdepth: 2

  install
  examples
  Tutorials <tutorials>
```
