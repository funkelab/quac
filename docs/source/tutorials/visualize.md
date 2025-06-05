# Interacting with QuAC results

Once you've run QuAC on your data, how do you interact with the results? 
The main two objects that you will be interacting with are `Report` and `Explanation`. 
This guide will walk you through some of the options that you have in interacting with these objects, and what you can get out of them.

## The `Report`

The output of the previous (evaluation) step of QuAC is a `Report`, stored on disk in JSON format. 
There is one report per attribution method, and they can be loaded directly from their path:

```{code-block} python
from quac.report import Report

report = Report("/path/to/report/file.json")
```

However we will generally want to merge all of the evaluations together to pick the best option for each sample. QuAC has a way to do this too:

```{code-block} python
from quac.report import Report

report = Report.from_directory("/path/to/report/directory/")
```

Here, we provide the path to a parent directory holding all reports. They can be in sub-directories, as is default for QuAC. This function will take some time to run, but will also return a `Report` object.

<details><summary> What happens in <tt>from_directory</tt>?</summary>
QuAC with crawl through the directory and all of its sub-directories looking for JSON files, and try to load them as reports. Do not worry if there are JSON files in there that are *not* QuAC reports: you should see a warning for each of these files saying that it could not be loaded, but this should not impede the process.

After all the files have been loaded, this report will then filter through them, selecting the best possible option out of all attribution methods for each `query` image considered in the report. It will then order all of the samples by QuAC score.

In the end, you get a final `Report` describing only the best explanations. 

</details>
<br>

### Filtering the report
The `Report` contains information for every source-target class pair, and the whole range of QuAC scores even some which you may not want.
However, there are some standardized way of filtering these results to whatever it may be that you are looking for.

For example: 
- Selecting a specific source class, *e.g.* `0`: `report.from_source(0)`
- Selecting a specific target class, *e.g.* `1`: `report.to_target(1)`
- Thresholding only high scores: `report.score_threshold(0.9)`
- Selecting only the best 10 samples, by QuAC score: `report.top_n(10)`

These functions are all composable, and return filtered down `Report` object, so you can do something like choosing the 5 best explanations for the translation from source 2 to target 1 like this: 
```{code-block} python
filtered_report = report.from_source(2).to_target(1).top_n(5)
```

### Storing reports

You may want to store filtered, combined, or otherwise modified reports so that you don't need to re-compute them.
You can store a report to a particular output directory.
The file *name* will be the same as the `report.name`, which you can set freely.
For example:
```{code-block} python
filtered_report.name = "from_2_to_1_top_5"
filtered_report.store("/path/to/directory")
```

The filtered report will then be found at `/path/to/directory/from_2_to_1_top_5.json`.


## The `Explanation` objects

The main purpose of a `Report` is to hold and organize a series of `Explanation` objects. 
You can find out how many of these `Explanation` objects your report holds using `len(report)`.
These can be accessed very simply by indexing, *e.g.* `report[0]`.

Each `Explanation` holds the following information: 
- a `query` tensor
- a `counterfactual` tensor
- a `mask` tensor, showing where `query` and `counterfactual` differ
- a `source_class` integer label for the annotated class of the query
- a `target_class` integer label for the desired class for the counterfactual
- a `query_prediction` tensor: the output of the classifier on the query
- a `counterfactual_prediction` tensor: the output of the classifier on the counterfactual
- a `score`, the QuAC score for this particular explanation

The image and mask tensors are `torch.Tensor` objects, stored in `C,H,W` order. 
Generally, you will want to look at the images, with or without a mask overlay. 
You can use your favorite python plotting library to do so. 
For example, with `matplotlib`, the following function can be used to plot the explanation of your choice:

```{code-block} python
def plot_explanation(explanation, show_mask=True):
    """
    Plot the query and counterfactual images in an explanation.

    Plots a 2x2 grid, with the query in the top left corner, 
    the counterfactual in the bottom left corner, 
    and the predictions next to their corresponding image. 
    The mask is overlaid depending on the value of `show_mask`.
    The QuAC score is shown as a figure title.
    """
    fig, ((ax1, bax1), (ax2, bax2)) = plt.subplots(
        2, 2, gridspec_kw={"width_ratios": [1, 0.2], "wspace": -0.4}
    )

    # Plot the query and its accompanying prediction
    ax1.imshow(explanation.query.permute(1, 2, 0))
    ax1.axis("off")
    ax1.set_title(f"Query: {explanation.source_class}")
    bax1.bar(
        np.arange(len(explanation.query_prediction)), 
        explanation.query_prediction, color="gray"
    )

    # Plot the counterfactual and its accompanying prediction
    ax2.imshow(explanation.counterfactual.permute(1, 2, 0))
    ax2.set_title(f"Counterfactual: {explanation.target_class}")
    ax2.axis("off")
    bax2.bar(
        np.arange(len(explanation.counterfactual_prediction)),
        explanation.counterfactual_prediction,
        color="gray",
    )

    if show_mask:
        # Show a color-agnostic mask
        ax1.imshow(
            explanation.mask.sum(0), 
            alpha=0.3, 
            cmap="coolwarm"
        )
        ax2.imshow(
            explanation.mask.sum(0), 
            alpha=0.3, 
            cmap="coolwarm"
        )

    fig.tight_layout()
    fig.suptitle(f"Score: {explanation.score:.2f}")
    return
```

It can be used on all of the elements in the `filtered_report` above:

```{code-block} python
for i in range(len(filtered_report)):
    plot_explanation(filtered_report[i], show_mask=True)
```

## Final comments
You've made it to the end of the tutorial, and the beginning of a new set of discoveries.
QuAC can show you where to look and what to look at... what it means is up to you to find out!
To see how we extracted insights from QuAC outputs, we strongly encourage you to have a look at [the examples page](../examples.md) and [the pre-print](https://doi.org/10.1101/2024.11.26.625505). 
[Reach out](mailto:adjavond%40hhmi.org?subject=QuAC%20Comments) for any questions, comments, or interesting insights you've discovered while using this method!