# Visualizing the results

This guide will let you look at the best explanations you have generated, and discover interesting features in your data.
We recommend copying the code blocks into a `jupyter` notebook, and using it interactively.

## Setup

```{code-block} python
report_directory = "/path/to/report/directory/"
classifier_checkpoint = "/path/to/classifier/model.pt"
```

## Reports
Let's start by loading the reports obtained in the previous step.

```{code-block} python
:linenos:

from quac.report import Report

reports = {
    method: Report(name=method)
    for method in [
        "discriminative_deeplift",
        "discriminative_ig",
    ]
}

for method, report in reports.items():
    report.load(report_directory + method + "/default.json")
```

Next, we can plot the QuAC curves for each method.
This allows us to get an idea of how well each method is performing, overall.

```{code-block} python
:linenos:

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
for method, report in reports.items():
    report.plot_curve(ax=ax)
# Add the legend
plt.legend()
plt.show()
```

The plots show the mean and interquartile range of QuAC curves for all examples in your testing dataset. The closer the curve gets to the upper left corner, the better.

## Choosing the best attribution method for each sample

While one attribution method may be better than another on average, we can select the best method for each example on a case-by-case basis. 
We can do this by comparing the QuAC scores per-image, and choosing the best of our options.

```{code-block} python
:linenos:

quac_scores = pd.DataFrame(
    {method: report.quac_scores for method, report in reports.items()}
)
best_methods = quac_scores.idxmax(axis=1)
best_quac_scores = quac_scores.max(axis=1)
```

## Choosing the best examples
Next we want to choose the examples to look at.
Examples with a high QuAC score will be better explanations of the classifier's decisions.
We therefore order the images from best to worst.


We choose to look at the 11th best example below (remember that python indexing starts at 0).
You can change the `idx` to choose a better (smaller number) or worse (larger number) example if you wish.

```{code-block} python
:linenos:

# Order the quac scores
order = best_quac_scores.argsort()[::-1]
# Choose your example
idx = 10
# Get the corresponding report
report = reports[best_methods[order[idx]]]
```

## Looking at the corresponding images

The `Report` holds the path to that example, and its pair in the `generated_images`. 
It also holds the path to the `attribution` for that pair, which is stored as a `npy` file. 
Finally, it holds the classification of both the original image and the generated image.

```{code-block} python
:linenos:

from PIL import Image

# Load the images
image_path, generated_path = report.paths[order[idx]], report.target_paths[order[idx]]
image, generated_image = Image.open(image_path), Image.open(generated_path)

# Load the attribution
attribution_path = report.attribution_paths[order[idx]]
attribution = np.load(attribution_path)

# Load the predictions for these images
prediction = report.predictions[order[idx]]
target_prediction = report.target_predictions[order[idx]]
```

## Getting the optimal mask
The counterfactual image is created by masking in part of the generated image into the original.
The mask we use can be made from the `attribution` via thresholding by a `Processor`.
We can get the optimal threshold from the `Report`.

```{code-block} python
:linenos:

from quac.evaluation import Processor

thresh = report.optimal_thresholds()[order[idx]]
processor = Processor()

mask, _ = processor.create_mask(attribution, thresh)
channel_last_mask = mask.transpose(1, 2, 0)

counterfactual = np.array(generated_image) / 255 * channel_last_mask + np.array(image) / 255 * (1.0 - channel_last_mask)
```

Let's also get the classifier output for the counterfactual image.
To do this, we need to begin by loading the classifier itself.
Then, we pass our image to the classifier

```{note}
When generating the counterfactual image, we made sure that that its values are between `[0, 1]`. 
Below we will normalize that image to fall within `[-1, 1]`, which is what we have assumed as default for the classifier so far. 
If you decided to use other values for your own classifier, make sure to modify the `mean` and `std` accordingly, or write your own normalization function. 

```

```{code-block} python
:linenos:

import torch

classifier = torch.jit.load(classifier_checkpoint)
mean = 0.5
std = 0.5

counterfactual_tensor = torch.tensor(counterfactual).permute(2, 0, 1).float().unsqueeze(0).to(device)

# Normalize
counterfactual_tensor = (counterfactual - mean) * std

classifier_output = classifier(counterfactual_tensor)
counterfactual_prediction = softmax(classifier_output[0].detach().cpu().numpy())
```

## Visualizing the results
Finally, let's plot the three images, their classifications, and the corresponding mask.

```{code-block} python
:linenos:

fig, axes = plt.subplots(2, 4)
axes[1, 0].imshow(image)
axes[1, 0].set_xlabel("Original")
axes[0, 0].bar(np.arange(len(prediction)), prediction)
axes[1, 1].imshow(generated_image)
axes[1, 1].set_xlabel("Generated")
axes[0, 1].bar(np.arange(len(target_prediction)), target_prediction)
axes[0, 2].bar(np.arange(len(counterfactual_prediction)), counterfactual_prediction)
axes[1, 2].imshow(counterfactual)
axes[1, 2].set_xlabel("Counterfactual)
axes[1, 3].imshow(channel_last_mask)
axes[0, 3].axis("off")
fig.suptitle(f"QuAC Score: {report.quac_scores[order[idx]]:.2f}")
plt.show()
```

You can now see the original image, the generated image, the counterfactual image, and the mask.
By looking at the masked region in the original and the counterfactual, you will see the differences that are important to the classifier.
From here, you can choose to visualize other examples, or save the images for later use.
