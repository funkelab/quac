=======================
Visualizing the results
=======================

In this tutorial, we will show you how to visualize the results of the attribution and evaluation steps.
Make sure to modify the paths to the reports and the classifier to match your setup!

Obtaining the QuAC curves
=========================
Let's start by loading the reports obtained in the previous step.

.. code-block:: python
    :linenos:

    from quac.report import Report

    report_directory = "/path/to/report/directory/"
    reports = {
        method: Report(name=method)
        for method in [
            "DDeepLift",
            "DIntegratedGradients",
        ]
    }

    for method, report in reports.items():
        report.load(report_directory + method + "/default.json")

Next, we can plot the QuAC curves for each method.
This allows us to get an idea of how well each method is performing, overall.

.. code-block:: python
    :linenos:

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for method, report in reports.items():
        report.plot_curve(ax=ax)
    # Add the legend
    plt.legend()
    plt.show()


Choosing the best attribution method for each sample
====================================================

While one attribution method may be better than another on average, it is possible that the best method for a given example is different.
Therefore, we will make a list of the best method for each example by comparing the quac scores.

.. code-block:: python
    :linenos:

    quac_scores = pd.DataFrame(
        {method: report.quac_scores for method, report in reports.items()}
    )
    best_methods = quac_scores.idxmax(axis=1)
    best_quac_scores = quac_scores.max(axis=1)

We'll also want to load the classifier at this point, so we can look at the classifications of the counterfactual images.

.. code-block:: python
    :linenos:

    import torch

    classifier = torch.jit.load("/path/to/classifier/model.pt")


Choosing the best examples
==========================
Next we want to choose the best example, given the best method.
This is done by ordering the examples by the QuAC score, and then choosing the one with the highest score.

.. code-block:: python
    :linenos:

    order = best_quac_scores[::-1].argsort()

    # For example, choose the 10th best example
    idx = 10
    # Get the corresponding report
    report = reports[best_methods[order[idx]]]

We will then load that example and its counterfactual from its path, and visualize it.
We also want to see the classification of both the original and the counterfactual.

.. code-block:: python
    :linenos:

    # Transform to apply to the images so they match each other
    # loading
    from PIL import Image

    image_path, generated_path = report.paths[order[idx]], report.target_paths[order[idx]]
    image, generated_image = Image.open(image_path), Image.open(generated_path)

    prediction = report.predictions[order[idx]]
    target_prediction = report.target_predictions[order[idx]]

    image_path, generated_path = report.paths[order[idx]], report.target_paths[order[idx]]
    image, generated_image = Image.open(image_path), Image.open(generated_path)

    prediction = report.predictions[order[idx]]
    target_prediction = report.target_predictions[order[idx]]

Loading the attribution
=======================
We next want to load the attribution for the example, and visualize it.

.. code-block:: python
    :linenos:

    attribution_path = report.attribution_paths[order[idx]]
    attribution = np.load(attribution_path)

Getting the processor
=====================
We want to see the specific mask that was optimal in this case.
To do this, we will need to get the optimal threshold, and get the processor used for masking.

.. code-block:: python
    :linenos:

    from quac.evaluation import Processor

    gaussian_kernel_size = 11
    struc = 10
    thresh = report.optimal_thresholds()[order[idx]]
    print(thresh)
    processor = Processor(gaussian_kernel_size=gaussian_kernel_size, struc=struc)

    mask, _ = processor.create_mask(attribution, thresh)
    rgb_mask = mask.transpose(1, 2, 0)
    # zero-out the green and blue channels
    rgb_mask[:, :, 1] = 0
    rgb_mask[:, :, 2] = 0
    counterfactual = np.array(generated_image) / 255 * rgb_mask + np.array(image) / 255 * (1.0 - rgb_mask)

Let's also get the classifier output for the counterfactual image.

.. code-block:: python
    :linenos:

    classifier_output = classifier(
        torch.tensor(counterfactual).permute(2, 0, 1).float().unsqueeze(0).to(device)
    )
    counterfactual_prediction = softmax(classifier_output[0].detach().cpu().numpy())

Visualizing the results
=======================
Finally, we can visualize the results.

.. code-block:: python
    :linenos:

    fig, axes = plt.subplots(2, 4)
    axes[1, 0].imshow(image)
    axes[0, 0].bar(np.arange(len(prediction)), prediction)
    axes[1, 1].imshow(generated_image)
    axes[0, 1].bar(np.arange(len(target_prediction)), target_prediction)
    axes[0, 2].bar(np.arange(len(counterfactual_prediction)), counterfactual_prediction)
    axes[1, 2].imshow(counterfactual)
    axes[1, 3].imshow(rgb_mask)
    axes[0, 3].axis("off")
    fig.suptitle(f"QuAC Score: {report.quac_scores[order[idx]]}")
    plt.show()

You can now see the original image, the generated image, the counterfactual image, and the mask.
From here, you can choose to visualize other examples, of save the images for later use.
