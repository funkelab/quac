.. _sec_attribute:

================================================
Attribution and evaluation given counterfactuals
================================================

Attribution
===========

.. code-block:: python
    :linenos:

    # Load the classifier
    from quac.generate import load_classifier
    classifier = load_classifier(

    )

    # Defining attributions
    from quac.attribution import (
        DDeepLift,
        DIntegratedGradients,
        AttributionIO
    )
    from torchvision import transforms

    attributor = AttributionIO(
        attributions = {
            "deeplift" : DDeepLift(),
            "ig" : DIntegratedGradients()
        },
        output_directory = "my_attributions_directory"
    )

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(...)
        ]
    )

    # This will run attributions and store all of the results in the output_directory
    # Shows a progress bar
    attributor.run(
        source_directory="my_source_image_directory",
        counterfactual_directory="my_counterfactual_image_directory",
        transform=transform
    )

Evaluation
==========

Once you have attributions, you can run evaluations.
You may want to try different methods for thresholding and smoothing the attributions to get masks.


In this example, we evaluate the results from the DeepLift attribution method.

.. code-block:: python
    :linenos:

    # Defining processors and evaluators
    from quac.evaluation import Processor, Evaluator
    from sklearn.metrics import ConfusionMatrixDisplay

    classifier = load_classifier(...)

    evaluator = Evaluator(
        classifier,
        source_directory="my_source_image_directory",
        counterfactual_directory="my_counterfactual_image_directory",
        attribution_directory="my_attributions_directory/deeplift",
        transform=transform
    )


    cf_confusion_matrix = evaluator.classification_report(
                            data="counterfactuals",  # this is the default
                            return_classifications=False,
                            print_report=True,
                        )

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cf_confusion_matrix,
    )
    disp.show()

    # Run QuAC evaluation on your attribution and store a report
    report = evaluator.quantify(processor=Processor())
    # The report will be stored based on the processor's name, which is "default" by default
    report.store("my_attributions_directory/deeplift/reports")
