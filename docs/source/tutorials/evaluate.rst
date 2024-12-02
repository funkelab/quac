==========
Evaluation
==========

.. attention::
    This tutorial is still under construction. Come back soon for updates!

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
