.. _sec_attribute:

===============================================
Discriminative attribution from Counterfactuals
===============================================

.. attention::
    This tutorial is still under construction. Come back soon for updates!

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
