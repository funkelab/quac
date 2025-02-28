# Discriminative attribution from Counterfactuals

Now that we have generated counterfactuals, we will refine our **generated** images into **counterfactuals** using discriminative attribution.
Remember that although the conversion network is trained to keep as much of the image fixed as possible, it is not perfect.
This means that there may still be regions of the **generated** image that differ from the **query** image *even if they don't need to*.
Luckily, we have a classifier that can help us identify and keep only the necessary regions of change.

The first thing that we want to do is load the classifier.

```{code-block} python
    :linenos:

    classifier_checkpoint = "path/to/classifier/checkpoint"

    from quac.generate import load_classifier
    classifier = load_classifier(
        checkpoint_path=classifier_checkpoint
    )
```

Next, we will define the attribution that we want to use.
In this tutorial, we will use Discriminative Integrated Gradients, using the classifier as a baseline.
As a comparison, we will also use Vanilla Integrated Gradients, which uses a black image as a baseline.
This will allow us to identify the regions of the image that are most important for the classifier to make its decision.
Later in the [evaluation](evaluate) tutorial, we will process these attributions into masks, and finally get our counterfactuals.



```{code-block} python
    :linenos:

    # Parameters
    attribution_directory = "path/to/store/attributions"

    # Defining attributions
    from quac.attribution import (
        DIntegratedGradients,
        VanillaIntegratedGradients,
        AttributionIO
    )
    from torchvision import transforms

    attributor = AttributionIO(
        attributions = {
            "discriminative_ig" : DIntegratedGradients(classifier),
            "vanilla_ig" : VanillaIntegratedGradients(classifier)
        },
        output_directory = atttribution_directory
    )
```

Finally, we want to make sure that the images are processed as we would like for the classifier.
Here, we will simply define a set of `torchvision` transforms to do this, we will pass them to the `attributor` object.
Keep in mind that if you processed your data in a certain way when training your classfier, you will need to use the same processing here.

```{code-block} python
    :linenos:

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize(128),
            transforms.Normalize(0.5, 0.5),
        ]
    )
```

Finally, let's run the attributions.

```{code-block} python
    :linenos:

    data_directory = "path/to/data/directory"
    counterfactual_directory = "path/to/counterfactual/directory"

    # This will run attributions and store all of the results in the output_directory
    # Shows a progress bar
    attributor.run(
        source_directory=data_directory,
        counterfactual_directory=counterfactual_directory,
        transform=transform
    )
```

If you look into the `attribution_directory`, you should see a set of attributions.
They will be organized in the following way:

```{code-block} bash

    attribution_directory/
        attribution_method_name/
            source_class/
                target_class/
                    image_name.npy
```
In the next tutorial, we will use these attributions to generate masks and finally get our counterfactuals.
