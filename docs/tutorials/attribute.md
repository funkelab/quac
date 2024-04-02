# Attribution and evaluation given counterfactuals

## Attribution
```python
# Defining attributions
from quac.attribute import (
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
```

## Evaluation
Once you have attributions, you can run evaluations.
You may want to try different methods for thresholding and smoothing the attributions to get masks.


In this example, we evaluate the results from the DeepLift attribution method.

```python
# Defining processors and evaluators
from quac.evaluate import Processor, Evaluator

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
                        split_by_source=False,
                        split_by_target=False
                    )

# TODO plot the confusion matrix

report = evaluator.quantify(processor=Processor())
report.store("where_my_report_goes")
```
