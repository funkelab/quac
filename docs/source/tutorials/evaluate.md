# Evaluation

Finally, we have the **generated** images and the **attributions**, let's run the QuAC evaluation to get and score our final **counterfactuals**.

Once again, we will need the classifier!
Indeed, it is using the change in the classifier's output that we will decide on the quality of our counterfactual.
We also want to use the correct classifier transform, so we will define it here.


```{code-block} python
    :linenos:

    from quac.generate import load_classifier

    classifier_checkpoint = "path/to/classifier/checkpoint"
    classifier = load_classifier(
        checkpoint_path=classifier_checkpoint
    )

    # Defining the transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize(128),
            transforms.Normalize(0.5, 0.5),
        ]
    )
```


Let's run evaluation for the discriminative version of integrated gradients.
If you have been following the tutorials exactly, you will also have run the vanilla version of integrated gradients.
Just swap the attribution directory in the below to run the vanilla version instead!


```{code-block} python
    :linenos:

    # Defining processors and evaluators
    from quac.evaluation import Processor, Evaluator

    attribution_method_name = "discriminative_ig"
    data_directory = "path/to/data/directory"
    counterfactual_directory = "path/to/counterfactual/directory"
    attribution_directory = "path/to/attributions/directory/" + attribution_method_name


    evaluator = Evaluator(
        classifier,
        source_directory=data_directory,
        counterfactual_directory=counterfactual_directory,
        attribution_directory=attribution_directory,
        transform=transform
    )
```


To run the evaluation, we will need to define a processor.
This is the object that takes an **attribution map** and turns it into a binary **attribution mask**.
QuAC provides a default processor that will work for most cases.
Finally, we'll need a place to store the results.

```{code-block} python
    :linenos:

    report_directory = "path/to/store/reports/" + attribution_method_name

    # Run QuAC evaluation on your attribution and store a report
    report = evaluator.quantify(processor=Processor())
    # The report will be stored based on the processor's name, which is "default" by default
    report.store(report_directory)
```

Done! Now all that is left is to go through the report and visualize your final results.


```{include} visualize.md
```