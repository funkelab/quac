# So you've decided to use QuAC...

Great! In these tutorials we'll go through the process of setting up QuAC for your images.
We built QuAC with biological images in mind, so those will be our analogies here. However, you're very welcome to use QuAC with any kind of image data!
If you're interested in using it with non-image data, please [contact us](mailto:adjavond%40hhmi.org?subject=QuAC%20Help)

But first, a quick overview of the method.

## What even is QuAC?
QuAC, or Quantitative Attributions with Counterfactuals, is a method for generating and scoring visual counterfactual explanations of an image clasifier.
Let's assume, for instance, that you have images of cells grown in two different conditions.
To your eye, the phenotypic difference between the two conditions is hidden within the cell-to-cell variability of the dataset, but you know it is there because you've trained a classifier to differentiate the two conditions and it works. So how do you pull out the differences?

Assuming that you already have a classifier that does your task, we begin by training a generative neural network to convert your images from one class to another. Here, we'll use a StarGAN. This allows us to go from our real, **query** image, to our **generated** image.
Using information learned from **reference** images, the StarGAN is trained in such a way that the **generated** image will have a different class!

While very powerful, these generative networks *can potentially* make some changes that are not necessary to the classification.
In our example below, the **generated** image's membrane has been unnecessarily changed.
We use Discriminative Attribution methods to generate a set of candidate attribution masks.
Among these, we are looking for the smallest mask that has the greatest change in the classification output.
By taking only the changes *within* that mask, we create the counterfactual image.
It is as close as possible to the original image, with only the necessary changes to turn its class!

```{image} assets/overview.png
    :width: 800
    :align: center
```

Before you begin, make sure you've installed QuAC by following the [Installation guide](install).

## The classifier
To use QuAC, we assume that you have a classifier trained on your data.
There are many different packages already to help you do that, for this code-base we will need you to have the weights to your classifier as a JIT-compiled `pytorch` model.

If you just want to try QuAC as a learning experience, you can use one of the datasets [in this collection](https://doi.org/10.25378/janelia.c.7620737.v1), and the pre-trained models we provide.

## The conversion network

Once you've set up your data and your classifier, you can move on to training the conversion network.
We'll use a StarGAN for this.
There are two options for training the StarGAN, but we recommend [training it using a YAML file](tutorials/train_yaml).
This will make it easier to keep track of your experiments!
Note that in both cases, you will need the JIT-compiled classifier model!

Once you've trained a decent model, you can generate a set of images using the [image generation tutorial](tutorials/generate).
We recommend taking a look at your generated images, to make sure that they look like what you expect.
If that is the case, you can move on to the next steps!

## Attribution and evaluation

With the generated images in hand, we can now run the [attribution](tutorials/attribute) step, then the [evaluation](tutorials/evaluate) step.
These two steps allow us to overcome the limitations of the generative network to create *truly* minimal counterfactual images, and to score the query-counterfactual pairs based on how well they explain the classifier.

## Visualizing results

Finally, we can visualize the results of the attribution and evaluation steps using the [visualization tutorial](tutorials/visualize).
This will allow you to see the quantification results, in the form of QuAC curves.
It will also help you choose the best attribution method for each example, and load the counterfactual visual explanations for these examples.

## Table of Contents
Here's a list of all available tutorials, in case you want to navigate directly to one of them.

```{toctree}
    :maxdepth: 1

    Data <tutorials/data>
    Training the generator <tutorials/train_yaml>
    Generating images <tutorials/generate>
    Attribution <tutorials/attribute>
    Evaluation <tutorials/evaluate>
    Visualizing results <tutorials/visualize>
```
