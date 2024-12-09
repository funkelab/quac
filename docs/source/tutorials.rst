================================
So you've decided to use QuAC...
================================

Great! In these tutorials we'll go through the process of setting up QuAC for your images.
We built QuAC with biological images in mind, so those will be our analogies here. However, you're very welcome to use QuAC with any kind of image data!
If you're interested in using it with non-image data, please :email:`contact us <adjavond@hhmi.org>`.

But first, a quick overview of the method.

What even is QuAC?
==================
QuAC, or Quantitative Attributions with Counterfactuals, is a method for generating and scoring visual counterfactual explanations of an image clasifier.
Let's assume, for instance, that you have images of cells grown in two different conditions.
To your eye, the phenotypic difference between the two conditions is hidden within the cell-to-cell variability of the dataset, but you know it is there because you've trained a classifier to differentiate the two conditions and it works. So how do you pull out the differences?

We begin by training a generative neural network to convert your images from one class to another. Here, we'll use a StarGAN. This allows us to go from our real, **query** image, to our **generated** image.
Using information learned from **reference** images, the StarGAN is trained in such a way that the **generated** image will have a different class!

While very powerful, these generative networks *can potentially* make some changes that are not necessary to the classification.
In our example below, the **generated** image's membrane has been unnecessarily changed.
We use Discriminative Attribution methods to generate a set of candidate attribution masks.
Among these, we are looking for the smallest mask that has the greatest change in the classification output.
By taking only the changes *within* that mask, we create the counterfactual image.
It is as close as possible to the original image, with only the necessary changes to turn its class!

.. image:: assets/overview.png
    :width: 800
    :align: center

Before you begin, download [the data]() and [the pre-trained models]() for an example.
Then, make sure you've installed QuAC by following the :doc:`Installation guide <install>`.


The conversion network
===============================

You have two options for training the StarGAN, you can either :doc:`define parameters directly in Python <tutorials/train>` or :doc:`train it using a YAML file <tutorials/train_yaml>`.
We recommend the latter, which will make it easier to keep track of your experiments!
Once you've trained a decent model, generate a set of images using the :doc:`image generation tutorial <tutorials/generate>` before moving on to the next steps.

.. toctree::
    :maxdepth: 1

    tutorials/train
    tutorials/train_yaml
    Generating images <tutorials/generate>

Attribution and evaluation
==========================

With the generated images in hand, we can now run the attribution and evaluation steps.
These two steps allow us to overcome the limitations of the generative network to create *truly* minimal counterfactual images, and to score the query-counterfactual pairs based on how well they explain the classifier.

.. toctree::
    :maxdepth: 1

    Attribution <tutorials/attribute>
    Evaluation <tutorials/evaluate>
    Visualizing results <tutorials/visualize>
