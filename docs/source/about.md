# What even is QuAC?
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
