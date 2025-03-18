# Glossary
```{glossary}
query image
  An image for which we want to generate an explanation.

source class
  The original class of the query image. This is the class we want to change from.

target class
  The target class for conversion. This is the class we want to change to.

generated image
  An image made by the conversion network by perturbing the query image enough to change its classification. 

conversion network
  A generative neural network that takes a query image and a style as an input, and outputs a generated image of a given class.

style
  A representation of class-dependent features. It can be obtained from a reference image or from a latent vector.

reference image
  An image of the target class, from which style can be extracted. 

latent
  A random (gaussian) tensor, from which style can be extracted.

counterfactual image
  A minimal perturbation to a query image to change its class, made by combining the query image with the generated image.
```
