# Generate images

`quac explain` generates images and evaluates counterfactuals in one pipeline.
Set `generator.sampling` to `latent` (default) or `reference`. Reference images
are sampled from the target class. `explanation.samples` defaults to three
generated images per source image and other target class. Each source/generated
image pair is shared across all configured attribution methods.

For Python extensions, implement the `quac.interfaces.ImageGenerator` contract
and pass an instance as `explain(config, generator=generator)`. Its `generate`
method accepts source images as unit-range NCHW tensors, target-class indices,
and an optional batch of reference images. It returns generated images as
unit-range NCHW tensors.
