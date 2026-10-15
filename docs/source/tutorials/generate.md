# Sample translations

`quac explain` samples translations and evaluates them in one pipeline. Set
`generator.sampling` to `latent` (default) or `reference`. References are sampled
from the target class. `explanation.samples` defaults to three translations per
image and other target class. Each generated pair is shared across all configured
attribution methods.

For Python extensions, implement the `quac.interfaces.Translator` contract and
pass an instance as `explain(config, generator=translator)`. The interface accepts
unit-range NCHW images, target labels, and an optional reference batch, and returns
unit-range NCHW translated images.
