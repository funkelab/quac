import torch

METHODS = (
    "DIntegratedGradients",
    "DDeepLift",
    "DInGrad",
    "VanillaIntegratedGradients",
    "VanillaDeepLift",
    "residual",
    "random",
)


def normalize(attribution_map: torch.Tensor) -> torch.Tensor:
    """Normalize a channel attribution map to a finite [0,1] spatial attribution map."""
    attribution_map = attribution_map.detach().abs()
    if attribution_map.ndim == 3:
        attribution_map = attribution_map.mean(dim=0)
    if attribution_map.ndim != 2 or not torch.isfinite(attribution_map).all():
        raise ValueError("Attribution map must be a finite HW or CHW array")
    span = attribution_map.max() - attribution_map.min()
    if span == 0:
        return torch.zeros_like(attribution_map)
    return (attribution_map - attribution_map.min()) / span


class PairAttribution:
    """Adapt supported Captum methods to the common pairwise interface."""

    def __init__(self, classifier, method: str, integration_steps: int = 50):
        if method not in METHODS:
            raise ValueError(f"Unknown attribution {method}; choose from {METHODS}")
        if "DeepLift" in method and isinstance(classifier, torch.jit.ScriptModule):
            raise ValueError(
                "DeepLift requires an eager PyTorch classifier; inject classifier= into explain"
            )
        self.classifier = classifier
        self.method = method
        self.integration_steps = integration_steps

    def attribute(self, source_image, generated_image, source_class, target_class):
        """Explain the source-class logit using the generated image as the D-method baseline."""
        if self.method == "residual":
            return normalize(source_image - generated_image)
        if self.method == "random":
            return normalize(torch.rand_like(source_image))
        from captum import attr

        image = source_image[None].detach().requires_grad_(True)
        baseline = generated_image[None].detach() if self.method.startswith("D") else 0
        self.classifier.zero_grad(set_to_none=True)
        if "IntegratedGradients" in self.method:
            result = attr.IntegratedGradients(self.classifier).attribute(
                image,
                baselines=baseline,
                target=source_class,
                n_steps=self.integration_steps,
                internal_batch_size=1,
            )
        elif "DeepLift" in self.method:
            result = attr.DeepLift(self.classifier).attribute(
                image, baselines=baseline, target=source_class
            )
        else:
            gradient = attr.Saliency(self.classifier).attribute(
                image, target=source_class, abs=False
            )
            result = gradient * (generated_image[None] - image)
        return normalize(result[0])
