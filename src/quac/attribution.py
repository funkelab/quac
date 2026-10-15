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


def normalize(importance: torch.Tensor) -> torch.Tensor:
    """Convert channel importance to a finite [0,1] spatial map."""
    importance = importance.detach().abs()
    if importance.ndim == 3:
        importance = importance.mean(dim=0)
    if importance.ndim != 2 or not torch.isfinite(importance).all():
        raise ValueError("Attribution must be a finite HW or CHW map")
    span = importance.max() - importance.min()
    if span == 0:
        return torch.zeros_like(importance)
    return (importance - importance.min()) / span


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

    def attribute(self, real, generated, source, target):
        """Explain the source-class logit, using generated images as D-method baselines."""
        if self.method == "residual":
            return normalize(real - generated)
        if self.method == "random":
            return normalize(torch.rand_like(real))
        from captum import attr

        image = real[None].detach().requires_grad_(True)
        baseline = generated[None].detach() if self.method.startswith("D") else 0
        self.classifier.zero_grad(set_to_none=True)
        if "IntegratedGradients" in self.method:
            result = attr.IntegratedGradients(self.classifier).attribute(
                image,
                baselines=baseline,
                target=source,
                n_steps=self.integration_steps,
                internal_batch_size=1,
            )
        elif "DeepLift" in self.method:
            result = attr.DeepLift(self.classifier).attribute(
                image, baselines=baseline, target=source
            )
        else:
            gradient = attr.Saliency(self.classifier).attribute(
                image, target=source, abs=False
            )
            result = gradient * (generated[None] - image)
        return normalize(result[0])
