from typing import Protocol
import torch


class ImageGenerator(Protocol):
    """Generate images from [0,1] NCHW source images using latent or reference sampling."""

    def generate(
        self,
        source_images: torch.Tensor,
        target_classes: torch.Tensor,
        reference_images: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class Attribution(Protocol):
    """Return a finite [0,1] attribution map for a source/generated image pair."""

    def attribute(
        self,
        source_image: torch.Tensor,
        generated_image: torch.Tensor,
        source_class: int,
        target_class: int,
    ) -> torch.Tensor: ...
