from typing import Protocol
import torch


class Translator(Protocol):
    """Translate [0,1] NCHW images using latent or reference sampling."""

    def translate(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class Attribution(Protocol):
    """Return a finite [0,1] spatial importance map for a real/generated pair."""

    def attribute(
        self, real: torch.Tensor, generated: torch.Tensor, source: int, target: int
    ) -> torch.Tensor: ...
