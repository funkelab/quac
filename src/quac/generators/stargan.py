"""StarGAN backend; signed image values never leave these network adapters."""

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from quac.config import GeneratorConfig
from quac.training.stargan import Generator, MappingNetwork, StyleEncoder, Discriminator


class ImageAdapter(nn.Module):
    """Pad and scale unit-range inputs; optionally undo generator output scaling."""

    def __init__(self, network: nn.Module, size: int, generates: bool = False):
        super().__init__()
        self.network = network
        self.size = size
        self.generates = generates

    def forward(self, image, condition):
        """Call a signed-range StarGAN network behind a unit-range boundary."""
        height, width = image.shape[-2:]
        signed = F.pad(
            image * 2 - 1,
            (0, self.size - width, 0, self.size - height),
            mode="replicate",
        )
        result = self.network(signed, condition)
        if self.generates:
            result = ((result[..., :height, :width] + 1) / 2).clamp(0, 1)
        return result


class StarGAN:
    """Own trainable networks and EMA networks for translation."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        classes: list[str],
        config: GeneratorConfig,
        device: torch.device,
    ):
        self.shape = tuple(shape)
        self.classes = classes
        self.config = config
        self.device = device
        size = max(32, 2 ** (max(shape[1:]) - 1).bit_length())
        if size > 16384:
            raise ValueError("StarGAN image dimensions must not exceed 16384")
        common = dict(
            img_size=size,
            input_dim=shape[0],
            max_conv_dim=config.max_channels,
            base_channels=config.base_channels,
        )
        self.nets = nn.ModuleDict(
            {
                "generator": ImageAdapter(
                    Generator(style_dim=config.style_dim, **common), size, True
                ),
                "style_encoder": ImageAdapter(
                    StyleEncoder(
                        style_dim=config.style_dim, num_domains=len(classes), **common
                    ),
                    size,
                ),
                "discriminator": ImageAdapter(
                    Discriminator(num_domains=len(classes), **common), size
                ),
                "mapping_network": MappingNetwork(
                    config.latent_dim,
                    config.style_dim,
                    len(classes),
                    config.mapping_width,
                ),
            }
        ).to(device)
        self.ema = nn.ModuleDict(
            {
                name: copy.deepcopy(self.nets[name]).eval().requires_grad_(False)
                for name in ("generator", "style_encoder", "mapping_network")
            }
        )

    @torch.no_grad()
    def translate(self, images, targets, reference=None):
        """Sample EMA translations using a reference image or a fresh latent code."""
        if reference is None:
            latent = torch.randn(
                len(images), self.config.latent_dim, device=self.device
            )
            style = self.ema["mapping_network"](latent, targets)
        else:
            style = self.ema["style_encoder"](reference, targets)
        return self.ema["generator"](images, style)

    @torch.no_grad()
    def update_ema(self, beta: float):
        """Update inference weights after an optimization step."""
        for name, network in self.ema.items():
            for average, current in zip(
                network.parameters(), self.nets[name].parameters()
            ):
                average.lerp_(current, 1 - beta)

    def save(self, path: Path, step: int, optimizers: dict):
        """Atomically save weights, architecture, optimizer states, and RNG state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            version=1,
            shape=self.shape,
            classes=self.classes,
            architecture=self._architecture(),
            step=step,
            nets=self.nets.state_dict(),
            ema=self.ema.state_dict(),
            optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
            rng=torch.get_rng_state(),
            rng_cuda=(
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        )
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path: Path) -> dict:
        """Load a native checkpoint or compatible legacy EMA state dictionaries."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        try:
            if "version" in payload:
                if (
                    payload["version"] != 1
                    or tuple(payload["shape"]) != self.shape
                    or payload["classes"] != self.classes
                    or payload["architecture"] != self._architecture()
                ):
                    raise ValueError(
                        "Checkpoint architecture, image shape, or class order does not match"
                    )
                self.nets.load_state_dict(payload["nets"])
                self.ema.load_state_dict(payload["ema"])
            else:
                for name, network in self.ema.items():
                    raw = (
                        network.network
                        if isinstance(network, ImageAdapter)
                        else network
                    )
                    state = {
                        key.removeprefix("module."): value
                        for key, value in payload[name].items()
                    }
                    raw.load_state_dict(state)
                payload = {"step": 0}
        except (KeyError, RuntimeError) as error:
            raise ValueError(f"Incompatible StarGAN checkpoint: {error}") from error
        return payload

    def _architecture(self):
        return self.config.model_dump(exclude={"checkpoint", "sampling"}, mode="json")
