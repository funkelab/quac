"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    """Sample `mu + sigma * eps`, eps ~ N(0, I) (VAE reparameterization trick)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ResBlk(nn.Module):
    """
    Residual Block.

    Parameters
    ----------
    dim_in: int
        Number of input channels.
    dim_out: int
        Number of output channels.
    actv: torch.nn.Module
        Activation function.
    normalize: bool
        If True, apply instance normalization. Default: False.
    downsample: bool
        If True, apply average pooling with stride 2. Default: False.
    """

    def __init__(
        self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample=False
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    """
    Adaptive Instance normalization.
    """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    """A Residual block with Adaptive Instance Normalization."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        style_dim: int = 64,
        actv: nn.Module = nn.LeakyReLU(0.2),
        upsample: bool = False,
    ):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class Generator(nn.Module):
    def __init__(
        self,
        img_size=256,
        style_dim=64,
        max_conv_dim=512,
        input_dim=1,
        final_activation=None,
        variational=False,
    ):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.variational = variational
        self.from_rgb = nn.Conv2d(input_dim, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, input_dim, 1, 1, 0),
        )
        if final_activation == "sigmoid":
            # print("Using sigmoid")
            self.final_activation = nn.Sigmoid()
        else:
            # print("Using tanh")
            self.final_activation = nn.Tanh()

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        dim_out = dim_in
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True)
            )  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim))

        # Shape of the content code `c` at the bottleneck: (dim_out, h, h).
        # Each downsampling block halves the spatial size (floored), so a
        # power-of-2 img_size lands at 16x16. Stored so callers can sample
        # `c ~ N(0, I)` of the right shape for variational generation.
        bottleneck_hw = img_size
        for _ in range(repeat_num):
            bottleneck_hw //= 2
        self.content_shape = (dim_out, bottleneck_hw, bottleneck_hw)

        # Variational heads: map the bottleneck feature map to the mean and
        # log-variance of q(c | x). Only built when variational, so the
        # non-variational state_dict is unchanged.
        if variational:
            self.mu = nn.Conv2d(dim_out, dim_out, 1, 1, 0)
            self.logvar = nn.Conv2d(dim_out, dim_out, 1, 1, 0)

    def encode_latent(self, x):
        """Encode an image into the bottleneck content code.

        Returns the feature map `h` when non-variational, or the
        `(mu, logvar)` of the posterior `q(c | x)` when variational.
        """
        h = self.from_rgb(x)
        for block in self.encode:
            h = block(h)
        if self.variational:
            return self.mu(h), self.logvar(h)
        return h

    def decode_latent(self, c, s):
        """Decode a content code `c` and style `s` into an image."""
        x = c
        for block in self.decode:
            x = block(x, s)
        return self.final_activation(self.to_rgb(x))

    def forward(self, x, s):
        if self.variational:
            mu, logvar = self.encode_latent(x)
            c = reparameterize(mu, logvar)
        else:
            c = self.encode_latent(x)
        return self.decode_latent(c, s)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        self.latent_dim = latent_dim
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, style_dim),
                )
            ]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        style_dim=64,
        num_domains=2,
        max_conv_dim=512,
        input_dim=3,
        variational=False,
    ):
        super().__init__()
        dim_in = 2**14 // img_size
        self.style_dim = style_dim
        self.variational = variational

        self.nearest_power = None
        if np.ceil(np.log2(img_size)) != np.floor(np.log2(img_size)):  # Not power of 2
            self.nearest_power = int(np.log2(img_size))

        blocks = []
        blocks += [nn.Conv2d(input_dim, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            # For img_size = 224, repeat_num = 5, dim_out = 256, 512, 512, 512, 512
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        # Variational heads emit (mu, logvar) per domain by doubling the width;
        # non-variational heads keep their original `style_dim` output so the
        # deterministic state_dict is unchanged.
        out_dim = style_dim * 2 if variational else style_dim
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, out_dim)]

    def encode_style(self, x, y):
        """Encode the style for domain `y`.

        Returns `(mu, logvar)` of the posterior `q(s | x, y)` when variational,
        otherwise the deterministic style vector `s`.
        """
        if self.nearest_power is not None:
            # Required for img_size=224 in the retina case
            # Resize input image to nearest power of 2
            x = F.interpolate(x, size=2**self.nearest_power, mode="bilinear")
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, out_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch, out_dim)
        if self.variational:
            mu, logvar = torch.chunk(out, 2, dim=1)
            return mu, logvar
        return out

    def forward(self, x, y):
        if self.variational:
            mu, logvar = self.encode_style(x, y)
            return reparameterize(mu, logvar)
        return self.encode_style(x, y)


class SingleOutputStyleEncoder(StyleEncoder, nn.Module):
    def __init__(
        self,
        img_size=256,
        style_dim=64,
        num_domains=2,
        max_conv_dim=512,
        input_dim=3,
        variational=False,
    ):
        super().__init__()
        dim_in = 2**14 // img_size
        self.style_dim = style_dim
        self.variational = variational

        self.nearest_power = None
        if np.ceil(np.log2(img_size)) != np.floor(np.log2(img_size)):  # Not power of 2
            self.nearest_power = int(np.log2(img_size))

        blocks = []
        blocks += [nn.Conv2d(input_dim, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            # For img_size = 224, repeat_num = 5, dim_out = 256, 512, 512, 512, 512
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        # Making this shared again, to try to learn new things from data.
        # Variational mode doubles the output to emit (mu, logvar).
        out_dim = style_dim * 2 if variational else style_dim
        self.output = nn.Linear(dim_out, out_dim)

    def encode_style(self, x, y):
        if self.nearest_power is not None:
            # Required for img_size=224 in the retina case
            # Resize input image to nearest power of 2
            x = F.interpolate(x, size=2**self.nearest_power, mode="bilinear")
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = self.output(h)
        if self.variational:
            mu, logvar = torch.chunk(out, 2, dim=1)
            return mu, logvar
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512, input_dim=3):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(input_dim, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


def build_model(
    img_size=128,
    style_dim=64,
    input_dim=3,
    latent_dim=16,
    num_domains=4,
    single_output_style_encoder=False,
    final_activation=None,
    variational=False,
    gpu_ids=[0],
):
    generator = nn.DataParallel(
        Generator(
            img_size,
            style_dim,
            input_dim=input_dim,
            final_activation=final_activation,
            variational=variational,
        ),
        device_ids=gpu_ids,
    )
    mapping_network = nn.DataParallel(
        MappingNetwork(latent_dim, style_dim, num_domains),
        device_ids=gpu_ids,
    )
    if single_output_style_encoder:
        print("Using single output style encoder")
        style_encoder = nn.DataParallel(
            SingleOutputStyleEncoder(
                img_size,
                style_dim,
                num_domains,
                input_dim=input_dim,
                variational=variational,
            ),
            device_ids=gpu_ids,
        )
    else:
        style_encoder = nn.DataParallel(
            StyleEncoder(
                img_size,
                style_dim,
                num_domains,
                input_dim=input_dim,
                variational=variational,
            ),
            device_ids=gpu_ids,
        )
    discriminator = nn.DataParallel(
        Discriminator(img_size, num_domains, input_dim=input_dim),
        device_ids=gpu_ids,
    )
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = torch.nn.ModuleDict(
        {
            "generator": generator,
            "mapping_network": mapping_network,
            "style_encoder": style_encoder,
            "discriminator": discriminator,
        }
    )

    nets_ema = torch.nn.ModuleDict(
        {
            "generator": generator_ema,
            "mapping_network": mapping_network_ema,
            "style_encoder": style_encoder_ema,
        }
    )
    return nets, nets_ema


def _unwrap(net):
    """Return the underlying module, peeling off DataParallel if present."""
    return net.module if isinstance(net, nn.DataParallel) else net


@torch.no_grad()
def _broadcast_heads(src_heads, dst_heads):
    """Tile per-domain heads: `dst` head i is initialised from `src` head i % S.

    With a single source domain (S=1) every target head starts from the same
    pretrained weights; they then diverge during fine-tuning.
    """
    n_src = len(src_heads)
    for i, dst_head in enumerate(dst_heads):
        dst_head.load_state_dict(src_heads[i % n_src].state_dict())


@torch.no_grad()
def expand_style_encoder(src, dst):
    """Copy a trained style encoder into one with (possibly) more domains."""
    dst.shared.load_state_dict(src.shared.state_dict())
    if hasattr(src, "unshared") and hasattr(dst, "unshared"):
        _broadcast_heads(src.unshared, dst.unshared)
    else:
        # SingleOutputStyleEncoder has one shared head, no per-domain dimension.
        dst.output.load_state_dict(src.output.state_dict())


@torch.no_grad()
def expand_mapping_network(src, dst):
    """Copy a trained mapping network into one with (possibly) more domains."""
    dst.shared.load_state_dict(src.shared.state_dict())
    _broadcast_heads(src.unshared, dst.unshared)


@torch.no_grad()
def expand_discriminator(src, dst):
    """Copy a trained discriminator into one with (possibly) more domains.

    Everything but the final 1x1 conv (whose output channels index the domain)
    has matching shape and is copied directly; that conv is tiled along its
    output dimension.
    """
    *src_body, src_head = list(src.main)
    *dst_body, dst_head = list(dst.main)
    for src_layer, dst_layer in zip(src_body, dst_body):
        dst_layer.load_state_dict(src_layer.state_dict())
    n_src = src_head.weight.size(0)
    for i in range(dst_head.weight.size(0)):
        dst_head.weight[i].copy_(src_head.weight[i % n_src])
        if dst_head.bias is not None:
            dst_head.bias[i].copy_(src_head.bias[i % n_src])


@torch.no_grad()
def transfer_to_more_domains(
    src_nets,
    dst_nets,
    transfer=("generator", "style_encoder", "discriminator"),
):
    """Initialise the `num_domains=N` `dst_nets` from a trained `src_nets`.

    `src_nets` typically comes from single-domain VAE-GAN pretraining (S=1).
    Domain-agnostic nets (the generator) are copied wholesale; per-domain heads
    are tiled across the new domains. The mapping network is excluded by default
    (introduced fresh at fine-tune time); pass it in `transfer` to broadcast it
    too. `dst_nets` must be built with the same `variational` setting as the
    source so the architectures match.

    `src_nets`/`dst_nets` are dict-like (e.g. the ModuleDict from
    ``build_model``); the generator/style encoder may be taken from the EMA nets
    and the discriminator from the live nets.
    """
    if "generator" in transfer:
        _unwrap(dst_nets["generator"]).load_state_dict(
            _unwrap(src_nets["generator"]).state_dict()
        )
    if "style_encoder" in transfer:
        expand_style_encoder(
            _unwrap(src_nets["style_encoder"]), _unwrap(dst_nets["style_encoder"])
        )
    if "mapping_network" in transfer:
        expand_mapping_network(
            _unwrap(src_nets["mapping_network"]), _unwrap(dst_nets["mapping_network"])
        )
    if "discriminator" in transfer:
        expand_discriminator(
            _unwrap(src_nets["discriminator"]), _unwrap(dst_nets["discriminator"])
        )
