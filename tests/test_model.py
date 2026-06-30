from quac.training.stargan import (
    build_model,
    Generator,
    StyleEncoder,
    reparameterize,
)
from quac.config import ModelConfig
import warnings

import pytest
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_model():
    args = ModelConfig()
    nets, nets_ema = build_model(**args.model_dump())
    example_input = torch.randn(4, 3, 128, 128)
    example_class = torch.randint(0, 5, (4,))
    example_latent = torch.randn(4, 16)
    # Ensure that the sizes of the outputs are as expected
    latent_style = nets.mapping_network(example_latent, example_class)
    assert latent_style.shape == (4, 64)
    style = nets.style_encoder(example_input, example_class)
    assert style.shape == (4, 64)
    out = nets.generator(example_input, style)
    assert out.shape == example_input.shape


def test_non_variational_generator_unchanged():
    # The default generator is deterministic: encode_latent returns a feature
    # map and forward(x, s) round-trips to an image of the same shape.
    gen = Generator(img_size=64, style_dim=64, input_dim=1)
    assert gen.variational is False
    x = torch.randn(2, 1, 64, 64)
    s = torch.randn(2, 64)
    assert gen(x, s).shape == x.shape
    h = gen.encode_latent(x)
    assert torch.is_tensor(h)
    assert h.shape[1:] == gen.content_shape


def test_variational_generator():
    # The variational generator exposes a VAE-style content posterior at the
    # bottleneck and can decode a content code sampled from the N(0, I) prior.
    gen = Generator(img_size=64, style_dim=64, input_dim=1, variational=True)
    x = torch.randn(4, 1, 64, 64)
    s = torch.randn(4, 64)

    # forward(x, s) still returns an image (samples c internally).
    assert gen(x, s).shape == x.shape

    # encode_latent returns (mu, logvar) at the bottleneck.
    mu, logvar = gen.encode_latent(x)
    assert mu.shape == logvar.shape == (4, *gen.content_shape)

    # reparameterize matches the posterior shape.
    c = reparameterize(mu, logvar)
    assert c.shape == mu.shape

    # generation path: decode a content code drawn from the prior.
    c_prior = torch.randn(4, *gen.content_shape)
    img = gen.decode_latent(c_prior, s)
    assert img.shape == x.shape


def test_non_variational_style_encoder_unchanged():
    # Default style encoder is deterministic: encode_style returns a vector and
    # forward(x, y) returns a style of shape (batch, style_dim).
    enc = StyleEncoder(img_size=64, style_dim=64, num_domains=5, input_dim=1)
    assert enc.variational is False
    x = torch.randn(4, 1, 64, 64)
    y = torch.randint(0, 5, (4,))
    assert enc(x, y).shape == (4, 64)
    s = enc.encode_style(x, y)
    assert torch.is_tensor(s) and s.shape == (4, 64)


def test_variational_style_encoder():
    # Variational style encoder exposes a per-domain (mu, logvar) posterior and
    # samples a style vector from it on forward.
    enc = StyleEncoder(img_size=64, style_dim=64, num_domains=5, input_dim=1,
                       variational=True)
    x = torch.randn(4, 1, 64, 64)
    y = torch.randint(0, 5, (4,))

    mu, logvar = enc.encode_style(x, y)
    assert mu.shape == logvar.shape == (4, 64)

    s = enc(x, y)
    assert s.shape == (4, 64)


def test_build_model_variational_flag():
    # The variational flag flows from config through build_model to both the
    # generator and the style encoder.
    args = ModelConfig(variational=True, input_dim=1, img_size=64)
    nets, _ = build_model(**args.model_dump())
    assert nets.generator.module.variational is True
    assert nets.style_encoder.module.variational is True


def test_generator_warns_and_resizes_on_non_divisible_size():
    # 178 is not divisible by 2**3, so the decoder lands at 176 and resizes to
    # 178 -- the generator warns about the implicit resize.
    with pytest.warns(UserWarning, match="not divisible"):
        gen = Generator(img_size=178, style_dim=64, input_dim=1, variational=True)
    s = torch.randn(2, 64)
    out = gen.decode_latent(torch.randn(2, *gen.content_shape), s)
    assert out.shape == (2, 1, 178, 178)


def test_generator_no_warning_on_clean_sizes():
    # Clean sizes (power-of-2 128, and 160 = 8*20) decode to img_size exactly,
    # so no resize and no warning.
    for img_size in (128, 160):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gen = Generator(img_size=img_size, style_dim=64, input_dim=1,
                            variational=True)
        out = gen.decode_latent(torch.randn(2, *gen.content_shape),
                                torch.randn(2, 64))
        assert out.shape == (2, 1, img_size, img_size)


def test_discriminator_collapses_any_size():
    # The discriminator must reduce to one score per sample regardless of size;
    # 160 (5x5 before the collapse) previously mis-indexed the domain.
    for img_size in (128, 160):
        nets, _ = build_model(img_size=img_size, num_domains=3, input_dim=1)
        x = torch.randn(2, 1, img_size, img_size)
        y = torch.randint(0, 3, (2,))
        # main() collapses spatial to 1x1 -> (B, num_domains, 1, 1)
        feat = nets.discriminator.module.main(x)
        assert feat.shape[-2:] == (1, 1)
        # forward picks one score per sample
        assert nets.discriminator(x, y).shape == (2,)
