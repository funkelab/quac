from quac.training.stargan import build_model, Generator
from quac.config import ModelConfig
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
    c = gen.reparameterize(mu, logvar)
    assert c.shape == mu.shape

    # generation path: decode a content code drawn from the prior.
    c_prior = torch.randn(4, *gen.content_shape)
    img = gen.decode_latent(c_prior, s)
    assert img.shape == x.shape


def test_build_model_variational_flag():
    # The variational flag flows from config through build_model to the net.
    args = ModelConfig(variational=True, input_dim=1, img_size=64)
    nets, _ = build_model(**args.model_dump())
    assert nets.generator.module.variational is True
