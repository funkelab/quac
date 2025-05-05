from quac.training.stargan import build_model
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
