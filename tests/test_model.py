from types import SimpleNamespace

import pytest
import torch

from quac.config import Config, GeneratorConfig
from quac.data import ImageDataset, write_image
from quac.generators.stargan import ImageAdapter, StarGAN
from quac.training.trainer import train_stargan


class Spy(torch.nn.Module):
    def forward(self, image, condition):
        self.image = image.detach()
        return image


def test_signed_boundary_and_padding():
    spy = Spy()
    adapter = ImageAdapter(spy, 32, generates=True)
    image = torch.rand(1, 3, 13, 19)
    output = adapter(image, None)
    assert spy.image.shape == (1, 3, 32, 32)
    torch.testing.assert_close(spy.image[..., :13, :19], image * 2 - 1)
    torch.testing.assert_close(output, image)


@pytest.mark.parametrize("channels", [1, 3])
def test_real_stargan_translation_and_ema_checkpoint(tmp_path, channels):
    options = GeneratorConfig(
        style_dim=4, latent_dim=2, base_channels=4, max_channels=8, mapping_width=8
    )
    model = StarGAN((channels, 13, 19), ["a", "b"], options, torch.device("cpu"))
    image = torch.rand(1, channels, 13, 19)
    for reference in [None, image]:
        output = model.generate(image, torch.tensor([1]), reference)
        assert output.shape == image.shape
        assert 0 <= output.min() <= output.max() <= 1
    path = tmp_path / "ema.ckpt"
    torch.save(
        {
            name: (net.network if isinstance(net, ImageAdapter) else net).state_dict()
            for name, net in model.ema.items()
        },
        path,
    )
    reloaded = StarGAN(model.shape, model.classes, options, torch.device("cpu"))
    reloaded.load(path)
    torch.testing.assert_close(
        model.generate(image, torch.tensor([1]), image),
        reloaded.generate(image, torch.tensor([1]), image),
    )


def test_real_training_checkpoint_and_resume(tmp_path):
    torch.set_num_threads(1)
    for name in ["a", "b"]:
        write_image(torch.rand(1, 13, 19), tmp_path / "images" / name / "one.tif")
    dataset = ImageDataset(tmp_path / "images")
    config = Config.model_validate(
        dict(
            data={"source": str(dataset.source)},
            generator=dict(
                style_dim=4,
                latent_dim=2,
                base_channels=4,
                max_channels=8,
                mapping_width=8,
            ),
            training=dict(steps=1, batch_size=1),
        )
    )
    model = StarGAN(
        dataset.shape, dataset.classes, config.generator, torch.device("cpu")
    )
    before = next(model.nets["generator"].parameters()).detach().clone()
    path = train_stargan(model, dataset, config, tmp_path / "checkpoints")
    assert not torch.equal(before, next(model.nets["generator"].parameters()))
    reloaded = StarGAN(
        dataset.shape, dataset.classes, config.generator, torch.device("cpu")
    )
    assert reloaded.load(path)["step"] == 1
    torch.testing.assert_close(
        next(model.ema["generator"].parameters()),
        next(reloaded.ema["generator"].parameters()),
    )
    config.generator.checkpoint = path
    config.training.steps = 2
    train_stargan(reloaded, dataset, config, tmp_path / "checkpoints")
    assert reloaded.load(path)["step"] == 2
    incompatible = StarGAN(
        dataset.shape, ["b", "a"], config.generator, torch.device("cpu")
    )
    with pytest.raises(ValueError, match="class order"):
        incompatible.load(path)
