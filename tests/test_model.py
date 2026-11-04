import sys
from types import SimpleNamespace

import pytest
import torch

from quac.config import Config, GeneratorConfig
from quac.data import ImageDataset, write_image
from quac.generators.stargan import ImageAdapter, StarGAN
from quac.training.trainer import _validate, train_stargan
from conftest import MeanClassifier


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


class FakeRun:
    def __init__(self):
        self.logs = {}

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.finished = True

    def log(self, payload, step):
        self.logs.setdefault(step, {}).update(payload)


@pytest.mark.parametrize("sampling", ["latent", "reference"])
def test_validation_logging_spares_training_rng(tmp_path, monkeypatch, sampling):
    torch.set_num_threads(1)
    for split in ["images", "validation"]:
        for name in ["a", "b"]:
            for index in range(2):
                write_image(
                    torch.rand(1, 13, 19), tmp_path / split / name / f"{index}.tif"
                )
    dataset = ImageDataset(tmp_path / "images")
    validation = ImageDataset(tmp_path / "validation")
    run = FakeRun()
    wandb = SimpleNamespace(
        init=lambda **options: run,
        Table=lambda columns, data: data,
        plot=SimpleNamespace(scatter=lambda table, x, y, title: table),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    config = Config.model_validate(
        dict(
            data={"source": str(dataset.source)},
            generator=dict(
                sampling=sampling,
                style_dim=4,
                latent_dim=2,
                base_channels=4,
                max_channels=8,
                mapping_width=8,
            ),
            training=dict(steps=2, batch_size=3, log_every=1, validate_every=1),
            wandb={"enabled": True},
        )
    )
    weights = []
    for options in [{}, dict(validation=validation, classifier=MeanClassifier())]:
        torch.manual_seed(0)
        model = StarGAN(
            dataset.shape, dataset.classes, config.generator, torch.device("cpu")
        )
        train_stargan(model, dataset, config, tmp_path / "checkpoints", **options)
        weights.append(next(model.nets["generator"].parameters()).detach())
    torch.testing.assert_close(weights[0], weights[1])
    assert run.finished and set(run.logs) == {1, 2}
    logged = run.logs[2]
    for name in ["D", "G", "D_latent", "G_reference"]:
        assert f"train/{name}" in logged and f"validation/{name}" in logged
    for pair in ["a->b", "b->a"]:
        assert len(logged[f"validation/scatter/{pair}"]) == 2
        assert 0 <= logged[f"validation/{pair}/translation_rate"] <= 1
    repeated = [_validate(model, validation, MeanClassifier(), config)[0] for _ in "ab"]
    assert repeated[0] == repeated[1]


def test_workers_and_resume_reproduce_uninterrupted_training(tmp_path):
    torch.set_num_threads(1)
    for name in ["a", "b"]:
        for index in range(3):
            write_image(
                torch.rand(1, 13, 19), tmp_path / "images" / name / f"{index}.tif"
            )
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
            training=dict(steps=2, batch_size=2),
        )
    )

    def train(steps, workers, checkpoint=None):
        options = config.model_copy(deep=True)
        options.training.steps = steps
        options.training.num_workers = workers
        options.generator.checkpoint = checkpoint
        torch.manual_seed(0)
        model = StarGAN(
            dataset.shape, dataset.classes, options.generator, torch.device("cpu")
        )
        directory = tmp_path / f"checkpoints-{steps}-{workers}"
        # Validating only while resuming also checks that it spares training RNG.
        validation = dict(validation=dataset, classifier=MeanClassifier())
        path = train_stargan(
            model, dataset, options, directory, **(validation if checkpoint else {})
        )
        return path, next(model.ema["generator"].parameters()).detach()

    _, uninterrupted = train(2, 0)
    checkpoint, _ = train(1, 2)
    _, resumed = train(2, 2, checkpoint)
    torch.testing.assert_close(uninterrupted, resumed, rtol=0, atol=0)
