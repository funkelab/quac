import imageio.v2 as imageio
import numpy as np
import pytest
import torch

import quac.attribution
from quac.attribution import AttributionIO, BaseAttribution
from quac.config import AttributionConfig
from quac.data import create_transform
from quac.generate import load_classifier

IMG_SHAPE = (1, 32, 32)
NUM_CLASSES = 3
# Same device selection as example_experiment/run_attribution.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Every attribution class in quac.attribution, so new methods get tested automatically
ATTRIBUTION_NAMES = sorted(
    name
    for name, obj in vars(quac.attribution).items()
    if isinstance(obj, type)
    and issubclass(obj, BaseAttribution)
    and obj is not BaseAttribution
)
DISCRIMINATIVE_NAMES = ["DDeepLift", "DInGrad", "DIntegratedGradients"]
VANILLA_NAMES = ["VanillaDeepLift", "VanillaIntegratedGradients"]


class SmallClassifier(torch.nn.Module):
    """
    Small CNN classifier. Each ReLU is a separate module, as required by DeepLift.
    """

    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.features(x)


@pytest.fixture(scope="module")
def classifier_checkpoint(tmp_path_factory):
    """A torchscript checkpoint, the format expected by `load_classifier`."""
    torch.manual_seed(0)
    model = SmallClassifier(in_channels=IMG_SHAPE[0])
    checkpoint = tmp_path_factory.mktemp("classifier") / "model.pt"
    torch.jit.script(model).save(str(checkpoint))
    return checkpoint


@pytest.fixture
def classifier(classifier_checkpoint):
    # Loaded the same way as in example_experiment/run_*.py
    classifier = load_classifier(
        checkpoint=str(classifier_checkpoint),
        scale=2.0,
        shift=-1.0,
    )
    classifier.to(DEVICE)
    classifier.eval()
    return classifier


@pytest.fixture
def images():
    """A real image, and a generated image that only differs inside `region`."""
    generator = torch.Generator().manual_seed(1)
    real = torch.rand(IMG_SHAPE, generator=generator)
    generated = real.clone()
    region = (slice(None), slice(8, 16), slice(10, 20))
    generated[region] = torch.rand(generated[region].shape, generator=generator)
    return real, generated, region


def test_all_attributions_discovered():
    assert set(DISCRIMINATIVE_NAMES + VANILLA_NAMES) <= set(ATTRIBUTION_NAMES)
    # The defaults used by the example experiment must be available
    assert set(AttributionConfig().attributions) <= set(ATTRIBUTION_NAMES)


@pytest.mark.parametrize("name", ATTRIBUTION_NAMES)
@pytest.mark.parametrize("batch_size", [None, 4])
def test_attribution_shape_and_range(name, batch_size, classifier, images):
    real, generated, _ = images
    if batch_size is not None:
        real = torch.stack([real] * batch_size)
        generated = torch.stack([generated] * batch_size)
    attribution = getattr(quac.attribution, name)(classifier)

    result = attribution.attribute(real, generated, 0, 1, device=DEVICE)

    assert isinstance(result, np.ndarray)
    assert result.shape == real.shape
    assert np.all(np.isfinite(result))
    # Normalized attributions are between 0 and 1, per sample
    assert result.min() >= 0
    assert result.max() <= 1
    per_sample = result.reshape(-1, int(np.prod(IMG_SHAPE)))
    np.testing.assert_allclose(per_sample.min(axis=1), 0, atol=1e-6)
    np.testing.assert_allclose(per_sample.max(axis=1), 1, atol=1e-6)


@pytest.mark.parametrize("name", DISCRIMINATIVE_NAMES)
def test_discriminative_attribution_is_localized(name, classifier, images):
    """Pixels where the real and generated images match get zero attribution."""
    real, generated, region = images
    attribution = getattr(quac.attribution, name)(classifier)

    result = attribution.attribute(real, generated, 0, 1, device=DEVICE)

    outside = np.ones(IMG_SHAPE, dtype=bool)
    outside[region] = False
    np.testing.assert_allclose(result[outside], 0, atol=1e-6)
    assert result[region].max() == pytest.approx(1)


@pytest.mark.parametrize("name", DISCRIMINATIVE_NAMES)
def test_discriminative_attribution_identical_images(name, classifier, images):
    """If nothing changed, there is nothing to attribute."""
    real, _, _ = images
    attribution = getattr(quac.attribution, name)(classifier, normalize=False)

    result = attribution.attribute(real, real.clone(), 0, 1, device=DEVICE)

    np.testing.assert_allclose(result, 0, atol=1e-6)


@pytest.mark.parametrize("name", ["DDeepLift", "DIntegratedGradients"])
def test_discriminative_attribution_completeness(name, classifier, images):
    """
    Integrated gradients and DeepLift attributions should sum to the change in the
    classifier's output for the source class, between the generated and real images.
    """
    real, generated, _ = images
    source_class = 0
    attribution = getattr(quac.attribution, name)(classifier, normalize=False)

    result = attribution.attribute(real, generated, source_class, 1, device=DEVICE)

    with torch.no_grad():
        logits = classifier(torch.stack([real, generated]).to(DEVICE)).cpu()
    expected = (logits[0, source_class] - logits[1, source_class]).item()
    assert abs(expected) > 1e-5, "Test images do not change the classifier output"
    # Integrated gradients is approximated with a Riemann sum, hence the tolerance
    assert result.sum() == pytest.approx(expected, rel=5e-2, abs=1e-7)


@pytest.mark.parametrize("name", VANILLA_NAMES)
def test_vanilla_attribution_ignores_generated(name, classifier, images):
    real, generated, _ = images
    attribution = getattr(quac.attribution, name)(classifier)

    result = attribution.attribute(real, generated, 0, 1, device=DEVICE)
    other = attribution.attribute(real, torch.rand(IMG_SHAPE), 0, 1, device=DEVICE)

    np.testing.assert_allclose(result, other, atol=1e-6)


def test_attribution_io_run(tmp_path, classifier):
    """Mirrors example_experiment/run_attribution.py on a tiny paired dataset."""
    source_directory = tmp_path / "source"
    generated_directory = tmp_path / "generated"
    output_directory = tmp_path / "attributions"
    rng = np.random.default_rng(2)
    classes = ["class_a", "class_b"]
    for source_class in classes:
        target_class = next(c for c in classes if c != source_class)
        (source_directory / source_class).mkdir(parents=True)
        (generated_directory / source_class / target_class).mkdir(parents=True)
        for i in range(2):
            image = rng.integers(0, 256, IMG_SHAPE[1:], dtype=np.uint8)
            imageio.imwrite(source_directory / source_class / f"{i}.png", image)
            image[8:16, 10:20] = rng.integers(0, 256, (8, 10), dtype=np.uint8)
            imageio.imwrite(
                generated_directory / source_class / target_class / f"{i}.png", image
            )

    attributions_dict = {
        name: getattr(quac.attribution, name)(classifier) for name in ATTRIBUTION_NAMES
    }
    attributor = AttributionIO(
        attributions=attributions_dict, output_directory=str(output_directory)
    )
    transform = create_transform(
        img_size=IMG_SHAPE[1], grayscale=True, rgb=False, scale=1, shift=0
    )
    attributor.run(
        source_directory=str(source_directory),
        generated_directory=str(generated_directory),
        transform=transform,
        device=DEVICE,
    )

    for name in ATTRIBUTION_NAMES:
        for source_class, target_class in [classes, classes[::-1]]:
            directory = output_directory / name / source_class / target_class
            files = sorted(directory.glob("*.npy"))
            assert [f.stem for f in files] == ["0", "1"]
            for f in files:
                result = np.load(f)
                assert result.shape == IMG_SHAPE
                assert np.all(np.isfinite(result))
