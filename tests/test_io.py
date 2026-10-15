import imageio.v3 as iio
import numpy as np
import pytest
import torch

from quac.data import ImageDataset, read_image, write_image


@pytest.mark.parametrize("channels", [1, 3])
def test_float_tiff_roundtrip(tmp_path, channels):
    original = torch.rand(channels, 9, 13)
    path = tmp_path / "image.tif"
    write_image(original, path)
    assert torch.equal(read_image(path), original)


def test_integer_normalization_preserves_brightness(tmp_path):
    path = tmp_path / "image.tif"
    iio.imwrite(path, np.full((5, 7), 32768, dtype=np.uint16))
    assert torch.allclose(read_image(path), torch.full((1, 5, 7), 32768 / 65535))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_float_image(tmp_path, value):
    path = tmp_path / "image.tif"
    iio.imwrite(path, np.full((5, 7), value, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        read_image(path)


def test_dataset_discovery(tmp_path):
    for name in ["zebra", "apple"]:
        write_image(torch.zeros(1, 9, 13), tmp_path / name / "one.tif")
    dataset = ImageDataset(tmp_path)
    assert dataset.classes == ["apple", "zebra"]
    assert dataset.shape == (1, 9, 13)
    write_image(torch.zeros(3, 9, 13), tmp_path / "apple/two.tif")
    with pytest.raises(ValueError, match="shape"):
        ImageDataset(tmp_path)
