import pytest
from quac.data import PairedImageDataset, read_image, write_image
import torch
from torchvision import transforms
from skimage.data import astronaut


@pytest.mark.skip("Skip test until we have usable data in CI/CD")
def test_paired_image_folders():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = PairedImageDataset(
        "/nrs/funke/adjavond/data/synapses/test",
        "/nrs/funke/adjavond/data/synapses/counterfactuals/stargan_invariance_v0/test",
        transform=transform,
    )
    x, xc, y, yc = dataset[0]
    assert x.shape == (1, 128, 128)
    assert xc.shape == x.shape
    assert y != yc
    assert len(dataset.classes) == 6


@pytest.fixture(scope="session")
def image_file_path(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data")
    return fn


@pytest.mark.parametrize(
    "format",
    [
        "png",
        "tif",
        pytest.param("jpg", marks=pytest.mark.xfail(reason="JPEG Compression")),
    ],
)
def test_read_write(format, image_file_path):
    """
    Make sure that the write function, followed by the read function, returns the same image.
    """
    image = astronaut().astype("float32")
    # Make it a float, values in 0-1
    image = image / 255.0
    image = torch.from_numpy(image).float().permute(2, 0, 1)  # Change to CxHxW
    path = str(image_file_path / f"test.{format}")
    write_image(image, path)

    # Read the image
    image2 = read_image(path)
    assert image2.shape == image.shape
    assert (image2 == image).all()
