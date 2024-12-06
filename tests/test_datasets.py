from quac.data import PairedImageDataset
from torchvision import transforms
import pytest


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
