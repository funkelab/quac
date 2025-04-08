from quac.data import read_image, write_image
from quac.training.stargan import build_model
import tifffile
import torch
import pytest


class TestNonSquareImages:
    """Test that non-square images are handled correctly."""

    non_square_image = torch.rand((3, 100, 200))  # Example non-square image in CHW

    def test_read_write(self, tmp_path):
        """Check that non-square images are read correctly."""
        write_image(self.non_square_image, tmp_path / "test.tif")
        read = read_image(tmp_path / "test.tif")
        assert read.shape == self.non_square_image.shape
        assert read.dtype == self.non_square_image.dtype
        assert torch.equal(read, self.non_square_image)

    @pytest.mark.skip(reason="non-square images not yet supported")
    def test_augmentation(self):
        """Check that non-square images are augmented correctly."""
        pass

    @pytest.mark.xfail(reason="Model does not yet accept square images.")
    def test_model(self):
        """Check that the model can handle non-square images."""
        model = build_model()
        # Assuming the model has a method to process images
        processed_image = model(self.non_square_image)
        assert processed_image.shape == self.non_square_image.shape
        assert processed_image.dtype == self.non_square_image.dtype


def test_non_normalized_tiffs(tmp_path):
    """Test that non-normalized TIFF images are handled correctly."""
    normed = torch.rand(3, 100, 100)  # Example normalized image in CHW
    normed[0, 0, 0] = 0.0
    normed[0, 99, 99] = 1.0
    non_normed = (normed * 255).type(torch.uint8)  # Example non-normalized image in CHW
    # use Tiffile to write the image
    path = tmp_path / "non_norm_image.tif"
    tifffile.imwrite(path, non_normed.permute(1, 2, 0).numpy())
    # Read the image using read_image function
    read = read_image(path)
    assert read.shape == non_normed.shape
    assert read.dtype == torch.float32
    # Not asserting equality because of the type switch and normalization
    # but we can check that the values are close
    assert torch.allclose(read, normed, atol=1e-2, rtol=1e-3)


@pytest.mark.xfail(reason="Two random images are not close.")
def test_tolerance(tmp_path):
    """
    Test that a significant shift in pixel values is detected.
    This test is expected to fail because the two images are not close.
    This ensures that the tolerance check above is reasonable.
    """
    image1 = torch.rand(3, 100, 100)
    image2 = image1.clone().detach() + 0.1

    # Save the images
    write_image(image1, tmp_path / "image1.tif")
    write_image(image2, tmp_path / "image2.tif")
    # Read the images
    read1 = read_image(tmp_path / "image1.tif")
    read2 = read_image(tmp_path / "image2.tif")
    # Check whether the images are the same
    assert torch.equal(read1, read2)
    assert torch.allclose(read1, read2, atol=1e-2, rtol=1e-3)
