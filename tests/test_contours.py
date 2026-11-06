import base64
from io import BytesIO

import numpy as np
from PIL import Image
from scipy.ndimage import label

from quac.contours import contour_source, raster_contour


def test_native_contour_uses_dominant_generated_pixels():
    mask = np.zeros((12, 14), dtype=np.float32)
    mask[2:10, 3:11] = 0.7
    mask[2, 3] = 0.5
    contour = raster_contour(mask)
    assert contour[2, 3, 3] == 0
    np.testing.assert_array_equal(contour[2, 4], [255, 20, 147, 255])
    assert contour[3, 3, 3] == 255
    assert contour[5, 6, 3] == 0
    assert np.all(contour[..., 3][mask <= 0.5] == 0)
    assert set(np.unique(contour[..., 3])) == {0, 255}


def test_width_is_based_on_the_smaller_dimension_and_at_least_one_pixel():
    mask = np.zeros((60, 120))
    mask[10:50, 20:100] = 1
    thin = raster_contour(mask, width_percent=0)
    default = raster_contour(mask)
    np.testing.assert_array_equal(thin, default)
    assert thin[10, 40, 3] == 255 and thin[11, 40, 3] == 0
    thick = raster_contour(mask, width_percent=5)
    assert np.all(thick[10:13, 40, 3] == 255)
    assert thick[13, 40, 3] == 0


def test_diagonal_native_contour_is_eight_connected():
    mask = np.eye(12)
    contour = raster_contour(mask, width_percent=0)
    np.testing.assert_array_equal(contour[..., 3] > 0, mask > 0.5)
    assert label(contour[..., 3] > 0, structure=np.ones((3, 3)))[1] == 1


def test_empty_full_and_holed_masks():
    assert not raster_contour(np.zeros((12, 14)))[..., 3].any()
    full = raster_contour(np.ones((12, 14)))
    assert np.all(full[0, :, 3] == 255)
    assert np.all(full[-1, :, 3] == 255)
    assert np.all(full[:, 0, 3] == 255)
    assert np.all(full[:, -1, 3] == 255)
    assert not full[1:-1, 1:-1, 3].any()
    mask = np.zeros((20, 20))
    mask[2:18, 2:18] = 1
    mask[8:12, 8:12] = 0
    contour = raster_contour(mask)
    assert not contour[8:12, 8:12, 3].any()
    assert np.all(contour[7, 8:12, 3] == 255)


def test_png_preserves_contour_pixels_and_transparency():
    mask = np.eye(12)
    encoded = contour_source(mask)
    image = np.asarray(Image.open(BytesIO(base64.b64decode(encoded.split(",", 1)[1]))))
    np.testing.assert_array_equal(image, raster_contour(mask))


def test_channel_contours_retain_nested_boundaries():
    masks = np.zeros((3, 20, 20))
    masks[0, 2:18, 2:18] = 0.8
    masks[1, 6:14, 6:14] = 0.9
    masks[2, 9:11, 9:11] = 1
    contour = raster_contour(masks)
    assert contour.shape == (20, 20, 4)
    assert contour[2, 9, 3] == contour[6, 9, 3] == contour[9, 9, 3] == 255
    assert contour[4, 9, 3] == 0
