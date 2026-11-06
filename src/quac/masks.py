import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def ellipse_kernel(size: int) -> np.ndarray:
    """Return a square raster footprint of an ellipse, anchored at size // 2."""
    if size < 1:
        raise ValueError("Closing size must be positive")
    radius = size // 2
    y, x = np.ogrid[:size, :size]
    half_width = np.rint(
        radius * np.sqrt(np.maximum(0, 1 - ((y - radius) / max(1, radius)) ** 2))
    )
    return abs(x - radius) <= half_width


def close_mask(mask, kernel):
    """Close an HW or CHW binary mask independently in each channel."""
    footprint = kernel[None] if mask.ndim == 3 else kernel
    dilated = maximum_filter(mask, footprint=footprint, mode="constant", cval=0)
    return minimum_filter(dilated, footprint=footprint, mode="constant", cval=1)
