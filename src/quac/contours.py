import base64
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageColor
from scipy.ndimage import distance_transform_cdt

DEFAULT_COLOR = "#ff1493"
DEFAULT_WIDTH = 1.0


def raster_contour(mask, color=DEFAULT_COLOR, width_percent=DEFAULT_WIDTH):
    """Outline mask values above 0.5 in native pixels, with a minimum width of one pixel."""
    mask = np.asarray(mask)
    width = math.ceil(max(1, min(mask.shape) * width_percent / 100))
    inside = mask > 0.5
    distance = distance_transform_cdt(np.pad(inside, 1), metric="taxicab")[1:-1, 1:-1]
    border = inside & (distance <= width)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[border, :3] = ImageColor.getrgb(color)
    rgba[border, 3] = 255
    return rgba


def contour_source(mask, color=DEFAULT_COLOR, width_percent=DEFAULT_WIDTH):
    """Encode a native-resolution contour as a transparent PNG."""
    buffer = BytesIO()
    Image.fromarray(raster_contour(mask, color, width_percent)).save(
        buffer, format="PNG"
    )
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def contour_metadata(mask):
    """Carry the stored blurred mask and its displayed contour style into the summary."""
    return {
        "quac_contour": {
            "mask": np.asarray(mask).tolist(),
            "style": f"{DEFAULT_COLOR}:1:0",
        }
    }
