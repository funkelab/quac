import base64
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageColor
from scipy.ndimage import distance_transform_cdt

DEFAULT_COLOR = "#ff1493"
DEFAULT_RGB_COLORS = ("#ff0000", "#00ff00", "#0000ff")
DEFAULT_WIDTH = 1.0


def raster_contour(mask, color=None, width_percent=DEFAULT_WIDTH):
    """Outline each channel above 0.5 in native pixels, with a minimum width of one pixel."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        colors = (
            color
            if color is not None
            else (DEFAULT_RGB_COLORS if len(mask) == 3 else [DEFAULT_COLOR] * len(mask))
        )
        if isinstance(colors, str):
            colors = [colors] * len(mask)
        if len(colors) != len(mask):
            raise ValueError("Provide one contour color per channel")
        return np.maximum.reduce(
            [
                raster_contour(channel, channel_color, width_percent)
                for channel, channel_color in zip(mask, colors)
            ]
        )
    width = math.ceil(max(1, min(mask.shape) * width_percent / 100))
    inside = mask > 0.5
    distance = distance_transform_cdt(np.pad(inside, 1), metric="taxicab")[1:-1, 1:-1]
    border = inside & (distance <= width)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[border, :3] = ImageColor.getrgb(color or DEFAULT_COLOR)
    rgba[border, 3] = 255
    return rgba


def contour_source(mask, color=None, width_percent=DEFAULT_WIDTH):
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
    mask = np.asarray(mask)
    colors = (
        DEFAULT_RGB_COLORS if mask.ndim == 3 and len(mask) == 3 else [DEFAULT_COLOR]
    )
    return {
        "quac_contour": {
            "mask": mask.tolist(),
            "style": f"{','.join(colors)}:1:0",
        }
    }
