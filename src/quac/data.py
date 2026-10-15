"""Image I/O and class-directory discovery with a single [0,1] contract."""

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def validate_image(image: torch.Tensor) -> None:
    """Require a finite grayscale/RGB CHW tensor in [0,1]."""
    if image.ndim != 3 or image.shape[0] not in (1, 3) or min(image.shape) < 1:
        raise ValueError(f"Expected grayscale/RGB CHW image, got {tuple(image.shape)}")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Image values must be finite and within [0,1]")


def read_image(path: str | Path) -> torch.Tensor:
    """Read an image as float32 CHW; scale integers by their dtype maximum."""
    image = np.asarray(iio.imread(path))
    if np.issubdtype(image.dtype, np.integer):
        if image.min() < 0:
            raise ValueError(f"Negative integer image values: {path}")
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    elif image.dtype == np.bool_:
        image = image.astype(np.float32)
    else:
        image = image.astype(np.float32)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected a 2D image: {path}")
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    validate_image(tensor)
    return tensor


def write_image(image: torch.Tensor, path: str | Path) -> None:
    """Write lossless float TIFF or quantized 8-bit display images."""
    validate_image(image)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    if path.suffix.lower() not in {".tif", ".tiff"}:
        array = np.rint(array * 255).astype(np.uint8)
    iio.imwrite(path, array)


@dataclass(frozen=True)
class Sample:
    """A source image and its class index."""

    path: Path
    label: int


class ImageDataset:
    """Discover sorted class directories and validate consistent image shapes."""

    def __init__(self, source: str | Path):
        self.source = Path(source)
        self.classes = sorted(
            path.name for path in self.source.iterdir() if path.is_dir()
        )
        if len(self.classes) < 2:
            raise ValueError("Source must contain at least two class directories")
        self.samples = []
        for label, name in enumerate(self.classes):
            paths = sorted(
                p
                for p in (self.source / name).rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            )
            if not paths:
                raise ValueError(f"No images found for class {name}")
            self.samples.extend(Sample(path, label) for path in paths)
        self.shape = tuple(read_image(self.samples[0].path).shape)
        for sample in self.samples[1:]:
            if tuple(read_image(sample.path).shape) != self.shape:
                raise ValueError(
                    f"All images must have shape {self.shape}: {sample.path}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return read_image(sample.path), sample.label

    def indices(self, label: int) -> list[int]:
        """Return source indices belonging to a class."""
        return [i for i, sample in enumerate(self.samples) if sample.label == label]
