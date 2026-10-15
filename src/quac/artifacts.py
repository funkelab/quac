from contextlib import contextmanager
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import tifffile


class ImageStore:
    """Write TIFF images under an output directory and resolve stored relative paths."""

    def __init__(self, report_path: Path, directory: Path | None = None):
        self.base = report_path.resolve().parent
        self.directory = directory or self.base / "images"

    @contextmanager
    def batch(self):
        """Remove newly created artifacts if their database transaction fails."""
        directory = self.directory / uuid4().hex
        directory.mkdir(parents=True)
        try:
            yield directory
        except BaseException:
            shutil.rmtree(directory)
            raise

    def write(self, path: Path, array) -> str:
        """Save a spatial map or CHW image without changing values or dtype."""
        array = np.asarray(array)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
            if array.shape[-1] == 1:
                array = array[..., 0]
        elif array.ndim != 2:
            raise ValueError("Expected an HW map or grayscale/RGB CHW image")
        tifffile.imwrite(
            path, array, photometric="rgb" if array.ndim == 3 else "minisblack"
        )
        return Path(os.path.relpath(path, self.base)).as_posix()

    def read(self, path: str, *, spatial: bool = False) -> np.ndarray:
        """Read a stored path as an HW map or CHW image, relative to the database."""
        array = tifffile.imread(self.base / path)
        if spatial:
            return array
        return array[None] if array.ndim == 2 else np.moveaxis(array, -1, 0)
