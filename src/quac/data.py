from dataclasses import dataclass
import imageio
from itertools import chain
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import (
    is_image_file,
    has_file_allowed_extension,
)
from torchvision import transforms
from typing import Optional, Callable, List, Tuple, Dict, Union, cast


class RGB:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.size(0) == 1:
            return torch.cat([img, img, img], dim=0)
        return img


class ScaleShift:
    def __init__(self, scale: float = 1, shift: float = 0):
        self.scale = scale
        self.shift = shift

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img * self.scale + self.shift


def create_transform(img_size, grayscale=True, rgb=False, scale=1, shift=0):
    assert not (grayscale and rgb), "Cannot use both grayscale and rgb"
    channel_transform = None
    if grayscale:
        channel_transform = transforms.Grayscale()
    elif rgb:
        channel_transform = RGB()

    transforms_list = [
        transforms.Resize([img_size, img_size]),
        ScaleShift(scale, shift),
    ]

    if channel_transform:
        transforms_list.append(channel_transform)
    return transforms.Compose(transforms_list)


def read_image(path: str) -> torch.Tensor:
    """Reads an image from a file path and returns it as a numpy array.

    Parameters
    ----------
    path: str
        Path to the image to read.

    Returns
    -------
    image: torch.Tensor
        The image read from the path.
        The image is always min-max normalized.
        Its values are between 0 and 1.
        Its dtype is np.float32.
        It is in CHW format (channels, height, width).
    """
    image = imageio.imread(path)
    # Check data type
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.min() < 0 or image.max() > 1:
            # min-max normalization, per-channel
            min_vals = image.min(axis=(0, 1), keepdims=True)
            max_vals = image.max(axis=(0, 1), keepdims=True)
            image = (image - min_vals) / (max_vals - min_vals)
    # Add a channel dimension if it is not there
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    # Permute to CHW format
    image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
    return image


def write_image(image: torch.Tensor, path: Union[str | Path]) -> None:
    """Writes an image to a file path.

    Parameters
    ----------
    path: str
        Path to save the image to.
    image: torch.Tensor
        The image to save. It is assumed to be of data type float32

    The image will be "normalized" to range `[0, 1]` before saving.
    If it is in the range [-1, 1], it will be shifted to [0, 1] using: $ x = (x + 1) / 2 $.
    If it is in any other range, it will be min-max normalized per channel.

    If the file is a JPEG or a PNG, the image will scaled to `[0, 255]` and converted to `np.uint8`.
    Else, the image will be saved as a float32.
    """
    if isinstance(path, str):
        path = Path(path)
    assert image.dtype == torch.float32, "Image must be of type float32"
    if image.ndim == 2:
        image = image.unsqueeze(
            0
        )  # grayscale, add channel dimension so that the rest is grayscale/rgb agnostic
    image = image.permute(1, 2, 0).numpy()  # CHW to HWC
    # Checks whether the data is in `[0, 1]`
    if image.min() < 0 or image.max() > 1:
        # Check if the data is in `[-1, 1]`
        if image.min() >= -1 and image.max() <= 1:
            # Shift the data to `[0, 1]`
            image = (image + 1) / 2
        else:
            # min-max normalization, per-channel
            min_vals = image.min(axis=(0, 1), keepdims=True)
            max_vals = image.max(axis=(0, 1), keepdims=True)
            image = (image - min_vals) / (max_vals - min_vals)
    # Now, data is in `[0, 1]`
    # Check output data type, based on the file format
    if path.suffix in [".jpg", ".jpeg", ".png", ".PNG", ".JPG", ".JPEG"]:
        image = (image * 255).astype(np.uint8)
    imageio.imwrite(path, image.squeeze())  # Remove channel dimension if it is 1


def find_classes(directory):
    """Finds the class folders in a dataset.

    Args:
        directory (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    """
    classes = [d.name for d in os.scandir(directory) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def check_requirements(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> Callable:
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError(
            "'class_to_index' must have at least one entry to collect any samples."
        )

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    return is_valid_file


def listdir(dname):
    fnames = list(
        chain(
            *[
                list(Path(dname).rglob("*." + ext))
                for ext in [
                    "png",
                    "jpg",
                    "jpeg",
                    "JPG",
                    "tiff",
                    "tif",
                ]
            ]
        )
    )
    return fnames


def make_converted_dataset(
    counterfactual_directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    allow_empty: bool = False,
) -> List[Tuple[str, int, int]]:
    """Generates a list of samples of a form (path_to_sample, source_class, target_class)
    for data organized in a counterfactual style directory.

    The dataset is organized in the following way:
    ```
    root_directory/
    ├── class_x
    |   └── class_y
    │       ├── xxx.ext
    │       ├── xxy.ext
    │       └── xxz.ext
    └── class_y
        └── class_x
            ├── 123.ext
            ├── nsdf3.ext
            └── ...
            └── asd932_.ext
    ```

    We want to use the most nested subdirectories as the class labels.
    """
    directory = os.path.expanduser(counterfactual_directory)
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)

    is_valid_file = check_requirements(
        counterfactual_directory,
        class_to_idx,
        extensions,
        is_valid_file,
    )

    instances = []
    available_classes = set()
    for source_class in sorted(class_to_idx.keys()):
        source_dir = os.path.join(directory, source_class)
        if not os.path.isdir(source_dir):
            continue
        target_directories = {}
        for target_class in sorted(class_to_idx.keys()):
            if target_class == source_class:
                continue
            target_dir = os.path.join(directory, source_class, target_class)
            if os.path.isdir(target_dir):
                target_directories[target_class] = target_dir
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                assert source_class != target_class
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = (
                            path,
                            class_to_idx[source_class],
                            class_to_idx[target_class],
                        )
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(source_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes and not allow_empty:
        msg = (
            f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        )
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


def make_paired_dataset(
    directory: str,
    paired_directory: str,
    class_to_idx: Optional[Dict[str, int]],
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    allow_empty: bool = False,
) -> List[Tuple[str, str, int, int]]:
    """Generates a list of samples of a form (path_to_sample, target_path, class_index, target_class_index).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)

    is_valid_file = check_requirements(
        directory, class_to_idx, extensions, is_valid_file
    )

    instances = []
    available_classes = set()
    for source_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[source_class]
        source_dir = os.path.join(directory, source_class)
        if not os.path.isdir(source_dir):
            continue
        target_directories = {}
        for target_class in sorted(class_to_idx.keys()):
            if target_class == source_class:
                continue
            target_dir = os.path.join(paired_directory, source_class, target_class)
            if os.path.isdir(target_dir):
                target_directories[target_class] = target_dir
            for root, _, fnames in sorted(os.walk(source_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        for target_class, target_dir in target_directories.items():
                            target_path = os.path.join(target_dir, fname)
                            if os.path.isfile(target_path) and is_valid_file(
                                target_path
                            ):
                                item = (
                                    path,
                                    target_path,
                                    class_index,
                                    class_to_idx[target_class],
                                )
                                instances.append(item)

                                if source_class not in available_classes:
                                    available_classes.add(source_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes and not allow_empty:
        msg = (
            f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        )
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


def make_paired_attribution_dataset(
    directory: str,
    paired_directory: str,
    attribution_directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    allow_empty: bool = False,
) -> List[Tuple[str, str, str, int, int]]:
    """Generates a list of samples of a form (path_to_sample, path_to_cf, path_to_attr, source_class, target_class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)

    is_valid_file = check_requirements(
        directory, class_to_idx, extensions, is_valid_file
    )

    instances = []
    available_classes = set()
    for source_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[source_class]
        source_dir = os.path.join(directory, source_class)
        if not os.path.isdir(source_dir):
            continue
        target_directories = {}
        attribution_directories = {}
        for target_class in sorted(class_to_idx.keys()):
            if target_class == source_class:
                continue
            target_dir = os.path.join(paired_directory, source_class, target_class)
            if os.path.isdir(target_dir):
                target_directories[target_class] = target_dir
            # Add the attribution directory as well. It is organized in the same way
            # as the paire directory is
            attribution_dir = os.path.join(
                attribution_directory, source_class, target_class
            )
            if os.path.isdir(attribution_dir):
                attribution_directories[target_class] = attribution_dir

            for root, _, fnames in sorted(os.walk(source_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        for target_class, target_dir in target_directories.items():
                            target_path = os.path.join(target_dir, fname)
                            # attribution path must replace the extension to npy
                            attr_filename = fname.split(".")[-2] + ".npy"
                            attr_path = os.path.join(
                                attribution_directories[target_class], attr_filename
                            )

                            if (
                                os.path.isfile(target_path)
                                and is_valid_file(target_path)
                                and os.path.isfile(attr_path)
                            ):
                                item = (
                                    path,
                                    target_path,
                                    attr_path,
                                    class_index,
                                    class_to_idx[target_class],
                                )
                                instances.append(item)

                                if source_class not in available_classes:
                                    available_classes.add(source_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes and not allow_empty:
        msg = (
            f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        )
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


@dataclass
class Sample:
    image: torch.Tensor
    source_class_index: int
    path: Optional[Path] = None
    source_class: Optional[str] = None


# TODO remove?
@dataclass
class ConvertedSample:
    generated: torch.Tensor
    target_class_index: int
    source_class_index: int
    path: Optional[Path] = None
    generated_path: Optional[Path] = None
    source_class: Optional[str] = None
    target_class: Optional[str] = None


@dataclass
class PairedSample:
    image: torch.Tensor
    generated: torch.Tensor
    source_class_index: int
    target_class_index: int
    path: Optional[Path] = None
    generated_path: Optional[Path] = None
    source_class: Optional[str] = None
    target_class: Optional[str] = None


@dataclass
class SampleWithAttribution:
    attribution: np.ndarray
    image: torch.Tensor
    generated: torch.Tensor
    source_class_index: int
    target_class_index: int
    path: Optional[Path] = None
    generated_path: Optional[Path] = None
    source_class: Optional[str] = None
    target_class: Optional[str] = None
    attribution_path: Optional[Path] = None


class DefaultDataset(Dataset):
    """
    A simple dataset that returns images and their names from a directory.
    """

    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = read_image(fname)
        if self.transform is not None:
            img = self.transform(img)
        return img, Path(fname).name

    def __len__(self):
        return len(self.samples)


class PairedImageDataset(Dataset):
    def __init__(
        self, source_directory, paired_directory, transform=None, allow_empty=True
    ):
        """A dataset that loads images from paired directories, where one has images
        generated based on the other.

        Source directory is expected to be of the form:
        ```
        directory/
        ├── class_x
        │   ├── xxx.ext
        │   ├── xxy.ext
        │   └── xxz.ext
        └── class_y
            ├── 123.ext
            ├── nsdf3.ext
            └── ...
            └── asd932_.ext
        ```

        Paired directory should be:
        ```
        directory/
        ├── class_x
        |   └── class_y
        │       ├── xxx.ext
        │       ├── xxy.ext
        │       └── xxz.ext
        └── class_y
            └── class_x
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext
        ```
        note:: this will not work if the file names do not match!

        note:: the transform is applied sequentially to the image, counterfactual, and attribution.
        This means that if there is any randomness in the transform, the three images will fail to match.
        Additionally, the attribution will be a torch tensor when the transform is applied, so no PIL-only transforms
        can be used.
        """
        classes, class_to_idx = find_classes(source_directory)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = make_paired_dataset(
            source_directory,
            paired_directory,
            class_to_idx,
            is_valid_file=is_image_file,
            allow_empty=allow_empty,
        )
        self.transform = transform

    def __getitem__(self, index):
        path, target_path, class_index, target_class_index = self.samples[index]
        sample = read_image(path)
        target_sample = read_image(target_path)
        if self.transform is not None:
            sample = self.transform(sample)
            target_sample = self.transform(target_sample)
        output = PairedSample(
            path=Path(path),
            generated_path=Path(target_path),
            image=sample,
            generated=target_sample,
            source_class_index=class_index,
            target_class_index=target_class_index,
            source_class=self.classes[class_index],
            target_class=self.classes[target_class_index],
        )
        return output

    def __len__(self):
        return len(self.samples)


class ConvertedDataset(Dataset):
    def __init__(self, counterfactual_directory, transform=None, allow_empty=True):
        classes, class_to_idx = find_classes(counterfactual_directory)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = make_converted_dataset(
            counterfactual_directory,
            class_to_idx,
            is_valid_file=is_image_file,
            allow_empty=allow_empty,
        )
        self.transform = transform

    def __getitem__(self, index):
        path, source_class_index, target_class_index = self.samples[index]
        sample = read_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
        output = ConvertedSample(
            generated_path=Path(path),
            generated=sample,
            source_class_index=source_class_index,
            source_class=self.classes[source_class_index],
            target_class_index=target_class_index,
            target_class=self.classes[target_class_index],
        )
        return output

    def __len__(self):
        return len(self.samples)


class PairedWithAttribution(Dataset):
    """This dataset returns both the original and counterfactual images,
    as well as an attribution heatmap.

    note:: the transform is applied sequentially to the image, counterfactual.
    This means that if there is any randomness in the transform, the images will fail to match.
    Additionally, no transform is applied to the attribution.
    """

    def __init__(
        self,
        source_directory,
        paired_directory,
        attribution_directory,
        transform=None,
        allow_empty=True,
    ):
        classes, class_to_idx = find_classes(source_directory)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = make_paired_attribution_dataset(
            source_directory,
            paired_directory,
            attribution_directory,
            class_to_idx,
            is_valid_file=is_image_file,
            allow_empty=allow_empty,
        )
        self.transform = transform

    def __getitem__(self, index) -> SampleWithAttribution:
        (
            path,
            target_path,
            attribution_path,
            class_index,
            target_class_index,
        ) = self.samples[index]
        sample = read_image(path)
        target_sample = read_image(target_path)
        attribution = np.load(attribution_path)
        if self.transform is not None:
            sample = self.transform(sample)
            target_sample = self.transform(target_sample)

        output = SampleWithAttribution(
            path=Path(path),
            generated_path=Path(target_path),
            attribution_path=Path(attribution_path),
            image=sample,
            generated=target_sample,
            attribution=attribution,
            source_class_index=class_index,
            target_class_index=target_class_index,
            source_class=self.classes[class_index],
            target_class=self.classes[target_class_index],
        )
        return output

    def __len__(self):
        return len(self.samples)
