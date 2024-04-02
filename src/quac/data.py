from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import (
    default_loader,
    is_image_file,
    has_file_allowed_extension,
)
from typing import Optional, Callable, List, Tuple, Dict, Union, cast


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
):
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


def make_counterfactual_dataset(
    counterfactual_directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class)
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
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target_class])
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(source_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
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
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, str, int, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

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
                        if os.path.isfile(target_path) and is_valid_file(target_path):
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
    if empty_classes:
        msg = (
            f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        )
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class PairedImageFolders(Dataset):
    def __init__(self, source_directory, paired_directory, transform=None):
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
        Note that this will not work if the file names do not match!
        """
        classes, class_to_idx = find_classes(source_directory)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = make_paired_dataset(
            source_directory,
            paired_directory,
            class_to_idx,
            is_valid_file=is_image_file,
        )
        self.transform = transform

    def __getitem__(self, index):
        path, target_path, class_index, target_class_index = self.samples[index]
        sample = default_loader(path)
        target_sample = default_loader(target_path)
        # TODO ensure that the transforms are applied the same way to both images!
        if self.transform is not None:
            sample = self.transform(sample)
            target_sample = self.transform(target_sample)
        output = Sample(
            path=Path(path),
            counterfactual_path=Path(target_path),
            image=sample,
            counterfactual=target_sample,
            source_class_index=class_index,
            target_class_index=target_class_index,
            source_class=self.classes[class_index],
            target_class=self.classes[target_class_index],
        )
        return output

    def __len__(self):
        return len(self.samples)


class CounterfactualDataset(Dataset):
    def __init__(self, counterfactual_directory, transform=None):
        classes, class_to_idx = find_classes(counterfactual_directory)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = make_counterfactual_dataset(
            counterfactual_directory,
            class_to_idx,
            is_valid_file=is_image_file,
        )
        self.transform = transform

    def __getitem__(self, index):
        path, class_index = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        output = CounterfactualSample(
            counterfactual_path=Path(path),
            counterfactual=sample,
            target_class_index=class_index,
            target_class=self.classes[class_index],
        )
        return output

    def __len__(self):
        return len(self.samples)


@dataclass
class Sample:
    path: Path
    image: torch.Tensor
    source_class_index: int
    source_class: str


@dataclass
class CounterfactualSample:
    counterfactual_path: Path
    counterfactual: torch.Tensor
    target_class_index: int
    target_class: str


@dataclass
class PairedSample(Sample, CounterfactualSample):
    pass


@dataclass
class SampleWithAttribution(PairedSample):
    attribution: np.array
