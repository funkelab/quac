"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import logging
from pathlib import Path
import random

import numpy as np

from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

from quac.data import read_image, listdir, DefaultDataset, create_transform


class LabelledDataset(data.Dataset):
    """A base dataset for QuAC."""

    def __init__(self, root, transform=None, augment=None):
        self.samples, self.targets = self._make_dataset(root)
        # Check if empty
        assert len(self.samples) > 0, "Dataset is empty, no files found."
        self.transform = transform

    def _make_dataset(self, root):
        # Get all subitems, sorted, ignore hidden
        domains = sorted(Path(root).glob("[!.]*"))
        # only directories, absolute paths
        domains = [d.absolute() for d in domains if d.is_dir()]
        # Get class names
        self.classes = [d.name for d in domains]
        fnames, labels = [], []
        for idx, class_dir in enumerate(domains):
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            labels += [idx] * len(cls_fnames)
        return fnames, labels

    def __getitem__(self, index):
        fname = self.samples[index]
        label = self.targets[index]
        img = read_image(fname)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.targets)


class AugmentedDataset(LabelledDataset):
    """Adds an augmented version of the input to the sample."""

    def __init__(self, root, transform=None, augment=None):
        super().__init__(root, transform, augment)  # Creates self.samples, self.targets
        if augment is None:
            # Default augmentation: random horizontal flip, random vertical flip
            augment = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            )
        self.augment = augment

    def __getitem__(self, index):
        fname = self.samples[index]
        label = self.targets[index]
        img = read_image(fname)
        if self.transform is not None:
            img = self.transform(img)
        # Augment the image to create a second image
        img2 = self.augment(img)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


class ReferenceDataset(LabelledDataset):
    """A dataset that returns a reference image and a target image."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)  # Creates self.samples, self.targets

    def select_from_class(self, label, idx):
        """Select a random image from a given class."""
        is_class = set(np.where(np.array(self.targets) == label)[0])
        idx2 = random.choice(list(is_class - {idx}))
        return self.samples[idx2]

    def __getitem__(self, index):
        # fname, fname2 = self.samples[index]
        fname = self.samples[index]
        label = self.targets[index]
        # Randomly select a second image from the same class
        fname2 = self.select_from_class(label, index)
        # Read the images
        img = read_image(fname)
        img2 = read_image(fname2)
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    assert np.all(class_counts > 0), f"Some of the classes are empty. {class_counts}"
    class_weights = 1.0 / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(
    root,
    which="source",
    img_size=256,
    batch_size=8,
    prob=0.5,
    num_workers=4,
    grayscale=False,
    rgb=True,
    scale=2,
    shift=-1,
    rand_crop_prob=0,
):
    logging.info(
        "Preparing DataLoader to fetch %s images during the training phase..." % which
    )
    # Basic image loading, resizing, and normalization
    transform = create_transform(img_size, grayscale, rgb, scale, shift)
    # Augmentations
    crop = transforms.RandomResizedCrop(img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < rand_crop_prob else x
    )
    # Combine
    transform = transforms.Compose(
        [
            transform,
            rand_crop,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    if which == "source":
        dataset = AugmentedDataset(root, transform)
    elif which == "reference":
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = make_balanced_sampler(dataset.targets)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_eval_loader(
    root,
    img_size=256,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=False,
    grayscale=False,
    rgb=True,
    scale=2,
    shift=-1,
):
    logging.info("Preparing DataLoader for the evaluation phase...")
    # Basic image loading, resizing, and normalization
    transform = create_transform(img_size, grayscale, rgb, scale, shift)

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


class TrainingData:
    def __init__(
        self,
        source,
        reference=None,
        img_size=128,
        batch_size=8,
        num_workers=4,
        grayscale=False,
        rgb=True,
        scale=2,
        shift=-1,
        rand_crop_prob=0,
    ):
        ref_root = reference or source  # if reference is None, use source as reference
        self.src = get_train_loader(
            root=source,
            which="source",
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            grayscale=grayscale,
            rgb=rgb,
            scale=scale,
            shift=shift,
            prob=rand_crop_prob,
        )
        self.reference = get_train_loader(
            root=ref_root,
            which="reference",
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            grayscale=grayscale,
            rgb=rgb,
            scale=scale,
            shift=shift,
            prob=rand_crop_prob,
        )
        self.iter = iter(self.src)
        self.iter_ref = iter(self.reference)

    def _fetch_inputs(self):
        try:
            x, x2, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.src)
            x, x2, y = next(self.iter)
        return x, x2, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.reference)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, x2, y = self._fetch_inputs()
        x_ref, x_ref2, y_ref = self._fetch_refs()
        inputs = dict(
            x_src=x,
            y_src=y,
            x_src2=x2,
            y_ref=y_ref,
            x_ref=x_ref,
            x_ref2=x_ref2,
        )
        return inputs


class ValidationData:
    """
    A data loader for validation.

    """

    def __init__(
        self,
        source,
        reference=None,
        mode="latent",
        img_size=128,
        batch_size=32,
        num_workers=4,
        grayscale=False,
        rgb=True,
        scale=2,
        shift=-1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        source_directory : str
            The directory containing the source images.
        ref_directory : str
            The directory containing the reference images, defaults to source_directory if None.
        mode : str
            The mode of the data loader, either "latent" or "reference".
            If "latent", the data loader will only load the source images.
            If "reference", the data loader will load both the source and reference images.
        image_size : int
            The size of the images; images of a different size will be resized.
        batch_size : int
            The batch size for source data.
        num_workers : int
            The number of workers for the data loader.
        grayscale : bool
            Whether the images are grayscale.
        scale : float
            The scale factor for the images.
        shift : float
            The shift factor for the images.
        kwargs : dict
            Unused keyword arguments, for compatibility with configuration.
        """
        assert mode in ["latent", "reference"]
        # parameters
        self.image_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grayscale = grayscale
        self.rgb = rgb
        self.scale = scale
        self.shift = shift
        # The source and target classes
        self.source = None
        self.target = None
        # The roots of the source and target directories
        self.source_root = Path(source)
        if reference is not None:
            self.ref_root = Path(reference)
        else:
            self.ref_root = self.source_root

        # Available classes
        self.available_sources = [
            subdir.name for subdir in self.source_root.iterdir() if subdir.is_dir()
        ]
        self._available_targets = None
        self.set_mode(mode)
        # Loaders, reuse
        self._loader_src = None
        self._loader_ref = None

    def set_mode(self, mode):
        assert mode in ["latent", "reference"]
        self.mode = mode

    @property
    def available_targets(self):
        if self.mode == "latent":
            return self.available_sources
        elif self._available_targets is None:
            self._available_targets = [
                subdir.name
                for subdir in Path(self.ref_root).iterdir()
                if subdir.is_dir()
            ]
        return self._available_targets

    def set_target(self, target):
        assert target in self.available_targets, (
            f"{target} not in {self.available_targets}"
        )
        self.target = target

    def set_source(self, source):
        assert source in self.available_sources, (
            f"{source} not in {self.available_sources}"
        )
        self.source = source

    @property
    def reference_directory(self):
        if self.mode == "latent":
            return None
        if self.target is None:
            raise (ValueError("Target not set."))
        return self.ref_root / self.target

    @property
    def source_directory(self):
        if self.source is None:
            raise (ValueError("Source not set."))
        return self.source_root / self.source

    def print_info(self):
        print(f"Available sources: {self.available_sources}")
        print(f"Available targets: {self.available_targets}")
        print(f"Mode: {self.mode}")
        try:
            print(f"Current source directory: {self.source_directory}")
        except ValueError:
            print("Source not set.")
        try:
            print(f"Current target directory: {self.reference_directory}")
        except ValueError:
            print("Target not set.")

    @property
    def loader_src(self):
        if self._loader_src is None:
            self._loader_src = get_eval_loader(
                self.source_directory,
                img_size=self.image_size,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                grayscale=self.grayscale,
                rgb=self.rgb,
                scale=self.scale,
                shift=self.shift,
                drop_last=True,
            )
        return self._loader_src

    @property
    def loader_ref(self):
        if self._loader_ref is None:
            self._loader_ref = get_eval_loader(
                self.reference_directory,
                img_size=self.image_size,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                grayscale=self.grayscale,
                rgb=self.rgb,
                scale=self.scale,
                shift=self.shift,
                drop_last=True,
            )
        return self._loader_ref
