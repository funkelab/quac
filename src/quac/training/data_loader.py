"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import random

import imageio
from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


class RGB:
    def __call__(self, img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        else:  # Tensor
            if img.size(0) == 1:
                return torch.cat([img, img, img], dim=0)
            return img


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


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class LabelledDataset(data.Dataset):
    """A base dataset for QuAC."""

    def __init__(self, root, transform=None, augment=None):
        self.samples, self.targets = self._make_dataset(root)
        # Check if empty
        assert len(self.samples) > 0, "Dataset is empty, no files found."
        self.transform = transform

    def _open_image(self, fname):
        array = imageio.imread(fname)
        # if no channel dimension, add it
        if len(array.shape) == 2:
            array = array[:, :, None]
        # data will be h,w,c, switch to c,h,w
        array = array.transpose(2, 0, 1)
        return torch.from_numpy(array)

    def _make_dataset(self, root):
        # Get all subitems, sorted, ignore hidden
        domains = sorted(Path(root).glob("[!.]*"))
        # only directories, absolute paths
        domains = [d.absolute() for d in domains if d.is_dir()]
        fnames, labels = [], []
        for idx, class_dir in enumerate(domains):
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            labels += [idx] * len(cls_fnames)
        return fnames, labels

    def __getitem__(self, index):
        fname = self.samples[index]
        label = self.targets[index]
        img = self._open_image(fname)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class AugmentedDataset(LabelledDataset):
    """Adds an augmented version of the input to the sample."""

    def __init__(self, root, transform=None, augment=None):
        super().__init__(root, transform, augment)  # Creates self.samples, self.targets
        if self.augment is None:
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
        img = self._open_image(fname)
        # Augment the image to create a second image
        img2 = self.augment(img)
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


class ReferenceDataset(LabelledDataset):
    """A dataset that returns a reference image and a target image."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)  # Creates self.samples, self.targets
        # Create a second set of samples
        fnames2 = []
        for fname in self.samples:
            # Get the class of the current image
            class_dir = Path(fname).parent
            # Get a random image from the same class
            cls_fnames = listdir(class_dir)
            fname2 = random.choice(cls_fnames)
            fnames2.append(fname2)
        self.samples = list(zip(self.samples, fnames2))

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = self._open_image(fname)
        img2 = self._open_image(fname2)
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
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
    mean=0.5,
    std=0.5,
):
    print(
        "Preparing DataLoader to fetch %s images during the training phase..." % which
    )

    crop = transforms.RandomResizedCrop(img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < prob else x)

    transform_list = [rand_crop]
    if grayscale:
        transform_list.append(transforms.Grayscale())
    else:
        transform_list.append(RGB())

    transform_list += [
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transform_list)

    if which == "source":
        # dataset = ImageFolder(root, transform)
        dataset = AugmentedDataset(root, transform)
    elif which == "reference":
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
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
    imagenet_normalize=False,
    shuffle=True,
    num_workers=4,
    drop_last=False,
    grayscale=False,
    mean=0.5,
    std=0.5,
):
    print("Preparing DataLoader for the evaluation phase...")
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size

    if mean is not None:
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        normalize = transforms.Lambda(lambda x: x)

    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale())
    else:
        transform_list.append(RGB())

    transform = transforms.Compose(
        [
            *transform_list,
            transforms.Resize([height, width]),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def get_test_loader(
    root,
    img_size=256,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=4,
    grayscale=False,
    mean=0.5,
    std=0.5,
    return_dataset=False,
):
    print("Preparing DataLoader for the generation phase...")
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale())
    else:
        transform_list.append(RGB())

    transform_list += [
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
    ]
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transform_list)

    dataset = ImageFolder(root, transform)
    if return_dataset:
        return dataset
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=""):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == "train":
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(
                x_src=x,
                y_src=y,
                y_ref=y_ref,
                x_ref=x_ref,
                x_ref2=x_ref2,
                z_trg=z_trg,
                z_trg2=z_trg2,
            )
        elif self.mode == "val":
            x_ref, y_ref = self._fetch_refs()
            inputs = Munch(x_src=x, y_src=y, x_ref=x_ref, y_ref=y_ref)
        elif self.mode == "test":
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device) for k, v in inputs.items()})


class AugmentedInputFetcher(InputFetcher):
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=""):
        super().__init__(loader, loader_ref, latent_dim, mode)

    def _fetch_inputs(self):
        try:
            x, x2, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, x2, y = next(self.iter)
        return x, x2, y

    def __next__(self):
        x, x2, y = self._fetch_inputs()
        if self.mode == "train":
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(
                x_src=x,
                y_src=y,
                x_src2=x2,
                y_ref=y_ref,
                x_ref=x_ref,
                x_ref2=x_ref2,
                z_trg=z_trg,
                z_trg2=z_trg2,
            )
        elif self.mode == "val":
            x_ref, _, y_ref = self._fetch_refs()
            inputs = Munch(x_src=x, y_src=y, x_ref=x_ref, y_ref=y_ref)
        elif self.mode == "test":
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device) for k, v in inputs.items()})


class TrainingData:
    def __init__(
        self,
        source,
        reference,
        img_size=128,
        batch_size=8,
        num_workers=4,
        grayscale=False,
        mean=None,
        std=None,
        rand_crop_prob=0,
    ):
        self.src = get_train_loader(
            root=source,
            which="source",
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            grayscale=grayscale,
            mean=mean,
            std=std,
            prob=rand_crop_prob,
        )
        self.reference = get_train_loader(
            root=reference,
            which="reference",
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            grayscale=grayscale,
            mean=mean,
            std=std,
            prob=rand_crop_prob,
        )


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
        mean=None,
        std=None,
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
        mean: float
            The mean for normalization, for the classifier.
        std: float
            The standard deviation for normalization, for the classifier.
        kwargs : dict
            Unused keyword arguments, for compatibility with configuration.
        """
        assert mode in ["latent", "reference"]
        # parameters
        self.image_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grayscale = grayscale
        self.mean = mean
        self.std = std
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
        return get_eval_loader(
            self.source_directory,
            img_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            grayscale=self.grayscale,
            mean=self.mean,
            std=self.std,
            drop_last=False,
        )

    @property
    def loader_ref(self):
        return get_eval_loader(
            self.reference_directory,
            img_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            grayscale=self.grayscale,
            mean=self.mean,
            std=self.std,
            drop_last=True,
        )
