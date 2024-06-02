"""Holds all of the discriminative attribution methods that are accepted by QuAC."""

from captum import attr
import numpy as np
import scipy
from pathlib import Path
from quac.data import PairedImageDataset
from tqdm import tqdm
import torch
from typing import Callable


def residual(real_img, fake_img):
    """Residual attribution method.

    This method just takes the standardized difference between the real and fake images.
    """
    res = np.abs(real_img - fake_img)
    res = (res - min(res)) / np.max(res)
    return res


def random(real_img, fake_img):
    """Random attribution method.

    This method randomly assigns attribution to each pixel in the image, then applies a Gaussian filter for smoothing.
    """
    rand = np.abs(np.random.randn(*np.shape(real_img)))
    rand = np.abs(scipy.ndimage.gaussian_filter(rand, 4))
    rand = (rand - min(rand)) / np.max(np.abs(rand))
    return rand


class BaseAttribution:
    # TODO use ABC?
    """
    Basic format of an attribution class.
    """

    def __init__(self, classifier, normalize=True):
        self.classifier = classifier
        self.normalize = normalize

    def _normalize(self, attribution):
        """Scale the attribution to be between 0 and 1.

        Note that this also takes the absolute value of the attribution.
        Generally in this frameowrk, we only care about the absolute value of the attribution,
        because if "negative changes" need to be made, this should be inherent in
        the counterfactual image.
        """
        attribution = np.abs(attribution)
        # We scale the attribution to be between 0 and 1
        return (attribution - np.min(attribution)) / (
            np.max(attribution) - np.min(attribution)
        )

    def _attribute(
        self, real_img, counterfactual_img, real_class, target_class, **kwargs
    ):
        raise NotImplementedError(
            "The base attribution class does not have an attribute method."
        )

    def attribute(
        self,
        real_img,
        counterfactual_img,
        real_class,
        target_class,
        device="cuda",
        **kwargs,
    ):
        self.classifier.zero_grad()
        attribution = self._attribute(
            real_img.to(device),
            counterfactual_img.to(device),
            real_class,
            target_class,
            **kwargs,
        )
        attribution = attribution.detach().cpu().numpy()
        if self.normalize:
            attribution = self._normalize(attribution)
        return attribution


class DIntegratedGradients(BaseAttribution):
    """
    Discriminative version of the Integrated Gradients attribution method.
    """

    def __init__(self, classifier, normalize=True):
        super().__init__(classifier, normalize=normalize)
        self.ig = attr.IntegratedGradients(classifier)

    def _attribute(self, real_img, counterfactual_img, real_class, target_class):
        # FIXME in the original DAPI code, the real and counterfactual were switched.
        attribution = self.ig.attribute(
            real_img[None, ...],  # .cuda(),
            baselines=counterfactual_img[None, ...],  # .cuda(),
            target=real_class,
        )
        return attribution[0]


class DDeepLift(BaseAttribution):
    """
    Discriminative version of the DeepLift attribution method.
    """

    def __init__(self, classifier, normalize=True):
        super().__init__(classifier, normalize=normalize)
        self.dl = attr.DeepLift(classifier)

    def _attribute(self, real_img, counterfactual_img, real_class, target_class):
        # FIXME in the original DAPI code, the real and counterfactual were switched.
        attribution = self.dl.attribute(
            real_img[None, ...],  # .cuda(),
            baselines=counterfactual_img[None, ...],  # .cuda(),
            target=real_class,
        )
        return attribution[0]


class DInGrad(BaseAttribution):
    """
    Discriminative version of the InputxGradient attribution method.
    """

    def __init__(self, classifier, normalize=True):
        super().__init__(classifier, normalize=normalize)
        self.saliency = attr.Saliency(self.classifier)

    def _attribute(self, real_img, counterfactual_img, real_class, target_class):
        # FIXME in the original DAPI code, the real and counterfactual were switched. See below.
        # grads_fake = self.saliency.attribute(counterfactual_img,
        #                                 target=target_class)
        # ingrad_diff_0 = grads_fake * (real_img - counterfactual_img)
        grads_real = (
            self.saliency.attribute(real_img[None, ...], target=real_class)  # .cuda(),
            .detach()
            .cpu()
        )
        ingrad_diff_1 = grads_real * (
            counterfactual_img[None, ...] - real_img[None, ...]
        )
        return ingrad_diff_1[0]


class VanillaIntegratedGradient(BaseAttribution):
    """Wrapper class for Integrated Gradients from Captum.

    Allows us to use it as a baseline.
    """

    def __init__(self, classifier, normalize=True):
        super().__init__(classifier, normalize=normalize)
        self.ig = attr.IntegratedGradients(classifier)

    def _attribute(self, real_img, counterfactual_img, real_class, target_class):
        batched_attribution = (
            self.ig.attribute(real_img[None, ...], target=real_class).detach().cpu()
        )
        return batched_attribution[0]


class VanillaDeepLift(BaseAttribution):
    """Wrapper class for DeepLift from Captum.

    Allows us to use it as a baseline.
    """

    def __init__(self, classifier, normalize=True):
        super().__init__(classifier, normalize=normalize)
        self.dl = attr.DeepLift(classifier)

    def _attribute(self, real_img, counterfactual_img, real_class, target_class):
        batched_attribution = (
            self.dl.attribute(real_img[None, ...], target=real_class).detach().cpu()
        )
        return batched_attribution[0]


class AttributionIO:
    """
    Running the attribution methods on the images.
    Storing the results in the output directory.

    """

    def __init__(self, attributions: dict[str, BaseAttribution], output_directory: str):
        self.attributions = attributions
        self.output_directory = Path(output_directory)

    def get_directory(self, attr_name: str, source_class: str, target_class: str):
        directory = self.output_directory / f"{attr_name}/{source_class}/{target_class}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def run(
        self,
        source_directory: str,
        counterfactual_directory: str,
        transform: Callable,
        device: str = "cuda",
    ):
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available on this machine.")
        print("Loading paired data")
        dataset = PairedImageDataset(
            source_directory, counterfactual_directory, transform=transform
        )
        print("Running attributions")
        for sample in tqdm(dataset, total=len(dataset)):
            for attr_name, attribution in self.attributions.items():
                attr = attribution.attribute(
                    sample.image,
                    sample.counterfactual,
                    sample.source_class_index,
                    sample.target_class_index,
                    device=device,
                )
                # Store the attribution
                np.save(
                    self.get_directory(
                        attr_name, sample.source_class, sample.target_class
                    )
                    / f"{sample.path.stem}.npy",
                    attr,
                )
