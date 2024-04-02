"""Holds all of the discriminative attribution methods that are accepted by QuAC."""

from captum import attr
import numpy as np
import scipy
from pathlib import Path
from quac.data import PairedImageDataset
from tqdm import tqdm


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

    def __init__(self, classifier):
        self.classifier = classifier

    def _attribute(
        self, real_img, counterfactual_img, real_class, target_class, **kwargs
    ):
        raise NotImplementedError(
            "The base attribution class does not have an attribute method."
        )

    def attribute(
        self, real_img, counterfactual_img, real_class, target_class, **kwargs
    ):
        self.classifier.zero_grad()
        attribution = self._attribute(
            real_img, counterfactual_img, real_class, target_class, **kwargs
        )
        return attribution.detach().cpu().numpy()


class DIntegratedGradients(BaseAttribution):
    """
    Discriminative version of the Integrated Gradients attribution method.
    """

    def __init__(self, classifier):
        super().__init__(classifier)
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

    def __init__(self, classifier):
        super().__init__(classifier)
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

    def __init__(self, classifier):
        super().__init__(classifier)
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
        self, source_directory: str, counterfactual_directory: str, transform: callable
    ):
        dataset = PairedImageDataset(
            source_directory, counterfactual_directory, transform=transform
        )

        for sample in tqdm(dataset):
            for attr_name, attribution in self.attributions.items():
                attr = attribution.attribute(
                    sample.image,
                    sample.counterfactual,
                    sample.class_index,
                    sample.target_class_index,
                )
                # Store the attribution
                np.save(
                    self.get_directory(
                        attr_name, sample.source_class, sample.target_class
                    )
                    / f"{sample.path.name}.npy",
                    attr,
                )
