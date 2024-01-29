"""Holds all of the discriminative attribution methods that are accepted by QuAC."""
from captum import attr 
import torch
import numpy as np
import scipy


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

    def attribute(self, real_img, counterfactual_img, real_class, target_class, **kwargs):
        return NotImplementedError("The base attribution class does not have an attribute method.")


class DIntegratedGradients(BaseAttribution):
    """
    Discriminative version of the Integrated Gradients attribution method.
    """
    def __init__(self, classifier):
        self.ig = attr.IntegratedGradients(classifier)

    def attribute(self, real_img, counterfactual_img, real_class, target_class):
        # FIXME in the original DAPI code, the real and counterfactual were switched.
        attribution = self.ig.attribute(
            real_img,
            counterfactual_img, 
            target=real_class
        )
        return attribution


class DDeepLift(BaseAttribution):
    """
    Discriminative version of the DeepLift attribution method.
    """
    def __init__(self, classifier):
        self.dl = attr.DeepLift(classifier)

    def attribute(self, real_img, counterfactual_img, real_class, target_class):
        # FIXME in the original DAPI code, the real and counterfactual were switched.
        attribution = self.dl.attribute(
            real_img,
            counterfactual_img,
            target=real_class
        )
        return attribution


class DInGrad(BaseAttribution):
    """
    Discriminative version of the InputxGradient attribution method.
    """
    def __init__(self, classifier):
        self.classifier = classifier  
        self.salicency = attr.Saliency(self.classifier)

    def attribute(self, real_img, counterfactual_img, real_class, target_class):
        self.classifier.zero_grad()  # TODO is this necessary?
        # FIXME in the original DAPI code, the real and counterfactual were switched. See below.
        # grads_fake = self.saliency.attribute(counterfactual_img,
        #                                 target=target_class)
        # ingrad_diff_0 = grads_fake * (real_img - counterfactual_img)
        grads_real = self.saliency.attribute(real_img,
                                        target=real_class)
        ingrad_diff_1 = grads_real * (counterfactual_img - real_img)
        return ingrad_diff_1