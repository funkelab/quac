import cv2
import numpy as np
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torch
from quac.data import PairedImageFolder, CounterfactualDataset
from quac.report import Report


def image_to_tensor(image, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = torch.tensor(image, device=device)
    if len(np.shape(image)) == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif len(np.shape(image)) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    else:
        raise ValueError("Input shape not understood")
    return image_tensor.float()


class Processor:
    """Class the turns attributions into masks."""

    def __init__(self, gaussian_kernel_size=11, struc=10, channel_wise=False):
        self.gaussian_kernel_size = gaussian_kernel_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (struc, struc))
        self.channel_wise = channel_wise

    def create_mask(self, attribution, threshold, return_size=True):
        channels, _, _ = attribution.shape
        mask_size = 0
        mask = []
        # construct mask channel by channel
        for c in range(channels):
            # threshold
            if self.channel_wise:
                channel_mask = attribution[c, :, :] > threshold
            else:
                channel_mask = np.any(attribution > threshold, axis=0)
            # morphological closing
            channel_mask = cv2.morphologyEx(
                channel_mask.astype(np.uint8), cv2.MORPH_CLOSE, self.kernel
            )
            # TODO This might be misleading, given the blur afterwards
            mask_size += np.sum(channel_mask)
            # TODO Add connected components
            # Blur
            mask.append(
                cv2.GaussianBlur(
                    channel_mask.astype(np.float32),
                    (self.gaussian_kernel_size, self.gaussian_kernel_size),
                    0,
                )
            )
            # TODO should we do this instead?
            # mask_size += np.sum(mask)
        if not return_size:
            return np.array(mask)
        return np.array(mask), mask_size


class Evaluator:
    """This class evaluates the quality of an attribution using the QuAC method.

    It it is based on the assumption that there exists a counterfactual for each image.
    """

    def __init__(
        self,
        classifier,
        source_directory,
        counterfactual_directory,
        attribution_directory,
        transform=None,
        num_thresholds=200,
        device=None,
    ):
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = classifier.to(self.device)
        self.num_thresholds = num_thresholds

        # TODO Check that they all exist
        self.source_directory = source_directory
        self.counterfactual_directory = counterfactual_directory
        self.attribution_directory = attribution_directory
        self.transform = transform

    @property
    def source_dataset(self):
        # Note: Recomputed each time
        dataset = ImageFolder(self.source_directory, transform=self.transform)
        return dataset

    @property
    def counterfactual_dataset(self):
        dataset = CounterfactualDataset(
            self.counterfactual_dataset, transform=self.transform
        )
        return dataset

    @property
    def paired_dataset(self):
        dataset = PairedImageFolder(
            self.source_directory,
            self.counterfactual_directory,
            transform=self.transform,
        )
        return dataset

    @property
    def dataset_with_attribution(self):
        dataset = ...  # TODO implement
        return dataset

    def quantify(self, processor=None):
        if processor is None:
            processor = Processor()
        report = Report(name=processor.name)
        for inputs in self.dataset_with_attribution:
            predictions = {
                "original": self.run_inference(inputs.image),
                "counterfactual": self.run_inference(inputs.counterfactual),
            }
            results = self.evaluate(
                inputs.image,
                inputs.counterfactual,
                inputs.source_class_index,
                inputs.target_class_index,
                inputs.attribution,
                predictions,
                processor,
            )
            report.accumuate(
                inputs,
                results,
                predictions,
            )
        return report

    def evaluate(
        self, x, x_t, y, y_t, attribution, predictions, processor, vmin=-1, vmax=1
    ):
        """
        Run QuAC evaluation on the data point.

        Parameters
        ----------
                x: the input image
                x_t: the counterfactual image
                y: the class of the input image
                y_t: the class of the counterfactual image
                attrihbution: the attribution map
                predictions: the predictions of the classifier
                processor: the attribution processing function (to get mask)
                vmin: the minimal possible value of the attribution, to be used for thresholding. Defaults to -1
                vmax: the maximal possible value of the attribution, to be used for thresholding. Defaults to 1.
        """
        # copy parts of "fake" into "real", see how much the classification of
        # "real" changes into "fake_class"
        classification_real = predictions["original"]

        # TODO remove the need for this
        results = {
            "thresholds": [],
            "hybrids": [],
            "mask_sizes": [],
            "score_change": [],
        }
        for threshold in np.arange(vmin, vmax, (vmax - vmin) / self.num_thresholds):
            # soft mask of the parts to copy
            mask, mask_size = processor.create_mask(attribution, threshold)

            # hybrid = real parts copied to fake
            hybrid = x_t * mask + x * (1.0 - mask)

            classification_hybrid = self.run_inference(hybrid)[0]

            score_change = classification_hybrid[y_t] - classification_real[y_t]

            # Append results
            # TODO Do we want to store the hybrid?
            results["thresholds"].append(threshold)
            results["hybrids"].append(hybrid)
            results["mask_sizes"].append(mask_size / np.prod(x.shape))
            results["score_change"].append(score_change)
        return results

    @torch.no_grad()
    def run_inference(self, im):
        """
        Net: network object
        input_image: Normalized 2D input image.
        """
        im_tensor = image_to_tensor(im, device=self.device)
        class_probs = F.softmax(self.classifier(im_tensor), dim=1).cpu().numpy()
        return class_probs
