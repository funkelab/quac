import cv2
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from quac.data import PairedImageDataset, CounterfactualDataset, PairedWithAttribution
from quac.report import Report
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torch
from tqdm import tqdm


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

    def __init__(
        self, gaussian_kernel_size=11, struc=10, channel_wise=True, name="default"
    ):
        self.gaussian_kernel_size = gaussian_kernel_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (struc, struc))
        self.channel_wise = channel_wise
        self.name = name

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
            # TODO explain the reasoning behind the morphological closing
            # Morphological closing
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

    Raises:
        FileNotFoundError: If the source, counterfactual or attribution directories do not exist.
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

        # Check that they all exist
        for directory in [
            source_directory,
            counterfactual_directory,
            attribution_directory,
        ]:
            if not Path(directory).exists():
                raise FileNotFoundError(f"Directory {directory} does not exist")
        self.source_directory = source_directory
        self.counterfactual_directory = counterfactual_directory
        self.attribution_directory = attribution_directory
        self.transform = transform

    @property
    def source_dataset(self):
        # NOTE: Recomputed each time, but should be used sparingly.
        dataset = ImageFolder(self.source_directory, transform=self.transform)
        return dataset

    @property
    def counterfactual_dataset(self):
        # NOTE: Recomputed each time, but should be used sparingly.
        dataset = CounterfactualDataset(
            self.counterfactual_directory, transform=self.transform
        )
        return dataset

    @property
    def paired_dataset(self):
        # NOTE: Recomputed each time, but should be used sparingly.
        dataset = PairedImageDataset(
            self.source_directory,
            self.counterfactual_directory,
            transform=self.transform,
        )
        return dataset

    @property
    def dataset_with_attribution(self):
        dataset = PairedWithAttribution(
            self.source_directory,
            self.counterfactual_directory,
            self.attribution_directory,
            transform=self.transform,
        )
        return dataset

    def _source_classification_report(
        self, return_classification=False, print_report=True
    ):
        """
        Classify the source data and return the confusion matrix.
        """
        pred = []
        target = []
        for sample in tqdm(self.source_dataset):
            pred.append(self.run_inference(sample.image).argmax())
            target.append(sample.source_class_index)

        if print_report:
            print(classification_report(target, pred))

        cm = confusion_matrix(target, pred, normalize="true")
        if return_classification:
            return cm, pred, target
        return cm

    def _counterfactual_classification_report(
        self,
        return_classification=False,
        print_report=True,
    ):
        """
        Classify the counterfactual data and return the confusion matrix.
        """
        pred = []
        source = []
        target = []
        for sample in tqdm(self.counterfactual_dataset):
            pred.append(self.run_inference(sample.counterfactual).argmax())
            target.append(sample.target_class_index)
            source.append(sample.source_class_index)

        if print_report:
            print(classification_report(target, pred))

        cm = confusion_matrix(target, pred, normalize="true")
        if return_classification:
            return cm, pred, source, target
        return cm

    def classification_report(
        self,
        data="counterfactuals",
        return_classification=False,
        print_report=True,
    ):
        """
        Classify the data and return the confusion matrix.
        """
        if data == "counterfactuals":
            return self._counterfactual_classification_report(
                return_classification=return_classification,
                print_report=print_report,
            )
        elif data == "source":
            return self._source_classification_report(
                return_classification=return_classification,
                print_report=print_report,
            )
        else:
            raise ValueError(f"Data must be 'counterfactuals' or 'source', not {data}")

    def quantify(self, processor=None):
        if processor is None:
            processor = Processor()
        report = Report(name=processor.name)
        for inputs in tqdm(self.dataset_with_attribution):
            predictions = {
                "original": self.run_inference(inputs.image)[0],
                "counterfactual": self.run_inference(inputs.counterfactual)[0],
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
            report.accumulate(
                inputs,
                predictions,
                results,
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
