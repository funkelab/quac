import cv2
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
    elif len(np.shape(image)) == 4:
        return image_tensor.float()
    else:
        raise ValueError(f"Input shape not understood, {image.shape}")
    return image_tensor.float()


class Processor:
    """Class that turns attributions into masks."""

    def __init__(
        self, gaussian_kernel_size=11, struct=10, channel_wise=True, name="default"
    ):
        self.gaussian_kernel_size = gaussian_kernel_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (struct, struct))
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


class UnblurredProcessor(Processor):
    """
    Processor without any blurring
    """

    def __init__(self, struct=10, channel_wise=True, name="no_blur"):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (struct, struct))
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
            # Morphological closing
            channel_mask = cv2.morphologyEx(
                channel_mask.astype(np.uint8), cv2.MORPH_CLOSE, self.kernel
            )
            mask.append(channel_mask)
            mask_size += np.sum(channel_mask)
        if not return_size:
            return np.array(mask)
        return np.array(mask), mask_size


class BaseEvaluator:
    """Base class for evaluating attributions."""

    def __init__(
        self,
        classifier,
        source_dataset=None,
        paired_dataset=None,
        attribution_dataset=None,
        num_thresholds=200,
        device=None,
    ):
        """Initializes the evaluator.

        It requires three different datasets: the source dataset, the counterfactual dataset and the attribution dataset.
        All of them must return objects in the forms of the dataclasses in `quac.data`.


        Parameters
        ----------
        classifier:
            The classifier to be used for the evaluation.
        source_dataset:
            The source dataset must returns a `quac.data.Sample` object in its `__getitem__` method.
        paired_dataset:
            The paired dataset must returns a `quac.data.PairedSample` object in its `__getitem__` method.
        attribution_dataset:
            The attribution dataset must returns a `quac.data.SampleWithAttribution` object in its `__getitem__` method.
        """
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = classifier.to(self.device)
        self.num_thresholds = num_thresholds
        self.classifier = classifier

        self._source_dataset = source_dataset
        self._paired_dataset = paired_dataset
        self._dataset_with_attribution = attribution_dataset

    @property
    def source_dataset(self):
        return self._source_dataset

    @property
    def paired_dataset(self):
        return self._paired_dataset

    @property
    def dataset_with_attribution(self):
        return self._dataset_with_attribution

    def _source_classification_report(
        self, return_classification=False, print_report=True
    ):
        """
        Classify the source data and return the confusion matrix.
        """
        pred = []
        target = []
        for image, source_class_index in tqdm(self.source_dataset):
            pred.append(self.run_inference(image).argmax())
            target.append(source_class_index)

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
                attribution: the attribution map
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
            # Append results
            # TODO Do we want to store the hybrid?
            results["thresholds"].append(threshold)
            results["hybrids"].append(hybrid)
            results["mask_sizes"].append(mask_size / np.prod(x.shape))

            # Classification
            # classification_hybrid = self.run_inference(hybrid)[0]
            # score_change = classification_hybrid[y_t] - classification_real[y_t]
            # results["score_change"].append(score_change)
        hybrid = np.stack(results["hybrids"], axis=0)
        classification_hybrid = self.run_inference(hybrid)
        score_change = classification_hybrid[:, y_t] - classification_real[y_t]
        results["score_change"] = score_change
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


class Evaluator(BaseEvaluator):
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
        # Check that they all exist
        for directory in [
            source_directory,
            counterfactual_directory,
            attribution_directory,
        ]:
            if not Path(directory).exists():
                raise FileNotFoundError(f"Directory {directory} does not exist")

        super().__init__(
            classifier, None, None, None, num_thresholds=num_thresholds, device=device
        )
        self.transform = transform
        self.source_directory = source_directory
        self.counterfactual_directory = counterfactual_directory
        self.attribution_directory = attribution_directory

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
