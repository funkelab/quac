import cv2
from functools import lru_cache
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from quac.data import (
    PairedImageDataset,
    CounterfactualDataset,
    PairedWithAttribution,
    read_image,
    write_image,
)
from quac.report import Report
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torch
from tqdm import tqdm
from typing import Union


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


class FinalReport:
    """
    Collates and stores the (best) results from multiple reports.
    """

    def __init__(self, reports, transform=None, classifier=None, output_dir=None):
        """
        Initialize the final report with a dictionary of reports.
        To read reports from a directory, use `Final_report.from_directory` instead.

        Parameters
        ----------
        reports: dict
            A dictionary mapping method names to their respective reports.
        transform: callable, optional
            A transform to be applied to the images when loading them.
            This can be used, for example, to crop the images to a specific size.
        classifier: nn.Module, optional
            A classifier to be used for the evaluation.
        output_dir: str, optional
            A directory where the masks and counterfactuals will be saved on-the-fly.
            If None, they will not be saved.
        """
        self._reports = reports
        self._final_report = None
        self._processor = Processor()
        self._classifier = classifier
        self._output_dir = output_dir
        self.transform = transform

    @property
    def final_report(self):
        if self._final_report is None:
            logging.warning("Merging reports to create final report.")
            self._final_report = self._merge()
        return self._final_report

    def read_image(self, path):
        """
        Read an image from disk and apply the transform if provided.
        """
        image = read_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    @classmethod
    def from_directory(cls, eval_directory, **kwargs):
        """
        Find and load all reports in a given directory.

        Parameters
        ----------
        eval_directory: str
            Path to the directory used for QuAC evaluation.
            We expect it to be organized as follows:

            ```
            eval_directory/
                method/
                    report.json
                method2/
                    report.json
            ```

        kwargs: additional arguments to be passed to the FinalReport constructor.

        Returns
        -------
        reports: dict
            A dictionary mapping method names to their respective reports.
        """
        reports = {}
        # Search for all json files in the directory or any subdirectory
        for json_file in Path(eval_directory).rglob("*.json"):
            name = str(json_file.parent)
            report = Report(name=name)
            try:
                report.load(json_file)
                reports[report.name] = report
            except KeyError:
                logging.warning(f"Could not load {json_file}, not a valid report.")
        return FinalReport(reports, **kwargs)

    def _merge(self):
        """
        Merge all available reports into a single, final report.
        This chooses the best attribution method for each sample, based on the QuAC score.
        It also sorts the samples by QuAC score.
        """
        # Create a dataframe with all of the QuAC scores
        quac_scores = pd.DataFrame(
            {method: report.quac_scores for method, report in self._reports.items()}
        )
        # Get the report "name" with the highest QuAC score for each sample
        best_methods = quac_scores.idxmax(axis=1)  # This is a pandas Series
        # Add the QuAC score, turning it into a pandas DataFrame
        best_methods = pd.DataFrame(best_methods, columns=["method"])
        best_methods["quac_score"] = quac_scores.max(axis=1)

        # Sort the samples by QuAC score
        best_methods = best_methods.sort_values("quac_score", ascending=False)

        # Merge all reports into a single one
        final_report = Report(name="final")
        final_report.quac_scores = best_methods["quac_score"].tolist()
        # Store the best results for each sample
        for idx, row in best_methods.iterrows():
            report = self._reports[row["method"]]
            final_report.paths.append(report.paths[idx])
            final_report.target_paths.append(report.target_paths[idx])
            final_report.labels.append(report.labels[idx])
            final_report.target_labels.append(report.target_labels[idx])
            final_report.predictions.append(report.predictions[idx])
            final_report.target_predictions.append(report.target_predictions[idx])
            final_report.attribution_paths.append(report.attribution_paths[idx])
            final_report.thresholds.append(report.thresholds[idx])
            final_report.normalized_mask_sizes.append(report.normalized_mask_sizes[idx])
            final_report.score_changes.append(report.score_changes[idx])

        # Add an empty column for the computed values
        final_report.counterfactual_predictions = [None] * len(final_report.paths)
        final_report.counterfactual_paths = [None] * len(final_report.paths)
        final_report.mask_paths = [None] * len(final_report.paths)

        # The data in the final report is now sorted by QuAC score
        return final_report

    @lru_cache(maxsize=10)
    def get_query(self, item: int) -> torch.Tensor:
        """
        Get the query image for a given explanation.
        """
        query_path = self.final_report.paths[item]
        return self.read_image(query_path)

    def get_query_prediction(self, item: int) -> torch.Tensor:
        """
        Get the classifier output for the query image.
        """
        return self.final_report.predictions[item]

    @lru_cache(maxsize=10)
    def get_generated(self, item: int) -> torch.Tensor:
        """
        Get the generated image for a given explanation.
        """
        generated_path = self.final_report.target_paths[item]
        return self.read_image(generated_path)

    @lru_cache(maxsize=10)
    def get_mask(self, item: int) -> torch.Tensor:
        """
        Get the mask for a given explanation, using the attribution and the threshold.
        """
        try:
            mask_path = self.final_report.mask_paths[item]
            mask = np.load(mask_path)
        except (KeyError, ValueError, TypeError):  # path is "None", or not defined
            # Get the attribution
            attribution_path = self.final_report.attribution_paths[item]
            attribution = np.load(attribution_path)
            # Get the threshold
            threshold = self.final_report.get_optimal_threshold(item)
            # process the attribution to get the mask
            mask, _ = self._processor.create_mask(attribution, threshold)
            self.save_mask(mask, item)
        return mask

    def save_mask(self, mask, item):
        """
        Save the mask to disk, if an output directory is provided.
        Else, do nothing.
        """
        if self._output_dir is not None:
            source_label = self.final_report.labels[item]
            target_label = self.final_report.target_labels[item]
            name = self.final_report.attribution_paths[item].name
            output_dir = Path(self._output_dir) / f"masks/{source_label}/{target_label}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / name
            np.save(output_path, mask)
            if "mask_paths" not in self.final_report.__dict__:
                self.final_report.mask_paths = [None] * len(
                    self.final_report.attribution_paths
                )
            self.final_report.mask_paths[item] = output_path
        else:
            logging.info("No output directory provided, not saving mask to disk.")

    @lru_cache(maxsize=10)
    def get_counterfactual(self, item: int) -> torch.Tensor:
        """
        Get the counterfactual image for a given explanation.
        """
        try:
            counterfactual_path = self.final_report.counterfactual_paths[item]
            counterfactual = self.read_image(counterfactual_path)
        except (KeyError, ValueError, OSError):  # path is "None", or not defined
            query = self.get_query(item)
            generated = self.get_generated(item)
            mask = self.get_mask(item)
            counterfactual = query * (1 - mask) + generated * mask
            self.save_counterfactual(counterfactual, item)
        return counterfactual

    def save_counterfactual(self, counterfactual, item):
        """
        Save the counterfactual image to disk, if an output directory is provided.
        Else, do nothing.
        """
        if self._output_dir is not None:
            source_label = self.final_report.labels[item]
            target_label = self.final_report.target_labels[item]
            name = self.final_report.paths[item].name
            output_dir = (
                Path(self._output_dir)
                / f"counterfactuals/{source_label}/{target_label}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / name
            write_image(counterfactual, output_path)
            # Add path to final report
            if "counterfactual_paths" not in self.final_report.__dict__:
                self.final_report.counterfactual_paths = [None] * len(
                    self.final_report.paths
                )
            self.final_report.counterfactual_paths[item] = output_path
        else:
            logging.info(
                "No output directory provided, not saving counterfactual to disk."
            )

    def get_counterfactual_prediction(self, item: int) -> Union[torch.Tensor, None]:
        """
        Get the classifier output for the counterfactual image.
        We first check if this has already been computed and stored in the final report.
        If not, we check if the classifier is provided and compute the output.
        Else, we return None.
        """
        counterfactual_prediction = self.final_report.counterfactual_predictions[item]
        if counterfactual_prediction is None and self._classifier is not None:
            device = next(self._classifier.parameters()).device
            counterfactual = self.get_counterfactual(item)
            with torch.no_grad():
                counterfactual_prediction = (
                    self._classifier(counterfactual[None, :].to(device)).cpu().detach()
                )
            counterfactual_prediction = F.softmax(counterfactual_prediction, dim=1)[0]
            self.final_report.counterfactual_predictions[item] = (
                counterfactual_prediction
            )
        return counterfactual_prediction

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: int
            Index of the explanation to return. The samples are ordered by QuAC score from best to worst.

        Returns
        -------
        query: torch.Tensor
            The query image for the explanation.
        counterfactual: torch.Tensor
            The counterfactual image for the explanation.
            This is generated by merging the query and the generated image, using the mask.
        mask: torch.Tensor
            The mask used to generate the counterfactual image.
            This is a binary mask with the same size as the query image, including the channel dimension.
            Pixels that are masked-in (1) are taken from the generated image.
            Pixels that are masked-out (0) are taken from the query image.
        query_prediction: torch.Tensor
            The classifier output (softmaxxed) for the query image.
        counterfactual_prediction: torch.Tensor
            The classifier output (softmaxxed) for the counterfactual image.
        source_class: int
            The index of the source class for the explanation.
        target_class: int
            The index of the target class for the explanation.
        quac_score: float
            The QuAC score for the explanation.
        """
        return {
            "query": self.get_query(item),
            "counterfactual": self.get_counterfactual(item),
            "mask": self.get_mask(item),
            "query_prediction": self.get_query_prediction(item),
            "counterfactual_prediction": self.get_counterfactual_prediction(item),
            "source_class": self.final_report.labels[item],
            "target_class": self.final_report.target_labels[item],
            "quac_score": self.final_report.quac_scores[item],
        }
