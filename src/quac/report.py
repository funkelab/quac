import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import torch
from tqdm import tqdm
import warnings


class Report:
    """This class stores the results of the evaluation.

    It contains the following information for each accumulated sample:
    - Thresholds: the thresholds used to generate masks from the attribution
    - Mask sizes: the size of the mask generated from the attribution, in pixels
    - Normed mask sizes: the size of the mask generated from the attribution, normalized between 0 and 1
    - Scores: the change in classification score for each threshold

    Optionally, it also stores the hybrids generated for each threshold.
    """

    def __init__(self, name=None, metadata={}):
        if name is None:
            self.name = "report"
        else:
            self.name = name
        # Shows where the attribution is, if needed
        # TODO check that the metadata is JSON serializable
        self.metadata = metadata
        # Initialize as empty
        self.paths = []
        self.target_paths = []
        self.labels = []
        self.target_labels = []
        # Predictions/Classifier output
        self.predictions = []
        self.target_predictions = []
        # Evaluations/Attribution outputs
        self.attribution_paths = []
        self.thresholds = []
        self.normalized_mask_sizes = []
        self.score_changes = []
        # QuAC scores
        self.quac_scores = None
        # Initialize interpolation values
        self.interp_mask_values = np.arange(0.0, 1.0001, 0.01)
        # Initialize the optimal thresholds
        self.optimal_thresholds = []
        # Initialize mask and counterfactual paths
        self.mask_paths = []
        self.counterfactual_paths = []
        # Initialize counterfactual predictions
        self.counterfactual_predictions = []

    def accumulate(self, inputs, predictions, evaluation_results):
        """
        Store a new result.
        """
        # Store the input information
        self.paths.append(inputs.path)
        self.target_paths.append(inputs.counterfactual_path)
        self.labels.append(inputs.source_class_index)
        self.target_labels.append(inputs.target_class_index)
        self.attribution_paths.append(inputs.attribution_path)
        # Store the prediction results
        self.predictions.append(predictions["original"])
        self.target_predictions.append(predictions["generated"])
        # Store the evaluation results
        self.thresholds.append(evaluation_results["thresholds"])
        self.normalized_mask_sizes.append(evaluation_results["mask_sizes"])
        self.score_changes.append(evaluation_results["score_change"])
        self.optimal_thresholds.append(evaluation_results["optimal_threshold"])
        # mask and counterfactual path
        self.mask_paths.append(evaluation_results.get("mask_path", None))
        self.counterfactual_paths.append(
            evaluation_results.get("counterfactual_path", None)
        )
        # counterfactual prediction
        self.counterfactual_predictions.append(
            evaluation_results.get("counterfactual_prediction", None)
        )

    def interpolate_score_values(self, normalized_mask_sizes, score_changes):
        """Computes the score changes interpolated at the desired mask sizes"""
        f = interp1d(
            np.concatenate([[1], normalized_mask_sizes, [0]]),
            np.concatenate([[score_changes[0]], score_changes, [0]]),
        )
        interp_score_values = [f(x) for x in self.interp_mask_values]
        return interp_score_values

    def get_quac_score(self, interp_score_values):
        """
        The QuAC score is the area under the mask-size vs. score-change curve.
        We use the interpolated mask and score values to compute the QuAC score.
        """
        # QuAC score = AUC of above x-y values
        quac_score = np.trapz(interp_score_values, self.interp_mask_values)
        return quac_score

    def compute_scores(self):
        """Compute all QuAC scores"""
        if self.quac_scores is None:
            quac_scores = []
            for normalized_mask_sizes, score_changes in tqdm(
                zip(self.normalized_mask_sizes, self.score_changes),
                total=len(self.normalized_mask_sizes),
            ):
                interp_score_values = self.interpolate_score_values(
                    normalized_mask_sizes, score_changes
                )
                quac_scores.append(self.get_quac_score(interp_score_values))
            self.quac_scores = quac_scores
        return self.quac_scores

    def make_json_serializable(self, obj):
        """Make an object JSON serializable"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (list, tuple)):
            return [self.make_json_serializable(x) for x in obj]
        return obj

    def store(self, save_dir):
        """Store report to disk"""
        if self.quac_scores is None:
            self.compute_scores()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f"{self.name}.json", "w") as fd:
            json.dump(
                {
                    "metadata": self.metadata,
                    "thresholds": self.make_json_serializable(self.thresholds),
                    "normalized_mask_sizes": self.make_json_serializable(
                        self.normalized_mask_sizes
                    ),
                    "score_changes": self.make_json_serializable(self.score_changes),
                    "paths": self.make_json_serializable(self.paths),
                    "target_paths": self.make_json_serializable(self.target_paths),
                    "labels": self.make_json_serializable(self.labels),
                    "target_labels": self.make_json_serializable(self.target_labels),
                    "predictions": self.make_json_serializable(self.predictions),
                    "target_predictions": self.make_json_serializable(
                        self.target_predictions
                    ),
                    "attribution_paths": self.make_json_serializable(
                        self.attribution_paths
                    ),
                    "quac_scores": self.make_json_serializable(self.quac_scores),
                    "optimal_thresholds": self.make_json_serializable(
                        self.optimal_thresholds
                    ),
                    "mask_paths": self.make_json_serializable(self.mask_paths),
                    "counterfactual_paths": self.make_json_serializable(
                        self.counterfactual_paths
                    ),
                    "counterfactual_predictions": self.make_json_serializable(
                        self.counterfactual_predictions
                    ),
                },
                fd,
            )

    def load(self, filename):
        """Load report from disk"""
        with open(filename, "r") as fd:
            data = json.load(fd)
            self.metadata = data.get("metadata", {})
            self.thresholds = data["thresholds"]
            self.normalized_mask_sizes = data["normalized_mask_sizes"]
            self.score_changes = data["score_changes"]
            self.paths = data.get("paths", [])
            self.target_paths = data.get("target_paths", [])
            self.labels = data.get("labels", [])
            self.target_labels = data.get("target_labels", [])
            self.predictions = data.get("predictions", [])
            self.target_predictions = data.get("target_predictions", [])
            self.attribution_paths = data.get("attribution_paths", [])
            self.quac_scores = data.get("quac_scores", None)
            self.optimal_thresholds = data.get("optimal_thresholds", [])
            self.mask_paths = data.get("mask_paths", [])
            self.counterfactual_paths = data.get("counterfactual_paths", [])
            self.counterfactual_predictions = data.get("counterfactual_predictions", [])

    def get_curve(self):
        """Gets the median and IQR of the QuAC curve"""
        # TODO Cache the results, takes forever otherwise
        plot_values = []
        for normalized_mask_sizes, score_changes in zip(
            self.normalized_mask_sizes, self.score_changes
        ):
            interp_score_values = self.interpolate_score_values(
                normalized_mask_sizes, score_changes
            )
            plot_values.append(interp_score_values)
        plot_values = np.array(plot_values)
        # mean = np.mean(plot_values, axis=0)
        # std = np.std(plot_values, axis=0)
        median = np.median(plot_values, axis=0)
        p25 = np.percentile(plot_values, 25, axis=0)
        p75 = np.percentile(plot_values, 75, axis=0)
        return median, p25, p75

    def plot_curve(self, ax=None):
        """Plot the QuAC curve

        We plot the median and IQR of the QuAC curve across all accumulated results.

        Parameters
        ----------
        ax: plt.axis
                Axis on which to plot. Defaults to None in which case a new figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots()

        mean, p25, p75 = self.get_curve()

        ax.plot(self.interp_mask_values, mean, label=self.name)
        ax.fill_between(self.interp_mask_values, p25, p75, alpha=0.2)
        if ax is None:
            plt.show()

    def get_optimal_thresholds(self, min_percentage=0.0):
        """Get the optimal threshold for each sample

        The optimal threshold has a minimal mask size, and maximizes the score change.
        We optimize $|m| - \delta f$ where $m$ is the mask size and $\delta f$ is the score change.

        Parameters
        ----------
        min_percentage: float
            The optimal threshold chosen needs to account for at least this percentage of total score change.
            Increasing this value will favor high percentage changes even when they require larger masks.
        """
        warnings.warn(
            "This function is deprecated, threshold computed during evaluation.",
            DeprecationWarning,
        )
        mask_scores = np.array(self.score_changes)
        mask_sizes = np.array(self.normalized_mask_sizes)
        thresholds = np.array(self.thresholds)
        tradeoff_scores = np.abs(mask_sizes) - mask_scores
        # Determine what to ignore
        if min_percentage > 0.0:
            min_value = np.min(mask_scores, axis=1)
            max_value = np.max(mask_scores, axis=1)
            threshold = min_value + min_percentage * (max_value - min_value)
            below_threshold = mask_scores < threshold[:, None]
            tradeoff_scores[below_threshold] = (
                np.inf
            )  # Ignores the points with not enough score change
        threshold_idx = np.argmin(tradeoff_scores, axis=1)

        optimal_thresholds = np.take_along_axis(
            thresholds, threshold_idx[:, None], axis=1
        ).squeeze()
        return optimal_thresholds
