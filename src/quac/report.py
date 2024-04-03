import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import torch
from tqdm import tqdm


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

    def accumulate(self, inputs, predictions, evaluation_results):
        """
        Store a new result.
        If `save_intermediates` is `True`, the hybrids are stored to disk.
        Otherwise they are discarded.
        """
        # Store the input information
        self.paths.append(inputs.path)
        self.target_paths.append(inputs.counterfactual_path)
        self.labels.append(inputs.source_class_index)
        self.target_labels.append(inputs.target_class_index)
        # Store the prediction results
        self.predictions.append(predictions["original"])
        self.target_predictions.append(predictions["counterfactual"])
        # Store the evaluation results
        self.thresholds.append(evaluation_results["thresholds"])
        self.normalized_mask_sizes.append(evaluation_results["mask_sizes"])
        self.score_changes.append(evaluation_results["score_change"])
        # TODO Store the hybrids to disk ?

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
                },
                fd,
            )

    def load(self, filename):
        """Load report from disk"""
        with open(filename, "r") as fd:
            data = json.load(fd)
            self.metadata = data["metadata"]
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

    def plot_curve(self, ax=None):
        """Plot the QuAC curve

        We plot the mean and standard deviation of the QuAC curve acrosss all accumulated results.

        Parameters
        ----------
        ax: plt.axis
                Axis on which to plot. Defaults to None in which case a new figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots()

        plot_values = []
        for normalized_mask_sizes, score_changes in zip(
            self.normalized_mask_sizes, self.score_changes
        ):
            interp_score_values = self.interpolate_score_values(
                normalized_mask_sizes, score_changes
            )
            plot_values.append(interp_score_values)
        plot_values = np.array(plot_values)
        mean = np.mean(plot_values, axis=0)
        std = np.std(plot_values, axis=0)
        ax.plot(self.interp_mask_values, mean, label=self.name)
        ax.fill_between(self.interp_mask_values, mean - std, mean + std, alpha=0.2)
        if ax is None:
            plt.show()

    def get_optimal_threshold(self, index, return_index=False):
        # TODO Check this function
        mask_scores = np.array(self.score_changes[index])
        mask_sizes = np.array(self.normalized_mask_sizes[index])

        pareto_scores = mask_sizes**2 + (1 - mask_scores) ** 2
        thr_idx = np.argmin(pareto_scores)
        if return_index:
            return thr_idx, self.thresholds[index][thr_idx]
        return self.thresholds[index][thr_idx]
