import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from quac.explanation import Explanation, explanation_encoder
from scipy.interpolate import interp1d


class Report:
    """This class stores the results of the evaluation.

    It contains the following information for each accumulated sample:
    - Thresholds: the thresholds used to generate masks from the attribution
    - Mask sizes: the size of the mask generated from the attribution, in pixels
    - Normed mask sizes: the size of the mask generated from the attribution, normalized between 0 and 1
    - Scores: the change in classification score for each threshold
    """

    def __init__(self, name=None, metadata={}):
        if name is None:
            self.name = "report"
        else:
            self.name = name
        self.metadata = metadata
        # Initialize as empty
        self.explanations = []
        self.interp_mask_values = np.arange(0.0, 1.0001, 0.01)

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

    def accumulate(self, inputs, predictions, evaluation_results):
        """
        Store a new result.
        """
        quac_score = self.get_quac_score(
            self.interpolate_score_values(
                evaluation_results["mask_sizes"], evaluation_results["score_change"]
            )
        )
        explanation = Explanation(
            query_path=inputs.path,
            counterfactual_path=evaluation_results["counterfactual_path"],
            mask_path=evaluation_results["mask_path"],
            query_prediction=predictions["original"],
            counterfactual_prediction=predictions["counterfactual"],
            source_class=inputs.source_class_index,
            target_class=inputs.target_class_index,
            score=quac_score,
            attribution_path=inputs.attribution_path,
            generated_path=inputs.target_path,
            generated_prediction=inputs.target_prediction,
            normalized_mask_changes=evaluation_results["mask_sizes"],
            score_changes=evaluation_results["score_change"],
            optimal_threshold=evaluation_results.get("optimal_threshold", None),
        )
        self.explanations.append(explanation)

    def store(self, save_dir):
        """Store the report in a JSON file."""
        output = {
            "name": self.name,
            "metadata": self.metadata,
            "results": self.explanations,
        }
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f"{self.name}.json", "w") as json_file:
            json.dump(output, json_file, default=explanation_encoder)

    def load(self, filename):
        """Load report from disk"""
        with open(filename, "r") as fd:
            data = json.load(fd)
            self.metadata = data.get("metadata", {})
            self.name = data["name"]
            self.explanations = [Explanation(**result) for result in data["results"]]

    def get_curve(self):
        """Gets the median and IQR of the QuAC curve"""
        plot_values = []
        normalized_mask_sizes = [
            explanation.normalized_mask_changes for explanation in self.explanations
        ]
        score_changes = [explanation.score_changes for explanation in self.explanations]
        for normalized_mask, score_change in zip(normalized_mask_sizes, score_changes):
            interp_score_values = self.interpolate_score_values(
                normalized_mask, score_change
            )
            plot_values.append(interp_score_values)
        plot_values = np.array(plot_values)
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
