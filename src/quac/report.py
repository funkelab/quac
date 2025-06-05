import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from quac.explanation import Explanation, explanation_encoder
from scipy.interpolate import interp1d
import logging


def merge_reports(reports, **kwargs):
    """
    Merge all available reports into a single, final report.
    This chooses the best attribution method for each sample, based on the QuAC score.
    It also sorts the samples by QuAC score.
    """
    # Make sure that each report has the same number of samples
    num_samples = len(reports[list(reports.keys())[0]])
    for name, report in reports.items():
        if len(report) != num_samples:
            raise ValueError(
                f"Reports have different number of samples: {len(report)} vs {num_samples}"
            )

    final_report = Report(**kwargs)
    for i in range(num_samples):
        options = [report[i] for report in reports.values()]  # Gets you an explanation
        # Select one explanation for each sample, based on the QuAC score
        best_explanation = max(options, key=lambda x: x.score)
        best_explanation.method = name
        final_report.explanations.append(best_explanation)

    # Sort the items in the final report by QuAC score, from highest to lowest
    final_report.explanations.sort(key=lambda x: x.score, reverse=True)
    # The data in the final report is now sorted by QuAC score
    return final_report


class Report:
    """This class stores the results of the evaluation.

    The report combines a set of Explanation objects, and can be used to store and load them from disk.
    It also has several filtering methods, to help interact with the results.

    For example, given the output of the QuAC evaluation, stored in a directory, we can do the following:
    ```
    report = Report.from_directory("eval_directory", name="final_report")

    # Select a specific source and target, and look at the QuAC curve
    report.from_source(0).to_target(1).plot_curve()

    # OR select the top 10 explanations for the conversion from 0 to 1 and store them
    report.from_source(0).to_target(1).top_n(10).store("top_10_explanations")
    ```
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

    def __len__(self):
        return len(self.explanations)

    def __getitem__(self, idx):
        return self.explanations[idx]

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
        quac_score = np.trapezoid(interp_score_values, self.interp_mask_values)
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
            query_path=str(inputs.path),
            counterfactual_path=str(evaluation_results["counterfactual_path"]),
            mask_path=str(evaluation_results["mask_path"]),
            query_prediction=predictions["original"],
            counterfactual_prediction=predictions["counterfactual"],
            source_class=inputs.source_class_index,
            target_class=inputs.target_class_index,
            score=quac_score,
            attribution_path=str(inputs.attribution_path),
            generated_path=str(inputs.generated_path),
            generated_prediction=predictions["generated"],
            normalized_mask_sizes=evaluation_results["mask_sizes"],
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
            self.name = data.get("name", "default_report")
            self.explanations = [Explanation(**result) for result in data["results"]]

    def get_curve(self):
        """Gets the median and IQR of the QuAC curve"""
        # TODO explanation might not have the right attributes
        # -- maybe remove this from this class and make it a utility function instead
        plot_values = []
        normalized_mask_sizes = [
            explanation._normalized_mask_sizes for explanation in self.explanations
        ]
        score_changes = [
            explanation._score_changes for explanation in self.explanations
        ]
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

    # Filtering functions
    def from_source(self, source_class, name=None) -> "Report":
        """
        Create a Report containing only the explanations with the given source class.

        This filters explanations based on both the original and the *predicted* source class.
        That is, if an explanation has a source of class 0, but the model predicted it as class 1, it will not be included.
        """
        filtered_report = Report(name=name)
        filtered_report.explanations = [
            explanation
            for explanation in self.explanations
            if explanation.source_class == source_class
            and np.argmax(explanation.query_prediction) == source_class
        ]
        return filtered_report

    def to_target(self, target_class, name=None) -> "Report":
        """
        Create a filtered Report containing only the explanations with the given target class.

        This filters explanations based on both the original and the *predicted* target class.
        That is, if an explanation has a target of class 0, but the model predicted it as class 1, it will not be included.
        """
        filtered_report = Report(name=name)
        filtered_report.explanations = [
            explanation
            for explanation in self.explanations
            if explanation.target_class == target_class
            and np.argmax(explanation.counterfactual_prediction) == target_class
        ]
        return filtered_report

    def score_threshold(self, threshold, name=None) -> "Report":
        """
        Create a filtered Report containing only the explanations with a QuAC score above the given threshold.
        """
        filtered_report = Report(name=name)
        filtered_report.explanations = [
            explanation
            for explanation in self.explanations
            if explanation.score > threshold
        ]
        return filtered_report

    def top_n(self, n, name=None) -> "Report":
        """
        Create a filtered Report containing only the top n explanations.
        """
        filtered_report = Report(name=name)
        filtered_report.explanations = sorted(
            self.explanations, key=lambda x: x.score, reverse=True
        )[:n]
        return filtered_report

    @classmethod
    def from_directory(cls, eval_directory, **kwargs) -> "Report":
        """
        Find and load all reports in a given directory, merging them into a single report.
        The best attribution method for each sample is chosen based on the QuAC score.
        The final report selects only the best results for each sample and sorts them by QuAC score.

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

        kwargs: additional arguments to be passed to the Report constructor, specifying the name and metadata.

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
        return merge_reports(reports, **kwargs)
