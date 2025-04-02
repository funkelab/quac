import pytest
import json
from typing import List
from quac.explanation import Explanation, explanation_encoder


class DummyReport:
    def __init__(self, explanations: List[Explanation]):
        self.results = explanations

    def store(self, json_file):
        """Store the report in a JSON file."""
        json.dump(self.results, json_file, default=explanation_encoder)

    @classmethod
    def load(cls, json_file):
        """Load the report from a JSON file."""
        with open(json_file, "r") as file:
            results = json.load(file)
            # Convert the loaded data back to DummyExplanation objects
            results = [Explanation(**result) for result in results]
        return cls(results)

    def __eq__(self, value):
        """Check equality based on the results."""
        if isinstance(value, DummyReport) and len(self.results) == len(value.results):
            for i in range(len(self.results)):
                if self.results[i] != value.results[i]:
                    return False
            return True
        return False


@pytest.fixture
def explanation() -> Explanation:
    """
    Create a simple explanation for testing.
    """
    return Explanation(
        query_path="path/to/query",
        counterfactual_path="path/to/counterfactual",
        mask_path="path/to/mask",
        query_prediction=[0.1, 0.9],
        counterfactual_prediction=[0.2, 0.8],
        source_class=0,
        target_class=1,
        score=0.5,
    )


def test_explanation_equality(explanation: Explanation):
    """
    Test equality of two explanations.
    """
    explanation2 = Explanation(
        query_path="path/to/query",
        counterfactual_path="path/to/counterfactual",
        mask_path="path/to/mask",
        query_prediction=[0.1, 0.9],
        counterfactual_prediction=[0.2, 0.8],
        source_class=0,
        target_class=1,
        score=0.5,
    )
    assert explanation == explanation2
    explanation2.score = 0.75
    assert explanation != explanation2


def test_report_equality(explanation: Explanation):
    """
    Test equality of two reports.
    """
    report1 = DummyReport([explanation])
    report2 = DummyReport([explanation])
    assert report1 == report2


@pytest.mark.parametrize("repeats", [1])
def test_json(explanation: Explanation, tmp_path, repeats: int):
    """
    Test saving a set of explanations to JSON, using __repr__.
    """
    json_path = tmp_path / "explanation.json"
    write_report = DummyReport([explanation] * repeats)
    with open(json_path, "w") as json_file:
        write_report.store(json_file)

    read_report = DummyReport.load(json_path)
    # Compare the loaded report with the original
    assert read_report == write_report


def test_malformed_report(tmp_path):
    """
    Test loading a malformed report.
    """
    json_path = tmp_path / "malformed_report.json"
    with open(json_path, "w") as json_file:
        json.dump({"a": 1, "b": "test"}, json_file)
    with pytest.raises(TypeError):
        DummyReport.load(json_path)


def test_missing_fields_report(tmp_path):
    # Test loading a report with missing fields
    json_path = tmp_path / "missing_fields_report.json"
    with open(json_path, "w") as json_file:
        json.dump([{"query_path": "path/to/query"}], json_file)
    with pytest.raises(TypeError):
        DummyReport.load(json_path)
