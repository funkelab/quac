import numpy as np
from typing import Optional, Union
import torch
from pathlib import Path
from quac.data import read_image


def serialize(obj):
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
        return [serialize(x) for x in obj]
    return obj


class Explanation:
    def __init__(
        self,
        query_path: str,
        counterfactual_path: str,
        mask_path: str,
        query_prediction: Union[list, np.ndarray],
        counterfactual_prediction: Union[list, np.ndarray],
        source_class: int,
        target_class: int,
        score: Optional[float] = None,
        # Optional
        attribution_path: Optional[str] = None,
        generated_path: Optional[str] = None,
        generated_prediction: Optional[Union[list, np.ndarray]] = None,
        normalized_mask_sizes: Optional[Union[list, np.ndarray]] = None,
        score_changes: Optional[Union[list, np.ndarray]] = None,
        optimal_threshold: Optional[float] = None,
        method: Optional[str] = None,
    ):
        self._query_path = query_path
        self._counterfactual_path = counterfactual_path
        self._mask_path = mask_path
        self.query_prediction = query_prediction
        self.counterfactual_prediction = counterfactual_prediction
        self.source_class = source_class
        self.target_class = target_class
        self.score = score
        # Optional
        self.generated_prediction = generated_prediction
        self._generated_path = generated_path
        self._attribution_path = attribution_path
        self._normalized_mask_sizes = normalized_mask_sizes
        self._score_changes = score_changes
        self._optimal_threshold = optimal_threshold
        self._method = method

        # Computed
        self._query: Optional[torch.Tensor] = None
        self._counterfactual: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None

    def __eq__(self, value):
        if isinstance(value, Explanation):
            return (
                self._query_path == value._query_path
                and self._counterfactual_path == value._counterfactual_path
                and self._mask_path == value._mask_path
                and np.array_equal(self.query_prediction, value.query_prediction)
                and np.array_equal(
                    self.counterfactual_prediction, value.counterfactual_prediction
                )
                and self.source_class == value.source_class
                and self.target_class == value.target_class
                and self.score == value.score
            )
            # The other, optional, attributes are not checked for equality
        return False

    def __repr__(self):
        return f"Explanation(query_path={self._query_path}, counterfactual_path={self._counterfactual_path}, mask_path={self._mask_path}, source_class={self.source_class}, target_class={self.target_class})"

    @property
    def query(self) -> torch.Tensor:
        if self._query is None:
            self._query = self.read_image(self._query_path)
        return self._query

    @property
    def counterfactual(self) -> torch.Tensor:
        if self._counterfactual is None:
            self._counterfactual = self.read_image(self._counterfactual_path)
        return self._counterfactual

    @property
    def mask(self) -> torch.Tensor:
        if self._mask is None:
            self._mask = torch.from_numpy(np.load(self._mask_path))
        return self._mask

    def read_image(self, path: str) -> torch.Tensor:
        """
        Read an image from a given path.
        """
        return read_image(path)  # Assuming read_image is defined elsewhere


def explanation_encoder(explanation: Explanation):
    """Custom JSON encoder for the Explanation class.

    Stores full dict representation of the Explanation object, so that it can be re-loaded exactly as is later.
    """
    if isinstance(explanation, Explanation):
        return {
            "query_path": explanation._query_path,
            "counterfactual_path": explanation._counterfactual_path,
            "mask_path": explanation._mask_path,
            "query_prediction": serialize(explanation.query_prediction),
            "counterfactual_prediction": serialize(
                explanation.counterfactual_prediction
            ),
            "source_class": explanation.source_class,
            "target_class": explanation.target_class,
            "score": explanation.score,
            # Optionals
            "normalized_mask_sizes": serialize(explanation._normalized_mask_sizes),
            "score_changes": serialize(explanation._score_changes),
            "generated_path": explanation._generated_path,
            "attribution_path": explanation._attribution_path,
            "generated_prediction": serialize(explanation.generated_prediction),
            "optimal_threshold": explanation._optimal_threshold,
            "method": explanation._method,
        }
