from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .classifier import predict
from .data import validate_image


@dataclass
class Evaluation:
    """Complete stored counterfactual sweep for one attribution map."""

    areas: np.ndarray
    cutoffs: np.ndarray
    masks: np.ndarray
    counterfactuals: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    deltas: np.ndarray
    optimal_index: int
    auc: float


def area_masks(attribution_map: torch.Tensor, levels: int, sigma: float):
    """Select ranked pixels at equidistant areas, with stable ties and spatial blur."""
    if attribution_map.ndim != 2 or not torch.isfinite(attribution_map).all():
        raise ValueError("Expected a finite HW attribution map")
    if (
        attribution_map.min() < 0
        or attribution_map.max() > 1
        or levels < 2
        or sigma < 0
    ):
        raise ValueError(
            "Require a [0,1] attribution map, at least two levels, and nonnegative sigma"
        )
    values = attribution_map.detach().cpu().numpy()
    order = np.argsort(-values.ravel(), kind="stable")
    counts = np.rint(np.linspace(0, values.size, levels)).astype(int)
    masks, cutoffs = [], []
    for count in counts:
        mask = np.zeros(values.size, dtype=np.float32)
        mask[order[:count]] = 1
        mask = mask.reshape(values.shape)
        masks.append(gaussian_filter(mask, sigma=sigma) if sigma else mask)
        cutoffs.append(1.0 if count == 0 else float(values.ravel()[order[count - 1]]))
    return counts / values.size, np.array(cutoffs), np.stack(masks)


def optimal_threshold_index(areas, deltas) -> int:
    """Maximize delta/positive area, choosing the smallest area on ties."""
    areas, deltas = np.asarray(areas), np.asarray(deltas)
    valid = np.flatnonzero(areas > 0)
    if not len(valid):
        raise ValueError("At least one positive mask area is required")
    valid = valid[np.argsort(areas[valid], kind="stable")]
    return int(valid[np.argmax(deltas[valid] / areas[valid])])


def evaluate(
    classifier,
    source_image,
    generated_image,
    attribution_map,
    target_class,
    classes,
    source_target_probability,
    levels=101,
    sigma=1.0,
    batch_size=16,
) -> Evaluation:
    """Blend and classify every stored area level without further generation."""
    validate_image(source_image)
    validate_image(generated_image)
    if source_image.shape != generated_image.shape or tuple(
        attribution_map.shape
    ) != tuple(source_image.shape[-2:]):
        raise ValueError(
            "Source image, generated image, and attribution map must have matching shapes"
        )
    areas, cutoffs, masks = area_masks(attribution_map, levels, sigma)
    counterfactual_batches, logits, probabilities = [], [], []
    for start in range(0, levels, batch_size):
        mask = torch.from_numpy(masks[start : start + batch_size]).to(
            source_image.device
        )[:, None]
        counterfactuals = source_image[None] * (1 - mask) + generated_image[None] * mask
        output, probability = predict(classifier, counterfactuals, classes, batch_size)
        counterfactual_batches.append(counterfactuals.cpu().numpy())
        logits.append(output.numpy())
        probabilities.append(probability.numpy())
    logits, probabilities = np.concatenate(logits), np.concatenate(probabilities)
    deltas = probabilities[:, target_class] - source_target_probability
    optimal = optimal_threshold_index(areas, deltas)
    return Evaluation(
        areas,
        cutoffs,
        masks,
        np.concatenate(counterfactual_batches),
        logits,
        probabilities,
        deltas,
        optimal,
        float(np.trapezoid(deltas, areas)),
    )
