from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .classifier import predict
from .data import validate_image
from .masks import close_mask, ellipse_kernel


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


def area_masks(
    attribution_map: torch.Tensor, levels: int, sigma: float, closing_size: int = 10
):
    """Select ranked channel pixels, close each mask, then blur spatially."""
    if attribution_map.ndim not in (2, 3) or not torch.isfinite(attribution_map).all():
        raise ValueError("Expected a finite HW or CHW attribution map")
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
    kernel = ellipse_kernel(closing_size)
    blur_sigma = (0, sigma, sigma) if values.ndim == 3 else sigma
    masks, cutoffs, areas = [], [], []
    for count in counts:
        mask = np.zeros(values.size, dtype=np.float32)
        mask[order[:count]] = 1
        mask = close_mask(mask.reshape(values.shape), kernel)
        areas.append(np.count_nonzero(mask) / mask.size)
        masks.append(gaussian_filter(mask, sigma=blur_sigma) if sigma else mask)
        cutoffs.append(1.0 if count == 0 else float(values.ravel()[order[count - 1]]))
    return np.array(areas), np.array(cutoffs), np.stack(masks)


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
    closing_size=10,
) -> Evaluation:
    """Blend and classify every stored area level without further generation."""
    validate_image(source_image)
    validate_image(generated_image)
    if attribution_map.ndim == 2:
        attribution_map = attribution_map[None].expand_as(source_image)
    if (
        source_image.shape != generated_image.shape
        or attribution_map.shape != source_image.shape
    ):
        raise ValueError(
            "Source image, generated image, and attribution map must have matching shapes"
        )
    areas, cutoffs, masks = area_masks(attribution_map, levels, sigma, closing_size)
    counterfactual_batches, logits, probabilities = [], [], []
    for start in range(0, levels, batch_size):
        mask = torch.from_numpy(masks[start : start + batch_size]).to(
            source_image.device
        )
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
