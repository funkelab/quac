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
    images: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    deltas: np.ndarray
    optimal_index: int
    auc: float


def area_masks(attribution: torch.Tensor, levels: int, sigma: float):
    """Select ranked pixels at equidistant areas, with stable ties and spatial blur."""
    if attribution.ndim != 2 or not torch.isfinite(attribution).all():
        raise ValueError("Expected a finite HW attribution map")
    if attribution.min() < 0 or attribution.max() > 1 or levels < 2 or sigma < 0:
        raise ValueError(
            "Require [0,1] attribution, at least two levels, and nonnegative sigma"
        )
    values = attribution.detach().cpu().numpy()
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
    real,
    generated,
    attribution,
    target,
    classes,
    original_probability,
    levels=101,
    sigma=1.0,
    batch_size=16,
) -> Evaluation:
    """Blend and classify every stored area level without further generation."""
    validate_image(real)
    validate_image(generated)
    if real.shape != generated.shape or tuple(attribution.shape) != tuple(
        real.shape[-2:]
    ):
        raise ValueError(
            "Real/generated images and spatial attribution must have matching shapes"
        )
    areas, cutoffs, masks = area_masks(attribution, levels, sigma)
    images, logits, probabilities = [], [], []
    for start in range(0, levels, batch_size):
        mask = torch.from_numpy(masks[start : start + batch_size]).to(real.device)[
            :, None
        ]
        hybrids = real[None] * (1 - mask) + generated[None] * mask
        output, probability = predict(classifier, hybrids, classes, batch_size)
        images.append(hybrids.cpu().numpy())
        logits.append(output.numpy())
        probabilities.append(probability.numpy())
    logits, probabilities = np.concatenate(logits), np.concatenate(probabilities)
    deltas = probabilities[:, target] - original_probability
    optimal = optimal_threshold_index(areas, deltas)
    return Evaluation(
        areas,
        cutoffs,
        masks,
        np.concatenate(images),
        logits,
        probabilities,
        deltas,
        optimal,
        float(np.trapezoid(deltas, areas)),
    )
