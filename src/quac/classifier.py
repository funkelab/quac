"""Classifier loading and batched inference without image rescaling."""

from pathlib import Path
import torch


def load_classifier(path: Path, device: torch.device) -> torch.nn.Module:
    """Load a TorchScript classifier in evaluation mode."""
    return torch.jit.load(str(path), map_location=device).eval()


@torch.no_grad()
def predict(classifier, images: torch.Tensor, classes: int, batch_size: int = 16):
    """Return exact logits and softmax probabilities for NCHW images."""
    outputs = []
    for batch in images.split(batch_size):
        logits = classifier(batch)
        if logits.shape != (len(batch), classes) or not torch.isfinite(logits).all():
            raise ValueError(f"Classifier must return finite [N,{classes}] logits")
        outputs.append(logits.detach().cpu())
    logits = torch.cat(outputs)
    return logits, logits.softmax(dim=1)
