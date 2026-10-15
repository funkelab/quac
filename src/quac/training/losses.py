import torch
from torch.nn import functional as F


def adversarial(logits, real):
    """Binary logistic adversarial loss."""
    return F.binary_cross_entropy_with_logits(
        logits, torch.full_like(logits, float(real))
    )


def discriminator_loss(nets, real, source, target, style, regularization):
    """Real/fake discrimination with an R1 penalty in the signed StarGAN domain."""
    real = real.detach().requires_grad_(True)
    output = nets["discriminator"](real, source)
    gradient = torch.autograd.grad(output.sum(), real, create_graph=True)[0]
    # Unit-range network adapters double derivatives relative to signed inputs.
    penalty = gradient.square().flatten(1).sum(1).mean() / 8
    with torch.no_grad():
        generated = nets["generator"](real, style.detach())
    loss = adversarial(output, True) + adversarial(
        nets["discriminator"](generated, target), False
    )
    return loss + regularization * penalty


def generator_loss(nets, real, source, target, style, lambda_style, lambda_cycle):
    """Adversarial, style, and cycle objectives using unit-range adapters."""
    generated = nets["generator"](real, style)
    augmented = generated
    for axis in (-1, -2):
        if torch.rand(()) < 0.5:
            augmented = augmented.flip(axis)
    reconstructed_style = nets["style_encoder"](augmented, target)
    original_style = nets["style_encoder"](real, source)
    reconstructed = nets["generator"](generated, original_style)
    return (
        adversarial(nets["discriminator"](generated, target), True)
        + lambda_style * (reconstructed_style - style).abs().mean()
        + lambda_cycle * 2 * (reconstructed - real).abs().mean()
    )
