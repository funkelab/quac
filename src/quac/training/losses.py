import torch
from torch.nn import functional as F


def adversarial(logits, is_source):
    """Binary logistic adversarial loss."""
    return F.binary_cross_entropy_with_logits(
        logits, torch.full_like(logits, float(is_source))
    )


def discriminator_loss(
    nets, source_images, source_classes, target_classes, style, regularization
):
    """Discriminate source and generated images with an R1 penalty."""
    source_images = source_images.detach().requires_grad_(True)
    output = nets["discriminator"](source_images, source_classes)
    gradient = torch.autograd.grad(output.sum(), source_images, create_graph=True)[0]
    # Unit-range network adapters double derivatives relative to signed inputs.
    penalty = gradient.square().flatten(1).sum(1).mean() / 8
    with torch.no_grad():
        generated_images = nets["generator"](source_images, style.detach())
    loss = adversarial(output, True) + adversarial(
        nets["discriminator"](generated_images, target_classes), False
    )
    return loss + regularization * penalty


def generator_loss(
    nets,
    source_images,
    source_classes,
    target_classes,
    style,
    lambda_style,
    lambda_cycle,
):
    """Adversarial, style, and cycle objectives using unit-range adapters."""
    generated_images = nets["generator"](source_images, style)
    augmented_generated_images = generated_images
    for axis in (-1, -2):
        if torch.rand(()) < 0.5:
            augmented_generated_images = augmented_generated_images.flip(axis)
    reconstructed_style = nets["style_encoder"](
        augmented_generated_images, target_classes
    )
    source_style = nets["style_encoder"](source_images, source_classes)
    generated_source_images = nets["generator"](generated_images, source_style)
    return (
        adversarial(nets["discriminator"](generated_images, target_classes), True)
        + lambda_style * (reconstructed_style - style).abs().mean()
        + lambda_cycle * 2 * (generated_source_images - source_images).abs().mean()
    )
