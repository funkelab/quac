"""Generator factories, kept independent of pipeline orchestration."""


def build_generator(dataset, config, device):
    """Construct the configured generator; custom Python callers may inject their own."""
    if config.kind != "stargan":
        raise ValueError(f"Unknown generator backend: {config.kind}")
    from .stargan import StarGAN

    return StarGAN(dataset.shape, dataset.classes, config, device)
