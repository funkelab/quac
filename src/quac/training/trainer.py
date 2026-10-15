import torch

from .losses import discriminator_loss, generator_loss


def _batch(dataset, indices, device):
    images, labels = zip(*(dataset[index] for index in indices))
    return torch.stack(images).to(device), torch.tensor(labels, device=device)


def _step(loss, optimizers, names):
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name in names:
        optimizers[name].step()


def train_stargan(model, dataset, config, directory):
    """Train latent/reference branches and save resumable checkpoints."""
    options = config.training
    nets = model.nets
    optimizers = {
        name: torch.optim.Adam(
            network.parameters(),
            lr=options.mapping_lr if name == "mapping_network" else options.lr,
            betas=(options.beta1, options.beta2),
            weight_decay=options.weight_decay,
        )
        for name, network in nets.items()
    }
    start = 0
    if config.generator.checkpoint:
        payload = model.load(config.generator.checkpoint)
        if "optimizers" not in payload:
            raise ValueError(
                "Resuming training requires a checkpoint with optimizer states"
            )
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        start = payload["step"]
        torch.set_rng_state(payload["rng"].cpu())
        if model.device.type == "cuda" and payload.get("rng_cuda"):
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["rng_cuda"]])
        if start >= options.steps:
            raise ValueError("training.steps must exceed the resumed checkpoint step")
    class_indices = [dataset.indices(i) for i in range(len(dataset.classes))]
    for step in range(start + 1, options.steps + 1):
        # Draw classes uniformly so small domains receive equal training weight.
        source_labels = torch.randint(len(dataset.classes), (options.batch_size,))
        target_labels = (
            source_labels + torch.randint(1, len(dataset.classes), source_labels.shape)
        ) % len(dataset.classes)
        source_indices = [
            class_indices[y][torch.randint(len(class_indices[y]), ()).item()]
            for y in source_labels
        ]
        reference_indices = [
            class_indices[y][torch.randint(len(class_indices[y]), ()).item()]
            for y in target_labels
        ]
        real, source = _batch(dataset, source_indices, model.device)
        reference, target = _batch(dataset, reference_indices, model.device)
        latent = torch.randn(
            len(real), config.generator.latent_dim, device=model.device
        )
        losses = []
        for mode in ("latent", "reference"):
            with torch.no_grad():
                style = (
                    nets["mapping_network"](latent, target)
                    if mode == "latent"
                    else nets["style_encoder"](reference, target)
                )
            loss = discriminator_loss(
                nets, real, source, target, style, options.lambda_reg
            )
            _step(loss, optimizers, ["discriminator"])
            losses.append(loss.item())
        for mode in ("latent", "reference"):
            style = (
                nets["mapping_network"](latent, target)
                if mode == "latent"
                else nets["style_encoder"](reference, target)
            )
            loss = generator_loss(
                nets,
                real,
                source,
                target,
                style,
                options.lambda_style,
                options.lambda_cycle,
            )
            names = (
                ["generator", "mapping_network", "style_encoder"]
                if mode == "latent"
                else ["generator"]
            )
            _step(loss, optimizers, names)
            losses.append(loss.item())
        model.update_ema(options.ema_beta)
        if step % options.log_every == 0 or step == options.steps:
            print(
                f"step {step}/{options.steps}: D={sum(losses[:2]):.4f} G={sum(losses[2:]):.4f}",
                flush=True,
            )
        if step % options.save_every == 0 or step == options.steps:
            model.save(directory / f"{step:06d}.pt", step, optimizers)
            model.save(directory / "latest.pt", step, optimizers)
    return directory / "latest.pt"
