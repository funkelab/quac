import contextlib
import sys

import torch
from tqdm import tqdm

from ..classifier import predict
from .losses import discriminator_loss, generator_loss

MODES = ("latent", "reference")


def _batch(dataset, indices, device):
    images, labels = zip(*(dataset[index] for index in indices))
    return torch.stack(images).to(device), torch.tensor(labels, device=device)


def _sample(indices):
    return indices[torch.randint(len(indices), ()).item()]


def _style(nets, mode, latent_vectors, reference_images, target_classes):
    if mode == "latent":
        return nets["mapping_network"](latent_vectors, target_classes)
    return nets["style_encoder"](reference_images, target_classes)


def _loss_metrics(losses):
    """Name losses ordered as discriminator then generator, each latent then reference."""
    names = [f"{network}_{mode}" for network in "DG" for mode in MODES]
    return {"D": sum(losses[:2]), "G": sum(losses[2:]), **dict(zip(names, losses))}


def _wandb_run(config):
    if not config.wandb.enabled:
        return contextlib.nullcontext()
    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            "wandb logging requires the optional extra: pip install 'quac[wandb]'"
        ) from error
    options = config.wandb
    return wandb.init(
        project=options.project,
        entity=options.entity,
        name=options.name,
        tags=options.tags,
        mode=options.mode,
        config=config.model_dump(mode="json"),
    )


@torch.no_grad()
def _translation_probabilities(model, dataset, classifier, config):
    """Return classifier target probabilities before and after every pair translation."""
    options = config.training
    classes = dataset.classes
    class_indices = [dataset.indices(i) for i in range(len(classes))]
    pairs = {}
    for source, indices in enumerate(class_indices):
        source_images, _ = _batch(
            dataset, indices[: options.validation_samples], model.device
        )
        _, before = predict(classifier, source_images, len(classes), options.batch_size)
        for target in range(len(classes)):
            if target == source:
                continue
            generated = []
            for images in source_images.split(options.batch_size):
                target_classes = torch.full((len(images),), target, device=model.device)
                reference_images = None
                if config.generator.sampling == "reference":
                    reference_images, _ = _batch(
                        dataset,
                        [_sample(class_indices[target]) for _ in images],
                        model.device,
                    )
                latent_vectors = torch.randn(
                    len(images), config.generator.latent_dim
                ).to(model.device)
                generated.append(
                    model.generate(
                        images, target_classes, reference_images, latent_vectors
                    )
                )
            _, after = predict(
                classifier, torch.cat(generated), len(classes), options.batch_size
            )
            pairs[classes[source], classes[target]] = (
                before[:, target],
                after[:, target],
                after.argmax(1) == target,
            )
    return pairs


def _validate(model, dataset, classifier, config):
    """Score validation losses and translations with a fixed seed, sparing training RNG."""
    options = config.training
    nets = model.nets
    classes = len(dataset.classes)
    class_indices = [dataset.indices(i) for i in range(classes)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        totals = [0.0] * 4
        for start in range(0, len(dataset), options.batch_size):
            indices = range(start, min(start + options.batch_size, len(dataset)))
            source_images, source_classes = _batch(dataset, indices, model.device)
            target_labels = (
                source_classes.cpu() + torch.randint(1, classes, (len(indices),))
            ) % classes
            reference_images, target_classes = _batch(
                dataset,
                [_sample(class_indices[y]) for y in target_labels],
                model.device,
            )
            latent_vectors = torch.randn(len(indices), config.generator.latent_dim).to(
                model.device
            )
            for index, mode in enumerate(MODES):
                with torch.no_grad():
                    style = _style(
                        nets, mode, latent_vectors, reference_images, target_classes
                    )
                    generator = generator_loss(
                        nets,
                        source_images,
                        source_classes,
                        target_classes,
                        style,
                        options.lambda_style,
                        options.lambda_cycle,
                    )
                # The R1 penalty needs input gradients; no parameters are updated.
                discriminator = discriminator_loss(
                    nets,
                    source_images,
                    source_classes,
                    target_classes,
                    style,
                    options.lambda_reg,
                )
                totals[index] += discriminator.item() * len(indices)
                totals[2 + index] += generator.item() * len(indices)
        metrics = _loss_metrics([total / len(dataset) for total in totals])
        pairs = {}
        if classifier is not None:
            pairs = _translation_probabilities(model, dataset, classifier, config)
    for (source, target), (_, after, translated) in pairs.items():
        metrics[f"{source}->{target}/translation_rate"] = (
            translated.float().mean().item()
        )
        metrics[f"{source}->{target}/mean_target_probability"] = after.mean().item()
    if pairs:
        metrics["translation_rate"] = sum(
            translated.float().mean().item() for *_, translated in pairs.values()
        ) / len(pairs)
    return metrics, pairs


def _log_validation(run, step, metrics, pairs):
    import wandb

    payload = {f"validation/{name}": value for name, value in metrics.items()}
    for (source, target), (before, after, _) in pairs.items():
        table = wandb.Table(
            columns=["source_probability", "generated_probability"],
            data=torch.stack([before, after], 1).tolist(),
        )
        payload[f"validation/scatter/{source}->{target}"] = wandb.plot.scatter(
            table,
            "source_probability",
            "generated_probability",
            title=f"p({target}): {source} source vs generated",
        )
    run.log(payload, step=step)


def _write(progress, line):
    """Print above the progress bar, flushing so redirected logs stay current."""
    progress.write(line)
    sys.stdout.flush()


def _step(loss, optimizers, names):
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name in names:
        optimizers[name].step()


def train_stargan(model, dataset, config, directory, validation=None, classifier=None):
    """Train latent vector and reference image branches and save resumable checkpoints.

    A validation dataset is scored every ``training.validate_every`` steps; with a
    classifier, translations of its images between every class pair are also scored.
    """
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
    # Without a terminal (e.g. logs redirected to a file) the bar is disabled and
    # only the logged lines are printed.
    progress = tqdm(
        range(start + 1, options.steps + 1),
        initial=start,
        total=options.steps,
        desc="training",
        unit="step",
        dynamic_ncols=True,
        disable=None,
    )
    with _wandb_run(config) as run, progress:
        for step in progress:
            # Draw classes uniformly so small classes receive equal training weight.
            source_labels = torch.randint(len(dataset.classes), (options.batch_size,))
            target_labels = (
                source_labels
                + torch.randint(1, len(dataset.classes), source_labels.shape)
            ) % len(dataset.classes)
            source_indices = [
                class_indices[y][torch.randint(len(class_indices[y]), ()).item()]
                for y in source_labels
            ]
            reference_indices = [
                class_indices[y][torch.randint(len(class_indices[y]), ()).item()]
                for y in target_labels
            ]
            source_images, source_classes = _batch(
                dataset, source_indices, model.device
            )
            reference_images, target_classes = _batch(
                dataset, reference_indices, model.device
            )
            latent_vectors = torch.randn(
                len(source_images), config.generator.latent_dim, device=model.device
            )
            losses = []
            for mode in MODES:
                with torch.no_grad():
                    style = _style(
                        nets, mode, latent_vectors, reference_images, target_classes
                    )
                loss = discriminator_loss(
                    nets,
                    source_images,
                    source_classes,
                    target_classes,
                    style,
                    options.lambda_reg,
                )
                _step(loss, optimizers, ["discriminator"])
                losses.append(loss.item())
            for mode in MODES:
                style = _style(
                    nets, mode, latent_vectors, reference_images, target_classes
                )
                loss = generator_loss(
                    nets,
                    source_images,
                    source_classes,
                    target_classes,
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
                metrics = _loss_metrics(losses)
                progress.set_postfix(
                    D=f"{metrics['D']:.4f}", G=f"{metrics['G']:.4f}", refresh=False
                )
                _write(
                    progress,
                    f"step {step}/{options.steps}: D={metrics['D']:.4f} G={metrics['G']:.4f}",
                )
                if run is not None:
                    run.log(
                        {f"train/{name}": value for name, value in metrics.items()},
                        step=step,
                    )
            if validation is not None and (
                step % options.validate_every == 0 or step == options.steps
            ):
                metrics, pairs = _validate(model, validation, classifier, config)
                translation = (
                    f" translation_rate={metrics['translation_rate']:.3f}"
                    if pairs
                    else ""
                )
                _write(
                    progress,
                    f"validation {step}/{options.steps}: "
                    f"D={metrics['D']:.4f} G={metrics['G']:.4f}{translation}",
                )
                if run is not None:
                    _log_validation(run, step, metrics, pairs)
            if step % options.save_every == 0 or step == options.steps:
                model.save(directory / f"{step:06d}.pt", step, optimizers)
                model.save(directory / "latest.pt", step, optimizers)
    return directory / "latest.pt"
