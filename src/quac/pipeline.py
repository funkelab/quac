"""High-level training and explanation orchestration."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .attribution import METHODS, PairAttribution, normalize
from .classifier import load_classifier, predict
from .config import Config
from .data import ImageDataset, validate_image
from .evaluation import evaluate
from .generators import build_generator
from .report import Report
from .runtime import initialize


def _checkpoint_identity(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def train(config: Config) -> Path:
    """Train the configured backend and return the latest checkpoint path."""
    from .training.trainer import train_stargan

    config = config.resolve_paths(Path.cwd())
    device = initialize(config)
    dataset = ImageDataset(config.data.source)
    model = build_generator(dataset, config.generator, device)
    return train_stargan(
        model, dataset, config, config.output.directory / "checkpoints"
    )


def explain(
    config: Config, *, generator=None, classifier=None, attributions=None
) -> Path:
    """Run every pair/method and persist a report and its image artifacts; components may be injected."""
    config = config.resolve_paths(Path.cwd())
    device = initialize(config)
    if classifier is None and config.classifier.checkpoint is None:
        raise ValueError(
            "explain requires classifier.checkpoint (a TorchScript model returning logits)"
        )
    methods = config.explanation.methods
    if len(set(methods)) != len(methods):
        raise ValueError("Attribution method names must be unique")
    if attributions is None and any(name not in METHODS for name in methods):
        raise ValueError(f"Unknown attribution method; choose from {METHODS}")
    if attributions is not None and set(attributions) != set(methods):
        raise ValueError("Injected attribution names must match explanation.methods")
    if config.output.report.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Report already exists: {config.output.report}; set output.overwrite: true"
        )
    dataset = ImageDataset(config.data.source)
    checkpoint = (
        config.generator.checkpoint
        or config.output.directory / "checkpoints" / "latest.pt"
    )
    if generator is None:
        generator = build_generator(dataset, config.generator, device)
        generator.load(checkpoint)
        generator_identity = _checkpoint_identity(checkpoint)
    else:
        generator_identity = {"injected": type(generator).__qualname__}
    if classifier is None:
        classifier = load_classifier(config.classifier.checkpoint, device)
        classifier_identity = _checkpoint_identity(config.classifier.checkpoint)
    else:
        classifier_identity = {"injected": type(classifier).__qualname__}
        classifier = classifier.to(device).eval()
    attributions = attributions or {
        name: PairAttribution(classifier, name, config.explanation.integration_steps)
        for name in methods
    }
    metadata = dict(
        config=config.model_dump(mode="json"),
        classes=dataset.classes,
        shape=dataset.shape,
        generator=generator_identity,
        classifier=classifier_identity,
        delta="target_probability - original_target_probability",
        optimum="max(delta / positive_binary_area)",
    )
    with Report.create(
        config.output.report,
        metadata,
        config.output.overwrite,
        image_directory=config.output.directory / "images",
    ) as report:
        try:
            _explain_pairs(
                config, dataset, generator, classifier, attributions, report, device
            )
        except BaseException as error:
            report.finish(f"{type(error).__name__}: {error}")
            raise
        report.finish()
    return config.output.report


def _explain_pairs(
    config, dataset, generator, classifier, attributions, report, device
):
    options = config.explanation
    for index, sample in enumerate(dataset.samples):
        real, source = dataset[index]
        real = real.to(device)
        original_logits, original_probs = predict(
            classifier, real[None], len(dataset.classes)
        )
        for target in range(len(dataset.classes)):
            if target == source:
                continue
            for number in range(options.samples):
                reference, reference_path = None, None
                if config.generator.sampling == "reference":
                    candidates = dataset.indices(target)
                    chosen = candidates[np.random.randint(len(candidates))]
                    reference = dataset[chosen][0][None].to(device)
                    reference_path = dataset.samples[chosen].path
                with torch.no_grad():
                    generated = generator.translate(
                        real[None], torch.tensor([target], device=device), reference
                    )[0].detach()
                validate_image(generated)
                generated_logits, generated_probs = predict(
                    classifier, generated[None], len(dataset.classes)
                )
                pair_id = report.add_pair(
                    sample.path,
                    source,
                    target,
                    number,
                    real.cpu().numpy(),
                    generated.cpu().numpy(),
                    original_logits[0].numpy(),
                    original_probs[0].numpy(),
                    generated_logits[0].numpy(),
                    generated_probs[0].numpy(),
                    reference_path,
                )
                for name, attribution in attributions.items():
                    importance = normalize(
                        attribution.attribute(real, generated, source, target)
                    )
                    result = evaluate(
                        classifier,
                        real,
                        generated,
                        importance,
                        target,
                        len(dataset.classes),
                        float(original_probs[0, target]),
                        options.levels,
                        options.sigma,
                        options.batch_size,
                    )
                    report.add_explanation(
                        pair_id, name, importance.cpu().numpy(), result
                    )
        print(f"explained {index + 1}/{len(dataset)}: {sample.path.name}", flush=True)
