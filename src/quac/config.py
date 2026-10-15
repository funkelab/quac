"""Validated configuration shared by the Python API and command line."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class Settings(BaseModel):
    """Reject misspelled settings rather than silently ignoring them."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DataConfig(Settings):
    """Source directory containing one subdirectory per class."""

    source: Path


class GeneratorConfig(Settings):
    """StarGAN architecture and translation sampling settings."""

    kind: str = "stargan"
    checkpoint: Path | None = None
    sampling: Literal["latent", "reference"] = "latent"
    style_dim: int = Field(64, ge=1)
    latent_dim: int = Field(16, ge=1)
    max_channels: int = Field(512, ge=1)
    base_channels: int | None = Field(None, ge=1)
    mapping_width: int = Field(512, ge=1)


class TrainingConfig(Settings):
    """Optimization, logging, and checkpoint intervals."""

    steps: int = Field(100000, ge=1)
    batch_size: int = Field(4, ge=1)
    lr: float = Field(1e-4, gt=0)
    mapping_lr: float = Field(1e-6, gt=0)
    beta1: float = Field(0.0, ge=0, lt=1)
    beta2: float = Field(0.99, ge=0, lt=1)
    weight_decay: float = Field(1e-4, ge=0)
    lambda_style: float = Field(1.0, ge=0)
    lambda_cycle: float = Field(1.0, ge=0)
    lambda_reg: float = Field(1.0, ge=0)
    ema_beta: float = Field(0.999, ge=0, lt=1)
    log_every: int = Field(100, ge=1)
    save_every: int = Field(10000, ge=1)


class ClassifierConfig(Settings):
    """TorchScript checkpoint returning logits in directory-label order."""

    checkpoint: Path | None = None


class ExplanationConfig(Settings):
    """Attribution and counterfactual evaluation parameters."""

    methods: list[str] = Field(
        default_factory=lambda: ["DIntegratedGradients"], min_length=1
    )
    samples: int = Field(3, ge=1)
    levels: int = Field(101, ge=2)
    sigma: float = Field(1.0, ge=0)
    batch_size: int = Field(16, ge=1)
    integration_steps: int = Field(50, ge=2)


class OutputConfig(Settings):
    """Config-relative artifact locations."""

    directory: Path = Path("quac-output")
    report: Path | None = None
    summary: Path | None = None
    overwrite: bool = False


class SummaryConfig(Settings):
    """Interactive summary layout."""

    top_k: int = Field(5, ge=1)


class Config(Settings):
    """Complete run configuration; only data.source is required for training."""

    data: DataConfig
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    explanation: ExplanationConfig = Field(default_factory=ExplanationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    device: str = "auto"
    seed: int = Field(0, ge=0)

    def resolve_paths(self, base: Path) -> "Config":
        """Return a copy with absolute paths relative to base."""
        config = self.model_copy(deep=True)
        for section, name in [
            (config.data, "source"),
            (config.generator, "checkpoint"),
            (config.classifier, "checkpoint"),
            (config.output, "directory"),
            (config.output, "report"),
            (config.output, "summary"),
        ]:
            path = getattr(section, name)
            if path is not None:
                setattr(section, name, (base / path).resolve())
        config.output.report = (
            config.output.report or config.output.directory / "report.sqlite"
        )
        config.output.summary = (
            config.output.summary or config.output.directory / "summary.html"
        )
        return config


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read YAML and resolve artifact paths against its containing directory."""
    path = Path(path).resolve()
    with path.open() as stream:
        return Config.model_validate(yaml.safe_load(stream)).resolve_paths(path.parent)
