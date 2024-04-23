from pydantic import BaseModel
from typing import Optional


class ModelConfig(BaseModel):
    img_size: int = 128
    style_dim: int = 64
    latent_dim: int = 16
    num_domains: int = 5


class DataConfig(BaseModel):
    source: str
    reference: str
    validation: str
    img_size: int = 128
    batch_size: int = 16
    num_workers: int = 4
    grayscale: bool = False
    latent_dim: int = 16


class RunConfig(BaseModel):
    resume_iter: int = 0
    total_iter: int = 100000
    log_every: int = 1000
    save_every: int = 10000
    eval_every: int = 10000


class ValConfig(BaseModel):
    classifier_checkpoint: str
    num_outs_per_domain: int = 10
    mean: Optional[float] = 0.5
    std: Optional[float] = 0.5


class LossConfig(BaseModel):
    lambda_ds: float = 1.0
    lambda_sty: float = 1.0
    lambda_cyc: float = 1.0
    lambda_reg: float = 1.0
    ds_iter: int = 100000


class SolverConfig(BaseModel):
    checkpoint_dir: str
    f_lr: float = 1e-4
    lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.99
    weight_decay: float = 0.1


class ExperimentConfig(BaseModel):
    # Some input required
    data: DataConfig
    solver: SolverConfig
    val: ValConfig
    # No input required
    model: ModelConfig = ModelConfig()
    run: RunConfig = RunConfig()
    loss: LossConfig = LossConfig()
