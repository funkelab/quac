from pydantic import BaseModel
from typing import Optional, Union, Literal


class ModelConfig(BaseModel):
    img_size: int = 128
    style_dim: int = 64
    latent_dim: int = 16
    num_domains: int = 5
    input_dim: int = 3
    final_activation: str = "tanh"


class DataConfig(BaseModel):
    source: str
    reference: str
    img_size: int = 128
    batch_size: int = 1
    num_workers: int = 4
    grayscale: bool = False
    mean: Optional[float] = 0.5
    std: Optional[float] = 0.5
    rand_crop_prob: Optional[float] = 0


class RunConfig(BaseModel):
    resume_iter: int = 0
    total_iters: int = 100000
    log_every: int = 1000
    save_every: int = 10000
    eval_every: int = 10000


class ValConfig(BaseModel):
    classifier_checkpoint: str
    num_outs_per_domain: int = 10
    mean: Optional[float] = 0.5
    std: Optional[float] = 0.5
    img_size: int = 128
    val_batch_size: int = 16
    assume_normalized: bool = False


class LossConfig(BaseModel):
    lambda_ds: float = 0.0  # No diversity by default
    lambda_sty: float = 1.0
    lambda_cyc: float = 1.0
    lambda_reg: float = 1.0
    ds_iter: int = 100000


class SolverConfig(BaseModel):
    root_dir: str
    f_lr: float = 1e-6
    lr: float = 1e-4
    beta1: float = 0.0
    beta2: float = 0.99
    weight_decay: float = 1e-4


class WandBLogConfig(BaseModel):
    project: str = "default"
    name: str = "default"
    notes: str = ""
    tags: list = []


class TensorboardLogConfig(BaseModel):
    log_dir: str
    comment: str = ""


class ExperimentConfig(BaseModel):
    # Metadata for keeping track of experiments
    log_type: Literal["wandb", "tensorboard"] = "wandb"
    log: Union[WandBLogConfig, TensorboardLogConfig] = WandBLogConfig()
    # Some input required
    data: DataConfig
    solver: SolverConfig
    validation_data: DataConfig
    validation_config: ValConfig
    # No input required
    model: ModelConfig = ModelConfig()
    run: RunConfig = RunConfig()
    loss: LossConfig = LossConfig()
