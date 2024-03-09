from pydantic import BaseModel


class ModelConfig(BaseModel):
    img_size: int = 128
    style_dim: int = 64
    latent_dim: int = 16
    num_domains: int = 5


class DataConfig(BaseModel):
    train_img_dir: str
    img_size: int = 128
    batch_size: int = 16
    randcrop_prob: float = 0.0
    num_workers: int = 4


class TrainConfig(BaseModel):
    f_lr: float = 1e-4  # Learning rate for the mapping network
    lr: float = 1e-4  # Learning rate for the other networks
    beta1: float = 0.5  # Beta1 for Adam optimizer
    beta2: float = 0.999  # Beta2 for Adam optimizer
    weight_decay: float = 1e-4  # Weight decay for Adam optimizer
    latent_dim: int = 16  # Latent dimension for the mapping network
    resume_iter: int = 0  # Iteration to resume training from
    lamdba_ds: float = 1.0  # Weight for the diversity sensitive loss
    total_iters: int = 100000  # Total number of iterations to train the model
    ds_iter: int = 1000  # Number of iterations to optimize the diversity sensitive loss
    log_every: int = 1000  # How often (iterations) to log training progress
    save_every: int = 10000  # How often (iterations) to save the model
    eval_every: int = 10000  # How often (iterations) to evaluate the model
