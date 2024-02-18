from pydantic import BaseModel


class ModelConfig(BaseModel):
    img_size: int = 128
    style_dim: int = 64
    latent_dim: int = 16
    num_domains: int = 5
