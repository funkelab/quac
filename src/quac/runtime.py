import random
import numpy as np
import torch


def initialize(config) -> torch.device:
    """Seed CPU/device generators and choose the requested compute device."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)
