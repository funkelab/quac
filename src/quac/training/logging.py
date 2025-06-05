from numbers import Number
import numpy as np
from typing import Union, Optional
import torch

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

try:
    from torch.utils.tensorboard import SummaryWriter

    tensorboard_available = True
except ImportError:
    tensorboard_available = False


class Logger:
    def create(log_type, resume_iter=0, hparams={}, **kwargs):
        if log_type == "wandb":
            if not wandb_available:
                raise ImportError("wandb is not installed.")
            resume = "allow" if resume_iter > 0 else False
            return WandBLogger(hparams=hparams, resume=resume, **kwargs)
        elif log_type == "tensorboard":
            if not tensorboard_available:
                raise ImportError("Tensorboard is not available.")
            purge_step = resume_iter if resume_iter > 0 else None
            return TensorboardLogger(hparams=hparams, purge_step=purge_step, **kwargs)
        else:
            raise NotImplementedError


class WandBLogger:
    def __init__(
        self,
        hparams: dict,
        project: str,
        name: str,
        notes: str,
        tags: list,
        resume: bool = False,
        id: Optional[str] = None,
    ):
        self.run = wandb.init(
            project=project,
            name=name,
            notes=notes,
            tags=tags,
            config=hparams,
            resume=resume,
            id=id,
        )

    def log(self, data: dict[str, Number], step: int = 0):
        self.run.log(data, step=step)

    def log_images(
        self, data: dict[str, Union[torch.Tensor, np.ndarray]], step: int = 0
    ):
        for key, value in data.items():
            self.run.log({key: wandb.Image(value)}, step=step)


class TensorboardLogger:
    # NOTE: Not tested
    def __init__(
        self,
        log_dir: str,
        comment: str,
        hparams: dict,
        purge_step: Union[int, None] = None,
    ):
        self.writer = SummaryWriter(log_dir, comment=comment, purge_step=purge_step)
        self.writer.add_hparams(hparams, {})

    def log(self, data: dict[str, Number], step: int = 0):
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def log_images(
        self, data: dict[str, Union[torch.Tensor, np.ndarray]], step: int = 0
    ):
        for key, value in data.items():
            self.writer.add_images(key, value, step)
