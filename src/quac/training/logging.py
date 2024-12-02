import json
from numbers import Number
import numpy as np
from pathlib import Path
from typing import Union, Optional
import torch
import torchvision

try:
    import wandb
except ImportError:
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class Logger:
    def create(log_type, resume_iter=0, hparams={}, **kwargs):
        if log_type == "wandb":
            if wandb is None:
                raise ImportError("wandb is not installed.")
            resume = "allow" if resume_iter > 0 else False
            return WandBLogger(hparams=hparams, resume=resume, **kwargs)
        elif log_type == "tensorboard":
            if SummaryWriter is None:
                raise ImportError("Tensorboard is not available.")
            purge_step = resume_iter if resume_iter > 0 else None
            return TensorboardLogger(hparams=hparams, purge_step=purge_step, **kwargs)
        elif log_type == "local":
            return LocalLogger(hparams=hparams, **kwargs)
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


class LocalLogger:
    def __init__(
        self,
        log_dir: str,
        project: str = "",
        name: str = "",
        notes: str = "",
        tags: list = [],
        hparams: dict = {},
        **kwargs,
    ):
        self.log_dir = Path(log_dir)
        self.images_dir = self.log_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        # Store the metadata
        metadata = {
            "project": project,
            "name": name,
            "notes": notes,
            "tags": tags,
            "config": hparams,
        }
        with open(self.log_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

    def log(self, data: dict[str, Number], step: int = 0):
        """Add metrics to a CSV file."""
        for key, value in data.items():
            if not (self.log_dir / f"{key}.csv").exists():
                with open(self.log_dir / f"{key}.csv", "w") as f:
                    f.write("step,value\n")
                    f.write(f"{step},{value}\n")
            else:
                with open(self.log_dir / f"{key}.csv", "a") as f:
                    f.write(f"{step},{value}\n")

    def log_images(
        self, data: dict[str, Union[torch.Tensor, np.ndarray]], step: int = 0
    ):
        """Save images to a directory using torchvision."""
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu()
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            torchvision.utils.save_image(value, self.images_dir / f"{step}_{key}.png")
