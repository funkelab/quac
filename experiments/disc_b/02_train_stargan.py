# Training the StarGAN
from quac.training.config import ExperimentConfig
import git
from quac.training.data_loader import TrainingData, ValidationData
from quac.training.stargan import build_model
from quac.training.solver import Solver
from quac.training.logging import Logger
import torch
import typer
import warnings
import yaml

torch.backends.cudnn.benchmark = True


def main(config_file: str = "configs/stargan.yml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    experiment = ExperimentConfig(**config)

    logger = Logger.create(
        experiment.log_type,
        hparams=experiment.model_dump(),
        resume_iter=experiment.run.resume_iter,
        **experiment.log.model_dump(),
    )

    ## Defining the dataset
    dataset = TrainingData(**experiment.data.model_dump())

    ## Defining the models
    nets, nets_ema = build_model(**experiment.model.model_dump())

    ## Defining the Solver
    solver = Solver(nets, nets_ema, **experiment.solver.model_dump(), run=logger)

    # Setup classifier and data for validation
    val_dataset = ValidationData(**experiment.validation_data.model_dump())

    # Training the model
    solver.train(
        dataset,
        **experiment.run.model_dump(),
        **experiment.loss.model_dump(),
        val_loader=val_dataset,
        val_config=experiment.validation_config,
    )


if __name__ == "__main__":
    typer.run(main)
