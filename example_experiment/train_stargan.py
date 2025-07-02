# Training the StarGAN
from argparse import ArgumentParser
from quac.config import ExperimentConfig
from quac.training.data_loader import TrainingData, ValidationData
from quac.training.stargan import build_model
from quac.training.solver import Solver
from quac.training.logging import Logger
import torch
import yaml

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Load the configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    experiment = ExperimentConfig(**config)

    # Setting up the logger
    logger = Logger.create(
        experiment.log_type,
        hparams=experiment.model_dump(),
        resume_iter=experiment.run.resume_iter,
        **experiment.log.model_dump(),
    )

    # Defining the datasets for training and validation
    dataset = TrainingData(**experiment.data.model_dump())
    val_dataset = ValidationData(**experiment.validation_data.model_dump())

    # Defining the models
    nets, nets_ema = build_model(**experiment.model.model_dump())

    # Defining the Solver
    solver = Solver(nets, nets_ema, **experiment.solver.model_dump(), run=logger)

    # Training the model
    solver.train(
        dataset,
        **experiment.run.model_dump(),
        **experiment.loss.model_dump(),
        val_loader=val_dataset,
        val_config=experiment.validation_config,
    )
