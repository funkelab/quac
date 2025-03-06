from argparse import ArgumentParser
from pathlib import Path
from quac.training.config import ExperimentConfig
from quac.generate import load_classifier
from quac.generate import load_data
from quac.generate import get_counterfactual
from quac.generate import load_stargan
from torchvision.utils import save_image
from tqdm import tqdm
import torch
import warnings
import yaml


# Computed setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--source_class",
        type=str,
        required=True,
        help="Name of the source class to take images from.",
    )
    parser.add_argument(
        "--target_class",
        type=str,
        required=True,
        help="Name of the target class to convert images to.",
    )
    parser.add_argument(
        "--checkpoint_iter",
        type=int,
        default=50000,
        help="Iteration of the checkpoint to load.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="test",
        help="""
        Which dataset to use. 
        If not provided, we will check if test exists, else use validation.
        Options are ["train", "validation", "test"]
        """,
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory to save the images.",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="latent",
        help="Method for obtaining the style.",
        choices=["latent", "reference"],
    )
    return parser.parse_args()


def get_data_config(experiment, dataset="test"):
    dataset = dataset or "test"
    if dataset == "train":
        return experiment.data
    elif dataset == "test":
        if experiment.test_data:
            return experiment.test_data
        warnings.warn("No test data found, using validation data.")
    return experiment.validation_data


def get_target_index(source_directory, target_class):
    # Figure out which "domain" the target class is, by how the directories are structured
    source_directory = Path(source_directory)
    subdirs = [d for d in source_directory.iterdir() if d.is_dir()]
    subdirs = sorted(subdirs)
    try:
        target_index = subdirs.index(source_directory / target_class)
    except ValueError:
        raise ValueError(f"Target class {target_class} not found in {source_directory}")
    return target_index


if __name__ == "__main__":
    args = parse_args()
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    experiment = ExperimentConfig(**config)

    data_config = get_data_config(experiment, args.dataset)

    dataset = load_data(
        data_directory=Path(data_config.source) / args.source_class,
        img_size=data_config.img_size,
        grayscale=data_config.grayscale,
    )

    reference_dataset = None
    if args.kind == "reference":
        reference_dataset = load_data(
            data_directory=Path(data_config.reference) / args.target_class,
            img_size=data_config.img_size,
            grayscale=data_config.grayscale,
        )

    target_index = get_target_index(data_config.source, args.target_class)

    if args.output:
        output_directory = Path(args.output)
    else:
        output_directory = (
            Path(experiment.solver.root_dir) / "generated_images" / args.kind
        )
    # Add subdirectories for the source and target classes
    output_directory = (output_directory / args.source_class) / args.target_class
    output_directory.mkdir(parents=True, exist_ok=True)

    classifier_config = experiment.validation_config

    classifier = load_classifier(
        classifier_config.classifier_checkpoint,
        mean=classifier_config.mean,
        std=classifier_config.std,
        eval=True,
        device=device,
    )

    inference_model = load_stargan(
        checkpoint_dir=experiment.solver.root_dir,
        **experiment.model.model_dump(),
        checkpoint_iter=args.checkpoint_iter,
        kind=args.kind,
    )

    for x, name in tqdm(dataset):
        xcf = get_counterfactual(
            classifier,
            inference_model,
            x,
            target=target_index,
            kind=args.kind,
            dataset_ref=reference_dataset,
            device=device,
            max_tries=10,
            batch_size=10,
        )
        # Save the images here
        save_image(xcf, output_directory / name)
