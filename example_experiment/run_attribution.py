from argparse import ArgumentParser
import torch
from quac.config import ExperimentConfig, get_data_config
from quac.generate import load_classifier
from quac.attribution import DIntegratedGradients, DDeepLift, AttributionIO
from quac.data import create_transform
import yaml


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset to use for generating images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="""
        Output directory for the generated attributions. 
        Defaults to an `attributions` folder in the experiment root directory, 
        based on the config file.""",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="""
        Directory holding the generated images (converted).
        Defaults to the `generated_images` directory in the experiment root directory,
        as defined in the config file.
        """,
    )
    parser.add_argument(
        "-k",
        "--kind",
        type=str,
        choices=["latent", "reference"],
        default="latent",
        help="""
        Kind of image generation that was done. Used to find the generated images. 
        """,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    experiment = ExperimentConfig(**config)

    data_config = get_data_config(experiment, args.dataset)
    classifier_config = experiment.validation_config

    data_directory = data_config.source
    attribution_directory = args.output or f"{experiment.solver.root_dir}/attributions"
    generated_directory = (
        args.input or f"{experiment.solver.root_dir}/generated_images/{args.kind}"
    )

    # Load the classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = load_classifier(
        checkpoint=classifier_config.classifier_checkpoint,
        scale=classifier_config.scale,
        shift=classifier_config.shift,
    )
    classifier.to(device)
    classifier.eval()

    # Defining attributions
    # TODO Offer a way to select which attributions to run
    attributor = AttributionIO(
        attributions={
            "discriminative_ig": DIntegratedGradients(classifier),
            "discriminative_deeplift": DDeepLift(classifier),
        },
        output_directory=attribution_directory,
    )

    transform = create_transform(
        img_size=data_config.img_size,
        grayscale=data_config.grayscale,
        rgb=data_config.rgb,
        scale=data_config.scale,
        shift=data_config.shift,
    )

    # This will run attributions and store all of the results in the output_directory
    # Shows a progress bar
    attributor.run(
        source_directory=data_directory,
        generated_directory=generated_directory,
        transform=transform,
    )
