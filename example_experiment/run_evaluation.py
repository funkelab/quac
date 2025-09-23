from argparse import ArgumentParser
from pathlib import Path
from quac.evaluation import Processor, Evaluator
from quac.generate import load_classifier
from quac.config import ExperimentConfig, get_data_config
from quac.data import create_transform
import yaml


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset to use for generating images.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="""
        Input directory for the generated images (conversions).
        Defaults to a `generated_images` folder in the experiment root directory, 
        based on the config file.""",
    )
    parser.add_argument(
        "-a",
        "--attrs",
        type=str,
        default=None,
        help="""
        Input directory for the attributions.
        Defaults to an `attributions` folder in the experiment root directory, 
        based on the config file.""",
    )
    parser.add_argument(
        "-n",
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="""
        Names of the attributions to evaluate.
        Defaults to all directories in the attribution directory.""",
    )
    parser.add_argument(
        "-k",
        "--kind",
        type=str,
        default="latent",
        choices=["latent", "reference"],
        help="""
        Kind of generated images to use for attribution.
        `latent` for images generated from the latent space,
        `reference` for images generated from the reference space.
        """,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Load the configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    experiment = ExperimentConfig(**config)

    data_config = get_data_config(experiment, args.dataset)
    classifier_config = experiment.validation_config

    data_directory = data_config.source
    attribution_directory = Path(
        args.attrs or f"{experiment.solver.root_dir}/attributions"
    )
    generated_directory = Path(
        args.input or f"{experiment.solver.root_dir}/generated_images/{args.kind}"
    )
    counterfactual_directory = Path(f"{experiment.solver.root_dir}/counterfactuals")
    mask_directory = Path(f"{experiment.solver.root_dir}/masks")
    report_directory = Path(f"{experiment.solver.root_dir}/reports")

    # Load the classifier
    classifier = load_classifier(
        checkpoint=classifier_config.classifier_checkpoint,
        scale=classifier_config.scale,
        shift=classifier_config.shift,
    )

    # Defining how to transform images for the classifier
    transform = create_transform(
        img_size=data_config.img_size,
        grayscale=data_config.grayscale,
        rgb=data_config.rgb,
        scale=data_config.scale,
        shift=data_config.shift,
    )

    attribution_directory = Path(attribution_directory)
    attribution_names = args.names or [
        d for d in attribution_directory.iterdir() if d.is_dir()
    ]

    for name in attribution_names:
        print(f"Evaluating attributions from {name}")
        # Defining the evaluator
        evaluator = Evaluator(
            classifier,
            source_directory=data_directory,
            generated_directory=generated_directory,
            attribution_directory=attribution_directory / name,
            counterfactual_directory=counterfactual_directory / name,
            mask_directory=mask_directory / name,
            transform=transform,
        )

        # Run QuAC evaluation on your attribution and store a report
        report = evaluator.quantify(processor=Processor())
        # The report will be stored based on the processor's name, which is "default" by default
        report.store(report_directory / name)
