from argparse import ArgumentParser
from quac.training.config import ExperimentConfig
from quac.generate import load_classifier
from quac.attribution import (
    DIntegratedGradients,
    DDeepLift,
    AttributionIO
)
from torchvision import transforms
import warnings
import yaml


def get_data_config(experiment, dataset="test"):
    # TODO this is duplicated with generate_images.py 
    # and should be moved to a common place
    dataset = dataset or "test"
    if dataset == "train":
        return experiment.data
    elif dataset == "test":
        if experiment.test_data:
            return experiment.test_data
        warnings.warn("No test data found, using validation data.")
    return experiment.validation_data


def create_transform(img_size, mean, std, grayscale):
    # TODO, this copies code from quac.generate (__init__.py)
    # and should be moved to a common utilities file
    transform=transforms.Compose(
        [
            transforms.Resize([img_size, img_size]),
            transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    ),
    return transform 

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
        based on the config file."""
    )
    parser.add_argument(
        "-i",
        "--input_fake",
        type=str,
        default=None,
        help="""
        Directory holding the generated images (converted).
        Defaults to the `generated_images` directory in the experiment root directory,
        as defined in the config file.
        """
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
    counterfactual_directory = args.input or f"{experiment.solver.root_dir}/generated_images"

    # Load the classifer
    classifier = load_classifier(
        checkpoint_path=classifier_config.classifier_checkpoint
    )

    # Defining attributions
    # TODO Offer a way to select which attributions to run
    attributor = AttributionIO(
        attributions = {
            "discriminative_ig" : DIntegratedGradients(classifier),
            "discriminative_deeplift": DDeepLift(classifier), 
        },
        output_directory = attribution_directory
    )

    transform = create_transform(
        img_size=data_config.img_size,
        mean=classifier_config.mean,
        std=classifier_config.std,
        grayscale=data_config.grayscale
    )

    # This will run attributions and store all of the results in the output_directory
    # Shows a progress bar
    attributor.run(
        source_directory=data_directory,
        counterfactual_directory=counterfactual_directory,
        transform=transform
    )