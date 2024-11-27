# Load the classifier
from quac.training.config import ExperimentConfig
from funlib.learn.torch.models import Vgg2D
from quac.attribution import DDeepLift, DIntegratedGradients, AttributionIO
from torchvision import transforms
from pathlib import Path
import torch
import yaml


def main(config_file: str = "configs/stargan.yml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    experiment = ExperimentConfig(**config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_checkpoint = "/nrs/funke/adjavond/projects/duplex/disc_b/vgg_checkpoint"
    classifier = Vgg2D(
        input_size=(128, 128),
        input_fmaps=1,
        fmaps=12,
        output_classes=3,
    )
    classifier.load_state_dict(torch.load(input_checkpoint))
    classifier = classifier.to(device)
    classifier.eval()

    data_directory = Path(experiment.test_data.source)
    cf_directory = data_directory.parent / "generated/test"

    # Defining attributions

    attributor = AttributionIO(
        attributions={
            "ig": DIntegratedGradients(classifier),
            "deeplift": DDeepLift(classifier),
        },
        output_directory="/nrs/funke/adjavond/projects/duplex/disc_b/attributions",
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize(128),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    # This will run attributions and store all of the results in the output_directory
    # Shows a progress bar
    attributor.run(
        source_directory=data_directory,
        counterfactual_directory=cf_directory,
        transform=transform,
    )


if __name__ == "__main__":
    main()
