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
    experiment_name = "horses_zebras"

    input_checkpoint = (
        f"/nrs/funke/adjavond/projects/duplex/{experiment_name}/vgg_checkpoint"
    )
    classifier = Vgg2D(
        input_size=(256, 256),
        input_fmaps=3,
        fmaps=12,
        output_classes=2,
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
        output_directory=f"/nrs/funke/adjavond/projects/duplex/{experiment_name}/attributions",
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # TODO TO RGB
            transforms.Resize(256),
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
