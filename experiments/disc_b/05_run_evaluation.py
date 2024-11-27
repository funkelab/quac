# Defining processors and evaluators
from quac.evaluation import Processor, Evaluator
from sklearn.metrics import ConfusionMatrixDisplay
from quac.training.config import ExperimentConfig
from funlib.learn.torch.models import Vgg2D
from torchvision import transforms
from pathlib import Path
import torch
import yaml


def main(config_file: str = "configs/stargan.yml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    experiment = ExperimentConfig(**config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = "disc_b"

    input_checkpoint = (
        f"/nrs/funke/adjavond/projects/duplex/{experiment_name}/vgg_checkpoint"
    )
    classifier = Vgg2D(
        input_size=(128, 128),
        input_fmaps=1,
        fmaps=12,
        output_classes=3,
    )
    classifier.load_state_dict(torch.load(input_checkpoint))
    classifier = classifier.to(device)
    classifier.eval()

    reports_dir = Path(f"/nrs/funke/adjavond/projects/duplex/{experiment_name}/reports")
    data_directory = Path(experiment.test_data.source)
    cf_directory = data_directory.parent / "generated/test"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize(128),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    for attribution in ["deeplift", "ig"]:
        attribution_dir = (
            Path(f"/nrs/funke/adjavond/projects/duplex/{experiment_name}/attributions")
            / attribution
        )
        evaluator = Evaluator(
            classifier,
            source_directory=data_directory,
            counterfactual_directory=cf_directory,
            attribution_directory=attribution_dir,
            transform=transform,
        )

        # Run QuAC evaluation on your attribution and store a report
        report = evaluator.quantify(processor=Processor())
        # The report will be stored based on the processor's name, which is "default" by default
        report.store(reports_dir / f"{attribution}")


if __name__ == "__main__":
    main()
