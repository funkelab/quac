# Generating images for DupLEX
import numpy as np
from pathlib import Path
from quac.training.config import ExperimentConfig
from quac.training.data_loader import RGB
from quac.generate.model import LatentInferenceModel
import torch
import typer
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import yaml

torch.backends.cudnn.benchmark = True


def main(
    config_file: str = "configs/stargan.yml",
    resume_iter: int = 100000,
    split: str = "train",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    experiment = ExperimentConfig(**config)

    data_dir = Path(experiment.data.source).parent / split
    checkpoint_dir = Path(experiment.solver.root_dir) / "checkpoints"

    model = LatentInferenceModel(
        checkpoint_dir,
        **experiment.model.model_dump(),
    )

    transform = transforms.Compose(
        [
            transforms.Resize((experiment.data.img_size, experiment.data.img_size)),
            RGB(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = ImageFolder(
        data_dir,
        transform=transform,
    )

    model.load_checkpoint(resume_iter)
    model.to(device)
    model.eval()

    output_dir = Path(experiment.data.source).parent / f"generated/{split}"
    print("Output directory:", output_dir)

    classes = dataset.classes

    for source_class in classes:
        source_indices = np.where(
            np.array(dataset.targets) == classes.index(source_class)
        )[0]
        source_dataset = Subset(dataset, source_indices)
        print(f"There are {len(source_dataset)} images for {source_class}")
        dl = DataLoader(source_dataset, batch_size=experiment.data.batch_size)
        for target_class in classes:
            if target_class == source_class:
                continue
            print(f"Generating images for {source_class} -> {target_class}")
            target_dir = output_dir / source_class / target_class
            target_dir.mkdir(parents=True, exist_ok=True)
            for i, (x, _) in enumerate(dl):
                y = torch.full(
                    (x.size(0),), classes.index(target_class), dtype=torch.long
                )
                x = x.to(device)
                x_fake = model(x, y)
                for j in range(x_fake.size(0)):
                    save_index = i * experiment.data.batch_size + j
                    source_index = source_indices[save_index]
                    image_name = (
                        dataset.imgs[source_index][0].split("/")[-1].split(".")[0]
                    )
                    save_path = target_dir / f"{image_name}.png"
                    # Images are -1, 1. We need to convert them to 0, 1
                    x_fake[j] = (x_fake[j] + 1) / 2
                    save_image(x_fake[j], save_path)


if __name__ == "__main__":
    typer.run(main)
