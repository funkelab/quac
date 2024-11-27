# %%
from pathlib import Path
import torch
import yaml
from quac.training.config import ExperimentConfig
from funlib.learn.torch.models import Vgg2D
from quac.evaluation import Processor

# %%
config_file = "configs/stargan.yml"
with open(config_file, "r") as f:
    config = yaml.safe_load(f)
experiment = ExperimentConfig(**config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attribution_dir = Path("attributions")
reports_dir = Path("reports")
data_directory = Path(experiment.test_data.source)
cf_directory = data_directory.parent / "generated/test"
#  %%

# %%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from quac.evaluation import Processor


processor = Processor()


def plot_witness(image_path, attribution_method="deeplift", threshold=0.5):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    og = Image.open(image_path)
    cf = Image.open(cf_directory / "0/1" / image_path.name)

    attr_dir = attribution_dir / attribution_method
    attr = np.load(attr_dir / "0/1" / image_path.name.replace(".png", ".npy"))
    mask = processor.create_mask(attr, threshold, return_size=False)
    # Plot
    ax1.imshow(og)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(cf)
    ax2.set_title("Counterfactual")
    ax2.axis("off")
    ax3.imshow(mask.squeeze(), cmap="gray")
    ax3.set_title("Attribution")
    ax3.axis("off")
    plt.show()


# %%
for image_path in list((data_directory / "0").iterdir())[10:20]:
    plot_witness(image_path, "deeplift", threshold=0.1)

# %%
