# %%
from quac.training.data_loader import TrainingData, AugmentedInputFetcher
from quac.training.config import ExperimentConfig
import yaml

root = "/nrs/funke/adjavond/data/duplex/horses_zebras/train"
with open("configs/stargan.yml", "r") as f:
    config = yaml.safe_load(f)
experiment = ExperimentConfig(**config)
# %%
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random

prob = 0.0
img_size = 256
height = 256
width = 256
crop = transforms.RandomResizedCrop(img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < prob else x)

transform_list = [rand_crop]
transform = transforms.Compose(
    [
        *transform_list,
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)
dataset = ImageFolder(root, transform)
for i, (img, _) in tqdm(enumerate(dataset), total=len(dataset)):
    assert img.shape[0] == 3
    assert img.shape[1] == height
    assert img.shape[2] == width
# %% Change the number of workers
experiment.data.num_workers = 1
# %%
loader = TrainingData(**experiment.data.model_dump())

# %%
latent_dim = 16
fetcher = AugmentedInputFetcher(
    loader.src,
    loader.reference,
    latent_dim=latent_dim,
    mode="train",
)

# %%
from tqdm import tqdm

for i in tqdm(range(100)):
    inputs = next(fetcher)


# %%
import matplotlib.pyplot as plt

plt.imshow(inputs.x_src[0].cpu().numpy().transpose(1, 2, 0))
# %%

# %%
