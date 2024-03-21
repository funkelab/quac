# How to generate images from a pre-trained network

## Defining the dataset

We will be generating images one source-target pair at a time.
As such, we need to point to the subdirectory that holds the source class that we are interested in.
For example, below, we are going to be using the validation data, and our source class will be class `0` which has no Diabetic Retinopathy.

```python
from pathlib import Path
from quac.generate import load_data

img_size = 224
data_directory = Path("root_directory/val/0_No_DR")
dataset = load_data(data_directory, img_size, grayscale=False)
```
## Loading the classifier

Next we need to load the pre-trained classifier, and wrap it in the correct pre-processing step.
The classifier is expected to be saved as a `torchscript` checkpoint. This allows us to use it without having to redefine the python class from which it was generated.

We also have a wrapper around the classifier that re-normalizes images to the range that it expects. The assumption is that these images come from the StarGAN trained with `quac`, so the images will have values in `[-1, 1]`.
Here, our pre-trained classifier expects images with the ImageNet normalization, for example.

Finally, we need to define the device, and whether to put the classifier in `eval` mode.

```python
from quac.generate import load_classifier

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = load_classifier(classifier_checkpoint, mean=mean, std=std, eval=True, device=device)
```

## Inference from random latents

The StarGAN model used to generate images can have two sources for the style.
The first and simplest one is to use a random latent vector to create style.

### Loading the StarGAN

```python
from quac.generate import load_stargan

latent_model_checkpoint_dir = Path("/path/to/directory/holding/the/stargan/checkpoints")

inference_model = load_stargan(
    latent_model_checkpoint_dir,
    img_size=224,
    input_dim=1,
    style_dim=64,
    latent_dim=16,
    num_domains=5,
    checkpoint_iter=100000,
    kind = "latent"
)
```

### Running the image generation

```python
from quac.generate import get_counterfactual
from torchvision.utils import save_image

output_directory = Path("/path/to/output/latent/0_No_DR/1_Mild/")

for x, name in tqdm(dataset):
    xcf = get_counterfactual(
        classifier,
        inference_model,
        x,
        target=1,
        kind="latent",
        device=device,
        max_tries=10,
        batch_size=10
    )
    # For example, you can save the images here
    save_image(xcf, output_directory / name)
```

## Inference using a reference dataset

The alternative image generation method of a StarGAN is to use an image of the target class to generate the style using the `StyleEncoder`.
Although the structure is similar as above, there are a few key differences.


### Generating the reference dataset

The first thing we need to do is to get the reference images.

```python
reference_data_directory = Path("root_directory/val/1_Mild")
reference_dataset = load_data(reference_data_directory, img_size, grayscale=False)
```

### Loading the StarGAN
This time, we will be creating a `ReferenceInferenceModel`.

```python
inference_model = load_stargan(
    latent_model_checkpoint_dir,
    img_size=224,
    input_dim=1,
    style_dim=64,
    latent_dim=16,
    num_domains=5,
    checkpoint_iter=100000,
    kind = "reference"
)
```

### Running the image generation

Finally, we combine the two by changing the `kind` in our counterfactual generation, and giving it the reference dataset to use.

```python
from torchvision.utils import save_image

output_directory = Path("/path/to/output/reference/0_No_DR/1_Mild/")

for x, name in tqdm(dataset):
    xcf = get_counterfactual(
        classifier,
        inference_model,
        x,
        target=1,
        kind="reference",   # Change the kind of inference being done
        dataset_ref=reference_dataset,  # Add the reference dataset
        device=device,
        max_tries=10,
        batch_size=10
    )
    # For example, you can save the images here
    save_image(xcf, output_directory / name)
```
