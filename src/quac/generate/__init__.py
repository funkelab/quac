"""Utilities for generating counterfactual images."""

from .model import LatentInferenceModel, ReferenceInferenceModel, InferenceModel

import logging
from quac.training.classification import ClassifierWrapper
from quac.data import DefaultDataset
import torch
from torchvision import transforms
from typing import Union, Optional

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CounterfactualNotFound(Exception):
    pass


def load_classifier(checkpoint, scale=1.0, shift=0.0, eval=True, device=None):
    """
    Load a classifier from a torchscript checkpoint.

    This also creates a wrapper around the classifier, which applies a scale and shift to the input.

    Parameters
    ----------
    checkpoint: str
        The path to the checkpoint, which must be torchscript.
    scale: float
        The scale factor applied to the input before classification. Defaults to 1.
    shift: float
        The shift factor applied to the input after scaling. Defaults to 0.
    eval: bool
        Whether to put the classifier in evaluation mode, defaults to True
    device:
        The device to use, defaults to None, in which case the device will be chosen
        based on the model checkpoint, or can be changed later.
    """
    classifier = ClassifierWrapper(checkpoint, scale=scale, shift=shift)
    if device:
        classifier.to(device)
    if eval:
        classifier.eval()
    return classifier


def load_data(
    data_directory, img_size, grayscale=True, mean=0.5, std=0.5
) -> DefaultDataset:
    """
    Load a dataset from a directory.

    This assumes that the images are in a folder, with no subfolders, and no labels.
    The images are resized to `img_size`, and normalized with the given `mean` and `std`.
    If `grayscale` is True, the images are converted to grayscale.

    The returned dataset will return the image file name as the second element of the tuple.

    Parameters:
        data_directory: the directory to load the images from
        img_size: the size to resize the images to
        grayscale: whether to convert the images to grayscale, defaults to True
        mean: the mean to normalize the images, defaults to 0.5
        std: the standard deviation to normalize the images, defaults to 0.5
    """
    dataset = DefaultDataset(
        root=data_directory,
        transform=transforms.Compose(
            [
                transforms.Resize([img_size, img_size]),
                transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        ),
    )
    return dataset


def load_stargan(
    latent_model_checkpoint_dir: str,
    img_size: int = 128,
    input_dim: int = 1,
    style_dim: int = 64,
    latent_dim: int = 16,
    num_domains: int = 6,
    checkpoint_iter: int = 100000,
    kind="latent",
    single_output_encoder: bool = False,
    final_activation: Union[str, None] = None,
) -> InferenceModel:
    """
    Load an inference version of the StarGANv2 model from a checkpoint.

    Parameters:
        latent_model_checkpoint_dir: the directory of the checkpoint
        img_size: the size of the model input
        input_dim: the number of input channels
        style_dim: the dimension of the style
        latent_dim: the dimension of the latent space
        num_domains: the number of domains
        checkpoint_iter: the iteration of the checkpoint to load
        kind: the kind of style to use, either "latent" or "reference"
        single_output_encoder: whether to use a single output encoder, only used if kind is "reference"

    Returns:
        the loaded inference model
    """
    if kind == "reference":
        latent_inference_model: InferenceModel = ReferenceInferenceModel(
            checkpoint_dir=latent_model_checkpoint_dir,
            img_size=img_size,
            input_dim=input_dim,
            style_dim=style_dim,
            latent_dim=latent_dim,
            num_domains=num_domains,
            single_output_encoder=single_output_encoder,
            final_activation=final_activation,
        )
    else:
        latent_inference_model: InferenceModel = LatentInferenceModel(  # type: ignore[no-redef]
            checkpoint_dir=latent_model_checkpoint_dir,
            img_size=img_size,
            input_dim=input_dim,
            style_dim=style_dim,
            latent_dim=latent_dim,
            num_domains=num_domains,
            final_activation=final_activation,
        )
    latent_inference_model.load_checkpoint(checkpoint_iter)  # type: ignore
    latent_inference_model.eval()
    return latent_inference_model


@torch.no_grad()
def get_counterfactual(  # type: ignore
    classifier,
    latent_inference_model,
    x,
    target,
    kind="latent",  # or "reference"
    dataset_ref=None,
    batch_size=10,
    device=None,
    max_tries=100,
    best_pred_so_far: Optional[torch.Tensor] = None,
    best_cf_so_far: Optional[torch.Tensor] = None,
    best_cf_path_so_far: Optional[str] = None,
    error_if_not_found=False,
) -> tuple[Optional[torch.Tensor], Optional[str], Optional[torch.Tensor]]:
    """
    Tries to find a counterfactual for the given sample, given the target.
    It creates a batch, and returns one of the samples if it is classified correctly.

    Parameters:
        classifier: the classifier to use
        latent_inference_model: the latent inference model to use
        x: the sample to find a counterfactual for
        target: the target class
        kind: the kind of style to use, either "latent" or "reference"
        dataset_ref: the dataset of reference images to use, required if kind is "reference"
        batch_size: the number of counterfactuals to generate
        device: the device to use
        max_tries: the maximum number of tries to find a counterfactual
        error_if_not_found: whether to raise an error if no counterfactual is found, if set to False, the best counterfactual found so far is returned
        return_path: whether to return the path of the reference used to create best counterfactual found so far,
            only used if kind is "reference"

    Returns:
        a counterfactual

    Raises:
        CounterfactualNotFound: if no counterfactual is found after max_tries tries
    """
    if best_pred_so_far is None:
        best_pred_so_far = torch.zeros(target + 1)
    # Copy x batch_size times
    x_multiple = torch.stack([x] * batch_size)
    if kind == "reference":
        assert dataset_ref is not None, (
            "Reference dataset required for reference style."
        )
        if len(dataset_ref) // batch_size < max_tries:
            max_tries = len(dataset_ref) // batch_size
            logger.warning(
                f"Not enough reference images, reducing max_tries to {max_tries}."
            )
        # Get a batch of reference images, starting from batch_size * max_tries, of size batch_size
        ref_batch_tuples, ref_paths = zip(
            *[
                dataset_ref[i]
                for i in range(batch_size * (max_tries - 1), batch_size * max_tries)
            ]
        )
        ref_batch = torch.stack(ref_batch_tuples)
        # Generate batch_size counterfactuals
        xcf = latent_inference_model(
            x_multiple.to(device),
            ref_batch.to(device),
            torch.tensor([target] * batch_size).to(device),
        )
    else:  # kind == "latent"
        # Generate batch_size counterfactuals from random latents
        xcf = latent_inference_model(
            x_multiple.to(device),
            torch.tensor([target] * batch_size).to(device),
        )

    # Evaluate the counterfactuals
    p = torch.softmax(classifier(xcf), dim=-1)
    # Get the predictions
    predictions = torch.argmax(p, dim=-1)
    # Get best so far
    best_idx_so_far = torch.argmax(p[:, target])

    if p[best_idx_so_far, target] > best_pred_so_far[target]:
        best_pred_so_far = p[best_idx_so_far]  # , target]
        best_cf_so_far = xcf[best_idx_so_far].cpu()
        if kind == "reference":
            best_cf_path_so_far = ref_paths[best_idx_so_far]
        else:
            best_cf_path_so_far = None
    # Get the indices of the correct predictions
    indices = torch.where(predictions == target)[0]

    if len(indices) == 0:
        if max_tries > 0:
            logger.info(
                f"Counterfactual not found, trying again. {max_tries} tries left."
            )
            return get_counterfactual(
                classifier,
                latent_inference_model,
                x,
                target,
                kind,
                dataset_ref,
                batch_size,
                device,
                max_tries - 1,
                best_pred_so_far=best_pred_so_far,
                best_cf_so_far=best_cf_so_far,
                best_cf_path_so_far=best_cf_path_so_far,
            )
        else:
            if error_if_not_found:
                raise CounterfactualNotFound(
                    "Counterfactual not found after max_tries tries."
                )
            logger.info(
                f"Counterfactual not found after {max_tries} tries, using best so far."
            )
    # Return the best counterfactual so far
    return best_cf_so_far, best_cf_path_so_far, best_pred_so_far
