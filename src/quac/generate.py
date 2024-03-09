"""Utilities for generating counterfacual images."""
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from starganv2.inference.model import LatentInferenceModel
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StarGANInferenceModel:
    def __init__(
        self,
        save_dir: str,
        checkpoint: str,
        classifier: torch.nn.Module,
        name=None,
        batch_size=100,
        max_tries=10,
    ):
        self.save_dir = save_dir
        self.name = name
        self.batch_size = batch_size
        self.max_tries = max_tries
        self.classifier = classifier

        # TODO Choose device
        #
        self._load_stargan(checkpoint)
        self.latent_inference_model = None

    def _load_stargan(
        self,
        latent_model_checkpoint_dir: str,
        img_size: int = 128,
        style_dim: int = 64,
        latent_dim: int = 16,
        num_domains: int = 6,
        checkpoint_iter: int = 100000,
    ):
        # Inference model
        self.latent_inference_model = LatentInferenceModel(
            checkpoint_dir=latent_model_checkpoint_dir,
            img_size=img_size,
            style_dim=style_dim,
            latent_dim=latent_dim,
            num_domains=num_domains,
            w_hpf=0.0,
        )
        self.latent_inference_model.load_checkpoint(checkpoint_iter)
        self.latent_inference_model.eval()

    @torch.no_grad()
    def _get_counterfactual(
        self,
        x: np.ndarray,
        target: int,
        tries: int,
        best_pred_so_far=0,
        best_cf_so_far=None,
    ):
        """
        Tries to find a counterfactual for the given sample, given the target.
        It creates a batch, and returns one of the samples if it is classified correctly.

        Parameters:
        -----------
        x: the sample to find a counterfactual for
        target: the target class
        batch_size: the number of counterfactuals to generate
        device: the device to use
        max_tries: the maximum number of tries to find a counterfactual

        Returns
        --------
        The best performing counterfactual

        ::note:: This function is recursive, and will call itself if it does not find a counterfactual.
        """
        # Copy x batch_size times
        x_multiple = torch.stack([x] * self.batch_size)
        # Generate batch_size counterfactuals
        xcf = self.latent_inference_model(
            x_multiple.to(self.device),
            torch.tensor([target] * self.batch_size).to(self.device),
        )
        # Evaluate the counterfactuals
        p = torch.softmax(self.classifier(xcf), dim=-1)
        # Get the predictions
        predictions = torch.argmax(p, dim=-1)
        # Get best so far
        best_idx_so_far = torch.argmax(p[:, target])
        if p[best_idx_so_far, target] > best_pred_so_far:
            best_pred_so_far = p[best_idx_so_far, target]
            best_cf_so_far = xcf[best_idx_so_far].cpu().numpy()
        # TODO Just return best overall counterfactual
        # Get the indices of the correct predictions
        indices = torch.where(predictions == target)[0]
        if len(indices) == 0:
            if tries > 0:
                logger.info(
                    f"Counterfactual not found, trying again. {tries} tries left."
                )
                return self.get_counterfactual(
                    x,
                    target,
                    tries - 1,
                    best_pred_so_far=best_pred_so_far,
                    best_cf_so_far=best_cf_so_far,
                )
            else:
                logger.info(
                    f"Counterfactual not found after {tries} tries, using best so far."
                )
                return best_cf_so_far
        # Choose the best of the correct predictions
        index = np.argmax(p[indices].cpu().numpy()[:, target])
        # Get the counterfactual
        xcf = xcf[index].cpu().numpy()
        # logger.info(f"Number of counterfactuals: {len(indices)}")
        return xcf

    def make_counterfactual(self, x, y, y_t):
        """Generate a counterfactual image.

        Parameters
        ----------
        x : np.ndarray
            The input image.
        y : int
            The input label.
        y_t : int
            The target label.

        Returns
        -------
        np.ndarray
            The counterfactual image. The model makes `batch_size*max_tries` attempts to generate a counterfactual image.
            It chooses the best among these.
        """
        # TODO prepare x to be used with the model
        xcf = self._get_counterfactual(x, y_t, self.max_tries)
        # TODO check normalization

    def store_counterfactual(self, counterfactual, filename, source=None, target=None):
        """Store the counterfactual image to disk.

        The image is stored in the directory `save_dir` in subdirectory `name`.
        If `source` and `target` are given, the image is stored in subdirectories `source` and `target` respectively.

        So, for example:
        ```
        save_dir/name/source/target/filename.png
        ```

        Otherwise it is simply stored in the directory `name`:
        ```
        save_dir/name/filename.png
        ```
        ::warning:: If there are more than two classes, the `source` and `target` parameters are required, otherwise the images will be overwritten.

        Parameters
        ----------
        counterfactual : np.ndarray
            The counterfactual image.
        filename : str
            The name of the file to store, without the extension.
        source : str
            The source class label.
        target : str
            The target class label.
        """
        # TODO Check normalization
        if source and target:
            plt.imsave(
                os.path.join(
                    self.save_dir, self.name, source, target, filename + ".png"
                ),
                counterfactual,
            )
        else:
            plt.imsave(
                os.path.join(self.save_dir, self.name, filename + ".png"),
                counterfactual,
            )
