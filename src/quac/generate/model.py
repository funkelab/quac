"""Reduces the model into just want is needed for inference."""

from os.path import join as ospj
from quac.training.stargan import (
    Generator,
    MappingNetwork,
    StyleEncoder,
    SingleOutputStyleEncoder,
)
from quac.training.checkpoint import CheckpointIO
import torch


class InferenceModel(torch.nn.Module):
    """A superclass for inference models.

    Useful for type-checking.
    """

    # TODO add checkpoint loading to this class
    def __init__(self) -> None:
        super().__init__()
        pass


class LatentInferenceModel(InferenceModel):
    def __init__(
        self,
        checkpoint_dir,
        img_size,
        style_dim,
        latent_dim,
        input_dim=1,
        num_domains=6,
        final_activation=None,
    ) -> None:
        super().__init__()
        generator = Generator(
            img_size, style_dim, input_dim=input_dim, final_activation=final_activation
        )
        mapping_network = MappingNetwork(latent_dim, style_dim, num_domains=num_domains)

        self.nets = torch.nn.ModuleDict(
            {
                "generator": generator,
                "mapping_network": mapping_network,
            }
        )

        self.checkpoint_io = CheckpointIO(
            ospj(checkpoint_dir, "{:06d}_nets_ema.ckpt"),
            data_parallel=False,
            **self.nets,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets.to(self.device)
        self.latent_dim = latent_dim
        self.style_dim = style_dim

    def load_checkpoint(self, step):
        self.checkpoint_io.load(step)

    def forward(self, x_src, y_trg):
        z = torch.randn(x_src.size(0), self.latent_dim).to(self.device)
        s = self.nets.mapping_network(z, y_trg)
        x_fake = self.nets.generator(x_src, s)
        return x_fake


class ReferenceInferenceModel(InferenceModel):
    def __init__(
        self,
        checkpoint_dir,
        img_size,
        style_dim=64,
        latent_dim=16,
        input_dim=1,
        num_domains=6,
        single_output_encoder=False,
        final_activation=None,
    ) -> None:
        super().__init__()
        generator = Generator(
            img_size, style_dim, input_dim=input_dim, final_activation=final_activation
        )
        if single_output_encoder:
            style_encoder: StyleEncoder = SingleOutputStyleEncoder(
                img_size, style_dim, num_domains, input_dim=input_dim
            )
        else:
            style_encoder = StyleEncoder(
                img_size, style_dim, num_domains, input_dim=input_dim
            )

        self.nets = torch.nn.ModuleDict(
            {"generator": generator, "style_encoder": style_encoder}
        )

        self.checkpoint_io = CheckpointIO(
            ospj(checkpoint_dir, "{:06d}_nets_ema.ckpt"),
            data_parallel=False,
            **self.nets,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets.to(self.device)
        self.latent_dim = latent_dim
        self.style_dim = style_dim

    def load_checkpoint(self, step):
        self.checkpoint_io.load(step)

    def forward(self, x_src, x_ref, y_trg):
        s = self.nets.style_encoder(x_ref, y_trg)
        x_fake = self.nets.generator(x_src, s)
        return x_fake
