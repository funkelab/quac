"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
from pathlib import Path
from munch import Munch
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


from quac.training.checkpoint import CheckpointIO
import quac.training.utils as utils
from quac.training.eval import calculate_metrics

import wandb


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
)


class Solver(nn.Module):
    def __init__(
        self,
        nets,
        nets_ema,
        f_lr: float,
        lr: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        root_dir: str,
        run=None,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets = nets
        self.nets_ema = nets_ema
        self.run = run
        self.root_dir = root_dir
        self.checkpoint_dir = Path(root_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_dir = str(self.checkpoint_dir)

        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + "_ema", module)

        self.optims = Munch()
        for net in self.nets.keys():
            self.optims[net] = torch.optim.Adam(
                params=self.nets[net].parameters(),
                lr=f_lr if net == "mapping_network" else lr,
                betas=[beta1, beta2],
                weight_decay=weight_decay,
            )

            self.ckptios = [
                CheckpointIO(
                    ospj(checkpoint_dir, "{:06d}_nets.ckpt"),
                    data_parallel=True,
                    **self.nets,
                ),
                CheckpointIO(
                    ospj(checkpoint_dir, "{:06d}_nets_ema.ckpt"),
                    data_parallel=True,
                    **self.nets_ema,
                ),
                CheckpointIO(ospj(checkpoint_dir, "{:06d}_optims.ckpt"), **self.optims),
            ]
        else:
            self.ckptios = [
                CheckpointIO(
                    ospj(checkpoint_dir, "{:06d}_nets_ema.ckpt"),
                    data_parallel=True,
                    **self.nets_ema,
                )
            ]

        self.to(self.device)
        # TODO The EMA doesn't need to be in named_childeren()
        for name, network in self.named_children():
            if "ema" not in name:
                print("Initializing %s..." % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(
        self,
        loader,
        resume_iter: int = 0,
        total_iters: int = 100000,
        log_every: int = 100,
        save_every: int = 10000,
        eval_every: int = 10000,
        # sample_dir: str = "samples",
        lambda_ds: float = 1.0,
        ds_iter: int = 10000,
        lambda_reg: float = 1.0,
        lambda_sty: float = 1.0,
        lambda_cyc: float = 1.0,
    ):
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = loader.train_fetcher

        # resume training if necessary
        if resume_iter > 0:
            self._load_checkpoint(resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = lambda_ds

        print("Start training...")
        for i in range(resume_iter, total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, x_aug, y_org = inputs.x_src, inputs.x_src2, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, x_real, y_org, y_trg, z_trg=z_trg, lambda_reg=lambda_reg
            )
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, x_real, y_org, y_trg, x_ref=x_ref, lambda_reg=lambda_reg
            )
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent, fake_x_latent = compute_g_loss(
                nets,
                x_real,
                y_org,
                y_trg,
                z_trgs=[z_trg, z_trg2],
                lambda_sty=lambda_sty,
                lambda_ds=lambda_ds,
                lambda_cyc=lambda_cyc,
            )
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref, fake_x_reference = compute_g_loss(
                nets,
                x_real,
                y_org,
                y_trg,
                x_refs=[x_ref, x_ref2],
                x_aug=x_aug,
                lambda_sty=lambda_sty,
                lambda_ds=lambda_ds,
                lambda_cyc=lambda_cyc,
            )
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            nets_ema.update(nets)

            # decay weight for diversity sensitive loss
            if lambda_ds > 0:
                lambda_ds -= initial_lambda_ds / ds_iter

            if (i + 1) % eval_every == 0:
                self.evaluate(loader, iteration=i + 1, mode="latent", val_config=None)
                self.evaluate(
                    loader, iteration=i + 1, mode="reference", val_config=None
                )

            # save model checkpoints
            if (i + 1) % save_every == 0:
                self._save_checkpoint(step=i + 1)

            # print out log losses, images
            if (i + 1) % log_every == 0:
                self.log(
                    d_losses_latent,
                    d_losses_ref,
                    g_losses_latent,
                    g_losses_ref,
                    lambda_ds,
                    x_real,
                    x_ref,
                    fake_x_latent,
                    fake_x_reference,
                    step=i + 1,
                )

    def log(
        self,
        d_losses_latent,
        d_losses_ref,
        g_losses_latent,
        g_losses_ref,
        lambda_ds,
        x_real,
        x_ref,
        fake_x_latent,
        fake_x_reference,
        step,
    ):
        all_losses = dict()
        for loss, prefix in zip(
            [d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
            ["D/latent_", "D/ref_", "G/latent_", "G/ref_"],
        ):
            for key, value in loss.items():
                all_losses[prefix + key] = value
        all_losses["G/lambda_ds"] = lambda_ds
        # log all losses to wandb or print them
        if self.run:
            self.run.log(all_losses, step=step)
            for name, img in zip(
                ["x_real", "x_ref", "fake_x_latent", "fake_x_reference"],
                [x_real, x_ref, fake_x_latent, fake_x_reference],
            ):
                self.run.log({name: [wandb.Image(img)]}, step=step)
        else:
            print(all_losses)
            # TODO store images

    def generate_images(self, val_loader, eval_dir, num_outs_per_domain, mode="latent"):
        """
        Generates images for evaluation and stores them to disk.

        Parameters
        ----------
        val_loader
        """
        assert mode in ["latent", "reference"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        val_loader.set_mode(mode)

        domains = val_loader.available_targets
        print("Number of domains: %d" % len(domains))

        for trg_idx, trg_domain in enumerate(domains):
            src_domains = [x for x in val_loader.available_sources if x != trg_domain]
            val_loader.set_target(trg_domain)

            for src_idx, src_domain in enumerate(src_domains):
                task = "%s_to_%s" % (src_domain, trg_domain)
                # Creating the path
                path_fake = os.path.join(eval_dir, task)
                shutil.rmtree(path_fake, ignore_errors=True)
                os.makedirs(path_fake)

                # Setting the source domain
                val_loader.set_source(src_domain)
                loader_src = val_loader.loader_src

                for i, x_src in enumerate(tqdm(loader_src, total=len(loader_src))):
                    N = x_src.size(0)
                    x_src = x_src.to(device)
                    y_trg = torch.tensor([trg_idx] * N).to(device)

                    # generate num_outs_per_domain outputs from the same input
                    for j in range(num_outs_per_domain):
                        if mode == "latent":
                            latent_dim = self.nets_ema.mapping_network.latent_dim
                            z_trg = torch.randn(N, latent_dim).to(device)
                            s_trg = self.nets_ema.mapping_network(z_trg, y_trg)
                        else:
                            try:
                                x_ref = next(iter_ref).to(device)
                            except:
                                iter_ref = iter(loader_ref)
                                x_ref = next(iter_ref).to(device)

                            if x_ref.size(0) > N:
                                x_ref = x_ref[:N]
                            s_trg = self.nets_ema.style_encoder(x_ref, y_trg)

                        x_fake = self.nets_ema.generator(x_src, s_trg)

                        for k in range(N):
                            # Get the index of the image as:
                            # i * batch_size * spot_in_batch
                            # Append to it the index of this output for the given batch
                            filename = os.path.join(
                                path_fake,
                                "%.4i_%.2i.png" % (i * N + (k + 1), j + 1),
                            )
                            utils.save_image(x_fake[k], ncol=1, filename=filename)
                del loader_src
                if mode == "reference":
                    del loader_ref
                    del iter_ref

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        iteration=None,
        num_outs_per_domain=4,
        mode="latent",
        val_config=None,
    ):
        if iteration is None:  # Choose the iteration to evaluate
            resume_iter = resume_iter
            self._load_checkpoint(resume_iter)
        # Generate images for evaluation
        eval_dir = self.root_dir / "eval"
        eval_dir.mkdir(exist_ok=True, parents=True)  # TODO add iteration?
        self.generate_images(loader, eval_dir, num_outs_per_domain, mode=mode)
        # Calculate metrics
        # TODO maybe launch a subprocess for evaluation?
        calculate_metrics(eval_dir, step=iteration, mode=mode, **val_config)


def compute_d_loss(
    nets, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, lambda_reg=1.0
):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + lambda_reg * loss_reg
    return loss, Munch(
        real=loss_real.item(), fake=loss_fake.item(), reg=loss_reg.item()
    )


def compute_g_loss(
    nets,
    x_real,
    y_org,
    y_trg,
    z_trgs=None,
    x_refs=None,
    x_aug=None,
    lambda_sty: float = 1.0,
    lambda_ds: float = 1.0,
    lambda_cyc: float = 1.0,
):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    # Adds random augmentation to x_fake before passing to style encoder
    s_pred = nets.style_encoder(transform(x_fake), y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # style invariance loss
    if x_aug is not None:
        s_pred2 = nets.style_encoder(x_aug, y_org)
        loss_sty2 = torch.mean(torch.abs(s_pred2 - s_org))
        loss_sty = (loss_sty + loss_sty2) / 2

    loss = (
        loss_adv + lambda_sty * loss_sty - lambda_ds * loss_ds + lambda_cyc * loss_cyc
    )
    return (
        loss,
        Munch(
            adv=loss_adv.item(),
            sty=loss_sty.item(),
            ds=loss_ds.item(),
            cyc=loss_cyc.item(),
        ),
        x_fake,
    )


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg