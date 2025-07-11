"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import datetime
import json
import numpy as np
import os
from os.path import join as ospj
from pathlib import Path
from quac.training.checkpoint import CheckpointIO
from quac.training.classification import ClassifierWrapper
from quac.training.data_loader import TrainingData
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
)


def save_json(json_file, filename):
    with open(filename, "w") as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


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
        self.root_dir = Path(root_dir)
        self.checkpoint_dir = self.root_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_dir = str(self.checkpoint_dir)

        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + "_ema", module)

        self.optims = dict()
        for name, net in self.nets.named_children():
            self.optims[name] = torch.optim.Adam(
                params=net.parameters(),
                lr=f_lr if name == "mapping_network" else lr,
                betas=(beta1, beta2),
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
        # TODO The EMA doesn't need to be in named_children()
        for name, network in self.named_children():
            if "ema" not in name:
                print("Initializing %s..." % name)
                network.apply(he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    @property
    def latent_dim(self):
        try:
            latent_dim = self.nets.mapping_network.latent_dim
        except AttributeError:
            # it's a data parallel model
            latent_dim = self.nets.mapping_network.module.latent_dim
        return latent_dim

    def train(  # type: ignore
        self,
        loader: TrainingData,
        resume_iter: int = 0,
        total_iters: int = 100000,
        log_every: int = 100,
        save_every: int = 10000,
        eval_every: int = 10000,
        # sample_dir: str = "samples",
        ds_iter: int = 10000,
        lambda_reg: float = 1.0,
        lambda_sty: float = 1.0,
        lambda_cyc: float = 1.0,
        # Validation things
        val_loader=None,
        val_config=None,
    ):
        start = datetime.datetime.now()

        # resume training if necessary
        if resume_iter > 0:
            self._load_checkpoint(resume_iter)

        print("Start training...")
        for i in range(resume_iter, total_iters):
            # fetch images and labels
            inputs = next(loader)
            x_real, x_aug, y_org = inputs["x_src"], inputs["x_src2"], inputs["y_src"]
            x_ref, x_ref2, y_trg = inputs["x_ref"], inputs["x_ref2"], inputs["y_ref"]
            z_trg = torch.randn(x_real.size(0), self.latent_dim)
            z_trg2 = torch.randn(x_real.size(0), self.latent_dim)

            # Put everything on the right device
            for item in [x_real, x_ref, x_ref2, x_aug, y_org, y_trg, z_trg, z_trg2]:
                item.to(self.device)

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                self.nets, x_real, y_org, y_trg, z_trg=z_trg, lambda_reg=lambda_reg
            )
            self._reset_grad()
            d_loss.backward()
            self.optims["discriminator"].step()

            d_loss, d_losses_ref = compute_d_loss(
                self.nets, x_real, y_org, y_trg, x_ref=x_ref, lambda_reg=lambda_reg
            )
            self._reset_grad()
            d_loss.backward()
            self.optims["discriminator"].step()

            # train the generator
            g_loss, g_losses_latent, fake_x_latent = compute_g_loss(
                self.nets,
                x_real,
                y_org,
                y_trg,
                z_trgs=[z_trg, z_trg2],
                lambda_sty=lambda_sty,
                lambda_cyc=lambda_cyc,
            )
            self._reset_grad()
            g_loss.backward()
            self.optims["generator"].step()
            self.optims["mapping_network"].step()
            self.optims["style_encoder"].step()

            g_loss, g_losses_ref, fake_x_reference = compute_g_loss(
                self.nets,
                x_real,
                y_org,
                y_trg,
                x_refs=[x_ref, x_ref2],
                x_aug=x_aug,
                lambda_sty=lambda_sty,
                lambda_cyc=lambda_cyc,
            )
            self._reset_grad()
            g_loss.backward()
            self.optims["generator"].step()

            # compute moving average of network parameters
            moving_average(self.nets.generator, self.nets_ema.generator, beta=0.999)
            moving_average(
                self.nets.style_encoder, self.nets_ema.style_encoder, beta=0.999
            )
            moving_average(
                self.nets.mapping_network, self.nets_ema.mapping_network, beta=0.999
            )

            if (i + 1) % eval_every == 0 and val_loader is not None:
                self.evaluate(
                    val_loader, iteration=i + 1, mode="reference", val_config=val_config
                )
                self.evaluate(
                    val_loader, iteration=i + 1, mode="latent", val_config=val_config
                )

            # save model checkpoints
            if (i + 1) % save_every == 0:
                self._save_checkpoint(step=i + 1)

            # print out log losses, images
            if (i + 1) % log_every == 0:
                elapsed = datetime.datetime.now() - start
                # Log the images made by the EMA model!
                with torch.no_grad():
                    ema_fake_x_latent = self.nets_ema.generator(
                        x_real, self.nets_ema.mapping_network(z_trg, y_trg)
                    )
                    ema_fake_x_reference = self.nets_ema.generator(
                        x_real, self.nets_ema.style_encoder(x_ref, y_trg)
                    )
                self.log(
                    d_losses_latent,
                    d_losses_ref,
                    g_losses_latent,
                    g_losses_ref,
                    x_real,
                    x_ref,
                    fake_x_latent,
                    fake_x_reference,
                    ema_fake_x_latent,
                    ema_fake_x_reference,
                    y_org,  # Source classes
                    y_trg,  # Target classes
                    step=i + 1,
                    total_iters=total_iters,
                    elapsed_time=elapsed,
                )

    def log(
        self,
        d_losses_latent,
        d_losses_ref,
        g_losses_latent,
        g_losses_ref,
        x_real,
        x_ref,
        fake_x_latent,
        fake_x_reference,
        ema_fake_x_latent,
        ema_fake_x_reference,
        y_source,
        y_target,
        step,
        total_iters,
        elapsed_time,
    ):
        all_losses = dict()
        for loss, prefix in zip(
            [d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
            ["D/latent_", "D/ref_", "G/latent_", "G/ref_"],
        ):
            for key, value in loss.items():
                all_losses[prefix + key] = value
        # log all losses to wandb or print them
        if self.run:
            self.run.log(all_losses, step=step)
            for name, img, label in zip(
                [
                    "x_real",
                    "x_ref",
                    "fake_x_latent",
                    "fake_x_reference",
                    "ema_fake_x_latent",
                    "ema_fake_x_reference",
                ],
                [
                    x_real,
                    x_ref,
                    fake_x_latent,
                    fake_x_reference,
                    ema_fake_x_latent,
                    ema_fake_x_reference,
                ],
                [y_source, y_target, y_target, y_target, y_target, y_target],
            ):
                # TODO put captions back in somehow
                self.run.log_images({name: img}, step=step)

        print(
            f"[{elapsed_time}]: {step}/{total_iters}",
            flush=True,
        )
        g_losses = "\t".join(
            [
                f"{key}: {value:.4f}"
                for key, value in all_losses.items()
                if not key.startswith("D/")
            ]
        )
        d_losses = "\t".join(
            [
                f"{key}: {value:.4f}"
                for key, value in all_losses.items()
                if key.startswith("D/")
            ]
        )
        print(f"G Losses: {g_losses}", flush=True)
        print(f"D Losses: {d_losses}", flush=True)

    @torch.no_grad()
    def evaluate(
        self,
        val_loader,
        iteration,
        num_outs_per_domain=10,
        mode="latent",
        val_config=None,
    ):
        """
        Generates images for evaluation and stores them to disk.

        Parameters
        ----------
        val_loader
        """

        # Generate images for evaluation
        eval_dir = self.root_dir / "eval"
        eval_dir.mkdir(exist_ok=True, parents=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load classifier
        model = torch.jit.load(val_config.classifier_checkpoint)
        classifier = ClassifierWrapper(
            model=model,
            shift=val_config.shift,
            scale=val_config.scale,
        )
        classifier.to(device)
        assert mode in ["latent", "reference"]

        val_loader.set_mode(mode)
        iter_ref = None

        domains = val_loader.available_targets
        print("Number of domains: %d" % len(domains))

        conversion_rate_values = {}
        translation_rate_values = {}

        for trg_idx, trg_domain in enumerate(domains):
            src_domains = [x for x in val_loader.available_sources if x != trg_domain]
            val_loader.set_target(trg_domain)
            if mode == "reference":
                loader_ref = val_loader.loader_ref

            for src_idx, src_domain in enumerate(src_domains):
                task = "%s/%s" % (src_domain, trg_domain)
                # Creating the path
                path_fake = os.path.join(eval_dir, task)
                shutil.rmtree(path_fake, ignore_errors=True)
                os.makedirs(path_fake)

                # Setting the source domain
                val_loader.set_source(src_domain)
                loader_src = val_loader.loader_src

                for i, (x_src, _) in enumerate(tqdm(loader_src, total=len(loader_src))):
                    N = x_src.size(0)
                    x_src = x_src.to(device)
                    y_trg = torch.tensor([trg_idx] * N).to(device)

                    predictions = []
                    # generate num_outs_per_domain outputs from the same input
                    for j in range(num_outs_per_domain):
                        if mode == "latent":
                            z_trg = torch.randn(N, self.latent_dim).to(device)
                            s_trg = self.nets_ema.mapping_network(z_trg, y_trg)
                        else:
                            try:
                                # TODO don't need to re-do this every time, just use
                                # the same set of reference images for the whole dataset!
                                x_ref, _ = next(iter_ref)
                                x_ref = x_ref.to(device)
                            except (TypeError, StopIteration):  # iter_ref is None
                                iter_ref = iter(loader_ref)
                                x_ref, _ = next(iter_ref)
                                x_ref = x_ref.to(device)

                            if x_ref.size(0) > N:
                                x_ref = x_ref[:N]
                            elif x_ref.size(0) < N:
                                raise ValueError(
                                    "Not enough reference images."
                                    "Make sure that the batch size of the validation loader is bigger than `num_outs_per_domain`."
                                )
                            s_trg = self.nets_ema.style_encoder(x_ref, y_trg)

                        x_fake = self.nets_ema.generator(x_src, s_trg)
                        # Run the classification
                        pred = classifier(x_fake)
                        predictions.append(pred.cpu().numpy())
                predictions = np.stack(predictions, axis=0)
                assert len(predictions) > 0
                # Do it in a vectorized way, by reshaping the predictions
                predictions = predictions.reshape(
                    -1, num_outs_per_domain, predictions.shape[-1]
                )
                predictions = predictions.argmax(axis=-1)
                #
                at_least_one = np.any(predictions == trg_idx, axis=1)
                #
                conversion_rate = np.mean(at_least_one)
                translation_rate = np.mean(predictions == trg_idx)

                # STORE
                conversion_rate_values[f"conversion_rate_{mode}/" + task] = (
                    conversion_rate
                )
                translation_rate_values[f"translation_rate_{mode}/" + task] = (
                    translation_rate
                )

        # Add average conversion rate and translation rate
        conversion_rate_values[f"conversion_rate_{mode}/average"] = np.mean(
            [conversion_rate_values[key] for key in conversion_rate_values.keys()]
        )
        translation_rate_values[f"translation_rate_{mode}/average"] = np.mean(
            [translation_rate_values[key] for key in translation_rate_values.keys()]
        )

        # report conversion rate values
        filename = os.path.join(
            eval_dir, "conversion_rate_%.5i_%s.json" % (iteration, mode)
        )
        save_json(conversion_rate_values, filename)
        # report translation rate values
        filename = os.path.join(
            eval_dir, "translation_rate_%.5i_%s.json" % (iteration, mode)
        )
        save_json(translation_rate_values, filename)
        if self.run is not None:
            self.run.log(conversion_rate_values, step=iteration)
            self.run.log(translation_rate_values, step=iteration)


def compute_d_loss(nets, x_real, y_org, y_trg, z_trg=None, x_ref=None, lambda_reg=1.0):
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

        x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + lambda_reg * loss_reg
    return loss, dict(real=loss_real.item(), fake=loss_fake.item(), reg=loss_reg.item())


def compute_g_loss(
    nets,
    x_real,
    y_org,
    y_trg,
    z_trgs=None,
    x_refs=None,
    x_aug=None,
    lambda_sty: float = 1.0,
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

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # style invariance loss
    if x_aug is not None:
        s_pred2 = nets.style_encoder(x_aug, y_org)
        loss_sty2 = torch.mean(torch.abs(s_pred2 - s_org))
        loss_sty = (loss_sty + loss_sty2) / 2

    loss = loss_adv + lambda_sty * loss_sty + lambda_cyc * loss_cyc
    return (
        loss,
        dict(
            adv=loss_adv.item(),
            sty=loss_sty.item(),
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
    grad_d_out = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_d_out2 = grad_d_out.pow(2)
    assert grad_d_out2.size() == x_in.size()
    reg = 0.5 * grad_d_out2.view(batch_size, -1).sum(1).mean(0)
    return reg
