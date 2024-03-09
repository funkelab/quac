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
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


from quac.training.stargan import build_model
from quac.training.checkpoint import CheckpointIO
from quac.training.data_loader import InputFetcher, AugmentedInputFetcher
import quac.training.utils as utils

# from metrics.eval import calculate_metrics


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
)


class Solver(nn.Module):
    def __init__(self, nets, nets_ema, run, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets = nets
        self.nets_ema = nets_ema
        self.run = run

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
                lr=args.f_lr if net == "mapping_network" else args.lr,
                betas=[args.beta1, args.beta2],
                weight_decay=args.weight_decay,
            )

            self.ckptios = [
                CheckpointIO(
                    ospj(args.checkpoint_dir, "{:06d}_nets.ckpt"),
                    data_parallel=True,
                    **self.nets,
                ),
                CheckpointIO(
                    ospj(args.checkpoint_dir, "{:06d}_nets_ema.ckpt"),
                    data_parallel=True,
                    **self.nets_ema,
                ),
                CheckpointIO(
                    ospj(args.checkpoint_dir, "{:06d}_optims.ckpt"), **self.optims
                ),
            ]
        else:
            self.ckptios = [
                CheckpointIO(
                    ospj(args.checkpoint_dir, "{:06d}_nets_ema.ckpt"),
                    data_parallel=True,
                    **self.nets_ema,
                )
            ]

        self.to(self.device)
        # TODO The EMA doesn't need to be in named_childeren()
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
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

    def train(self, loaders, run=None):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = AugmentedInputFetcher(
            loaders.src, loaders.ref, args.latend_dim, "train"
        )
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, "val")
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print("Start training...")
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, x_aug, y_org = inputs.x_src, inputs.x_src2, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg
            )
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref
            )
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2]
            )
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], x_aug=x_aug
            )
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # TODO do this with timm
            # compute moving average of network parameters
            nets_ema.update(nets)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= initial_lambda_ds / args.ds_iter

            # print out log info
            if (i + 1) % args.log_every == 0:
                # TODO replace with wandb
                all_losses = dict()
                for loss, prefix in zip(
                    [d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                    ["D/latent_", "D/ref_", "G/latent_", "G/ref_"],
                ):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses["G/lambda_ds"] = args.lambda_ds
                run.log(all_losses)

                # TODO put this on wandb instead
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i + 1)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                self._save_checkpoint(step=i + 1)

            # compute FID and LPIPS if necessary
            if (i + 1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i + 1, mode="latent")
                calculate_metrics(nets_ema, args, i + 1, mode="reference")

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode="latent")
        calculate_metrics(nets_ema, args, step=resume_iter, mode="reference")


def compute_d_loss(
    nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None
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

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(
        real=loss_real.item(), fake=loss_fake.item(), reg=loss_reg.item()
    )


def compute_g_loss(
    nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None, x_aug=None
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

    x_fake = nets.generator(x_real, s_trg, masks=masks)
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
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # style invariance loss
    if x_aug is not None:
        s_pred2 = nets.style_encoder(x_aug, y_org)
        loss_sty2 = torch.mean(torch.abs(s_pred2 - s_org))
        loss_sty = (loss_sty + loss_sty2) / 2

    # TODO Triplet loss with anchor=x_fake (from reference), neg=x_fake2 (from reference 2), and pos=x_fake3 (from augmented x_ref1)
    # e.g.
    # from torch.nn.functional import triplet_margin_loss
    # if x_trg is not None:
    #   x_ref3 = transform(x_ref)
    #   s_trg3 = nets.style_encoder(x_ref3, y_trg)
    #   x_fake3 = nets.generator(x_real, s_trg3, masks=masks)
    #   loss_triplet = triplet_margin_loss(x_fake, x_fake2, x_fake3)
    # I'm hoping this has the effect of a invariance-diversity double whammy

    loss = (
        loss_adv
        + args.lambda_sty * loss_sty
        - args.lambda_ds * loss_ds
        + args.lambda_cyc * loss_cyc
    )
    return loss, Munch(
        adv=loss_adv.item(), sty=loss_sty.item(), ds=loss_ds.item(), cyc=loss_cyc.item()
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
