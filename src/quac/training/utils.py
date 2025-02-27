"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import re
import torch
import torch.nn as nn
import torchvision.utils as vutils


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


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


class Logger:
    def __init__(self, log_dir, nets, num_outs_per_domain=10) -> None:
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)
        self.nets = nets

    @torch.no_grad()
    def translate_and_reconstruct(self, x_src, y_src, x_ref, y_ref, filename):
        N, C, H, W = x_src.size()
        s_ref = self.nets.style_encoder(x_ref, y_ref)
        x_fake = self.nets.generator(x_src, s_ref)
        s_src = self.nets.style_encoder(x_src, y_src)
        x_rec = self.nets.generator(x_fake, s_src)
        x_concat = [x_src, x_ref, x_fake, x_rec]
        x_concat = torch.cat(x_concat, dim=0)
        save_image(x_concat, N, filename)
        del x_concat

    @torch.no_grad()
    def translate_using_latent(self, x_src, y_trg_list, z_trg_list, psi, filename):
        N, C, H, W = x_src.size()
        latent_dim = z_trg_list[0].size(1)
        x_concat = [x_src]

        for i, y_trg in enumerate(y_trg_list):
            z_many = torch.randn(10000, latent_dim).to(x_src.device)
            y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
            s_many = self.nets.mapping_network(z_many, y_many)
            s_avg = torch.mean(s_many, dim=0, keepdim=True)
            s_avg = s_avg.repeat(N, 1)

            for z_trg in z_trg_list:
                s_trg = self.nets.mapping_network(z_trg, y_trg)
                s_trg = torch.lerp(s_avg, s_trg, psi)
                x_fake = self.nets.generator(x_src, s_trg)
                x_concat += [x_fake]

        x_concat = torch.cat(x_concat, dim=0)
        save_image(x_concat, N, filename)

    @torch.no_grad()
    def translate_using_reference(self, x_src, x_ref, y_ref, filename):
        N, C, H, W = x_src.size()
        wb = torch.ones(1, C, H, W).to(x_src.device)
        x_src_with_wb = torch.cat([wb, x_src], dim=0)

        s_ref = self.nets.style_encoder(x_ref, y_ref)
        s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
        x_concat = [x_src_with_wb]
        for i, s_ref in enumerate(s_ref_list):
            x_fake = self.nets.generator(x_src, s_ref)
            x_fake_with_ref = torch.cat([x_ref[i : i + 1], x_fake], dim=0)
            x_concat += [x_fake_with_ref]

        x_concat = torch.cat(x_concat, dim=0)
        save_image(x_concat, N + 1, filename)
        del x_concat


###########################
# LOSS PLOTTING FUNCTIONS #
###########################
def get_epoch_number(f):
    # Get the epoch number from the filename and sort the files by epoch
    # The epoch number is the only number in the filename, and can be 5 or 6 digits long
    return int(re.findall(r"\d+", f.name)[0])


def load_json_files(files):
    # Load the data from the json files, store everything into a dictionary, with the epoch number as the key
    data = {}
    for f in files:
        with open(f, "r") as file:
            data[get_epoch_number(f)] = json.load(file)

    df = pd.DataFrame.from_dict(data, orient="index")
    df.columns = [col.split("/")[-1] for col in df.columns]
    df.sort_index(inplace=True)
    return df


def plot_from_data(
    reference_conversion_data,
    latent_conversion_data,
    reference_translation_data,
    latent_translation_data,
):
    n_cols = int(np.ceil(len(reference_conversion_data.columns) / 7))
    fig, axes = plt.subplots(7, n_cols, figsize=(15, 15))
    for col, ax in zip(reference_conversion_data.columns, axes.ravel()):
        reference_conversion_data[col].plot(
            ax=ax, label="Reference Conversion", color="black"
        )
        latent_conversion_data[col].plot(
            ax=ax, label="Latent Conversion", color="black", linestyle="--"
        )
        reference_translation_data[col].plot(
            ax=ax, label="Reference Translation", color="gray"
        )
        latent_translation_data[col].plot(
            ax=ax, label="Latent Translation", color="gray", linestyle="--"
        )
        ax.set_ylim(0, 1)
        # format the title only if there are any non-numeric characters in the column name
        alphabetic_title = any([c.isalpha() for c in col])
        if alphabetic_title:
            # Split the words in the column by the underscore, remove all numbers, and add the word "to" between the words
            title = " to ".join(
                [word.capitalize() for word in col.split("_") if not word.isdigit()]
            )
            # Remove all remaining numbers from the title
            title = "".join([i for i in title if not i.isdigit()])
        else:
            # The title is \d2\d and we want it to be \d to \d, unfortunately sometimes \d is the number 2, so we need to be careful
            title = re.sub(r"(\d)2(\d)", r"\1 to \2", col)
        ax.set_title(title)
    # Hide all of the extra axes if any
    for ax in axes.ravel()[len(reference_conversion_data.columns) :]:
        ax.axis("off")
    # Add an x-axis label to the bottom row of plots that still has a visible x-axis
    num_axes = len(reference_conversion_data.columns)
    for ax in axes.ravel()[num_axes - n_cols :]:
        ax.set_xlabel("Iteration")
    # Add a y-axis label to the left column of plots
    for ax in axes[:, 0]:
        ax.set_ylabel("Rate")
    # Make a legend for the whole figure, assuming that the labels are the same for all subplots
    fig.legend(
        *ax.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(1.15, 1)
    )
    fig.tight_layout()

    return fig


def plot_metrics(root, show: bool = True, save: str = ""):
    """
    root: str
        The root directory containing the json files with the metrics.
    show: bool
        Whether to show the plot.
    save: str
        The path to save the plot to.
    """
    # Goal is to get a better idea of the success/failure of conversion as training goes on.
    # Will be useful to understand how it changes over time, and how the translation/conversion/diversity tradeoff plays out.

    files = list(Path(root).rglob("*.json"))

    # Split files by whether they are LPIPS, conversion_rate, or translation_rate
    conversion_files = [f for f in files if "conversion_rate" in f.name]
    translation_files = [f for f in files if "translation_rate" in f.name]

    # Split files by whether they are reference or latent
    reference_conversion = [f for f in conversion_files if "reference" in f.name]
    latent_conversion = [f for f in conversion_files if "latent" in f.name]

    reference_translation = [f for f in translation_files if "reference" in f.name]
    latent_translation = [f for f in translation_files if "latent" in f.name]

    # Load the data from the json files
    reference_conversion_data = load_json_files(reference_conversion)
    latent_conversion_data = load_json_files(latent_conversion)
    reference_translation_data = load_json_files(reference_translation)
    latent_translation_data = load_json_files(latent_translation)

    fig = plot_from_data(
        reference_conversion_data,
        latent_conversion_data,
        reference_translation_data,
        latent_translation_data,
    )
    if show:
        plt.show()
    if save is not None:
        fig.savefig(save)
