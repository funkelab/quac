"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from collections import OrderedDict

import numpy as np
from pathlib import Path
from starganv2.metrics.conversion import calculate_conversion_given_path
from starganv2.core import utils
import torch


@torch.no_grad()
def calculate_metrics(
    eval_dir,
    step,
    mode,
    classifier_checkpoint,
    img_size,
    val_batch_size,
    num_outs_per_domain,
    mean,
    std,
    input_dim=3,
):
    print("Calculating conversion rate for all tasks...")
    translation_rate_values = (
        OrderedDict()
    )  # How many output images are of the right class
    conversion_rate_values = (
        OrderedDict()
    )  # How many input samples have a valid counterfactual

    domains = [subdir.name for subdir in Path(eval_dir).iterdir() if subdir.is_dir()]

    for subdir in Path(eval_dir).iterdir():
        if not subdir.is_dir() or subdir.startswith("."):  # Skip hidden files
            continue
        src_domain = subdir.name

        for subdir2 in Path(subdir).iterdir():
            if not subdir2.is_dir() or subdir2.startswith("."):
                continue
            trg_domain = subdir2.name

            task = "%s_to_%s" % (src_domain, trg_domain)
            print("Calculating conversion rate for %s..." % task)
            target_class = domains.index(trg_domain)

            translation_rate, conversion_rate = calculate_conversion_given_path(
                subdir2,
                model_checkpoint=classifier_checkpoint,
                target_class=target_class,
                img_size=img_size,
                batch_size=val_batch_size,
                num_outs_per_domain=num_outs_per_domain,
                mean=mean,
                std=std,
                grayscale=(input_dim == 1),
            )
            conversion_rate_values[
                "conversion_rate_%s/%s" % (mode, task)
            ] = conversion_rate
            translation_rate_values[
                "translation_rate_%s/%s" % (mode, task)
            ] = translation_rate

    # calculate the average conversion rate for all tasks
    conversion_rate_mean = 0
    translation_rate_mean = 0
    for _, value in conversion_rate_values.items():
        conversion_rate_mean += value / len(conversion_rate_values)
    for _, value in translation_rate_values.items():
        translation_rate_mean += value / len(translation_rate_values)

    conversion_rate_values["conversion_rate_%s/mean" % mode] = conversion_rate_mean
    translation_rate_values["translation_rate_%s/mean" % mode] = translation_rate_mean

    # report conversion rate values
    filename = os.path.join(eval_dir, "conversion_rate_%.5i_%s.json" % (step, mode))
    utils.save_json(conversion_rate_values, filename)
    # report translation rate values
    filename = os.path.join(eval_dir, "translation_rate_%.5i_%s.json" % (step, mode))
    utils.save_json(translation_rate_values, filename)
