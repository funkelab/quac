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
from quac.training import utils
from quac.training.classification import ClassifierWrapper
from quac.training.data_loader import get_eval_loader
import torch
from tqdm import tqdm


@torch.no_grad()
def calculate_metrics(
    eval_dir,
    step=0,
    mode="latent",
    classifier_checkpoint=None,
    img_size=128,
    val_batch_size=16,
    num_outs_per_domain=10,
    mean=None,
    std=None,
    input_dim=3,
    run=None,
):
    print("Calculating conversion rate for all tasks...", flush=True)
    translation_rate_values = (
        OrderedDict()
    )  # How many output images are of the right class
    conversion_rate_values = (
        OrderedDict()
    )  # How many input samples have a valid counterfactual

    domains = [subdir.name for subdir in Path(eval_dir).iterdir() if subdir.is_dir()]

    for subdir in Path(eval_dir).iterdir():
        if not subdir.is_dir() or subdir.name.startswith("."):  # Skip hidden files
            continue
        src_domain = subdir.name

        for subdir2 in Path(subdir).iterdir():
            if not subdir2.is_dir() or subdir2.name.startswith("."):
                continue
            trg_domain = subdir2.name

            task = "%s/%s" % (src_domain, trg_domain)
            print("Calculating conversion rate for %s..." % task, flush=True)
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
            conversion_rate_values["conversion_rate_%s/%s" % (mode, task)] = (
                conversion_rate
            )
            translation_rate_values["translation_rate_%s/%s" % (mode, task)] = (
                translation_rate
            )

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
    if run is not None:
        run.log(conversion_rate_values, step=step)
        run.log(translation_rate_values, step=step)


@torch.no_grad()
def calculate_conversion_given_path(
    path,
    model_checkpoint,
    target_class,
    img_size=128,
    batch_size=50,
    num_outs_per_domain=10,
    mean=0.5,
    std=0.5,
    grayscale=False,
):
    print("Calculating conversion given path %s..." % path, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ClassifierWrapper(model_checkpoint, mean=mean, std=std)
    classifier.to(device)
    classifier.eval()

    loader = get_eval_loader(
        path,
        img_size=img_size,
        batch_size=batch_size,
        imagenet_normalize=False,
        shuffle=False,
        grayscale=grayscale,
    )

    predictions = []
    for x in tqdm(loader, total=len(loader)):
        x = x.to(device)
        predictions.append(classifier(x).cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    # Do it in a vectorized way, by reshaping the predictions
    predictions = predictions.reshape(-1, num_outs_per_domain, predictions.shape[-1])
    predictions = predictions.argmax(axis=-1)
    #
    at_least_one = np.any(predictions == target_class, axis=1)
    #
    conversion_rate = np.mean(at_least_one)  # (sum(at_least_one) / len(at_least_one)
    translation_rate = np.mean(predictions == target_class)
    return translation_rate, conversion_rate
