.. _sec_train_yaml:

=========================
Training from a YAML file
=========================

.. attention::
    This tutorial is still under construction. Come back soon for updates!

Example dataset
===============

In this tutorial, we will be applying QuAC to a simple but non-trivial dataset.

The majority of the work will be defining a YAML file that holds all of the details for your experiment.
We will begin with the data.

The data is expected to be split into at least two sets: train and validation.
If you have it, you can also add a test dataset.
For each, the details need to be defined anew.

The `source` and `reference` values point to the data. The data in `source` is used as the **query** image, and the data in `reference` as the **reference** image. They do not need to be different!

The `mean` and `std` values will be used to normalize your data before passing it into the StarGAN. Note that we expect the data to be normalized in the same way for your StarGAN and your classifier.

.. code-block:: yaml

    data:
        source: "/nrs/funke/adjavond/data/duplex/disc_b/train"
        reference: "/nrs/funke/adjavond/data/duplex/disc_b/train"
        img_size: 128
        batch_size: 16
        num_workers: 12
        mean: 0.5
        std: 0.5
        grayscale: true

    validation_data:
        source: "/nrs/funke/adjavond/data/duplex/disc_b/val"
        reference: "/nrs/funke/adjavond/data/duplex/disc_b/val"
        img_size: 128
        batch_size: 16
        num_workers: 12
        mean: 0.5
        std: 0.5
        grayscale: true

    test_data:
        source: "/nrs/funke/adjavond/data/duplex/disc_b/test"
        reference: "/nrs/funke/adjavond/data/duplex/disc_b/test"
        img_size: 128
        batch_size: 16
        num_workers: 12
        mean: 0.5
        std: 0.5
        grayscale: true

Behind the scenes, all of these will be absorned into a `quac.training.config.DataConfig` object.
You can have a look at that object to see the default values, and types of the parameters.

The model
=========

Next we will want to define the model that we will be training.

Make sure that you match your model to the data that you are putting in! For example, the `img_size` should be the same.
The `num_domains` parameter refers to the number of classes in your dataset. For this dataset there are three classes.
If `grayscale` is `true`, we need to confirm that `input_dim`, which corresponds to the number of channels in the input images, is `1`.
Finally, since we want our output to be images with the same range as our inputs, we want to use `tanh` as an activation here (our input images will be in `[-1, 1]`).

.. code-block:: yaml

    model:
        img_size: 128
        style_dim: 64
        latent_dim: 16
        num_domains: 3
        input_dim: 1
        final_activation: "tanh"

This will be ingested into a `quac.training.config.ModelConfig` object. Have a look to see what the parameter defaults are!

The Solver
==========

Let's next set up the required information for the `Solver`.
For this, we need to determine where the model checkpoints and any intermediate evaluation outputs will be saved.
This is defined in the `solver.root_dir`.

Next, we need to choose how to balance our losses.
For stability, we will ignore the diversity loss completely by setting its coefficient `lambda_ds` to `0.0`.
The style consistency loss, the cycle consistency loss, and the identity loss will all be considered equally important and set to `1.0`.

Next, since we want to run evaluation on our validation dataset, we need to define some details for that.
Specifically, we need to define where the checkpoint is located. This will need to be a `torchscript` compiled checkpoint, so that we don't need to explicitly define the classifier model.
We also define the batch size for validation, and since we have made sure that the data normalization for the StarGAN matches that of the classifier, we `do_nothing` before passing images to the classifier.
This setup ensures that during training we can check the StarGAN's *translation rate* and *conversion rate*.
The *translation rate* shows how many of the StarGAN's output are classified by the pre-trained classifier as the `target` class.
The *conversion rate* is similarly, except it gives the model several tries to correctly convert an image. This is possible because the StarGAN includes some randomness.
In our case, it gets `num_outs_per_domain` tries, so 10.

.. code-block:: yaml

    solver:
        root_dir: "/nrs/funke/adjavond/projects/quac/disc_b_example/stargan"

    loss:
        lambda_ds: 0.0
        lambda_sty: 1.  # Default
        lambda_cyc: 1.  # Default
        lambda_id: 1.  # Default

    validation_config:
        classifier_checkpoint: "/nrs/funke/adjavond/projects/duplex/disc_b/vgg_checkpoint_jit.pt"
        val_batch_size: 16
        num_outs_per_domain: 10 # Default
        do_nothing: true # Pass the image directly to the classifier


The run
=======

Finally, we need to decide on some details of our run, and how we're going to log run details.
We will save our logs locally, but there is also the option to use Tensorboard or WandB.

Then we decide how long to train, how often to save checkpoints, how often to run evaluation, and how often to log results.
All of these values are in number of batches.

.. code-block:: yaml

    log_type: local

    log:
        log_dir: "tutorial_logs"
        project: "quac_example_project"
        name: "disc_b"
        notes: "Stargan training on Disc B dataset"
        tags:
            - stargan
            - training

    run:
        log_every: 1000
        total_iters: 5000
        save_every: 1000
        eval_every: 1000

Putting it all together
=======================

We can save all of the configuration details in a YAML file `config.yaml`.
We will load all of this together, and then start training! To avoid training from within a Jupyter notebook, you can also put the following cell into a script and run that.

.. code-block:: python
    :linenos:

    # Training the StarGAN
    from quac.training.config import ExperimentConfig
    import git
    from quac.training.data_loader import TrainingData, ValidationData
    from quac.training.stargan import build_model
    from quac.training.solver import Solver
    from quac.training.logging import Logger
    import torch
    import typer
    import warnings
    import yaml

    torch.backends.cudnn.benchmark = True

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    experiment = ExperimentConfig(**config)

    # Setting up the logger
    logger = Logger.create(
        experiment.log_type,
        hparams=experiment.model_dump(),
        resume_iter=experiment.run.resume_iter,
        **experiment.log.model_dump(),
    )

    # Defining the datasets for training and validation
    dataset = TrainingData(**experiment.data.model_dump())
    val_dataset = ValidationData(**experiment.validation_data.model_dump())

    # Defining the models
    nets, nets_ema = build_model(**experiment.model.model_dump())

    # Defining the Solver
    solver = Solver(nets, nets_ema, **experiment.solver.model_dump(), run=logger)

    # Training the model
    solver.train(
        dataset,
        **experiment.run.model_dump(),
        **experiment.loss.model_dump(),
        val_loader=val_dataset,
        val_config=experiment.validation_config,
    )
