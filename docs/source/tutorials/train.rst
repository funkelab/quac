.. _sec_train:

=====================
Training the StarGAN
=====================

.. attention::
    This tutorial is still under construction. Come back soon for updates!

In this tutorial, we go over the basics of how to train a (slightly modified) StarGAN for use in QuAC.

Defining the dataset
====================

The data is expected to be in the form of image files with a directory structure denoting the classification.
For example:

.. code-block:: bash

    data_folder/
        crow/
            crow1.png
            crow2.png
        raven/
            raven1.png
            raven2.png

A training dataset is defined in `quac.training.data` which will need to be given two directories: a `source` and a `reference`. These directories can be the same.

The validation dataset will need the same information.

For example:

.. code-block:: python
    :linenos:

    from quac.training.data import TrainingDataset

    dataset = TrainingDataset(
        source="path/to/training/data",
        reference="path/to/training/data",
        img_size=128,
        batch_size=4,
        num_workers=4
    )

    # Setup data for validation
    val_dataset = ValidationData(
        source="path/to/training/data",
        reference="path/to/training/data",
        img_size=128,
        batch_size=16,
        num_workers=16
    )


Defining the models
===================

The models can be built using a function in `quac.training.stargan`.

.. code-block:: python
    :linenos:

    from quac.training.stargan import build_model

    nets, nets_ema = build_model(
        img_size=256,  # Images are made square
        style_dim=64,  # The size of the style vector
        input_dim=1,  # Number of channels in the input
        latent_dim=16,  # The size of the random latent
        num_domains=4,  # Number of classes
        single_output_style_encoder=False
    )
    ## Defining the models
    nets, nets_ema = build_model(**experiment.model.model_dump())

If using multiple or specific GPUs, it may be necessary to add the `gpu_ids` argument.

The `nets_ema` are a copy of the `nets` that will not be trained but rather will be an exponential moving average of the weight of the `nets`.
The sub-networks of both can be accessed in a dictionary-like manner.

Creating a logger
=================

.. code-block:: python
    :linenos:

    # Example using WandB
    logger = Logger.create(
        log_type="wandb",
        project="project-name",
        name="experiment name",
        tags=["experiment", "project", "test", "quac", "stargan"],
        hparams={ # this holds all of the hyperparameters you want to store for your run
            "hyperparameter_key": "Hyperparameter values"
        }
    )

    # TODO example using tensorboard

Defining the Solver
===================

It is now time to initiate the `Solver` object, which will do the bulk of the work in training.

.. code-block:: python
    :linenos:

    solver = Solver(
        nets,
        nets_ema,
        # Checkpointing
        checkpoint_dir="path/to/store/checkpoints",
        # Parameters for the Adam optimizers
        lr=1e-4,
        beta1=0.5,
        beta2=0.99,
        weight_decay=0.1,
    )

    # TODO
    solver = Solver(nets, nets_ema, **experiment.solver.model_dump(), run=logger)

Training
========
We use the solver to train on the data as follows:

.. code-block:: python
    :linenos:

    from quac.training.options import ValConfig
    val_config=ValConfig(
        classifier_checkpoint="/path/to/classifier/", mean=0.5, std=0.5
    )

    solver.train(dataset, val_config)

All results will be stored in the `checkpoint_directory` defined above.
Validation will be done during training at regular intervals (by default, every 10000 iterations).

BONUS: Training with a Config file
==================================

.. code-block:: python
    :linenos:

    run_config=RunConfig(
        # All of these are default
        resume_iter=0,
        total_iter=100000,
        log_every=1000,
        save_every=10000,
        eval_every=10000,
    )
    val_config=ValConfig(
        classifier_checkpoint="/path/to/classifier/",
        # The below is default
        val_batch_size=32
        num_outs_per_domain=10,
        mean=0.5,
        std=0.5,
        grayscale=True,
    )
    loss_config=LossConfig(
        # The following should probably not be changed
        # unless you really know what you're doing :)
        # All of these are default
        lambda_ds=1.,
        lambda_reg=1.,
        lambda_sty=1.,
        lambda_cyc=1.,
    )
