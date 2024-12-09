.. _sec_train:

=====================
Training the StarGAN
=====================

.. attention::
    It is recommended to use the YAML configuration method to train the conversion model.

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

    training_directory = "path/to/training/data"
    validation_directory = "path/to/validation/data"

    dataset = TrainingDataset(
        source=training_directory,
        reference=training_directory,
        img_size=128,
        batch_size=4,
        num_workers=4
    )

    # Setup data for validation
    val_dataset = ValidationData(
        source=validation_directory,
        reference=validation_directory,
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

    solver = Solver(nets, nets_ema, **experiment.solver.model_dump(), run=logger)



Training
========
Once we've created the solver, we also need to define how we're going to train and validate.
This is done through three different configuations.

The `ValConfig` determines how validation will be done.
It especially tells us

.. code-block:: python
    :linenos:

    val_config=ValConfig(
        classifier_checkpoint="/path/to/classifier/",
        # The below is default
        val_batch_size=32
        num_outs_per_domain=10,
        mean=0.5,
        std=0.5,
        grayscale=True,
    )

.. code-block:: python
    :linenos:

    loss_config=LossConfig(
        lambda_ds=0.,
        lambda_reg=1.,
        lambda_sty=1.,
        lambda_cyc=1.,
    )

    run_config=RunConfig(
        # All of these are default
        resume_iter=0,
        total_iter=100000,
        log_every=1000,
        save_every=10000,
        eval_every=10000,
    )

Finally, we can train the model!

.. code-block:: python
    :linenos:

    from quac.training.options import ValConfig

    solver.train(dataset, val_config)

All results will be stored in the `checkpoint_directory` defined above.

Once your model is trained, you can move on to generating images with it.
