# The conversion network

The central model in QuAC is a generator that converts data from one class to another.

To train the model, you need a YAML file that holds all of the details for your experiment.

## Data configuration

We will begin with the data configuration: this needs to be set for both the `train` and `validation` data sets.

Here is an example data loading configuration in YAML format.
```{code-block} yaml
data:
    source: "</path/to/your/source/data/train>"
    reference: "</path/to/your/source/data/train>" 
    img_size: 128
    batch_size: 16
    num_workers: 12
    mean: 0.5 
    std: 0.5
    grayscale: true

validation_data:
    source: "</path/to/your/source/data/val>"
    reference: "</path/to/your/source/data/val>" 
    img_size: 128
    batch_size: 16
    num_workers: 12
    mean: 0.5
    std: 0.5
    grayscale: true
```

- The `source` and `reference` values hold the (absolute) path your data. The data in `source` is used as the **query** image, and the data in `reference` as the **reference** image. 
- The `mean` and `std` values will be used to normalize your data before passing it into the StarGAN. These are passed to a [`torchvision.transforms.Normalize`](https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html?highlight=normalize#torchvision.transforms.Normalize).We *strongly* recommend `mean=0.5, std=0.5`, which will put your data in range `[-1, 1]`.
- If you have RGB data, set `grayscale` to `false`. Else, set it to `true`. 
- Set `img_size` to the input size expected by your classifier. Your images will be resized accordingly by bi-cubic interpolation.
- `batch_size` and `num_workers` are passed to a [`torch.utils.data.Dataloader`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

If you have a `test` data set, add it to the configuration under `test_data`.
In most cases, `train`, `validation` and `test` will have the same configuration. 
However, you may want to increase `batch_size` for `validation` and `test` for more efficient inference.

## The model

Next we will want to define parameters for the model that we will be training.

Here is an example YAML file, which you can modify to your purposes.
```{code-block} yaml

model:
    img_size: 128
    style_dim: 64
    latent_dim: 16
    num_domains: 3
    input_dim: 1
    final_activation: "tanh"
```

- Set `img_size` to the same value as above, in the data loading configuration. 
- `style_dim` defines the size of the learned style space. This is the latent representation of class features present in an image, and is used to condition the conversion of images from one class to another. Small, simple datasets can have a smaller style style. 
- `latent_dim` defines the size of the randomly sampled value from which `style` is made. It can be smaller than `style_dim`.
- `num_domains` defines the number of classes. This must match what is in your data.
- `input_dim` defines the number of channels in the input, and should be `1` if you data is grayscale, or `3` if you have RGB data. 
- `final_activation` defines the final layer of the model. You should use an activation that will put your output images within the same range as our inputs. Here, we use `tanh` because we assume that the input range is `[-1, 1]`.


## The Solver

```{code-block} yaml

solver:
    root_dir: "/directory/to/save/your/results"

loss:
    lambda_ds: 0.0
    lambda_sty: 1.  # Default
    lambda_cyc: 1.  # Default
    lambda_id: 1.  # Default

validation_config:
    classifier_checkpoint: "/path/to/your/torchscript/checkpoint"
    val_batch_size: 16
    num_outs_per_domain: 10 # Default
    do_nothing: true # Pass the image directly to the classifier
```

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


## The run

Finally, we need to decide on some details of our run, and how we're going to log run details.
We will save our logs locally, but there is also the option to use Tensorboard or WandB.

Then we decide how long to train, how often to save checkpoints, how often to run evaluation, and how often to log results.
All of these values are in number of batches.

```{code-block} yaml

    log_type: local

    log:
        log_dir: "tutorial_logs"
        project: "quac_example_project"
        name: "disc_b"
        notes: "Stargan training on my dataset"
        tags:
            - stargan
            - training

    run:
        log_every: 1000
        total_iters: 5000
        save_every: 1000
        eval_every: 1000
```

## Putting it all together

We can save all of the configuration details in a YAML file `config.yaml`.
We will load all of this together, and then start training! To avoid training from within a Jupyter notebook, you can also put the following cell into a script and run that.

```{code-block} python
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
```
