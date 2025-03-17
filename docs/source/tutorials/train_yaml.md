# The conversion network

The central model in QuAC is a generator that converts data from one class to another.

To get started, make sure you've copied the `example_experiment` directory, giving it a descriptive name for your experiment (*e.g.* `date_experiment-name_dataset`).
Then, modify the enclosed `config.yaml` file as you follow along this how-to.  

## Data loading configuration

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

## Model configuration

Next we will want to define parameters for the model that we will be training.

Here is some example YAML, which you can modify to your purposes.
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


## Training and validation configuration
Next we need to set a some details  

Here is some example YAML to modify and use for your own data.  
```{code-block} yaml
solver:
    root_dir: "/directory/to/save/your/results"

validation_config:
    classifier_checkpoint: "/path/to/your/torchscript/checkpoint"
    do_nothing: true # Pass the image directly to the classifier
    val_batch_size: 16
```

- `solver.root_dir`: this is an (absolute) path to the directory where you want to store the results. The checkpoints, logs, and other training artifacts will be stored there. 
- `validation_config` defines how we will use the classifier to validate the quality of our conversion network. 
    - `classifier_checkpoint` points to the path where your torchscript classifier is. This should be the path you decided on [when you prepared your classifier](classifier.md)
    - `do_nothing` is a boolean value (`true` or `false`) that defines whether to modify the output of the conversion network before you pass it to the classifier for validation. If you have correctly configured the conversion model to output the range of data that your classifier expects, then you should set this to `true`. If, however, your classifier requires a different normalization, then you should set it to `false`

```{note}
If you set `do_nothing` to `false`, you will have to set the `mean` and `std` arguments in the `validation_config`. The output of the conversion model will the be first shifted to `[0, 1]` and then we will apply `torchvision`-style normalization: `x = (x - mean) / std`.

For example: if your data is in `[-1, 1]`, the following configurations are equivalent: 
1. `do_nothing: true`
2. `do_nothing: false, mean: 0.5, std: 0.5`
```

## The run

Finally, we need to decide on some details of our run, and how we're going to log metrics and example batches.

Logging will happen using `wandb`.
Before you can begin, you must configure `wandb` for yourself using their [quickstart documentation](https://docs.wandb.ai/quickstart/). 

Here is some example YAML for the run configuration, to modify for your purposes.
```{code-block} yaml
run:
    total_iters: 50000
    log_every: 1000
    save_every: 1000
    eval_every: 1000

log:
    project: "quac_example_project"
    name: "name_of_your_run"
    notes: "Stargan training on my dataset"
    tags:
        - stargan
        - training
```

- All of the arguments in `run` are defined in number of batches:
    - `total_iters` corresponds to the total number of batches to train on. This is how you set the length of training. 
    - Every `log_every` batches, we will log losses, some metrics, and input/output images from the `train` set to `wandb`. You should see them appear over time. 
    - Every `save_every` batches, we will save a model checkpoint. 
    - Every `eval_every` batches, a validation loop is run. This will go through the validation dataset, run conversion from each class to every other class, and pass the output images to the pre-trained classifier for prediction. The resulting validation metrics will also be saved to `wandb`. Note that if you have many classes and a large `validation` dataset, `validation` can be a lengthy process.  

## Training
Once you have fully edited the `config.yaml` file, you are ready to start a training run. 
In your experiment directory, simply run `python train_stargan.py`. 
This script will read the arguments from the configuration file you have just written, and begin training a StarGAN network to convert your images from one class to another.

The training script will  also begin a run on [Weights and Biases](https://wandb.ai).
Connect to your account there to follow the run.
