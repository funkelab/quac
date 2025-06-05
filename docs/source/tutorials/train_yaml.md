# The conversion network

The {term}`conversion network` is the central model in QuAC.
It is a generator that converts data from one class to another.
Specifically, it turns a {term}`query image` into a {term}`generated image` by applying a {term}`style` to it.
Here, we will train a [StarGAN](http://arxiv.org/abs/1912.01865) model to do the job.

To get started, make sure you've copied the `example_experiment` directory, giving it a descriptive name for your experiment (*e.g.* `date_experiment-name_dataset`).
Then, modify the enclosed `config.yaml` file as you follow along this how-to.  

## Data loading configuration

We will begin with the data configuration: this needs to be set for both the `train` and `validation` data sets.

Here is an example data loading configuration in YAML format.
```{code-block} yaml
data:
    source: "</path/to/your/source/data/train>"
    img_size: 128
    batch_size: 16
    num_workers: 12
    grayscale: true
    rgb: false

validation_data:
    source: "</path/to/your/source/data/val>"
    img_size: 128
    batch_size: 16
    num_workers: 12
    grayscale: true
    rgb: false
```

- The `source` value holds (absolute) path your data.
- If you have RGB data, set `grayscale` to `false` and `rgb` to true. If you do not set these, RGB data is assumed. 
- Set `img_size` to the input size expected by your classifier. Your images will be resized accordingly by bi-cubic interpolation.
- `batch_size` and `num_workers` are passed to a [`torch.utils.data.Dataloader`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

If you have a `test` data set, add it to the configuration under `test_data`.
In most cases, `train`, `validation` and `test` will have the same configuration. 
However, you may want to increase `batch_size` for `validation` and `test` for more efficient inference.

```{note}
If you choose to, you can also add a `scale` and `shift` parameter to your data parameters. 
The data will be read into a `[0, 1]` range, so the `scale` and `shift` parameters can help you move it into a different rante. By default, these are set to `scale=2` and `shift=-1` to get data in the `[-1, 1]` range. Since generative adversarial networks are sensitive to train, it is not recommended to deviate from this range.
```

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
    val_batch_size: 16
    scale: 1
    shift: 0
```

- `solver.root_dir`: this is an (absolute) path to the directory where you want to store the results. The checkpoints, logs, and other training artifacts will be stored there. 
- `validation_config` defines how we will use the classifier to validate the quality of our conversion network. 
    - `classifier_checkpoint` points to the path where your torchscript classifier is. This should be the path you decided on [when you prepared your classifier](classifier.md)
    - `scale` and `shift` parameters are used to (optionally) change the range of data before passing it to the classifier. The default 
```{note}
Ideally, your classifier and your conversion network should expect data in the same range.

If that is not the case, you will need to add the `scale` and `shift` parameters to your `validation_config`. 
Any image passed to the classifier will first be scaled and then shifted `scale * x + shift` before.

For example: the conversion network creates data in the `[-1, 1]` range, so if your classifier wants data, so set: 
- `scale: 0.5`
- `shift: 0.5` 
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
This script will read the arguments from the configuration file you have just written, and begin training a {term}`conversion network` to convert your images from one class to another.

The training script will also begin a run on [Weights and Biases](https://wandb.ai).
Connect to your account there to follow the run.

## Output
The QuAC outputs will be organized in the `solver.root_dir` that you defined in your configuration. 
After training, that directory should look something like this: 
```{code-block} bash
<solver.root_directory>/
└── checkpoints/
    ├── 005000_nets_ema.ckpt
    ├── 010000_nets_ema.ckpt
    ├── ...
    └── 050000_nets_ema.ckpt
```