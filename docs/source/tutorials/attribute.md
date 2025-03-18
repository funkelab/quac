# Discriminative attribution from Counterfactuals

Now that we have generated images, we will refine each {term}`generated image` into a {term}`counterfactual image` using discriminative attribution.
Remember that although the conversion network is trained to keep as much of the image fixed as possible, it is not perfect.
This means that there may still be regions of the {term}`generated image` that differ from the {term}`query image` *even if they don't need to*.
Luckily, we have a classifier that can help us identify and keep only the necessary regions of change.

We will get candidate regions of change by running several discriminative attribution methods using the `run_attribution.py` script. 
It takes the following arguments, all of which are optional: 
- `dataset`: Which of the datasets to run the translation on. By default this will be the "test" dataset, if that does not exist it will revert to the "validation" dataset.
- `output`: Where to store the generated attributions. They will be further organized as sub-directories named after the different methods that we will try. By default, the attributions will go in the experiment root directory under `attributions`.
- `input_fake`: Where the generated images are. You should only set these if you used a custom `output` argument in the previous step. By default, this is in the experiment root directory under `generated_images`

The metadata regarding the real input images will be read from the configuration file, as before.

To use the `config.yaml` defaults:
```{code-block} bash
python run_attribution.py
```

You can use the following command to get help with formatting your arguments.
```{code-block} bash
python run_attribution.py -h
```

The script will create candidates using Discriminative [Integrated Gradients](https://arxiv.org/abs/1703.01365) and Discriminative [DeepLift](https://arxiv.org/abs/1704.02685) as attribution methods. 
If you look into the `attribution_directory`, you should see the results stored as `numpy` arrays.
They will be organized in the following way:
```{code-block} bash
attribution_directory/
    discriminative_ig/
        source_class/
            target_class/
                image_name.npy 
                [...]
    discriminative_deeplift/
        source_class/
            target_class/
                image_name.npy
                [...]
```
