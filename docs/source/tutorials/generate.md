# Generate images

## Select a checkpoint using run logs

The validation step that occurs regularly during training generates a set of examples and metrics. 
We can use these to choose the best model checkpoint.

You can choose your checkpoint based on the following metrics, which you will find in the `wandb` logs: 
- The *translation rate* shows how many of the StarGAN's output are classified by the pre-trained classifier as the `target` class.
- The *conversion rate* is similar, except it gives the model several tries to correctly convert an image. This is possible because the StarGAN includes some randomness.

```{note}
Generally, we choose the checkpoint with the highest average conversion rate.

Make sure that the generated images, which are also in the logs as `ema_fake_x_reference` and `ema_fake_x_latent` , look realistic at that point.
```

## Run image generation 

The `generate_images.py` script will use the model that you have chosen to try to convert your images from one class to another. 

It requires an additional set of parameters: 
- `dataset`: Which of the datasets to run the translation on. By default this will be the "test" dataset, if that does not exist it will revert to the "validation" dataset.
- `source_class`: The name of the class to take images from.
- `target_class`: The name of the class to convert images to.
- `checkpoint_iter`: The number of the checkpoint you chose above.

For example, to convert the `validation` dataset from `class_A` to `class_B` using checkpoint `50000` you would run: 
```{code-block} bash
python generate_images.py --dataset validation --source_class class_A --target_class class_B --checkpoint_iter 50000
```

If you want to have a look at the other modifiable options, run the following: 
```{code-block} bash
python generate_images.py --help
```
