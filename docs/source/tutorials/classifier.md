# The classifier

## Training a classifier 

The purpose of QuAC is to explain the decisions of a pre-trained classifier. 
As such, you need a classifier before you can use QuAC. 

We will need the classifier to be a `pytorch` model, so we recommend using that. 
Here are some tutorials describing how to train a classifier: 
1. [From Pytorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
2. [From MONAI](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb)

```{attention}
Pay particular attention to the data normalization you use when training your classifier.
We will be chaining networks in QuAC, and incorrect data ranges at the input of any of these networks will lead to incorrect results. 

We *strongly* recommend making sure that your data lies in `[-1, 1]`.

While you can set the StarGAN to return data in different ranges (such as `[0, 1]` or `[-inf, inf]`), the hyper-parameters in QuAC have been optimized for `[-1, 1]`. 
GANs are very finnicky creatures, so you will likely have to do extensive tuning outside of the defaults.
```

## Compiling to torchscript

To avoid the need to modify the code for every new type of classifier, we convert these models to `torchscript`, which includes a description of the architecture. 

```{code-block} python
    :linenos:

    # TODO set your checkpoints
    input_checkpoint = "path/to/pytorch/model/checkpoint"
    output_checkpoint = "path/to/store/jit-compiled/checkpoint"

    model = ... # TODO create your model as you do for training

    # Load your desired checkpoint checkpoint
    model.load_state_dict(torch.load(input_checkpoint))

    # Turn your model to torch-script and save it
    torch.jit.save(torch.jit.script(model), output_checkpoint)
```

Every time QuAC requires a classifier checkpoint, you should now point it to the `output_checkpoint`.