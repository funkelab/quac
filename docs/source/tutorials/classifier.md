# The classifier

## Training a classifier 

The purpose of QuAC is to explain the decisions of a pre-trained classifier. 
As such, you need a classifier before you can use QuAC. 

We will need the classifier to be a `pytorch` model.
If you don't already have a classifier, here are some tutorials describing how to train one: 
1. [From Pytorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
2. [From MONAI](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb)

```{attention}
Pay particular attention to the data normalization you use when training your classifier.
We will be chaining networks in QuAC, and incorrect data ranges at the input of any of these networks will lead to incorrect results. 

We *strongly* recommend making sure that your classifier expects input data that lies in `[-1, 1]`.

While you can set the {term}`conversion network` to return data in different ranges the hyper-parameters in QuAC have been optimized for `[-1, 1]`. 
Generative adversarial networks are very finnicky creatures, so you will likely have to do extensive tuning outside of the defaults.
```

## Compiling to torchscript

To avoid the need to modify the code for every new type of classifier, you must convert your model to `torchscript`. This is a format which includes a description of the architecture. 

Modify and run the code below to do this.
```{code-block} python
:linenos:

# TODO set your checkpoint paths
input_checkpoint = "path/to/pytorch/model/checkpoint"
output_checkpoint = "path/to/store/jit-compiled/checkpoint"

model = ... # TODO create your model as you do for training

# Load your desired checkpoint checkpoint
model.load_state_dict(torch.load(input_checkpoint))

# Turn your model to torch-script and save it
torch.jit.save(torch.jit.script(model), output_checkpoint)
```

Every time QuAC requires a classifier checkpoint, you should now point it to the `output_checkpoint`.