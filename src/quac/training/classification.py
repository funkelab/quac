import torch
from torchvision import transforms


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class ClassifierWrapper(torch.nn.Module):
    """
    This class expects a torchscript model. See [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format)
    for how to convert a model to torchscript.
    """

    def __init__(self, model_checkpoint, mean=None, std=None, assume_normalized=False):
        """Wraps a torchscript model, and applies normalization."""
        super().__init__()
        self.model = torch.jit.load(model_checkpoint)
        self.model.eval()
        self.transform = transforms.Normalize(mean, std)
        if mean is None:
            self.transform = Identity()
        self.assume_normalized = assume_normalized

    def forward(self, x, assume_normalized=False):
        """Assumes that x is between -1 and 1."""
        # TODO it would be even better if the range was between 0 and 1 so we wouldn't have to do the below
        if not self.assume_normalized and not assume_normalized:
            x = (x + 1) / 2
        x = self.transform(x)
        return self.model(x)
