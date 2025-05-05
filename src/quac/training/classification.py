import torch


class ClassifierWrapper(torch.nn.Module):
    """
    Wraps a torchscript model, and applies scale and shift in forward.

    Parameters
    ----------
    model: :class:`torch.jit.ScriptModule`
        A loaded torchscript model.
    scale: float
        Scale factor. Defaults to 1. Applied to the data before passing it to the model.
    shift: float
        Shift factor. Defaults to 0. Applied to the data after the scale.
    """

    def __init__(
        self,
        model,
        scale=1,
        shift=0,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        """
        Scales and shifts data, then runs it through the model.

        Scale is applied first.
        """
        x = self.scale * x + self.shift
        return self.model(x)
