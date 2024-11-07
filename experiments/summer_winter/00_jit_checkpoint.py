import torch
from funlib.learn.torch.models import Vgg2D


if __name__ == "__main__":
    input_checkpoint = (
        "/nrs/funke/adjavond/projects/duplex/summer_winter/vgg_checkpoint"
    )
    output_checkpoint = (
        "/nrs/funke/adjavond/projects/duplex/summer_winter/vgg_checkpoint_jit.pt"
    )
    model = Vgg2D(
        input_size=(256, 256),
        input_fmaps=3,
        fmaps=12,
        output_classes=2,
    )

    model.load_state_dict(torch.load(input_checkpoint))
    # JIT
    torch.jit.save(torch.jit.script(model), output_checkpoint)
