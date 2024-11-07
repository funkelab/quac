import torch
from funlib.learn.torch.models import Vgg2D


if __name__ == "__main__":
    input_checkpoint = "/nrs/funke/adjavond/projects/duplex/disc_a/vgg_checkpoint"
    output_checkpoint = (
        "/nrs/funke/adjavond/projects/duplex/disc_a/vgg_checkpoint_jit.pt"
    )
    model = Vgg2D(
        input_size=(128, 128),
        input_fmaps=1,
        fmaps=12,
        output_classes=2,
    )

    model.load_state_dict(torch.load(input_checkpoint))
    # JIT
    torch.jit.save(torch.jit.script(model), output_checkpoint)
