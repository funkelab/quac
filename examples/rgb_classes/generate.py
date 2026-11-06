import argparse
from pathlib import Path

import torch

from quac.data import write_image


class ColorClassifier(torch.nn.Module):
    """Detect localized color markers; output order is blue, green, red."""

    def forward(self, images):
        blue = images[:, 2, 6:12, 22:28].mean(dim=(1, 2))
        green = images[:, 1, 6:12, 4:10].mean(dim=(1, 2))
        red = images[:, 0, 22:28, 13:19].mean(dim=(1, 2))
        return 10 * (torch.stack([blue, green, red], dim=1) - 0.5)


def generate(directory: Path, count: int = 4):
    """Create RGB classes sharing a noisy disk, with a distinct color marker per class."""
    if count < 1:
        raise ValueError("count must be positive")
    generator = torch.Generator().manual_seed(17)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    markers = [("blue", 2, 6, 22), ("green", 1, 6, 4), ("red", 0, 22, 13)]
    for index in range(count):
        center = 15 + torch.randint(-2, 3, (2,), generator=generator)
        disk = (y - center[0]).square() + (x - center[1]).square() < 60
        image = 0.1 + torch.rand((3, 32, 32), generator=generator) * 0.1
        image[:, disk] += 0.3
        for name, channel, top, left in markers:
            marked = image.clone()
            marked[channel, top : top + 6, left : left + 6] = 0.95
            write_image(marked, directory / "images" / name / f"{index:03d}.tif")
    torch.jit.script(ColorClassifier()).save(str(directory / "classifier.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate three RGB image classes and a classifier."
    )
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    generate(args.directory, args.count)
