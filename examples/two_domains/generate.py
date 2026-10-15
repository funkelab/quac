import argparse
from pathlib import Path

import torch

from quac.data import write_image


class DetailClassifier(torch.nn.Module):
    """Classify the small upper-right marker; output order is marked, plain."""

    def forward(self, images):
        score = (images[:, :, 6:10, 22:26].mean(dim=(1, 2, 3)) - 0.5) * 10
        return torch.stack([score, -score], dim=1)


def generate(directory: Path, count: int = 8):
    """Generate paired domains differing only in a small bright square."""
    generator = torch.Generator().manual_seed(7)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    for index in range(count):
        center = 15 + torch.randint(-2, 3, (2,), generator=generator)
        disk = ((y - center[0]).square() + (x - center[1]).square()) < 60
        image = 0.1 + torch.rand((1, 32, 32), generator=generator) * 0.1
        image[:, disk] += 0.35
        write_image(image, directory / "images/plain" / f"{index:03d}.tif")
        marked = image.clone()
        marked[:, 6:10, 22:26] = 0.9
        write_image(marked, directory / "images/marked" / f"{index:03d}.tif")
    directory.mkdir(parents=True, exist_ok=True)
    torch.jit.script(DetailClassifier()).save(str(directory / "classifier.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate two image domains and a classifier."
    )
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    generate(args.directory, args.count)
