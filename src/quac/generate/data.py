from PIL import Image
from pathlib import Path
import torch


class LabelFreePngFolder(torch.utils.data.Dataset):
    # TODO Move to quac.data
    """Get all images in a folder, no subfolders, no labels."""

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.samples = [
            path
            for path in self.root.iterdir()
            if path.is_file()
            and path.name.endswith(".png")
            or path.name.endswith(".jpg")
        ]
        assert len(self.samples) > 0, f"No images found in {self.root}."

    def load_image(self, path):
        return Image.open(path)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path.name

    def __len__(self):
        return len(self.samples)
