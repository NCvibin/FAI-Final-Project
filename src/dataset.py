"""Step 3: PyTorch dataset over the split CSVs."""
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def resolve_data_root(data_root=None) -> Path:
    """Folder holding train/ and valid/. CSV paths are relative to it."""
    root = data_root or os.environ.get("MURA_ROOT")
    if not root:
        raise ValueError("Give --data-root /path/to/MURA-v1.1 or set the MURA_ROOT environment variable")
    root = Path(root).expanduser()
    if not (root / "train").is_dir():
        raise FileNotFoundError(f"{root} has no train/ folder: point it at the MURA-v1.1 folder")
    return root


class MuraDataset(Dataset):
    """Returns (image, label, sample_weight, index). Look up study_id/region via .df."""

    def __init__(self, csv_path: str, transform=None, regions=None, data_root=None):
        df = pd.read_csv(csv_path)
        if regions:
            df = df[df.region.isin(regions)]
        if df.empty:
            raise ValueError(f"No images left in {csv_path} for regions={regions}")
        self.df = df.reset_index(drop=True)
        self.root = resolve_data_root(data_root)
        self.transform = transform
        self.weights = torch.ones(len(self.df))

    def set_weights(self, weights):
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row.path).convert("RGB")  # grayscale -> 3 channels
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row.label, dtype=torch.float32), self.weights[i], i
