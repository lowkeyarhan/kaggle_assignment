"""Clip dataset: 30-frame dashcam folders -> sampled frame tensors."""
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms(size=224):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def eval_transforms(size=224, hflip=False):
    ops = [transforms.Resize((256, 256)), transforms.CenterCrop(size)]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def sample_indices(n_total=30, n_pick=8, jitter=True, rng=None):
    rng = np.random if rng is None else rng
    base = np.linspace(0, n_total - 1, n_pick)
    if jitter:
        base = base + rng.uniform(-1.0, 1.0, size=n_pick)
    return np.clip(np.round(base), 0, n_total - 1).astype(int)


class ClipDataset(Dataset):
    """split='train' needs train_labels.csv; split='test' reads ids from sample csv."""

    def __init__(self, data_root, split="train", n_frames=8, transform=None,
                 jitter=True, ids=None, seed=0):
        self.clip_dir = os.path.join(data_root, split)
        self.n_frames = n_frames
        self.transform = transform
        self.jitter = jitter
        self.rng = np.random.RandomState(seed)
        if ids is not None:
            self.df = pd.DataFrame({"clip_id": list(ids)})
        elif split == "train":
            self.df = pd.read_csv(os.path.join(data_root, "train_labels.csv"))
        else:
            self.df = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))[["clip_id"]]
        self.has_label = "label" in self.df.columns

    def __len__(self):
        return len(self.df)

    def _load_frame(self, clip_id, idx):
        p = os.path.join(self.clip_dir, clip_id, f"frame_{idx:03d}.jpg")
        return Image.open(p).convert("RGB")

    def __getitem__(self, i):
        row = self.df.iloc[i]
        clip_id = row["clip_id"]
        idxs = sample_indices(30, self.n_frames, self.jitter, self.rng)
        frames = [self.transform(self._load_frame(clip_id, j)) for j in idxs]
        import torch
        x = torch.stack(frames)  # (T, C, H, W)
        if self.has_label:
            return x, torch.tensor(float(row["label"])), clip_id
        return x, clip_id
