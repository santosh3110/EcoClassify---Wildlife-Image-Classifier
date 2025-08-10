import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from ecoclassify import logger
from ecoclassify.utils.common import load_json, save_json


class WildlifeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label"]) if "label" in row else -1
        img_id = os.path.splitext(os.path.basename(row["filepath"]))[0]
        if self.transform:
            image = self.transform(image)
        return image, label, img_id

def compute_mean_std(df: pd.DataFrame) -> tuple:
    logger.info("📊 Computing mean and std for training images...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_images = len(df)

    for i in tqdm(range(n_images), desc="🔍 Scanning images for stats"):
        img_path = df.iloc[i]["filepath"]
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        mean += img.mean(dim=(1, 2))
        std += img.std(dim=(1, 2))

    mean /= n_images
    std /= n_images

    logger.info(f"✅ Mean: {mean.tolist()} | Std: {std.tolist()}")
    return mean.tolist(), std.tolist()

def get_transforms(mean, std):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, val_test_transform


def get_dataloaders(train_df, val_df, batch_size, log_dir, num_workers=2):
    stats_path = os.path.join(log_dir, "mean_std.json")

    if os.path.exists(stats_path):
        stats = load_json(Path(stats_path))
        mean, std = stats["mean"], stats["std"]
        logger.info(f"📁 Loaded mean and std from cache: {stats_path}")
    else:
        mean, std = compute_mean_std(train_df)
        save_json(Path(stats_path), {"mean": mean, "std": std})

    train_transform, val_transform = get_transforms(mean, std)

    train_dataset = WildlifeDataset(train_df, transform=train_transform)
    val_dataset = WildlifeDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(f"✅ Dataloaders ready | Batch size: {batch_size}")
    return train_loader, val_loader