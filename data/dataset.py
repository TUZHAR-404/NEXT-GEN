import logging
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

def get_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(data_root / 'train', transform=train_transforms)
    val_dataset   = datasets.ImageFolder(data_root / 'val',   transform=val_test_transforms)
    test_dataset  = datasets.ImageFolder(data_root / 'test',  transform=val_test_transforms)

    class_counts = np.bincount(train_dataset.targets)
    weights = 1.0 / class_counts[train_dataset.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,   num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,   num_workers=2, pin_memory=True)

    logger.info(f"Classes: {train_dataset.classes}")
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
