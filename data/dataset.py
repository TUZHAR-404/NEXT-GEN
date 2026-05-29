"""
Dataset loading, preprocessing, and splitting for COVID-19 chest X-ray classification.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for COVID-19 chest X-ray classification.
    
    Args:
        image_paths: List of paths to images
        labels: List of class labels (0: COVID, 1: NORMAL, 2: VIRAL_PNEUMONIA)
        transform: torchvision transforms to apply
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = ["COVID", "NORMAL", "VIRAL_PNEUMONIA"]
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            image: Tensor of shape (3, 224, 224)
            label: Class index
            path: Image file path
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and convert to RGB (handle grayscale)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def load_dataset(
    data_root: Path,
    image_size: int = 224,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load and prepare COVID-19 dataset with train/val/test splits.
    
    Args:
        data_root: Root directory containing COVID/, NORMAL/, Viral Pneumonia/ folders
        image_size: Target image size (default 224 for ImageNet models)
        train_split: Proportion for training
        val_split: Proportion for validation
        test_split: Proportion for testing
        seed: Random seed for reproducibility
        batch_size: Batch size for DataLoaders
        
    Returns:
        train_loader, val_loader, test_loader, metadata_dict
    """
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    class_names = ["COVID", "NORMAL", "VIRAL_PNEUMONIA"]
    class_dirs = ["COVID", "NORMAL", "Viral Pneumonia"]
    
    image_paths = []
    labels = []
    
    logger.info(f"Loading dataset from {data_root}")
    
    # Collect all images
    for class_idx, class_dir in enumerate(class_dirs):
        class_path = data_root / class_dir
        if not class_path.exists():
            logger.warning(f"Class directory not found: {class_path}")
            continue
        
        files = sorted([f for f in class_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        logger.info(f"Class {class_names[class_idx]}: {len(files)} images")
        
        for file_path in files:
            image_paths.append(str(file_path))
            labels.append(class_idx)
    
    logger.info(f"Total images: {len(image_paths)}")
    
    # Stratified split: train+val, then val, test
    indices = np.arange(len(labels))
    labels_array = np.array(labels)
    
    # First split: train+val (85%) vs test (15%)
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_split,
        stratify=labels_array,
        random_state=seed
    )
    
    # Second split: train (70%) vs val (15%)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_split / (1 - test_split),
        stratify=labels_array[train_val_indices],
        random_state=seed
    )
    
    # Create split datasets
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    train_dataset = ChestXrayDataset(train_paths, train_labels, train_transform)
    val_dataset = ChestXrayDataset(val_paths, val_labels, val_test_transform)
    test_dataset = ChestXrayDataset(test_paths, test_labels, val_test_transform)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Compute class weights for balancing
    train_labels_array = np.array(train_labels)
    class_counts = np.bincount(train_labels_array, minlength=3)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    sample_weights = class_weights[train_labels_array]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    metadata = {
        "class_names": class_names,
        "num_classes": 3,
        "image_size": image_size,
        "class_counts": dict(zip(class_names, class_counts.tolist())),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_paths": train_paths,
        "val_paths": val_paths,
        "test_paths": test_paths,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
    }
    
    return train_loader, val_loader, test_loader, metadata
