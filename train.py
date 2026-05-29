"""
Full training loop for ResNet-50 and Vision Transformer models.
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import model and data classes
from models.resnet_model import create_resnet_model
from models.vit_model import create_vit_model
from data.dataset import load_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    'data_root': Path('aims-dtu/Covid-19 dataset'),
    'image_size': 224,
    'batch_size': 32,
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    
    # Training
    'num_epochs': 25,
    'learning_rate': 1e-4,
    'weight_decay': 1e-2,
    'label_smoothing': 0.1,
    'T_max': 20,
    'warmup_epochs': 5,  # Unfreeze after 5 epochs
    
    # Model
    'num_classes': 3,
    'pretrained': True,
    
    # Random seed
    'seed': 42,
    
    # Checkpoints
    'checkpoint_dir': Path('checkpoints'),
    'results_dir': Path('results'),
}


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def create_model(model_type: str, config: dict, device: torch.device) -> nn.Module:
    """Create model based on type."""
    if model_type == 'resnet':
        model = create_resnet_model(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        )
    elif model_type == 'vit':
        model = create_vit_model(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    current_epoch: int,
    total_epochs: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {current_epoch+1}/{total_epochs} [TRAIN]")
    
    for batch_idx, batch in enumerate(progress_bar):
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {current_epoch+1}/{total_epochs} - Train Loss: {avg_loss:.4f}")
    
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    current_epoch: int,
    total_epochs: int
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(val_loader, desc=f"Epoch {current_epoch+1}/{total_epochs} [VAL]")
    
    with torch.no_grad():
        for batch in progress_bar:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    logger.info(f"Epoch {current_epoch+1}/{total_epochs} - Val Loss: {avg_loss:.4f} | "
                f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list
) -> dict:
    """Test model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch in progress_bar:
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    logger.info(f"Test Results - Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


def train_model(
    model_type: str,
    config: dict = CONFIG,
    resume_from_checkpoint: str = None
) -> dict:
    """
    Complete training loop.
    
    Args:
        model_type: 'resnet' or 'vit'
        config: Configuration dictionary
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Dictionary with training history and final metrics
    """
    
    # Setup
    set_seed(config['seed'])
    device = get_device()
    config['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    config['results_dir'].mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, metadata = load_dataset(
        data_root=config['data_root'],
        image_size=config['image_size'],
        train_split=config['train_split'],
        val_split=config['val_split'],
        test_split=config['test_split'],
        seed=config['seed'],
        batch_size=config['batch_size']
    )
    
    class_names = metadata['class_names']
    logger.info(f"Classes: {class_names}")
    logger.info(f"Train: {metadata['train_samples']}, Val: {metadata['val_samples']}, Test: {metadata['test_samples']}")
    
    # Create model
    model = create_model(model_type, config, device)
    logger.info(f"Created {model_type} model")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['T_max'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_f1 = 0
    best_epoch = 0
    checkpoint_path = config['checkpoint_dir'] / f"{model_type}_best.pt"
    
    # Training loop
    for epoch in range(config['num_epochs']):
        
        # Unfreeze backbone after warmup
        if epoch == config['warmup_epochs'] and model_type == 'resnet':
            logger.info("Unfreezing backbone...")
            model.unfreeze_backbone()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, config['num_epochs'])
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, config['num_epochs'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1
            }, checkpoint_path)
            logger.info(f"Saved best model checkpoint (F1: {best_f1:.4f})")
        
        scheduler.step()
    
    # Load best model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {best_epoch+1}")
    
    # Test
    test_metrics = test(model, test_loader, device, class_names)
    
    # Save results
    results = {
        'model_type': model_type,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'history': history,
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
        'best_val_f1': best_f1,
        'metadata': metadata,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = config['results_dir'] / f"{model_type}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    return results, model, train_loader, val_loader, test_loader, metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['resnet', 'vit', 'both'], default='both',
                        help='Which model to train')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_root', type=str, default='D:/AIMS-DTU/COVID_19_dataset',
                        help='Path to dataset')
    
    args = parser.parse_args()
    
    # Update config
    config = CONFIG.copy()
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['data_root'] = Path(args.data_root)
    
    models_to_train = ['resnet', 'vit'] if args.model == 'both' else [args.model]
    
    for model_type in models_to_train:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {model_type.upper()}")
        logger.info(f"{'='*80}\n")
        
        train_model(model_type, config)
    
    logger.info("\nTraining complete!")


if __name__ == '__main__':
    main()
