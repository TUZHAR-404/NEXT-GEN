"""
Visualization utilities for saliency maps and comparison plots.
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
from PIL import Image
import torch

logger = logging.getLogger(__name__)


def create_saliency_grid(
    images: List[np.ndarray],
    saliency_maps: List[np.ndarray],
    titles: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create grid of images with overlaid saliency maps.
    
    Args:
        images: List of original images (H, W, 3)
        saliency_maps: List of saliency maps (H, W)
        titles: Optional titles for each subplot
        save_path: Path to save figure
        dpi: DPI for saving
        
    Returns:
        Figure and numpy array of visualization
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    if num_images == 1:
        axes = [axes]
    
    cmap = get_cmap('jet')
    
    for idx, (img, sal, ax) in enumerate(zip(images, saliency_maps, axes)):
        # Normalize image if needed
        if img.max() > 1:
            img = img / 255.0
        
        # Create overlay
        sal_colored = cmap(sal)[:, :, :3]
        overlay = 0.5 * img + 0.5 * sal_colored
        
        ax.imshow(overlay)
        ax.axis('off')
        
        if titles:
            ax.set_title(titles[idx], fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved saliency grid to {save_path}")
    
    fig.canvas.draw()
    result = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result = result.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return fig, result


def create_comparison_plot(
    base_image: np.ndarray,
    method_visualizations: dict,
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create side-by-side comparison of different XAI methods.
    
    Args:
        base_image: Original image (H, W, 3)
        method_visualizations: Dict of {method_name: visualization}
        save_path: Path to save
        dpi: DPI for saving
        
    Returns:
        Figure object
    """
    num_methods = len(method_visualizations) + 1
    fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 5))
    
    # Original image
    axes[0].imshow(base_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Method visualizations
    for idx, (method_name, viz) in enumerate(method_visualizations.items(), start=1):
        axes[idx].imshow(viz)
        axes[idx].set_title(method_name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    return fig


def plot_perturbation_curves(
    curves: dict,
    method_names: List[str],
    metric_type: str = 'insertion',
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot perturbation curves for multiple methods.
    
    Args:
        curves: Dict of {method_name: scores_array}
        method_names: List of method names
        metric_type: Type of metric ('insertion', 'deletion', 'aopc')
        save_path: Path to save
        dpi: DPI for saving
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in method_names:
        if method in curves:
            scores = curves[method]
            x = np.linspace(0, 100, len(scores))
            ax.plot(x, scores, marker='o', label=method, linewidth=2)
    
    ax.set_xlabel('Perturbation %', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Confidence', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_type.capitalize()} Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved perturbation curves to {save_path}")
    
    return fig


def create_metrics_heatmap(
    metrics_array: np.ndarray,
    method_names: List[str],
    metric_names: List[str],
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create heatmap of metrics across methods.
    
    Args:
        metrics_array: Array of shape (num_methods, num_metrics)
        method_names: List of method names
        metric_names: List of metric names
        save_path: Path to save
        dpi: DPI for saving
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(metrics_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticklabels(method_names)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(method_names)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{metrics_array[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('XAI Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved metrics heatmap to {save_path}")
    
    return fig


def create_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: List of class names
        save_path: Path to save
        dpi: DPI for saving
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=12, fontweight='bold')
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    return fig


def save_saliency_overlay(
    image: np.ndarray,
    saliency_map: np.ndarray,
    save_path: Path,
    colormap: str = 'jet',
    alpha: float = 0.5,
    dpi: int = 300
) -> None:
    """
    Save saliency map overlaid on original image.
    
    Args:
        image: Original image (H, W, 3)
        saliency_map: Saliency map (H, W)
        save_path: Output path
        colormap: Colormap name
        alpha: Blending factor
        dpi: DPI for saving
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Normalize image
    if image.max() > 1:
        image = image / 255.0
    
    # Create overlay
    cmap = get_cmap(colormap)
    sal_colored = cmap(saliency_map)[:, :, :3]
    overlay = (1 - alpha) * image + alpha * sal_colored
    
    ax.imshow(overlay)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved saliency overlay to {save_path}")
    plt.close(fig)
