"""
Gradient-based CAM methods for ResNet-50 explainability.
"""

import logging
from typing import Callable, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

logger = logging.getLogger(__name__)


class GradCAMBase:
    """Base class for Gradient-based CAM methods."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to generate CAM from
        """
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device
        self.gradients = None
        self.features = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.features = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate CAM for target class. To be implemented by subclasses.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            class_idx: Target class index
            
        Returns:
            CAM heatmap (H, W)
        """
        raise NotImplementedError
    
    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate CAM.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, use predicted class)
            
        Returns:
            CAM heatmap normalized to [0, 1]
        """
        batch_size, _, input_height, input_width = input_tensor.shape
        
        # Forward pass
        with torch.enable_grad():
            output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()
        
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Normalize to [0, 1]
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (input_width, input_height),
            Image.BILINEAR
        )).astype(np.float32) / 255.0
        
        return cam_resized


class GradCAM(GradCAMBase):
    """Gradient-weighted Class Activation Mapping."""
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Target class
            
        Returns:
            CAM heatmap
        """
        gradients = self.gradients[0].cpu().data.numpy()  # (C, H, W)
        features = self.features[0].cpu().data.numpy()    # (C, H, W)
        
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * features[i, :, :]
        
        return cam


class GradCAMPlusPlus(GradCAMBase):
    """Gradient-weighted Class Activation Mapping (Plus Plus)."""
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Target class
            
        Returns:
            CAM heatmap
        """
        gradients = self.gradients[0].cpu().data.numpy()  # (C, H, W)
        features = self.features[0].cpu().data.numpy()    # (C, H, W)
        
        # Compute alpha weights
        spatial_dim = gradients.shape[1:]
        second_derivatives = gradients ** 2
        third_derivatives = second_derivatives * gradients
        
        alpha = second_derivatives / (2 * third_derivatives + 1e-8)
        relu_alpha = np.maximum(alpha, 0)
        
        weights = np.sum(relu_alpha * np.maximum(gradients, 0), axis=(1, 2))
        weights = weights / (np.sum(relu_alpha, axis=(1, 2)) + 1e-8)
        
        cam = np.zeros(spatial_dim, dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i, :, :]
        
        return cam


class EigenCAM(GradCAMBase):
    """Eigenvalue-weighted Class Activation Mapping."""
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Args:
            input_tensor: Input tensor (B, C, H, W)
            class_idx: Target class
            
        Returns:
            CAM heatmap
        """
        features = self.features[0].cpu().data.numpy()  # (C, H, W)
        
        # Reshape features: (C, H*W)
        feature_matrix = features.reshape(features.shape[0], -1)
        
        # Compute covariance matrix and eigenvalues
        cov_matrix = np.cov(feature_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Use top eigenvector weighted by eigenvalue
        idx = np.argsort(eigenvalues)[-1]
        top_eigenvector = eigenvectors[:, idx]
        
        # Weight features by eigenvector
        weights = np.abs(top_eigenvector)
        weights = weights / (weights.sum() + 1e-8)
        
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i, :, :]
        
        return cam


class ScoreCAM(GradCAMBase):
    """Score-weighted Class Activation Mapping (non-gradient based)."""
    
    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Score-CAM.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index
            
        Returns:
            CAM heatmap
        """
        self.model.eval()
        batch_size, _, input_height, input_width = input_tensor.shape
        
        # Forward pass to get features
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        features = self.features[0]  # (C, H, W)
        C, H, W = features.shape
        
        # Get predicted class
        with torch.no_grad():
            output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Score each channel
        scores = []
        input_image_min = input_tensor.min()
        input_image_max = input_tensor.max()
        
        for c in range(C):
            # Normalize channel
            feature_map = features[c, :, :]
            feature_map_min = feature_map.min()
            feature_map_max = feature_map.max()
            
            if feature_map_max > feature_map_min:
                feature_map_normalized = (feature_map - feature_map_min) / (feature_map_max - feature_map_min + 1e-8)
            else:
                feature_map_normalized = torch.zeros_like(feature_map)
            
            # Upsample to input size
            feature_map_up = F.interpolate(
                feature_map_normalized.unsqueeze(0).unsqueeze(0),
                size=(input_height, input_width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Mask input
            masked_input = input_tensor[0] * feature_map_up.unsqueeze(0)
            masked_input = torch.clamp(masked_input, input_image_min, input_image_max)
            
            # Get prediction on masked input
            with torch.no_grad():
                score = self.model(masked_input.unsqueeze(0))[0, class_idx].item()
            
            scores.append(score)
        
        scores = np.array(scores)
        scores = np.maximum(scores, 0)
        scores = scores / (scores.sum() + 1e-8)
        
        # Generate CAM
        cam = np.zeros((H, W), dtype=np.float32)
        for c in range(C):
            feature_map = features[c].cpu().numpy()
            cam += scores[c] * feature_map
        
        # Normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (input_width, input_height),
            Image.BILINEAR
        )).astype(np.float32) / 255.0
        
        return cam_resized


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = 'jet'
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image (H, W, 3) in [0, 1] or [0, 255]
        heatmap: Heatmap (H, W) in [0, 1]
        alpha: Blending factor
        colormap_name: Matplotlib colormap name
        
    Returns:
        Blended image (H, W, 3) in [0, 255]
    """
    # Ensure image is in [0, 1]
    if image.max() > 1:
        image = image / 255.0
    
    # Apply colormap
    cmap = get_cmap(colormap_name)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Blend
    blended = (1 - alpha) * image + alpha * heatmap_colored
    
    return (blended * 255).astype(np.uint8)


def generate_gradcam_visualization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_idx: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
    method_name: str = "GradCAM"
) -> Tuple[np.ndarray, int]:
    """
    Generate and return Grad-CAM visualization.
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed input tensor (1, 3, H, W)
        original_image: Original image as numpy array (H, W, 3)
        class_idx: Target class (if None, use predicted)
        target_layer: Layer to visualize
        method_name: Name of method ("GradCAM", "GradCAMPlusPlus", "EigenCAM", "ScoreCAM")
        
    Returns:
        Visualization image, predicted class index
    """
    if target_layer is None:
        target_layer = model.get_target_layer()
    
    # Select method
    if method_name == "GradCAM":
        cam_generator = GradCAM(model, target_layer)
    elif method_name == "GradCAMPlusPlus":
        cam_generator = GradCAMPlusPlus(model, target_layer)
    elif method_name == "EigenCAM":
        cam_generator = EigenCAM(model, target_layer)
    elif method_name == "ScoreCAM":
        cam_generator = ScoreCAM(model, target_layer)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Generate CAM
    cam = cam_generator(input_tensor, class_idx)
    
    # Get predicted class if not provided
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
    
    # Overlay
    visualization = overlay_heatmap_on_image(original_image, cam, alpha=0.5, colormap_name='jet')
    
    return visualization, pred_class
