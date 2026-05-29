"""
Attribution-based explainability methods using captum library.
"""

import logging
from typing import Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from captum.attr import IntegratedGradients, LayerGradCam

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer:
    """
    Integrated Gradients from captum for model-agnostic attribution.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None
    ):
        """
        Args:
            model: PyTorch model
            target_layer: Optional layer to target (if None, uses input gradient)
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.ig = IntegratedGradients(model)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generate Integrated Gradients attribution.
        
        Args:
            input_tensor: Input tensor (1, 3, H, W)
            class_idx: Target class (if None, use predicted)
            n_steps: Number of integration steps
            baseline: Baseline image (if None, use black image)
            
        Returns:
            Attribution map (H, W) in [0, 1]
        """
        self.model.eval()
        
        # Get predicted class if not provided
        with torch.no_grad():
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
        
        # Set baseline
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Compute IG
        attributions = self.ig.attribute(
            input_tensor,
            baselines=baseline,
            target=class_idx,
            n_steps=n_steps,
            return_convergence_delta=False
        )
        
        # Aggregate over channels
        attr_map = np.abs(attributions[0].cpu().detach().numpy()).sum(axis=0)
        
        # Normalize
        attr_map = np.maximum(attr_map, 0)
        attr_map = attr_map / (attr_map.max() + 1e-8)
        
        return attr_map


class LayerGradCamExplainer:
    """
    Layer-based Grad-CAM from captum.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to target
        """
        self.model = model
        self.target_layer = target_layer
        self.device = next(model.parameters()).device
        self.lg = LayerGradCam(model, target_layer)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Layer Grad-CAM attribution.
        
        Args:
            input_tensor: Input tensor (1, 3, H, W)
            class_idx: Target class (if None, use predicted)
            
        Returns:
            Attribution map (H, W) in [0, 1]
        """
        self.model.eval()
        
        # Get predicted class if not provided
        with torch.no_grad():
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
        
        # Compute Layer Grad-CAM
        attributions = self.lg.attribute(
            input_tensor,
            target=class_idx
        )
        
        # Aggregate
        attr_map = attributions[0].cpu().detach().numpy().mean(axis=0)
        
        # Normalize
        attr_map = np.maximum(attr_map, 0)
        attr_map = attr_map / (attr_map.max() + 1e-8)
        
        return attr_map


def overlay_attribution_on_image(
    image: np.ndarray,
    attribution_map: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = 'hot'
) -> np.ndarray:
    """
    Overlay attribution map on original image.
    
    Args:
        image: Original image (H, W, 3) in [0, 1] or [0, 255]
        attribution_map: Attribution map (H, W) in [0, 1]
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
    attr_colored = cmap(attribution_map)[:, :, :3]
    
    # Blend
    blended = (1 - alpha) * image + alpha * attr_colored
    
    return (blended * 255).astype(np.uint8)


def generate_integrated_gradients_visualization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_idx: Optional[int] = None,
    n_steps: int = 50
) -> Tuple[np.ndarray, int]:
    """
    Generate Integrated Gradients visualization.
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed input tensor (1, 3, H, W)
        original_image: Original image as numpy array (H, W, 3)
        class_idx: Target class (if None, use predicted)
        n_steps: Number of integration steps
        
    Returns:
        Visualization image, predicted class index
    """
    ig_explainer = IntegratedGradientsExplainer(model)
    
    # Generate attribution map
    attribution_map = ig_explainer(input_tensor, class_idx, n_steps=n_steps)
    
    # Get predicted class
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
    
    # Overlay
    visualization = overlay_attribution_on_image(original_image, attribution_map, alpha=0.5, colormap_name='hot')
    
    return visualization, pred_class


def generate_layer_gradcam_visualization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_layer: nn.Module,
    class_idx: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Generate Layer Grad-CAM visualization.
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed input tensor (1, 3, H, W)
        original_image: Original image as numpy array (H, W, 3)
        target_layer: Layer to target
        class_idx: Target class (if None, use predicted)
        
    Returns:
        Visualization image, predicted class index
    """
    explainer = LayerGradCamExplainer(model, target_layer)
    
    # Generate attribution map
    attribution_map = explainer(input_tensor, class_idx)
    
    # Get predicted class
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
    
    # Overlay
    visualization = overlay_attribution_on_image(original_image, attribution_map, alpha=0.5, colormap_name='hot')
    
    return visualization, pred_class
