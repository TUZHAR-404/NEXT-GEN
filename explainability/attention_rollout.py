"""
Vision Transformer Attention Rollout for explainability.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

logger = logging.getLogger(__name__)


class AttentionRollout:
    """
    Attention Rollout for Vision Transformer explanations.
    
    Method: Multiply attention matrices across layers to get rollout attention,
    then apply to patch embeddings to generate class attribution map.
    """
    
    def __init__(self, model: nn.Module, attention_head_reduction: str = 'mean'):
        """
        Args:
            model: ViT model with attention maps
            attention_head_reduction: How to reduce multi-head attention ('mean', 'max')
        """
        self.model = model
        self.attention_head_reduction = attention_head_reduction
        self.device = next(model.parameters()).device
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate attention rollout visualization.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class (if None, use predicted)
            
        Returns:
            Attention rollout map (H, W) in [0, 1]
        """
        self.model.eval()
        batch_size, _, img_h, img_w = input_tensor.shape
        
        # Forward pass with attention capture
        with torch.no_grad():
            output, attention_maps = self.model.forward_with_attention(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Compute rollout
        rollout = self._compute_rollout(attention_maps)  # (num_patches, num_patches)
        
        # Reshape to spatial dimensions
        # ViT has 14x14 patches for 224x224 image (patch size 16)
        num_patches = int(np.sqrt(rollout.shape[0]))
        
        # Get class token attention (first token)
        class_attention = rollout[0, 1:]  # Exclude class token, keep patch tokens
        class_attention = class_attention.reshape(num_patches, num_patches)
        
        # Normalize
        class_attention = np.maximum(class_attention.cpu().numpy(), 0)
        class_attention = class_attention / (class_attention.max() + 1e-8)
        
        # Upsample to input size
        class_attention_up = np.array(Image.fromarray(
            (class_attention * 255).astype(np.uint8)
        ).resize((img_w, img_h), Image.BILINEAR)).astype(np.float32) / 255.0
        
        return class_attention_up
    
    def _compute_rollout(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute rollout by multiplying attention matrices across layers.
        
        Args:
            attention_maps: List of attention tensors (num_heads, seq_len, seq_len)
                            from each transformer block
            
        Returns:
            Rollout attention (seq_len, seq_len)
        """
        # Initialize with identity
        rollout = torch.eye(attention_maps[0].shape[-1]).to(attention_maps[0].device)
        
        for attention in attention_maps:
            # Average over heads if multi-head
            if attention.dim() == 3:
                attention = attention.mean(dim=0)
            
            # Skip highest eigenvalues as suggested in paper
            attention = attention + torch.eye(attention.shape[-1]).to(attention.device)
            attention = attention / attention.sum(dim=-1, keepdim=True)
            
            # Matrix multiplication
            rollout = torch.einsum('ij,jk->ik', rollout, attention)
        
        return rollout


class TransformerInterpretabilityBase:
    """Base class for Transformer interpretability methods."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device


def get_vit_attention_maps(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_indices: Optional[List[int]] = None
) -> List[np.ndarray]:
    """
    Extract attention maps from specific ViT layers.
    
    Args:
        model: ViT model
        input_tensor: Input image tensor (1, C, H, W)
        layer_indices: Indices of layers to extract (if None, use all)
        
    Returns:
        List of attention maps (H, W) from each layer
    """
    model.eval()
    
    with torch.no_grad():
        output, attention_maps = model.forward_with_attention(input_tensor)
    
    num_patches_side = 14  # For 224x224 with patch size 16
    attention_visualizations = []
    
    for i, attn in enumerate(attention_maps):
        if layer_indices and i not in layer_indices:
            continue
        
        # Average over heads
        if attn.dim() == 3:
            attn_mean = attn.mean(dim=0)
        else:
            attn_mean = attn
        
        # Get class token attention
        class_attn = attn_mean[0, 1:].numpy()
        class_attn = class_attn.reshape(num_patches_side, num_patches_side)
        
        # Normalize
        class_attn = np.maximum(class_attn, 0)
        class_attn = class_attn / (class_attn.max() + 1e-8)
        
        attention_visualizations.append(class_attn)
    
    return attention_visualizations


def overlay_attention_on_image(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = 'viridis'
) -> np.ndarray:
    """
    Overlay attention map on original image.
    
    Args:
        image: Original image (H, W, 3) in [0, 1] or [0, 255]
        attention_map: Attention map (H, W) in [0, 1]
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
    attention_colored = cmap(attention_map)[:, :, :3]  # Remove alpha channel
    
    # Blend
    blended = (1 - alpha) * image + alpha * attention_colored
    
    return (blended * 255).astype(np.uint8)


def generate_attention_rollout_visualization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_idx: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Generate Attention Rollout visualization.
    
    Args:
        model: ViT model
        input_tensor: Preprocessed input tensor (1, 3, H, W)
        original_image: Original image as numpy array (H, W, 3)
        class_idx: Target class (if None, use predicted)
        
    Returns:
        Visualization image, predicted class index
    """
    rollout_generator = AttentionRollout(model)
    
    # Generate rollout map
    rollout_map = rollout_generator(input_tensor, class_idx)
    
    # Get predicted class
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
    
    # Overlay
    visualization = overlay_attention_on_image(original_image, rollout_map, alpha=0.5, colormap_name='jet')
    
    return visualization, pred_class
