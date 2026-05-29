"""
Vision Transformer (ViT-B/16) model for COVID-19 chest X-ray classification.
"""

import logging
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class ViTClassifier(nn.Module):
    """
    Fine-tuned Vision Transformer-B/16 for 3-class COVID-19 chest X-ray classification.
    
    Uses timm library for pretrained ViT model.
    Stores attention maps during forward pass for Attention Rollout visualization.
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        """
        Args:
            num_classes: Number of output classes (default 3)
            pretrained: Whether to load ImageNet pretrained weights
        """
        super(ViTClassifier, self).__init__()
        
        # Load pretrained ViT-B/16 from timm
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        self.num_classes = num_classes
        self.attention_maps = []  # Store attention maps during forward pass
        
    def _register_attention_hooks(self) -> None:
        """Register hooks to capture attention maps from transformer blocks."""
        def create_attention_hook(layer_idx):
            def hook(module, input, output):
                # output shape: (B, num_heads, seq_len, seq_len)
                self.attention_maps.append(output.detach().cpu())
            return hook
        
        # Register hooks on all attention modules
        for block in self.model.blocks:
            block.attn.register_forward_hook(create_attention_hook(len(self.attention_maps)))
    
    def clear_attention_maps(self) -> None:
        """Clear stored attention maps."""
        self.attention_maps = []
    
    def forward(self, x: torch.Tensor, capture_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            capture_attention: Whether to capture attention maps
            
        Returns:
            Logits of shape (B, num_classes)
        """
        if capture_attention:
            self.clear_attention_maps()
            self._register_attention_hooks()
        
        return self.model(x)
    
    def get_attention_maps(self) -> List[torch.Tensor]:
        """
        Get captured attention maps.
        
        Returns:
            List of attention tensors, one per transformer block
        """
        return self.attention_maps
    
    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that returns logits and captured attention maps.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            logits: Output logits (B, num_classes)
            attention_maps: List of attention tensors from all blocks
        """
        self.clear_attention_maps()
        
        # Manually capture attention through forward hook registration
        hooks = []
        
        def create_hook():
            def hook(module, input, output):
                self.attention_maps.append(output.detach().cpu())
            return hook
        
        for block in self.model.blocks:
            h = block.attn.register_forward_hook(create_hook())
            hooks.append(h)
        
        logits = self.model(x)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        return logits, self.get_attention_maps()
    
    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate features for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of intermediate representations
        """
        features = {}
        
        # Patch embedding
        x = self.model._pos_embed(x)
        features['patch_embedding'] = x.detach().cpu()
        
        # Forward through blocks
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i % 4 == 0:  # Store every 4th block
                features[f'block_{i}'] = x.detach().cpu()
        
        # Layer norm and extract class token
        x = self.model.norm(x)
        features['final_norm'] = x.detach().cpu()
        features['class_token'] = x[:, 0, :].detach().cpu()
        
        return features


def create_vit_model(num_classes: int = 3, pretrained: bool = True) -> ViTClassifier:
    """
    Create and initialize Vision Transformer model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        Initialized ViTClassifier
    """
    model = ViTClassifier(num_classes=num_classes, pretrained=pretrained)
    logger.info(f"Created Vision Transformer model with {num_classes} classes")
    return model
