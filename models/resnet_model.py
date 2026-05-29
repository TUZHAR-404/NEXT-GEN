"""
ResNet-50 model for COVID-19 chest X-ray classification.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class ResNet50Classifier(nn.Module):
    """
    Fine-tuned ResNet-50 for 3-class COVID-19 chest X-ray classification.
    
    Architecture:
        - Pretrained ResNet-50 backbone
        - Custom final FC layer: (2048, 3)
        - Target layer for explainability: layer4[-1]
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        """
        Args:
            num_classes: Number of output classes (default 3)
            pretrained: Whether to load ImageNet pretrained weights
        """
        super(ResNet50Classifier, self).__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes
        self.freeze_backbone()
        
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters except FC layer."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both logits and intermediate features.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            logits: Output logits (B, num_classes)
            features: Feature maps from layer4[-1] before pooling
        """
        # Forward through backbone up to layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)  # (B, 2048, 7, 7)
        
        # Average pool and flatten
        x = self.backbone.avgpool(features)
        x = torch.flatten(x, 1)
        logits = self.backbone.fc(x)
        
        return logits, features
    
    def get_target_layer(self) -> nn.Module:
        """Get target layer for Grad-CAM visualization."""
        return self.backbone.layer4[-1]


def create_resnet_model(num_classes: int = 3, pretrained: bool = True) -> ResNet50Classifier:
    """
    Create and initialize ResNet-50 model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        Initialized ResNet50Classifier
    """
    model = ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)
    logger.info(f"Created ResNet-50 model with {num_classes} classes")
    return model
