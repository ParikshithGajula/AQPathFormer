"""
Baseline Models for AQPathFormer

Classical baseline models from timm.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Any, Optional


def create_model(
    model_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """Create baseline model from timm."""
    
    # Model name mapping
    model_map = {
        'resnet50': 'resnet50',
        'resnet18': 'resnet18',
        'vit': 'vit_base_patch16_224',
        'vit_small': 'vit_small_patch16_224',
        'swin': 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'convnext': 'convnext_tiny',
        'convnext_small': 'convnext_small',
        'efficientnet': 'efficientnet_b0',
        'efficientnet_b1': 'efficientnet_b1',
    }
    
    timm_name = model_map.get(model_name.lower(), model_name)
    
    # Create model
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=kwargs.get('dropout', 0.1),
        drop_path_rate=kwargs.get('drop_path', 0.1),
    )
    
    return model


# Specific model creation functions
def create_resnet50(num_classes: int = 5, pretrained: bool = True, **kwargs) -> nn.Module:
    return create_model('resnet50', num_classes, pretrained, **kwargs)

def create_vit(num_classes: int = 5, pretrained: bool = True, **kwargs) -> nn.Module:
    return create_model('vit', num_classes, pretrained, **kwargs)

def create_swin(num_classes: int = 5, pretrained: bool = True, **kwargs) -> nn.Module:
    return create_model('swin', num_classes, pretrained, **kwargs)

def create_convnext(num_classes: int = 5, pretrained: bool = True, **kwargs) -> nn.Module:
    return create_model('convnext', num_classes, pretrained, **kwargs)

def create_efficientnet(num_classes: int = 5, pretrained: bool = True, **kwargs) -> nn.Module:
    return create_model('efficientnet', num_classes, pretrained, **kwargs)


# Config-based creation
BASELINE_CONFIGS = {
    'resnet50': {
        'model_name': 'resnet50',
        'pretrained': True,
    },
    'vit': {
        'model_name': 'vit',
        'pretrained': True,
    },
    'swin': {
        'model_name': 'swin',
        'pretrained': True,
    },
    'convnext': {
        'model_name': 'convnext',
        'pretrained': True,
    },
    'efficientnet': {
        'model_name': 'efficientnet',
        'pretrained': True,
    },
}


def get_baseline_config(model_name: str) -> Dict[str, Any]:
    """Get default config for baseline model."""
    return BASELINE_CONFIGS.get(model_name.lower(), {'model_name': model_name})