"""
Cross-Cancer Representation Learning Module for AQPathFormer

Enables sharing representations across cancer types/datasets with dataset-specific heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict


class CrossCancerModule(nn.Module):
    """
    Cross-cancer representation learning with shared encoder and dataset-specific heads.
    
    Supports:
    - Joint multi-dataset training
    - Domain-specific feature alignment
    - Leave-one-dataset-out evaluation
    """
    
    def __init__(
        self,
        shared_encoder: nn.Module,
        dataset_configs: Dict[str, Dict[str, Any]],
        alignment_loss_weight: float = 0.1,
        alignment_method: str = 'mmd',  # 'mmd', 'coral', 'none'
    ):
        """
        Args:
            shared_encoder: Shared backbone encoder
            dataset_configs: Dict mapping dataset_name -> {'num_classes': int, 'head_config': dict}
            alignment_loss_weight: Weight for domain alignment loss
            alignment_method: Domain alignment method
        """
        super().__init__()
        
        self.shared_encoder = shared_encoder
        self.dataset_configs = dataset_configs
        self.alignment_loss_weight = alignment_loss_weight
        self.alignment_method = alignment_method
        
        # Create dataset-specific heads
        self.heads = nn.ModuleDict()
        for dataset_name, config in dataset_configs.items():
            num_classes = config['num_classes']
            head_config = config.get('head_config', {})
            
            # Default head: MLP
            hidden_dim = head_config.get('hidden_dim', 512)
            dropout = head_config.get('dropout', 0.1)
            
            self.heads[dataset_name] = nn.Sequential(
                nn.LayerNorm(self.shared_encoder.embed_dim if hasattr(self.shared_encoder, 'embed_dim') else 768),
                nn.Linear(self.shared_encoder.embed_dim if hasattr(self.shared_encoder, 'embed_dim') else 768, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        
        # Domain alignment
        if alignment_method == 'mmd':
            self.alignment_loss = self._mmd_loss
        elif alignment_method == 'coral':
            self.alignment_loss = self._coral_loss
        else:
            self.alignment_loss = None
    
    def forward(
        self,
        x: torch.Tensor,
        dataset_name: str,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input images
            dataset_name: Name of dataset for head selection
            return_features: If True, return shared features
        Returns:
            logits: Classification logits
            features: Shared features (if return_features)
        """
        # Shared encoding
        features = self.shared_encoder(x)
        
        # Dataset-specific classification
        head = self.heads[dataset_name]
        logits = head(features)
        
        if return_features:
            return logits, features
        return logits, None
    
    def compute_alignment_loss(
        self,
        features_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute domain alignment loss between dataset features.
        
        Args:
            features_dict: {dataset_name: features}
        Returns:
            alignment_loss: Scalar loss
        """
        if self.alignment_loss is None or len(features_dict) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        dataset_names = list(features_dict.keys())
        total_loss = 0.0
        count = 0
        
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                loss = self.alignment_loss(features_dict[name1], features_dict[name2])
                total_loss += loss
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0)
    
    def _mmd_loss(self, x: torch.Tensor, y: torch.Tensor, kernel: str = 'rbf') -> torch.Tensor:
        """Maximum Mean Discrepancy loss."""
        if kernel == 'rbf':
            # RBF kernel MMD
            xx = torch.cdist(x, x).pow(2)
            yy = torch.cdist(y, y).pow(2)
            xy = torch.cdist(x, y).pow(2)
            
            # Median heuristic for bandwidth
            sigma = xy.median()
            
            k_xx = torch.exp(-xx / (2 * sigma + 1e-8))
            k_yy = torch.exp(-yy / (2 * sigma + 1e-8))
            k_xy = torch.exp(-xy / (2 * sigma + 1e-8))
            
            mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
            return mmd
        else:
            # Linear MMD
            return (x.mean(0) - y.mean(0)).pow(2).sum()
    
    def _coral_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """CORAL (Correlation Alignment) loss."""
        # Center features
        x_centered = x - x.mean(0, keepdim=True)
        y_centered = y - y.mean(0, keepdim=True)
        
        # Covariance matrices
        cov_x = (x_centered.T @ x_centered) / (x.shape[0] - 1)
        cov_y = (y_centered.T @ y_centered) / (y.shape[0] - 1)
        
        # Frobenius norm of difference
        loss = (cov_x - cov_y).pow(2).sum() / (4 * x.shape[1] ** 2)
        return loss
    
    def get_domain_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get shared encoder features (for domain adaptation)."""
        return self.shared_encoder(x)
    
    def add_dataset_head(self, dataset_name: str, num_classes: int, head_config: Optional[Dict] = None):
        """Add new dataset head dynamically."""
        if dataset_name in self.heads:
            raise ValueError(f"Dataset {dataset_name} already exists")
        
        self.dataset_configs[dataset_name] = {
            'num_classes': num_classes,
            'head_config': head_config or {},
        }
        
        hidden_dim = (head_config or {}).get('hidden_dim', 512)
        dropout = (head_config or {}).get('dropout', 0.1)
        
        self.heads[dataset_name] = nn.Sequential(
            nn.LayerNorm(self.shared_encoder.embed_dim if hasattr(self.shared_encoder, 'embed_dim') else 768),
            nn.Linear(self.shared_encoder.embed_dim if hasattr(self.shared_encoder, 'embed_dim') else 768, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )


class DomainAdaptationModule(nn.Module):
    """
    Domain adaptation module for cross-cancer transfer.
    
    Uses gradient reversal layer for domain adversarial training (DANN).
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_domains: int,
        hidden_dim: int = 512,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.grl_lambda = grl_lambda
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward with gradient reversal."""
        # Gradient reversal: multiply gradient by -lambda during backprop
        features_grl = GradientReversalFunction.apply(features, self.grl_lambda)
        return self.domain_classifier(features_grl)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal for domain adversarial training."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def create_cross_cancer_module(config: Dict[str, Any], shared_encoder: nn.Module) -> CrossCancerModule:
    """Factory to create cross-cancer module from config."""
    return CrossCancerModule(
        shared_encoder=shared_encoder,
        dataset_configs=config.get('dataset_configs', {}),
        alignment_loss_weight=config.get('alignment_loss_weight', 0.1),
        alignment_method=config.get('alignment_method', 'mmd'),
    )