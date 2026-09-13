"""
Multi-Cancer Prediction Head for AQPathFormer

Configurable classification head supporting binary, multi-class, and multi-dataset settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import OrderedDict


class MultiCancerHead(nn.Module):
    """
    Multi-cancer prediction head with dataset-specific outputs.
    
    Supports:
    - Binary classification (per dataset)
    - Multi-class classification
    - Multi-label classification
    - Dynamic dataset addition
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        dataset_configs: Dict[str, Dict[str, Any]] = None,
        default_head: Optional[nn.Module] = None,
        shared_layers: bool = True,
        shared_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Input embedding dimension
            dataset_configs: {dataset_name: {'num_classes': int, 'task_type': 'binary'/'multiclass'/'multilabel', 'loss': 'ce'/'bce'/'focal'}}
            default_head: Default head for unknown datasets
            shared_layers: Use shared hidden layers before dataset-specific output
            shared_hidden_dim: Hidden dimension for shared layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.dataset_configs = dataset_configs or {}
        self.shared_layers = shared_layers
        
        # Shared feature extractor
        if shared_layers:
            self.shared_mlp = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, shared_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_hidden_dim, shared_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_input_dim = shared_hidden_dim
        else:
            self.shared_mlp = nn.Identity()
            head_input_dim = embed_dim
        
        # Dataset-specific output heads
        self.heads = nn.ModuleDict()
        self.task_types = {}  # 'binary', 'multiclass', 'multilabel'
        self.loss_functions = {}
        
        for dataset_name, config in self.dataset_configs.items():
            self._add_dataset_head(dataset_name, config, head_input_dim, dropout)
        
        # Default head
        self.default_head = default_head
        if default_head is not None:
            self.heads['default'] = default_head
    
    def _add_dataset_head(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        input_dim: int,
        dropout: float,
    ):
        """Add a new dataset head."""
        num_classes = config['num_classes']
        task_type = config.get('task_type', 'multiclass')
        loss_type = config.get('loss', 'ce')
        
        self.task_types[dataset_name] = task_type
        self.loss_functions[dataset_name] = loss_type
        
        # Create output layer
        if task_type == 'binary':
            # Binary: single output with sigmoid
            output_layer = nn.Linear(input_dim, 1)
        elif task_type == 'multiclass':
            # Multi-class: num_classes outputs with softmax
            output_layer = nn.Linear(input_dim, num_classes)
        elif task_type == 'multilabel':
            # Multi-label: num_classes outputs with sigmoid
            output_layer = nn.Linear(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        # Full head with optional hidden layer
        self.heads[dataset_name] = nn.Sequential(
            nn.Dropout(dropout),
            output_layer,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        dataset_name: str,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, embed_dim) or (batch, seq_len, embed_dim)
            dataset_name: Dataset to use head for
        Returns:
            logits: Classification logits
        """
        # Handle sequence input (take mean or first token)
        if x.dim() == 3:
            # Use first token (CLS-like) or mean pooling
            x = x[:, 0]  # First token
        
        # Shared feature extraction
        features = self.shared_mlp(x)
        
        # Dataset-specific head
        if dataset_name in self.heads:
            head = self.heads[dataset_name]
        elif 'default' in self.heads:
            head = self.heads['default']
        else:
            # Fallback to first available head
            head = list(self.heads.values())[0]
        
        logits = head(features)
        
        # Apply activation based on task type
        task_type = self.task_types.get(dataset_name, 'multiclass')
        if task_type == 'binary':
            logits = torch.sigmoid(logits)
        elif task_type == 'multilabel':
            logits = torch.sigmoid(logits)
        # multiclass: return raw logits (softmax applied in loss)
        
        return logits
    
    def add_dataset(
        self,
        dataset_name: str,
        num_classes: int,
        task_type: str = 'multiclass',
        loss_type: str = 'ce',
        head_config: Optional[Dict] = None,
    ):
        """Add new dataset head dynamically."""
        if dataset_name in self.heads:
            raise ValueError(f"Dataset {dataset_name} already exists")
        
        config = {
            'num_classes': num_classes,
            'task_type': task_type,
            'loss': loss_type,
        }
        if head_config:
            config.update(head_config)
        
        self.dataset_configs[dataset_name] = config
        
        input_dim = self.shared_mlp[-1].out_features if self.shared_layers else self.embed_dim
        dropout = (head_config or {}).get('dropout', 0.1)
        
        self._add_dataset_head(dataset_name, config, input_dim, dropout)
    
    def get_loss_fn(self, dataset_name: str) -> nn.Module:
        """Get appropriate loss function for dataset."""
        loss_type = self.loss_functions.get(dataset_name, 'ce')
        task_type = self.task_types.get(dataset_name, 'multiclass')
        
        if loss_type == 'ce':
            return nn.CrossEntropyLoss()
        elif loss_type == 'bce':
            if task_type == 'binary':
                return nn.BCEWithLogitsLoss()
            else:
                return nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            return FocalLoss(task_type=task_type)
        else:
            return nn.CrossEntropyLoss()
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dataset_name: str,
    ) -> torch.Tensor:
        """Compute loss for dataset."""
        loss_fn = self.get_loss_fn(dataset_name)
        task_type = self.task_types.get(dataset_name, 'multiclass')
        
        if task_type == 'binary':
            # BCE expects float targets
            targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
        elif task_type == 'multilabel':
            targets = targets.float()
        
        return loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        task_type: str = 'multiclass',
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.task_type = task_type
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type == 'multiclass':
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        elif self.task_type == 'binary':
            # BCE with logits
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        else:
            # Multi-label
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class EnsembleHead(nn.Module):
    """Ensemble of multiple heads for uncertainty estimation."""
    
    def __init__(
        self,
        base_head: MultiCancerHead,
        num_ensemble: int = 5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.base_head = base_head
        self.num_ensemble = num_ensemble
        
        # Create ensemble by adding dropout
        self.ensemble_dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        dataset_name: str,
        return_ensemble: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input features
            dataset_name: Dataset name
            return_ensemble: If True, return all ensemble predictions
        Returns:
            logits: Mean prediction (or all if return_ensemble)
            uncertainty: Optional uncertainty estimates
        """
        if not return_ensemble:
            return self.base_head(x, dataset_name)
        
        # Monte Carlo dropout ensemble
        ensemble_logits = []
        for _ in range(self.num_ensemble):
            # Enable dropout at eval time
            self.base_head.train()
            with torch.no_grad():
                logits = self.base_head(x, dataset_name)
                ensemble_logits.append(logits)
        
        ensemble_logits = torch.stack(ensemble_logits, dim=0)  # (ensemble, batch, ...)
        mean_logits = ensemble_logits.mean(dim=0)
        
        # Uncertainty: variance across ensemble
        uncertainty = ensemble_logits.var(dim=0)
        
        return mean_logits, uncertainty


def create_head(config: Dict[str, Any]) -> MultiCancerHead:
    """Factory to create multi-cancer head from config."""
    return MultiCancerHead(
        embed_dim=config.get('embed_dim', 768),
        dataset_configs=config.get('dataset_configs', {}),
        default_head=None,
        shared_layers=config.get('shared_layers', True),
        shared_hidden_dim=config.get('shared_hidden_dim', 512),
        dropout=config.get('dropout', 0.1),
    )