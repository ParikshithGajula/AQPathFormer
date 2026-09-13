"""
Adaptive Patch Generator for AQPathFormer

Learnable patch selection based on tissue content using attention scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class AdaptivePatchGenerator(nn.Module):
    """
    Adaptive patch generator that selects top-k most informative patches.
    
    Uses attention-based scoring with Gumbel-Softmax for differentiable selection.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_patches: int = 196,
        patch_size: int = 16,
        image_size: int = 224,
        selection_ratio: float = 0.25,  # Select top 25% patches
        temperature: float = 1.0,
        hard_selection: bool = True,
        score_network: str = 'mlp',
    ):
        """
        Args:
            embed_dim: Dimension of patch embeddings
            num_patches: Total number of patches (H*W/patch_size^2)
            patch_size: Size of each patch
            image_size: Input image size
            selection_ratio: Fraction of patches to select (0-1)
            temperature: Gumbel-Softmax temperature
            hard_selection: If True, use straight-through estimator for hard top-k
            score_network: 'mlp' or 'linear' for patch scoring
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
        self.selection_ratio = selection_ratio
        self.temperature = temperature
        self.hard_selection = hard_selection
        self.num_selected = max(1, int(num_patches * selection_ratio))
        
        # Patch scoring network
        if score_network == 'mlp':
            self.score_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
            )
        else:
            self.score_net = nn.Linear(embed_dim, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for m in self.score_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        patch_embeddings: torch.Tensor,
        return_scores: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patch_embeddings: (batch, num_patches, embed_dim)
            return_scores: If True, return patch scores
        Returns:
            selected_patches: (batch, num_selected, embed_dim)
            scores: (batch, num_patches) if return_scores else None
        """
        batch_size, num_patches, embed_dim = patch_embeddings.shape
        
        # Compute patch scores
        scores = self.score_net(patch_embeddings).squeeze(-1)  # (batch, num_patches)
        scores = torch.softmax(scores, dim=-1)  # Normalize to probabilities
        
        if self.training and self.hard_selection:
            # Gumbel-Softmax for differentiable top-k selection
            selected_indices = self._gumbel_top_k(scores, self.num_selected)
            # Hard selection with straight-through estimator
            selected_patches = self._gather_patches(patch_embeddings, selected_indices)
        else:
            # Soft selection: weighted sum
            # Or hard top-k for evaluation
            if self.training:
                # Soft: use scores as weights
                selected_patches = torch.einsum('bp,bpe->bpe', scores.unsqueeze(1), patch_embeddings)
                selected_patches = selected_patches[:, :self.num_selected]
            else:
                # Hard top-k for eval
                _, topk_indices = torch.topk(scores, self.num_selected, dim=-1)
                selected_patches = self._gather_patches(patch_embeddings, topk_indices)
        
        if return_scores:
            return selected_patches, scores
        return selected_patches, None
    
    def _gumbel_top_k(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """
        Differentiable top-k using Gumbel-Softmax trick.
        
        Args:
            scores: (batch, num_patches) probabilities
            k: Number of patches to select
        Returns:
            indices: (batch, k) selected patch indices
        """
        batch_size, num_patches = scores.shape
        
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        gumbel_scores = (scores.log() + gumbel_noise) / self.temperature
        
        # Soft top-k via sequential selection
        selected_indices = []
        remaining_scores = gumbel_scores.clone()
        
        for _ in range(k):
            # Softmax to get selection probabilities
            selection_probs = F.softmax(remaining_scores, dim=-1)
            
            # Sample or take argmax
            if self.training:
                # Gumbel-Softmax sample
                selected = torch.multinomial(selection_probs, 1).squeeze(-1)
            else:
                selected = torch.argmax(selection_probs, dim=-1)
            
            selected_indices.append(selected)
            
            # Mask out selected patches
            mask = torch.zeros_like(remaining_scores)
            mask.scatter_(1, selected.unsqueeze(1), 1)
            remaining_scores = remaining_scores.masked_fill(mask.bool(), -1e9)
        
        return torch.stack(selected_indices, dim=1)  # (batch, k)
    
    def _gather_patches(self, patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather patches by indices."""
        # patches: (batch, num_patches, embed_dim)
        # indices: (batch, k)
        batch_size, k = indices.shape
        embed_dim = patches.shape[-1]
        
        # Expand indices for gathering
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        selected = torch.gather(patches, 1, indices_expanded)
        
        return selected
    
    def get_selected_patch_ratio(self) -> float:
        return self.selection_ratio


class FixedPatchGenerator(nn.Module):
    """Fixed patch generator (baseline - no selection)."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_patches: int = 196,
        patch_size: int = 16,
        image_size: int = 224,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
    
    def forward(self, patch_embeddings: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Return all patches unchanged."""
        return patch_embeddings, None
    
    def get_selected_patch_ratio(self) -> float:
        return 1.0


def create_patch_generator(config: Dict[str, Any]) -> nn.Module:
    """Factory to create patch generator from config."""
    generator_type = config.get('type', 'adaptive')
    
    if generator_type == 'adaptive':
        return AdaptivePatchGenerator(
            embed_dim=config.get('embed_dim', 768),
            num_patches=config.get('num_patches', 196),
            patch_size=config.get('patch_size', 16),
            image_size=config.get('image_size', 224),
            selection_ratio=config.get('selection_ratio', 0.25),
            temperature=config.get('temperature', 1.0),
            hard_selection=config.get('hard_selection', True),
            score_network=config.get('score_network', 'mlp'),
        )
    else:
        return FixedPatchGenerator(
            embed_dim=config.get('embed_dim', 768),
            num_patches=config.get('num_patches', 196),
            patch_size=config.get('patch_size', 16),
            image_size=config.get('image_size', 224),
        )