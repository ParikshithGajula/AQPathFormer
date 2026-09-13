"""
Adaptive Quantum Multi-Head Attention for AQPathFormer

Implements patch-conditioned quantum measurement allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class AdaptiveQuantumAttention(nn.Module):
    """
    Adaptive Quantum Multi-Head Attention.
    
    Key innovation: Attention weights determine quantum measurement budget per patch.
    Patches with higher attention get more quantum shots for more precise expectation values.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_qubits: int = 4,
        total_shots: int = 1000,
        min_shots_per_patch: int = 10,
        dropout: float = 0.1,
        qkv_bias: bool = True,
        adaptive_mode: str = 'shots',  # 'shots', 'weights', 'hybrid'
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_qubits: Number of qubits in quantum encoder
            total_shots: Total measurement shots to distribute
            min_shots_per_patch: Minimum shots allocated to any patch
            dropout: Dropout rate
            qkv_bias: Use bias in QKV projections
            adaptive_mode: 'shots' (allocate shots), 'weights' (weight expectation values), 'hybrid'
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_qubits = num_qubits
        self.total_shots = total_shots
        self.min_shots_per_patch = min_shots_per_patch
        self.adaptive_mode = adaptive_mode
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # QKV projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Quantum encoder (placeholder - will be set externally)
        self.quantum_encoder = None
        
        # Shot allocation network (for adaptive_mode='shots')
        self.shot_allocator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Softplus(),  # Ensure positive
        )
    
    def set_quantum_encoder(self, encoder):
        """Set the quantum encoder to use for attention."""
        self.quantum_encoder = encoder
    
    def forward(
        self,
        x: torch.Tensor,
        quantum_features: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, embed_dim) - classical patch embeddings
            quantum_features: (batch, seq_len, num_qubits) - pre-computed quantum features
            return_attention: If True, return attention weights
        Returns:
            output: (batch, seq_len, embed_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len) if return_attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Standard attention
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x_attn = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x_attn = self.proj(x_attn)
        
        # If quantum features provided, apply adaptive quantum processing
        if quantum_features is not None and self.quantum_encoder is not None:
            x_attn = self._apply_adaptive_quantum(
                x_attn, quantum_features, attn
            )
        
        if return_attention:
            return x_attn, attn
        return x_attn, None
    
    def _apply_adaptive_quantum(
        self,
        x: torch.Tensor,
        quantum_features: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply adaptive quantum processing based on attention weights.
        
        Args:
            x: (batch, seq_len, embed_dim) - attended features
            quantum_features: (batch, num_qubits) or (batch, seq_len, num_qubits) - quantum expectation values
            attn_weights: (batch, num_heads, seq_len, seq_len) - attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Handle quantum features shape: if per-image, broadcast to all patches
        if quantum_features.dim() == 2:
            # (batch, num_qubits) -> (batch, seq_len, num_qubits)
            quantum_features = quantum_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Average attention over heads to get patch importance
        patch_importance = attn_weights.mean(dim=1).mean(dim=1)  # (batch, seq_len)
        
        if self.adaptive_mode == 'shots':
            # Allocate shots based on importance
            # Higher importance -> more shots -> more precise expectation
            # This is simulated by weighting the quantum features
            shot_weights = self._allocate_shots(patch_importance)
            # Weight quantum features by shot allocation
            quantum_features = quantum_features * shot_weights.unsqueeze(-1)
            
        elif self.adaptive_mode == 'weights':
            # Weight quantum features directly by attention
            quantum_features = quantum_features * patch_importance.unsqueeze(-1)
            
        elif self.adaptive_mode == 'hybrid':
            # Combine both
            shot_weights = self._allocate_shots(patch_importance)
            quantum_features = quantum_features * shot_weights.unsqueeze(-1) * patch_importance.unsqueeze(-1)
        
        # Concatenate classical and weighted quantum features
        combined = torch.cat([x, quantum_features], dim=-1)
        
        # Project back to embed_dim
        if not hasattr(self, 'quantum_projection'):
            self.quantum_projection = nn.Linear(
                self.embed_dim + self.num_qubits, self.embed_dim
            ).to(x.device)
        
        return self.quantum_projection(combined)
    
    def _allocate_shots(self, importance: torch.Tensor) -> torch.Tensor:
        """Allocate measurement shots based on patch importance."""
        # Normalize importance to [0, 1]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        # Compute shots per patch
        # Minimum shots for all patches
        base_shots = self.min_shots_per_patch
        remaining_shots = self.total_shots - base_shots * importance.shape[-1]
        
        if remaining_shots > 0:
            # Distribute remaining proportionally to importance
            extra_shots = importance * remaining_shots / (importance.sum(dim=-1, keepdim=True) + 1e-8)
            shots = base_shots + extra_shots
        else:
            shots = torch.full_like(importance, base_shots)
        
        # Normalize by total to get weights
        weights = shots / shots.sum(dim=-1, keepdim=True)
        
        return weights


class StandardMultiHeadAttention(nn.Module):
    """Standard multi-head attention (baseline)."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        
        if return_attention:
            return x, attn
        return x, None


def create_attention(config: Dict[str, Any]) -> nn.Module:
    """Factory to create attention module from config."""
    attn_type = config.get('type', 'standard')
    
    if attn_type == 'adaptive_quantum':
        return AdaptiveQuantumAttention(
            embed_dim=config.get('embed_dim', 768),
            num_heads=config.get('num_heads', 8),
            num_qubits=config.get('num_qubits', 4),
            total_shots=config.get('total_shots', 1000),
            min_shots_per_patch=config.get('min_shots_per_patch', 10),
            dropout=config.get('dropout', 0.1),
            qkv_bias=config.get('qkv_bias', True),
            adaptive_mode=config.get('adaptive_mode', 'shots'),
        )
    else:
        return StandardMultiHeadAttention(
            embed_dim=config.get('embed_dim', 768),
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1),
            qkv_bias=config.get('qkv_bias', True),
        )