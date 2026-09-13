"""
Hybrid Classical-Quantum Decoder for AQPathFormer

Processes fused quantum-classical features through classical transformer decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List


class HybridDecoder(nn.Module):
    """
    Hybrid classical-quantum decoder.
    
    Takes fused multi-scale quantum-classical features and processes them
    through classical transformer decoder layers for final classification.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True,
        use_cross_attention: bool = True,
        quantum_dim: int = 0,  # Additional quantum feature dimension
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout rate
            activation: 'gelu' or 'relu'
            norm_first: Use pre-norm (True) or post-norm (False)
            use_cross_attention: Use cross-attention with quantum features
            quantum_dim: Dimension of quantum features for cross-attention
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_cross_attention = use_cross_attention
        self.quantum_dim = quantum_dim
        
        # Cross-attention with quantum features
        if use_cross_attention and quantum_dim > 0:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(embed_dim)
            self.quantum_proj = nn.Linear(quantum_dim, embed_dim)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        quantum_features: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, embed_dim) - classical features
            quantum_features: (batch, num_qubits) - quantum expectation values
            return_attention: If True, return attention weights
        Returns:
            output: (batch, seq_len, embed_dim)
            attentions: List of attention weights if return_attention
        """
        attentions = [] if return_attention else None
        
        # Cross-attention with quantum features
        if self.use_cross_attention and quantum_features is not None:
            # Project quantum features to embed_dim
            q_proj = self.quantum_proj(quantum_features)  # (batch, embed_dim)
            q_proj = q_proj.unsqueeze(1)  # (batch, 1, embed_dim)
            
            # Cross-attention: classical queries, quantum keys/values
            x_cross, attn = self.cross_attention(x, q_proj, q_proj)
            x = self.cross_norm(x + x_cross)
            
            if return_attention:
                attentions.append(attn)
        
        # Decoder layers
        for layer in self.layers:
            x, attn = layer(x, return_attention=return_attention)
            if return_attention:
                attentions.append(attn)
        
        x = self.norm(x)
        
        return x, attentions


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and MLP."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True,
    ):
        super().__init__()
        
        self.norm_first = norm_first
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden = int(embed_dim * mlp_ratio)
        if activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass."""
        if self.norm_first:
            # Pre-norm
            attn_out, attn = self._sa_block(self.norm1(x), return_attention)
            x = x + attn_out
            x = x + self._mlp_block(self.norm2(x))
        else:
            # Post-norm
            attn_out, attn = self._sa_block(x, return_attention)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self._mlp_block(x))
        
        return x, attn if return_attention else None
    
    def _sa_block(self, x: torch.Tensor, return_attention: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Self-attention block."""
        attn_out, attn = self.self_attn(x, x, x, need_weights=return_attention)
        return self.dropout(attn_out), attn
    
    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        """MLP block."""
        return self.mlp(x)


class HybridDecoderWithQuantum(nn.Module):
    """
    Extended decoder that integrates quantum features at multiple levels.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        quantum_injection_layers: List[int] = [0, 2],  # Which layers get quantum injection
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.quantum_injection_layers = quantum_injection_layers
        
        # Quantum feature projections for each injection layer
        self.quantum_projections = nn.ModuleDict({
            str(layer): nn.Linear(4, embed_dim)  # 4 qubits -> embed_dim
            for layer in quantum_injection_layers
        })
        
        # Standard decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation='gelu',
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        quantum_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with quantum feature injection at specified layers."""
        
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            
            # Inject quantum features at specified layers
            if i in self.quantum_injection_layers and quantum_features is not None:
                proj = self.quantum_projections[str(i)]
                q_injected = proj(quantum_features)  # (batch, embed_dim)
                q_injected = q_injected.unsqueeze(1)  # (batch, 1, embed_dim)
                # Add to sequence (broadcast to all positions)
                x = x + q_injected
        
        x = self.norm(x)
        return x


def create_decoder(config: Dict[str, Any]) -> nn.Module:
    """Factory to create decoder from config."""
    decoder_type = config.get('type', 'hybrid')
    
    if decoder_type == 'hybrid':
        return HybridDecoder(
            embed_dim=config.get('embed_dim', 768),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'gelu'),
            norm_first=config.get('norm_first', True),
            use_cross_attention=config.get('use_cross_attention', True),
            quantum_dim=config.get('quantum_dim', 4),
        )
    elif decoder_type == 'hybrid_quantum_injection':
        return HybridDecoderWithQuantum(
            embed_dim=config.get('embed_dim', 768),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            quantum_injection_layers=config.get('quantum_injection_layers', [0, 2]),
        )
    else:
        # Standard transformer decoder
        return nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.get('embed_dim', 768),
                nhead=config.get('num_heads', 8),
                dim_feedforward=int(config.get('embed_dim', 768) * config.get('mlp_ratio', 4.0)),
                dropout=config.get('dropout', 0.1),
                activation=config.get('activation', 'gelu'),
                batch_first=True,
            ),
            num_layers=config.get('num_layers', 4),
        )