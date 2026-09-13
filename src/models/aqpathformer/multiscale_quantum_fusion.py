"""
Multi-Scale Quantum Feature Fusion for AQPathFormer

Fuses quantum features from multiple image scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from src.models.aqpathformer.quantum_patch_encoder import QuantumPatchEncoder, ClassicalProxyEncoder


class MultiScaleQuantumFusion(nn.Module):
    """
    Multi-scale quantum feature fusion.
    
    Extracts features at multiple scales, processes each through quantum encoder,
    then fuses with learnable weights.
    """
    
    def __init__(
        self,
        scales: List[int] = [4, 8, 16],  # Downsample factors
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_qubits: int = 4,
        circuit_depth: int = 2,
        feature_map: str = 'angle_embedding',
        ansatz: str = 'strongly_entangling_layers',
        classical_backbone: str = 'resnet18',
        fusion_method: str = 'weighted_sum',  # 'weighted_sum', 'concat', 'attention'
        use_quantum: bool = True,
        dropout: float = 0.1,
        projection_dim: int = 16,  # Quantum output dimension (2^num_qubits)
    ):
        """
        Args:
            scales: Downsample factors (e.g., 4 = 224/4 = 56x56)
            image_size: Input image size
            patch_size: Patch size for quantum encoder
            embed_dim: Output embedding dimension
            num_qubits: Qubits per quantum encoder
            circuit_depth: Variational layers
            feature_map: Quantum feature map type
            ansatz: Variational ansatz type
            classical_backbone: Shared or per-scale backbone
            fusion_method: How to fuse multi-scale features
            use_quantum: Use quantum encoders
            dropout: Dropout rate
            projection_dim: Quantum output dimension (2^num_qubits)
        """
        super().__init__()
        
        self.scales = scales
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.fusion_method = fusion_method
        self.num_scales = len(scales)
        
        # Create quantum encoder for each scale
        self.scale_encoders = nn.ModuleDict()
        
        # Determine output dimension for each encoder
        # Quantum encoders output projection_dim, classical encoders output embed_dim
        if use_quantum:
            encoder_output_dim = projection_dim
        else:
            encoder_output_dim = embed_dim
        
        for scale in scales:
            scale_size = image_size // scale
            num_patches = (scale_size // patch_size) ** 2
            
            encoder_config = {
                'in_channels': 3,
                'num_qubits': num_qubits,
                'circuit_depth': circuit_depth,
                'feature_map': feature_map,
                'ansatz': ansatz,
                'classical_backbone': classical_backbone,
                'projection_dim': projection_dim,
                'use_quantum': use_quantum,
            }
            
            if use_quantum:
                self.scale_encoders[str(scale)] = QuantumPatchEncoder(**encoder_config)
            else:
                self.scale_encoders[str(scale)] = ClassicalProxyEncoder(
                    in_channels=3,
                    output_dim=encoder_output_dim,
                    classical_backbone=classical_backbone,
                )
        
        # Fusion module
        # Encoder output dimension depends on use_quantum
        if use_quantum:
            fusion_input_dim = projection_dim
        else:
            fusion_input_dim = encoder_output_dim
        
        # Fusion module
        if fusion_method == 'weighted_sum':
            # Learnable scale weights
            self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
            self.fusion_proj = nn.Linear(fusion_input_dim, embed_dim)
        elif fusion_method == 'concat':
            self.fusion_proj = nn.Linear(fusion_input_dim * self.num_scales, embed_dim)
        elif fusion_method == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                fusion_input_dim, num_heads=4, dropout=dropout, batch_first=True
            )
            self.fusion_proj = nn.Linear(fusion_input_dim, embed_dim)
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (batch, 3, image_size, image_size)
        Returns:
            fused_features: (batch, embed_dim)
            scale_features: Dict of scale -> features
        """
        batch_size = x.shape[0]
        scale_features = {}
        
        # Extract features at each scale
        for scale in self.scales:
            scale_size = self.image_size // scale
            
            # Downsample
            x_scaled = F.interpolate(
                x, size=(scale_size, scale_size), 
                mode='bilinear', align_corners=False
            )
            
            # Encode
            encoder = self.scale_encoders[str(scale)]
            features = encoder(x_scaled)  # (batch, embed_dim)
            
            scale_features[f'scale_{scale}'] = features
        
        # Fuse features
        if self.fusion_method == 'weighted_sum':
            # Weighted sum with learnable weights
            weights = F.softmax(self.scale_weights, dim=0)
            fused = sum(
                weights[i] * scale_features[f'scale_{scale}']
                for i, scale in enumerate(self.scales)
            )
            fused = self.fusion_proj(fused)
            
        elif self.fusion_method == 'concat':
            # Concatenate and project
            concat_features = torch.cat(
                [scale_features[f'scale_{scale}'] for scale in self.scales], dim=-1
            )
            fused = self.fusion_proj(concat_features)
            
        elif self.fusion_method == 'attention':
            # Self-attention over scale features
            stacked = torch.stack(
                [scale_features[f'scale_{scale}'] for scale in self.scales], dim=1
            )  # (batch, num_scales, embed_dim)
            
            # Self-attention
            attn_out, _ = self.fusion_attention(stacked, stacked, stacked)
            fused = attn_out.mean(dim=1)  # Average over scales
            fused = self.fusion_proj(fused)
        
        fused = self.dropout(fused)
        fused = self.norm(fused)
        
        return fused, scale_features


class SingleScaleQuantumEncoder(nn.Module):
    """Single-scale quantum encoder (baseline for ablation)."""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_qubits: int = 4,
        circuit_depth: int = 2,
        feature_map: str = 'angle_embedding',
        ansatz: str = 'strongly_entangling_layers',
        classical_backbone: str = 'resnet18',
        use_quantum: bool = True,
        projection_dim: int = 16,
    ):
        super().__init__()
        
        encoder_config = {
            'in_channels': 3,
            'num_qubits': num_qubits,
            'circuit_depth': circuit_depth,
            'feature_map': feature_map,
            'ansatz': ansatz,
            'classical_backbone': classical_backbone,
            'projection_dim': projection_dim,
            'use_quantum': use_quantum,
        }
        
        if use_quantum:
            self.encoder = QuantumPatchEncoder(**encoder_config)
        else:
            self.encoder = ClassicalProxyEncoder(
                in_channels=3,
                output_dim=embed_dim,
                classical_backbone=classical_backbone,
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.encoder(x)
        return features, {'scale_1': features}


def create_multiscale_fusion(config: Dict[str, Any]) -> nn.Module:
    """Factory to create multi-scale fusion from config."""
    fusion_type = config.get('type', 'multiscale')
    
    if fusion_type == 'multiscale':
        return MultiScaleQuantumFusion(
            scales=config.get('scales', [4, 8, 16]),
            image_size=config.get('image_size', 224),
            patch_size=config.get('patch_size', 16),
            embed_dim=config.get('embed_dim', 768),
            num_qubits=config.get('num_qubits', 4),
            circuit_depth=config.get('circuit_depth', 2),
            feature_map=config.get('feature_map', 'angle_embedding'),
            ansatz=config.get('ansatz', 'strongly_entangling_layers'),
            classical_backbone=config.get('classical_backbone', 'resnet18'),
            fusion_method=config.get('fusion_method', 'weighted_sum'),
            use_quantum=config.get('use_quantum', True),
            dropout=config.get('dropout', 0.1),
            projection_dim=config.get('projection_dim', 16),
        )
    else:
        return SingleScaleQuantumEncoder(
            image_size=config.get('image_size', 224),
            patch_size=config.get('patch_size', 16),
            embed_dim=config.get('embed_dim', 768),
            num_qubits=config.get('num_qubits', 4),
            circuit_depth=config.get('circuit_depth', 2),
            feature_map=config.get('feature_map', 'angle_embedding'),
            ansatz=config.get('ansatz', 'strongly_entangling_layers'),
            classical_backbone=config.get('classical_backbone', 'resnet18'),
            use_quantum=config.get('use_quantum', True),
            projection_dim=config.get('projection_dim', 16),
        )