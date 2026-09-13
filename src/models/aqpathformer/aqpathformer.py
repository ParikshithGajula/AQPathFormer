"""
AQPathFormer: Adaptive Quantum Vision Transformer for Multi-Cancer Histopathological Intelligence

Main integration module combining all components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
import yaml
from pathlib import Path

from src.models.aqpathformer.adaptive_patch_generator import AdaptivePatchGenerator, FixedPatchGenerator, create_patch_generator
from src.models.aqpathformer.quantum_patch_encoder import QuantumPatchEncoder, ClassicalProxyEncoder, create_quantum_patch_encoder, create_classical_encoder
from src.models.aqpathformer.adaptive_quantum_attention import AdaptiveQuantumAttention, StandardMultiHeadAttention, create_attention
from src.models.aqpathformer.multiscale_quantum_fusion import MultiScaleQuantumFusion, SingleScaleQuantumEncoder, create_multiscale_fusion
from src.models.aqpathformer.cross_cancer_representation import CrossCancerModule, DomainAdaptationModule, create_cross_cancer_module
from src.models.aqpathformer.noise_aware_quantum_layer import NoiseAwareQuantumLayer, NoiseScheduler, create_noise_aware_layer
from src.models.aqpathformer.hybrid_decoder import HybridDecoder, HybridDecoderWithQuantum, create_decoder
from src.models.aqpathformer.multicancer_head import MultiCancerHead, create_head


class AQPathFormer(nn.Module):
    """
    Full AQPathFormer model integrating all components.
    
    Architecture:
    Input Image
        → Patch Embedding (CNN/ViT)
        → Adaptive Patch Generator (optional)
        → Quantum Patch Encoder (or Classical)
        → Adaptive Quantum Multi-Head Attention (optional)
        → Multi-Scale Quantum Feature Fusion (optional)
        → Cross-Cancer Representation (optional)
        → Hybrid Decoder
        → Multi-Cancer Prediction Head
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        model_config = config.get('model', {})
        
        # Core dimensions
        self.embed_dim = model_config.get('embed_dim', 768)
        self.num_classes = model_config.get('num_classes', 5)
        self.image_size = config.get('preprocessing', {}).get('image_size', 224)
        self.patch_size = config.get('preprocessing', {}).get('patch_size', 16)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Component flags (for ablations)
        self.use_adaptive_patches = model_config.get('use_adaptive_patches', True)
        self.use_quantum_encoder = model_config.get('use_quantum_encoder', True)
        self.use_adaptive_attention = model_config.get('use_adaptive_attention', True)
        self.use_multiscale = model_config.get('use_multiscale', True)
        self.use_cross_cancer = model_config.get('use_cross_cancer', False)
        self.use_noise_aware = model_config.get('use_noise_aware', False)
        
        # Classical patch embedding
        self.patch_embed = self._create_patch_embedding(model_config)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 1. Adaptive Patch Generator
        if self.use_adaptive_patches:
            self.patch_generator = create_patch_generator(model_config.get('patch_generator', {}))
        else:
            self.patch_generator = FixedPatchGenerator(
                embed_dim=self.embed_dim,
                num_patches=self.num_patches,
                patch_size=self.patch_size,
                image_size=self.image_size,
            )
        
        # 2. Quantum Patch Encoder
        if self.use_quantum_encoder:
            self.quantum_encoder = create_quantum_patch_encoder(model_config.get('quantum_encoder', {}))
        else:
            self.quantum_encoder = create_classical_encoder(model_config.get('classical_encoder', {}))
        
        # 3. Adaptive Quantum Attention
        if self.use_adaptive_attention:
            self.attention = create_attention(model_config.get('attention', {}))
            # Link quantum encoder to attention
            if hasattr(self.attention, 'set_quantum_encoder'):
                self.attention.set_quantum_encoder(self.quantum_encoder)
        else:
            self.attention = StandardMultiHeadAttention(
                embed_dim=self.embed_dim,
                num_heads=model_config.get('num_heads', 8),
                dropout=model_config.get('dropout', 0.1),
            )
        
        # 4. Multi-Scale Quantum Fusion
        if self.use_multiscale:
            self.multiscale_fusion = create_multiscale_fusion(model_config.get('multiscale_fusion', {}))
        else:
            self.multiscale_fusion = SingleScaleQuantumEncoder(
                image_size=self.image_size,
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                use_quantum=self.use_quantum_encoder,
            )
        
        # 5. Cross-Cancer Representation
        if self.use_cross_cancer:
            self.cross_cancer = create_cross_cancer_module(
                model_config.get('cross_cancer', {}), 
                self  # Self as shared encoder
            )
        else:
            self.cross_cancer = None
        
        # 6. Noise-Aware Layer (integrated into quantum encoder if needed)
        self.noise_aware_layer = None
        if self.use_noise_aware and self.use_quantum_encoder:
            self.noise_aware_layer = create_noise_aware_layer(
                self.quantum_encoder, 
                model_config.get('noise_aware', {})
            )
        
        # 7. Hybrid Decoder
        self.decoder = create_decoder(model_config.get('decoder', {}))
        
        # 8. Multi-Cancer Head
        self.head = create_head(model_config.get('head', {}))
        
        # Dropout
        self.dropout = nn.Dropout(model_config.get('dropout', 0.1))
        
        # Initialize
        self._init_weights()
    
    def _create_patch_embedding(self, config: Dict[str, Any]) -> nn.Module:
        """Create patch embedding layer."""
        embed_type = config.get('patch_embed_type', 'conv')
        
        if embed_type == 'conv':
            # Standard ViT patch embedding
            return nn.Conv2d(
                3, self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        elif embed_type == 'linear':
            # Linear projection of flattened patches
            return nn.Sequential(
                nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size),
                nn.Linear(self.patch_size * self.patch_size * 3, self.embed_dim),
            )
        else:
            raise ValueError(f"Unknown patch_embed_type: {embed_type}")
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        dataset_name: str = 'default',
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass.
        
        Args:
            x: (batch, 3, H, W) input images
            dataset_name: Dataset name for cross-cancer head
            return_features: If True, return intermediate features
        Returns:
            logits: Classification logits
            features: Dict of intermediate features (if return_features)
        """
        batch_size = x.shape[0]
        features = {}
        
        # Save raw image for multiscale fusion
        raw_image = x
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        features['patch_embed'] = x
        
        # 1. Adaptive Patch Generation
        if self.use_adaptive_patches:
            x, patch_scores = self.patch_generator(x)
            features['patch_scores'] = patch_scores
        
        # 2. Multi-Scale Quantum Fusion (handles full image at multiple scales)
        if self.use_multiscale:
            fused_features, scale_features = self.multiscale_fusion(raw_image)
            features['scale_features'] = scale_features
            features['fused_features'] = fused_features
        else:
            # Single scale - just pass through
            fused_features = x[:, 0]  # CLS token
        
        features['fused'] = fused_features
        
        # 3. Adaptive Quantum Attention (if enabled)
        if self.use_adaptive_attention:
            # For now, skip quantum features in attention (broadcasting causes issues)
            # The fused_features from multiscale are per-image, not per-patch
            attended, attn_weights = self.attention(
                x, 
                quantum_features=None,  # Disable for now
                return_attention=True
            )
            features['attention_weights'] = attn_weights
            x = attended
        else:
            x = x[:, 0]  # Use CLS token
        
        # 4. Cross-Cancer Representation
        if self.use_cross_cancer and self.cross_cancer is not None:
            # This would be used for multi-dataset training
            pass
        
        # 5. Decoder
        # Only pass quantum features if they are valid (from multiscale fusion)
        decoder_quantum_features = fused_features if (self.use_multiscale and x.dim() == 2) else None
        decoded, dec_attn = self.decoder(
            x.unsqueeze(1) if x.dim() == 2 else x,
            quantum_features=decoder_quantum_features,
            return_attention=False
        )
        
        # Use first token for classification
        if decoded.dim() == 3:
            decoded = decoded[:, 0]
        
        features['decoded'] = decoded
        
        # 5. Classification Head
        logits = self.head(decoded, dataset_name)
        
        if return_features:
            return logits, features
        return logits
    
    def forward_single_dataset(
        self,
        x: torch.Tensor,
        dataset_name: str = 'default',
    ) -> torch.Tensor:
        """Forward for single dataset (simplified)."""
        return self.forward(x, dataset_name=dataset_name, return_features=False)
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get quantum circuit information."""
        info = {
            'use_quantum_encoder': self.use_quantum_encoder,
            'num_qubits': self.config.get('model', {}).get('quantum_encoder', {}).get('num_qubits', 4),
            'circuit_depth': self.config.get('model', {}).get('quantum_encoder', {}).get('circuit_depth', 2),
        }
        
        if self.use_quantum_encoder and hasattr(self.quantum_encoder, 'get_circuit_info'):
            info['quantum_encoder'] = self.quantum_encoder.get_circuit_info()
        
        if self.use_multiscale and hasattr(self.multiscale_fusion, 'scale_encoders'):
            info['multiscale_encoders'] = {}
            for scale, encoder in self.multiscale_fusion.scale_encoders.items():
                if hasattr(encoder, 'get_circuit_info'):
                    info['multiscale_encoders'][scale] = encoder.get_circuit_info()
        
        return info
    
    def set_noise_level(self, level: float):
        """Set noise level for noise-aware training."""
        if self.noise_aware_layer is not None:
            self.noise_aware_layer.set_noise_level(level)
    
    def get_noise_config(self) -> Dict[str, Any]:
        """Get current noise configuration."""
        if self.noise_aware_layer is not None:
            return self.noise_aware_layer.get_noise_config()
        return {}


class ReducedAQPathFormer(AQPathFormer):
    """Reduced AQPathFormer for ablation (fewer components)."""
    
    def __init__(self, config: Dict[str, Any]):
        # Disable optional components for reduced version
        config = config.copy()
        model_config = config.get('model', {}).copy()
        model_config.update({
            'use_adaptive_patches': False,
            'use_adaptive_attention': False,
            'use_multiscale': False,
            'use_cross_cancer': False,
            'use_noise_aware': False,
        })
        config['model'] = model_config
        
        super().__init__(config)


def create_aqpathformer(config: Dict[str, Any]) -> AQPathFormer:
    """Factory to create AQPathFormer from config."""
    return AQPathFormer(config)


def create_reduced_aqpathformer(config: Dict[str, Any]) -> ReducedAQPathFormer:
    """Factory to create reduced AQPathFormer from config."""
    return ReducedAQPathFormer(config)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_from_config(config_path: str) -> nn.Module:
    """Create model from config file."""
    config = load_config(config_path)
    return create_aqpathformer(config)


# Model configurations for experiments
AQPATHFORMER_CONFIGS = {
    'full': {
        'model': {
            'name': 'aqpathformer',
            'embed_dim': 768,
            'num_classes': 5,
            'num_heads': 8,
            'dropout': 0.1,
            'use_adaptive_patches': True,
            'use_quantum_encoder': True,
            'use_adaptive_attention': True,
            'use_multiscale': True,
            'use_cross_cancer': False,
            'use_noise_aware': False,
            'patch_generator': {
                'type': 'adaptive',
                'selection_ratio': 0.25,
            },
            'quantum_encoder': {
                'num_qubits': 4,
                'circuit_depth': 2,
                'feature_map': 'angle_embedding',
                'ansatz': 'strongly_entangling_layers',
                'classical_backbone': 'resnet18',
                'projection_dim': 4,
            },
            'attention': {
                'type': 'adaptive_quantum',
                'num_qubits': 4,
                'total_shots': 1000,
            },
            'multiscale_fusion': {
                'type': 'multiscale',
                'scales': [4, 8, 16],
                'fusion_method': 'weighted_sum',
                'projection_dim': 4,
                'num_qubits': 4,
                'circuit_depth': 2,
            },
            'decoder': {
                'type': 'hybrid',
                'num_layers': 4,
            },
            'head': {
                'dataset_configs': {
                    'lc25000': {'num_classes': 5, 'task_type': 'multiclass'},
                },
            },
        },
        'preprocessing': {
            'image_size': 224,
            'patch_size': 16,
        },
    },
    'reduced': {
        'model': {
            'name': 'aqpathformer_reduced',
            'embed_dim': 768,
            'num_classes': 5,
            'num_heads': 8,
            'dropout': 0.1,
            'use_adaptive_patches': False,
            'use_quantum_encoder': True,
            'use_adaptive_attention': False,
            'use_multiscale': False,
            'use_cross_cancer': False,
            'use_noise_aware': False,
            'quantum_encoder': {
                'num_qubits': 4,
                'circuit_depth': 2,
            },
            'multiscale_fusion': {
                'type': 'single',
            },
            'decoder': {
                'type': 'hybrid',
                'num_layers': 2,
            },
            'head': {
                'dataset_configs': {
                    'lc25000': {'num_classes': 5, 'task_type': 'multiclass'},
                },
            },
        },
        'preprocessing': {
            'image_size': 224,
            'patch_size': 16,
        },
    },
    'fixed_patches': {
        'model': {
            'name': 'aqpathformer_fixed',
            'embed_dim': 768,
            'num_classes': 5,
            'num_heads': 8,
            'dropout': 0.1,
            'use_adaptive_patches': False,
            'use_quantum_encoder': True,
            'use_adaptive_attention': True,
            'use_multiscale': True,
            'quantum_encoder': {'num_qubits': 4, 'circuit_depth': 2, 'projection_dim': 4},
            'attention': {'type': 'adaptive_quantum', 'num_qubits': 4},
            'multiscale_fusion': {'type': 'multiscale', 'scales': [4, 8, 16], 'projection_dim': 4},
            'decoder': {'type': 'hybrid', 'num_layers': 4},
            'head': {'dataset_configs': {'lc25000': {'num_classes': 5}}},
        },
        'preprocessing': {'image_size': 224, 'patch_size': 16},
    },
    'classical_encoder': {
        'model': {
            'name': 'aqpathformer_classical_enc',
            'embed_dim': 768,
            'num_classes': 5,
            'num_heads': 8,
            'dropout': 0.1,
            'use_adaptive_patches': True,
            'use_quantum_encoder': False,  # Classical encoder
            'use_adaptive_attention': True,
            'use_multiscale': True,
            'quantum_encoder': {'num_qubits': 4, 'circuit_depth': 2, 'projection_dim': 4},
            'classical_encoder': {'output_dim': 16, 'classical_backbone': 'resnet18'},
            'attention': {'type': 'adaptive_quantum', 'num_qubits': 4},
            'multiscale_fusion': {'type': 'multiscale', 'scales': [4, 8, 16], 'use_quantum': False},
            'decoder': {'type': 'hybrid', 'num_layers': 4},
            'head': {'dataset_configs': {'lc25000': {'num_classes': 5}}},
        },
        'preprocessing': {'image_size': 224, 'patch_size': 16},
    },
}