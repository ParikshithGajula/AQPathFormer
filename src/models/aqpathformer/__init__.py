"""
AQPathFormer Models Package

Quantum-enhanced Vision Transformer components.
"""

from src.models.aqpathformer.adaptive_patch_generator import (
    AdaptivePatchGenerator,
    FixedPatchGenerator,
    create_patch_generator,
)

from src.models.aqpathformer.quantum_patch_encoder import (
    QuantumPatchEncoder,
    ClassicalProxyEncoder,
    create_quantum_patch_encoder,
    create_classical_encoder,
)

from src.models.aqpathformer.adaptive_quantum_attention import (
    AdaptiveQuantumAttention,
    StandardMultiHeadAttention,
    create_attention,
)

from src.models.aqpathformer.multiscale_quantum_fusion import (
    MultiScaleQuantumFusion,
    SingleScaleQuantumEncoder,
    create_multiscale_fusion,
)

from src.models.aqpathformer.cross_cancer_representation import (
    CrossCancerModule,
    DomainAdaptationModule,
    create_cross_cancer_module,
)

from src.models.aqpathformer.noise_aware_quantum_layer import (
    NoiseAwareQuantumLayer,
    NoiseScheduler,
    create_noise_aware_layer,
)

from src.models.aqpathformer.hybrid_decoder import (
    HybridDecoder,
    HybridDecoderWithQuantum,
    create_decoder,
)

from src.models.aqpathformer.multicancer_head import (
    MultiCancerHead,
    FocalLoss,
    EnsembleHead,
    create_head,
)

from src.models.aqpathformer.aqpathformer import (
    AQPathFormer,
    ReducedAQPathFormer,
    create_aqpathformer,
    create_reduced_aqpathformer,
    load_config,
    get_model_from_config,
    AQPATHFORMER_CONFIGS,
)

__all__ = [
    # Patch Generator
    'AdaptivePatchGenerator',
    'FixedPatchGenerator',
    'create_patch_generator',
    # Quantum Encoder
    'QuantumPatchEncoder',
    'ClassicalProxyEncoder',
    'create_quantum_patch_encoder',
    'create_classical_encoder',
    # Attention
    'AdaptiveQuantumAttention',
    'StandardMultiHeadAttention',
    'create_attention',
    # Multi-Scale Fusion
    'MultiScaleQuantumFusion',
    'SingleScaleQuantumEncoder',
    'create_multiscale_fusion',
    # Cross-Cancer
    'CrossCancerModule',
    'DomainAdaptationModule',
    'create_cross_cancer_module',
    # Noise-Aware
    'NoiseAwareQuantumLayer',
    'NoiseScheduler',
    'create_noise_aware_layer',
    # Decoder
    'HybridDecoder',
    'HybridDecoderWithQuantum',
    'create_decoder',
    # Head
    'MultiCancerHead',
    'FocalLoss',
    'EnsembleHead',
    'create_head',
    # Main Model
    'AQPathFormer',
    'ReducedAQPathFormer',
    'create_aqpathformer',
    'create_reduced_aqpathformer',
    'load_config',
    'get_model_from_config',
    'AQPATHFORMER_CONFIGS',
]