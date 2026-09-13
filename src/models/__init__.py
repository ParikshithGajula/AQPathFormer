"""
Models Package for AQPathFormer

Includes baseline models and AQPathFormer components.
"""

from src.models.baselines import (
    create_model,
    create_resnet50,
    create_vit,
    create_swin,
    create_convnext,
    create_efficientnet,
    BASELINE_CONFIGS,
    get_baseline_config,
)

from src.models.aqpathformer import (
    # Patch Generator
    AdaptivePatchGenerator,
    FixedPatchGenerator,
    create_patch_generator,
    # Quantum Encoder
    QuantumPatchEncoder,
    ClassicalProxyEncoder,
    create_quantum_patch_encoder,
    create_classical_encoder,
    # Attention
    AdaptiveQuantumAttention,
    StandardMultiHeadAttention,
    create_attention,
    # Multi-Scale Fusion
    MultiScaleQuantumFusion,
    SingleScaleQuantumEncoder,
    create_multiscale_fusion,
    # Cross-Cancer
    CrossCancerModule,
    DomainAdaptationModule,
    create_cross_cancer_module,
    # Noise-Aware
    NoiseAwareQuantumLayer,
    NoiseScheduler,
    create_noise_aware_layer,
    # Decoder
    HybridDecoder,
    HybridDecoderWithQuantum,
    create_decoder,
    # Head
    MultiCancerHead,
    FocalLoss,
    EnsembleHead,
    create_head,
    # Main Model
    AQPathFormer,
    ReducedAQPathFormer,
    create_aqpathformer,
    create_reduced_aqpathformer,
    load_config,
    get_model_from_config,
    AQPATHFORMER_CONFIGS,
)

__all__ = [
    # Baselines
    'create_model',
    'create_resnet50',
    'create_vit',
    'create_swin',
    'create_convnext',
    'create_efficientnet',
    'BASELINE_CONFIGS',
    'get_baseline_config',
    # AQPathFormer Components
    'AdaptivePatchGenerator',
    'FixedPatchGenerator',
    'create_patch_generator',
    'QuantumPatchEncoder',
    'ClassicalProxyEncoder',
    'create_quantum_patch_encoder',
    'create_classical_encoder',
    'AdaptiveQuantumAttention',
    'StandardMultiHeadAttention',
    'create_attention',
    'MultiScaleQuantumFusion',
    'SingleScaleQuantumEncoder',
    'create_multiscale_fusion',
    'CrossCancerModule',
    'DomainAdaptationModule',
    'create_cross_cancer_module',
    'NoiseAwareQuantumLayer',
    'NoiseScheduler',
    'create_noise_aware_layer',
    'HybridDecoder',
    'HybridDecoderWithQuantum',
    'create_decoder',
    'MultiCancerHead',
    'FocalLoss',
    'EnsembleHead',
    'create_head',
    'AQPathFormer',
    'ReducedAQPathFormer',
    'create_aqpathformer',
    'create_reduced_aqpathformer',
    'load_config',
    'get_model_from_config',
    'AQPATHFORMER_CONFIGS',
]