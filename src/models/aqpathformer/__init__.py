"""
AQPathFormer Models Package

Quantum-enhanced Vision Transformer components.
"""

# Core components that don't require PennyLane
from src.models.aqpathformer.adaptive_patch_generator import (
    AdaptivePatchGenerator,
    FixedPatchGenerator,
    create_patch_generator,
)

from src.models.aqpathformer.quantum_patch_encoder import (
    ClassicalProxyEncoder,
    create_classical_encoder,
)

from src.models.aqpathformer.adaptive_quantum_attention import (
    StandardMultiHeadAttention,
    create_attention,
)

from src.models.aqpathformer.cross_cancer_representation import (
    CrossCancerModule,
    DomainAdaptationModule,
    create_cross_cancer_module,
)

from src.models.aqpathformer.hybrid_decoder import (
    HybridDecoder,
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

# Lazy-loaded quantum components (only imported when explicitly requested)
_quantum_components_loaded = False

def _load_quantum_components():
    """Lazy load quantum components that require PennyLane."""
    global _quantum_components_loaded
    if not _quantum_components_loaded:
        from src.models.aqpathformer.quantum_patch_encoder import (
            QuantumPatchEncoder,
            create_quantum_patch_encoder,
        )
        from src.models.aqpathformer.adaptive_quantum_attention import (
            AdaptiveQuantumAttention,
        )
        from src.models.aqpathformer.multiscale_quantum_fusion import (
            MultiScaleQuantumFusion,
            SingleScaleQuantumEncoder,
            create_multiscale_fusion,
        )
        from src.models.aqpathformer.hybrid_decoder import (
            HybridDecoderWithQuantum,
        )
        
        # Add to globals
        globals().update({
            'QuantumPatchEncoder': QuantumPatchEncoder,
            'create_quantum_patch_encoder': create_quantum_patch_encoder,
            'AdaptiveQuantumAttention': AdaptiveQuantumAttention,
            'MultiScaleQuantumFusion': MultiScaleQuantumFusion,
            'SingleScaleQuantumEncoder': SingleScaleQuantumEncoder,
            'create_multiscale_fusion': create_multiscale_fusion,
            'HybridDecoderWithQuantum': HybridDecoderWithQuantum,
        })
        _quantum_components_loaded = True

# Always available
__all__ = [
    # Core Components
    'AdaptivePatchGenerator',
    'FixedPatchGenerator',
    'create_patch_generator',
    'ClassicalProxyEncoder',
    'create_classical_encoder',
    'StandardMultiHeadAttention',
    'create_attention',
    'CrossCancerModule',
    'DomainAdaptationModule',
    'create_cross_cancer_module',
    'HybridDecoder',
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
    
    # Quantum components (lazy-loaded)
    'QuantumPatchEncoder',
    'create_quantum_patch_encoder',
    'AdaptiveQuantumAttention',
    'MultiScaleQuantumFusion',
    'SingleScaleQuantumEncoder',
    'create_multiscale_fusion',
    'HybridDecoderWithQuantum',
]

def __getattr__(name):
    """Lazy load quantum components on first access."""
    if name in [
        'QuantumPatchEncoder', 'create_quantum_patch_encoder',
        'AdaptiveQuantumAttention', 'MultiScaleQuantumFusion',
        'SingleScaleQuantumEncoder', 'create_multiscale_fusion',
        'HybridDecoderWithQuantum',
    ]:
        _load_quantum_components()
        return globals()[name]
    raise AttributeError(f"module 'src.models.aqpathformer' has no attribute '{name}'")