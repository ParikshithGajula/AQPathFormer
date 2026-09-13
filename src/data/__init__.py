"""
Data Module for AQPathFormer

Dataset loaders and preprocessing for histopathology images.
"""

from src.data.base_dataset import (
    BaseHistopathologyDataset,
    ConcatDataset,
    compute_dataset_stats,
)

from src.data.lc25000 import (
    LC25000Dataset,
    LC25000,
    get_lc25000_transforms,
    create_lc25000_dataloaders,
)

from src.data.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipeline,
    PatchExtractor,
    MultiScalePatchExtractor,
    NormalizationType,
    AugmentationType,
    create_preprocessing_pipeline,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_EVAL_CONFIG,
    MULTI_SCALE_CONFIG,
    STAIN_NORM_CONFIG,
    visualize_preprocessing,
)

__all__ = [
    # Base
    'BaseHistopathologyDataset',
    'ConcatDataset',
    'compute_dataset_stats',
    # LC25000
    'LC25000Dataset',
    'LC25000',
    'get_lc25000_transforms',
    'create_lc25000_dataloaders',
    # Preprocessing
    'PreprocessingConfig',
    'PreprocessingPipeline',
    'PatchExtractor',
    'MultiScalePatchExtractor',
    'NormalizationType',
    'AugmentationType',
    'create_preprocessing_pipeline',
    'DEFAULT_TRAIN_CONFIG',
    'DEFAULT_EVAL_CONFIG',
    'MULTI_SCALE_CONFIG',
    'STAIN_NORM_CONFIG',
    'visualize_preprocessing',
]