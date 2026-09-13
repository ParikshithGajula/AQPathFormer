"""
Preprocessing Pipeline for Histopathology Images

Provides configurable preprocessing including:
- Resizing and cropping
- Normalization (ImageNet, custom, stain-aware)
- Augmentation (geometric, color, stain)
- Patch extraction (fixed grid, adaptive)
- Multi-scale extraction
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import Tuple, List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from PIL import Image
import cv2
import yaml


class NormalizationType(Enum):
    IMAGENET = "imagenet"
    CUSTOM = "custom"
    STAIN_NORM = "stain_norm"
    NONE = "none"


class AugmentationType(Enum):
    NONE = "none"
    BASIC = "basic"
    STRONG = "strong"
    STAIN_AWARE = "stain_aware"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    # Resize
    image_size: int = 224
    resize_method: str = "lanczos"  # lanczos, bilinear, bicubic
    center_crop: bool = False
    crop_size: Optional[int] = None
    
    # Normalization
    normalization: NormalizationType = NormalizationType.IMAGENET
    custom_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    custom_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Augmentation
    augmentation: AugmentationType = AugmentationType.BASIC
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotation_degrees: int = 90
    color_jitter: float = 0.1
    gaussian_blur_prob: float = 0.1
    gaussian_blur_kernel: int = 3
    
    # Stain normalization (Macenko/Reinhard)
    stain_normalize: bool = False
    stain_method: str = "macenko"  # macenko, reinhard, vahadane
    stain_target: Optional[np.ndarray] = None  # Target stain matrix
    
    # Patch extraction
    patch_size: int = 16
    patch_stride: Optional[int] = None  # None = patch_size (non-overlapping)
    extract_patches: bool = False
    
    # Multi-scale
    multi_scale: bool = False
    scales: List[int] = field(default_factory=lambda: [4, 8, 16])  # Downsample factors
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'image_size': self.image_size,
            'resize_method': self.resize_method,
            'center_crop': self.center_crop,
            'crop_size': self.crop_size,
            'normalization': self.normalization.value,
            'custom_mean': list(self.custom_mean),
            'custom_std': list(self.custom_std),
            'augmentation': self.augmentation.value,
            'hflip_prob': self.hflip_prob,
            'vflip_prob': self.vflip_prob,
            'rotation_degrees': self.rotation_degrees,
            'color_jitter': self.color_jitter,
            'gaussian_blur_prob': self.gaussian_blur_prob,
            'gaussian_blur_kernel': self.gaussian_blur_kernel,
            'stain_normalize': self.stain_normalize,
            'stain_method': self.stain_method,
            'patch_size': self.patch_size,
            'patch_stride': self.patch_stride,
            'extract_patches': self.extract_patches,
            'multi_scale': self.multi_scale,
            'scales': self.scales,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PreprocessingConfig':
        """Create from dictionary."""
        d = d.copy()
        d['normalization'] = NormalizationType(d['normalization'])
        d['augmentation'] = AugmentationType(d['augmentation'])
        d['custom_mean'] = tuple(d['custom_mean'])
        d['custom_std'] = tuple(d['custom_std'])
        return cls(**d)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PreprocessingConfig':
        """Load from YAML file."""
        with open(path, 'r') as f:
            return cls.from_dict(yaml.safe_load(f))
    
    def to_yaml(self, path: str):
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_resize_transform(config: PreprocessingConfig):
    """Get resize transform based on config."""
    interpolation_map = {
        'lanczos': InterpolationMode.LANCZOS,
        'bilinear': InterpolationMode.BILINEAR,
        'bicubic': InterpolationMode.BICUBIC,
        'nearest': InterpolationMode.NEAREST,
    }
    interpolation = interpolation_map.get(config.resize_method, InterpolationMode.LANCZOS)
    
    if config.center_crop and config.crop_size:
        return T.Compose([
            T.Resize(config.image_size, interpolation=interpolation),
            T.CenterCrop(config.crop_size),
        ])
    else:
        return T.Resize((config.image_size, config.image_size), interpolation=interpolation)


def get_normalization_transform(config: PreprocessingConfig):
    """Get normalization transform based on config."""
    if config.normalization == NormalizationType.IMAGENET:
        return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif config.normalization == NormalizationType.CUSTOM:
        return T.Normalize(mean=list(config.custom_mean), std=list(config.custom_std))
    elif config.normalization == NormalizationType.NONE:
        return T.Lambda(lambda x: x)
    else:
        # Stain normalization handled separately
        return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_augmentation_transform(config: PreprocessingConfig, is_train: bool = True):
    """Get augmentation transform based on config."""
    if not is_train or config.augmentation == AugmentationType.NONE:
        return T.Lambda(lambda x: x)
    
    transforms = []
    
    if config.augmentation in [AugmentationType.BASIC, AugmentationType.STRONG, AugmentationType.STAIN_AWARE]:
        if config.hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(p=config.hflip_prob))
        if config.vflip_prob > 0:
            transforms.append(T.RandomVerticalFlip(p=config.vflip_prob))
        if config.rotation_degrees > 0:
            transforms.append(T.RandomRotation(degrees=config.rotation_degrees))
    
    if config.augmentation in [AugmentationType.STRONG, AugmentationType.STAIN_AWARE]:
        if config.color_jitter > 0:
            jitter = config.color_jitter
            transforms.append(T.ColorJitter(
                brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter/2
            ))
        if config.gaussian_blur_prob > 0:
            transforms.append(T.RandomApply([
                T.GaussianBlur(kernel_size=config.gaussian_blur_kernel)
            ], p=config.gaussian_blur_prob))
    
    return T.Compose(transforms)


def get_stain_normalization_transform(config: PreprocessingConfig) -> Optional[Callable]:
    """Get stain normalization transform."""
    if not config.stain_normalize:
        return None
    
    def macenko_normalize(img: Image.Image) -> Image.Image:
        """Macenko stain normalization."""
        # Convert to numpy
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Optical density
        od = -np.log(img_np + 1e-8)
        
        # Remove transparent pixels
        od_flat = od.reshape(-1, 3)
        od_flat = od_flat[od_flat.min(axis=1) > 0.15]
        
        if len(od_flat) < 100:
            return img  # Not enough tissue pixels
        
        # SVD for stain matrix estimation
        _, _, vh = np.linalg.svd(od_flat, full_matrices=False)
        stain_matrix = vh[:2].T  # 3x2 matrix
        
        # Normalize stains
        stain_matrix = stain_matrix / (np.linalg.norm(stain_matrix, axis=0) + 1e-8)
        
        # Project to stain space
        concentrations = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)[0]
        
        # Use target concentrations if provided, else use 99th percentile
        if config.stain_target is not None:
            target_concentrations = config.stain_target
        else:
            target_concentrations = np.percentile(concentrations, 99, axis=1)
        
        # Reconstruct with target staining
        norm_od = stain_matrix @ np.diag(target_concentrations) @ (concentrations / (target_concentrations[:, None] + 1e-8))
        norm_img = np.exp(-norm_od.T).reshape(img_np.shape)
        norm_img = np.clip(norm_img * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(norm_img)
    
    def reinhard_normalize(img: Image.Image) -> Image.Image:
        """Reinhard color normalization in LAB space."""
        img_np = np.array(img).astype(np.float32)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Match mean and std to target
        if config.stain_target is not None:
            target_lab = cv2.cvtColor(config.stain_target.astype(np.uint8), cv2.COLOR_RGB2LAB)
            target_mean = target_lab.mean(axis=(0, 1))
            target_std = target_lab.std(axis=(0, 1))
        else:
            target_mean = np.array([128, 128, 128])
            target_std = np.array([30, 15, 15])
        
        img_mean = lab.mean(axis=(0, 1))
        img_std = lab.std(axis=(0, 1))
        
        lab = (lab - img_mean) * (target_std / (img_std + 1e-8)) + target_mean
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        
        norm_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(norm_img)
    
    if config.stain_method == "macenko":
        return T.Lambda(macenko_normalize)
    elif config.stain_method == "reinhard":
        return T.Lambda(reinhard_normalize)
    else:
        return None


class PreprocessingPipeline:
    """Complete preprocessing pipeline for histopathology images."""
    
    def __init__(self, config: PreprocessingConfig, is_train: bool = True):
        self.config = config
        self.is_train = is_train
        self.transform = self._build_transform()
    
    def _build_transform(self) -> T.Compose:
        """Build complete transform pipeline."""
        transforms = []
        
        # Resize
        transforms.append(get_resize_transform(self.config))
        
        # Stain normalization (before tensor conversion)
        stain_transform = get_stain_normalization_transform(self.config)
        if stain_transform:
            transforms.append(stain_transform)
        
        # Augmentation
        if self.is_train:
            transforms.append(get_augmentation_transform(self.config, is_train=True))
        
        # To tensor
        transforms.append(T.ToTensor())
        
        # Normalization
        transforms.append(get_normalization_transform(self.config))
        
        return T.Compose(transforms)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)
    
    def get_multi_scale_transforms(self) -> Dict[int, 'PreprocessingPipeline']:
        """Get pipelines for multi-scale processing."""
        if not self.config.multi_scale:
            return {1: self}
        
        pipelines = {}
        for scale in self.config.scales:
            scale_config = PreprocessingConfig(
                image_size=self.config.image_size // scale,
                normalization=self.config.normalization,
                custom_mean=self.config.custom_mean,
                custom_std=self.config.custom_std,
                augmentation=AugmentationType.NONE,  # No aug for multi-scale
                stain_normalize=self.config.stain_normalize,
                stain_method=self.config.stain_method,
            )
            pipelines[scale] = PreprocessingPipeline(scale_config, is_train=self.is_train)
        return pipelines


class PatchExtractor:
    """Extract patches from images for patch-based processing."""
    
    def __init__(
        self,
        patch_size: int = 16,
        stride: Optional[int] = None,
        padding: bool = True,
    ):
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.padding = padding
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from image tensor.
        
        Args:
            image: (C, H, W) or (B, C, H, W)
            
        Returns:
            patches: (N, C, patch_size, patch_size) where N = num_patches
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # Calculate padding
        if self.padding:
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            image = TF.pad(image, (0, 0, pad_w, pad_h))
            H, W = H + pad_h, W + pad_w
        
        # Extract patches using unfold
        patches = image.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        # patches: (B, C, n_h, n_w, patch_size, patch_size)
        
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # patches: (B, n_h, n_w, C, patch_size, patch_size)
        
        n_h, n_w = patches.shape[1], patches.shape[2]
        patches = patches.view(B, n_h * n_w, C, self.patch_size, self.patch_size)
        # patches: (B, N, C, patch_size, patch_size)
        
        return patches
    
    def get_num_patches(self, H: int, W: int) -> int:
        """Calculate number of patches for given image size."""
        if self.padding:
            H = H + (self.patch_size - H % self.patch_size) % self.patch_size
            W = W + (self.patch_size - W % self.patch_size) % self.patch_size
        n_h = (H - self.patch_size) // self.stride + 1
        n_w = (W - self.patch_size) // self.stride + 1
        return n_h * n_w


class MultiScalePatchExtractor:
    """Extract patches at multiple scales."""
    
    def __init__(
        self,
        base_patch_size: int = 16,
        scales: List[int] = [4, 8, 16],
        image_size: int = 224,
    ):
        self.base_patch_size = base_patch_size
        self.scales = scales
        self.image_size = image_size
        self.extractors = {
            scale: PatchExtractor(
                patch_size=base_patch_size,
                stride=base_patch_size,
            ) for scale in scales
        }
    
    def __call__(self, image: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract multi-scale patches.
        
        Args:
            image: (C, H, W) or (B, C, H, W)
            
        Returns:
            Dict mapping scale -> patches tensor
        """
        results = {}
        
        for scale in self.scales:
            # Downsample image
            scale_size = self.image_size // scale
            if image.dim() == 3:
                img_scaled = TF.resize(image.unsqueeze(0), (scale_size, scale_size)).squeeze(0)
            else:
                img_scaled = TF.resize(image, (scale_size, scale_size))
            
            # Extract patches
            patches = self.extractors[scale](img_scaled)
            results[scale] = patches
        
        return results


def create_preprocessing_pipeline(
    config_dict: Dict[str, Any],
    is_train: bool = True
) -> PreprocessingPipeline:
    """Factory function to create preprocessing pipeline from config dict."""
    config = PreprocessingConfig.from_dict(config_dict)
    return PreprocessingPipeline(config, is_train=is_train)


def visualize_preprocessing(
    image: Image.Image,
    pipeline: PreprocessingPipeline,
    save_path: Optional[str] = None
):
    """Visualize preprocessing steps."""
    import matplotlib.pyplot as plt
    
    # Original
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # After resize
    resize_transform = get_resize_transform(pipeline.config)
    resized = resize_transform(image)
    axes[1].imshow(resized)
    axes[1].set_title(f'Resized {pipeline.config.image_size}x{pipeline.config.image_size}')
    axes[1].axis('off')
    
    # Final processed
    processed = pipeline(image)
    # Denormalize for visualization
    if pipeline.config.normalization != NormalizationType.NONE:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        processed_vis = processed * std + mean
        processed_vis = torch.clamp(processed_vis, 0, 1)
    else:
        processed_vis = processed
    
    axes[2].imshow(processed_vis.permute(1, 2, 0).numpy())
    axes[2].set_title('Processed (normalized)')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Default configs for common use cases
DEFAULT_TRAIN_CONFIG = PreprocessingConfig(
    image_size=224,
    normalization=NormalizationType.IMAGENET,
    augmentation=AugmentationType.BASIC,
    hflip_prob=0.5,
    vflip_prob=0.5,
    rotation_degrees=90,
    color_jitter=0.1,
)

DEFAULT_EVAL_CONFIG = PreprocessingConfig(
    image_size=224,
    normalization=NormalizationType.IMAGENET,
    augmentation=AugmentationType.NONE,
)

MULTI_SCALE_CONFIG = PreprocessingConfig(
    image_size=224,
    normalization=NormalizationType.IMAGENET,
    augmentation=AugmentationType.BASIC,
    multi_scale=True,
    scales=[4, 8, 16],
)

STAIN_NORM_CONFIG = PreprocessingConfig(
    image_size=224,
    normalization=NormalizationType.IMAGENET,
    augmentation=AugmentationType.BASIC,
    stain_normalize=True,
    stain_method="macenko",
)