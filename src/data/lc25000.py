"""
LC25000 Dataset Loader

Lung and Colon Cancer Histopathological Images
Source: https://zenodo.org/records/3531430
Classes: lung_aca, lung_n, lung_scc, colon_aca, colon_n (5 classes)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data.base_dataset import BaseHistopathologyDataset


class LC25000Dataset(BaseHistopathologyDataset):
    """LC25000 Dataset for lung and colon cancer classification."""
    
    # Class names in fixed order for reproducibility
    CLASS_NAMES = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        cache_dir: Optional[str] = None,
        download: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing class subdirectories
            split: 'train', 'val', or 'test'
            transform: Image transforms
            target_transform: Label transforms
            split_ratios: Train/val/test split ratios
            seed: Random seed for reproducible splits
            cache_dir: Cache directory for split indices
            download: If True, attempt to download (not implemented)
        """
        super().__init__(
            root_dir=root_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
            split_ratios=split_ratios,
            seed=seed,
            cache_dir=cache_dir,
        )
    
    def _setup_class_mapping(self):
        """Set up class mapping (fixed order for reproducibility)."""
        self.class_to_idx = {name: i for i, name in enumerate(self.CLASS_NAMES)}
        self.idx_to_class = {i: name for i, name in enumerate(self.CLASS_NAMES)}
        self.num_classes = len(self.CLASS_NAMES)
    
    def _load_dataset(self):
        """Load all image paths and labels from class subdirectories."""
        self.image_paths = []
        self.labels = []
        self.patient_ids = []  # LC25000 doesn't have patient IDs
        
        # Supported image extensions
        extensions = {'.jpeg', '.jpg', '.png', '.tif', '.tiff', '.bmp'}
        
        for class_name in self.CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            
            class_idx = self.class_to_idx[class_name]
            
            # Find all images in class directory
            for ext in extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
                    # LC25000 doesn't provide patient IDs
                    self.patient_ids.append(f"{class_name}_{img_path.stem}")
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}. "
                             f"Expected class subdirectories: {self.CLASS_NAMES}")
        
        print(f"Loaded LC25000: {len(self.image_paths)} images, "
              f"{self.num_classes} classes")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class name to count mapping."""
        from collections import Counter
        counts = Counter(self.labels)
        return {self.idx_to_class[k]: v for k, v in counts.items()}
    
    @property
    def class_names(self) -> List[str]:
        return self.CLASS_NAMES
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced sampling."""
        from collections import Counter
        counts = Counter(self.labels)
        total = len(self.labels)
        weights = [total / (self.num_classes * counts[i]) for i in range(self.num_classes)]
        return torch.tensor(weights, dtype=torch.float32)


def get_lc25000_transforms(
    image_size: int = 224,
    is_train: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Callable:
    """Get standard transforms for LC25000."""
    import torchvision.transforms as T
    
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def create_lc25000_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 224,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, val, test dataloaders for LC25000."""
    
    train_dataset = LC25000Dataset(
        root_dir=root_dir,
        split='train',
        transform=get_lc25000_transforms(image_size, is_train=True),
        split_ratios=split_ratios,
        seed=seed,
    )
    
    val_dataset = LC25000Dataset(
        root_dir=root_dir,
        split='val',
        transform=get_lc25000_transforms(image_size, is_train=False),
        split_ratios=split_ratios,
        seed=seed,
    )
    
    test_dataset = LC25000Dataset(
        root_dir=root_dir,
        split='test',
        transform=get_lc25000_transforms(image_size, is_train=False),
        split_ratios=split_ratios,
        seed=seed,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, **kwargs
    )
    
    return train_loader, val_loader, test_loader


# For backward compatibility and direct use
class LC25000(Dataset):
    """Simple LC25000 dataset without splitting (for custom splits)."""
    
    CLASS_NAMES = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        self.class_to_idx = {name: i for i, name in enumerate(self.CLASS_NAMES)}
        self.idx_to_class = {i: name for i, name in enumerate(self.CLASS_NAMES)}
        
        self.image_paths = []
        self.labels = []
        
        extensions = {'.jpeg', '.jpg', '.png', '.tif', '.tiff', '.bmp'}
        
        for class_name in self.CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                class_idx = self.class_to_idx[class_name]
                for ext in extensions:
                    for img_path in class_dir.glob(f'*{ext}'):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
        
        print(f"LC25000: {len(self.image_paths)} images loaded")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label