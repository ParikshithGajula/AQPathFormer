"""
Base Dataset Class for Histopathology Datasets

Provides common interface for all dataset implementations.
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class BaseHistopathologyDataset(Dataset, ABC):
    """Abstract base class for histopathology datasets."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            root_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
            split_ratios: (train, val, test) ratios
            seed: Random seed for splitting
            cache_dir: Directory to cache split indices
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.split_ratios = split_ratios
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Class mapping (to be set by subclass)
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.num_classes: int = 0
        
        # Setup class mapping first (subclass responsibility)
        self._setup_class_mapping()
        
        # Data containers
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.patient_ids: List[str] = []  # For patient-level splitting
        
        # Load dataset
        self._initialize()
    
    @abstractmethod
    def _setup_class_mapping(self):
        """Set up class_to_idx, idx_to_class, num_classes. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _load_dataset(self):
        """Load image paths, labels, and patient IDs. Must be implemented by subclass."""
        pass
    
    def _initialize(self):
        """Initialize dataset after class mapping is set."""
        self._load_dataset()
        self._create_splits()
        self._apply_split()
    
    def _create_splits(self):
        """Create train/val/test splits with caching."""
        cache_file = self.cache_dir / f"splits_seed{self.seed}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                splits = json.load(f)
            self.train_indices = splits['train']
            self.val_indices = splits['val']
            self.test_indices = splits['test']
        else:
            n = len(self.image_paths)
            indices = list(range(n))
            
            # Patient-level split if patient_ids available
            if self.patient_ids and len(set(self.patient_ids)) > 1:
                self.train_indices, self.val_indices, self.test_indices = self._patient_level_split()
            else:
                # Stratified random split
                train_idx, temp_idx = train_test_split(
                    indices, train_size=self.split_ratios[0], 
                    stratify=self.labels, random_state=self.seed
                )
                val_size = self.split_ratios[1] / (self.split_ratios[1] + self.split_ratios[2])
                val_idx, test_idx = train_test_split(
                    temp_idx, train_size=val_size,
                    stratify=[self.labels[i] for i in temp_idx],
                    random_state=self.seed
                )
                self.train_indices = train_idx
                self.val_indices = val_idx
                self.test_indices = test_idx
            
            # Cache splits
            splits = {
                'train': self.train_indices,
                'val': self.val_indices,
                'test': self.test_indices,
                'seed': self.seed,
                'split_ratios': self.split_ratios,
                'class_distribution': self._get_split_distribution()
            }
            with open(cache_file, 'w') as f:
                json.dump(splits, f, indent=2)
    
    def _patient_level_split(self) -> Tuple[List[int], List[int], List[int]]:
        """Split by patient ID to prevent leakage."""
        patient_to_indices = {}
        for idx, pid in enumerate(self.patient_ids):
            patient_to_indices.setdefault(pid, []).append(idx)
        
        patients = list(patient_to_indices.keys())
        patient_labels = [self.labels[patient_to_indices[p][0]] for p in patients]
        
        train_patients, temp_patients = train_test_split(
            patients, train_size=self.split_ratios[0],
            stratify=patient_labels, random_state=self.seed
        )
        val_size = self.split_ratios[1] / (self.split_ratios[1] + self.split_ratios[2])
        val_patients, test_patients = train_test_split(
            temp_patients, train_size=val_size,
            stratify=[patient_labels[patients.index(p)] for p in temp_patients],
            random_state=self.seed
        )
        
        train_idx = [i for p in train_patients for i in patient_to_indices[p]]
        val_idx = [i for p in val_patients for i in patient_to_indices[p]]
        test_idx = [i for p in test_patients for i in patient_to_indices[p]]
        
        return train_idx, val_idx, test_idx
    
    def _get_split_distribution(self) -> Dict:
        """Get class distribution for each split."""
        dist = {}
        for split_name, indices in [('train', self.train_indices), 
                                     ('val', self.val_indices), 
                                     ('test', self.test_indices)]:
            labels = [self.labels[i] for i in indices]
            dist[split_name] = dict(Counter(labels))
        return dist
    
    def _apply_split(self):
        """Filter image_paths and labels to current split."""
        if self.split == 'train':
            indices = self.train_indices
        elif self.split == 'val':
            indices = self.val_indices
        elif self.split == 'test':
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        if self.patient_ids:
            self.patient_ids = [self.patient_ids[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution for current split."""
        return dict(Counter(self.labels))
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'num_samples': len(self),
            'num_classes': self.num_classes,
            'class_distribution': self.get_class_distribution(),
            'split': self.split,
            'split_ratios': self.split_ratios,
            'seed': self.seed,
        }
    
    @classmethod
    def get_dataloader(
        cls,
        dataset: 'BaseHistopathologyDataset',
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False,
        **kwargs
    ) -> DataLoader:
        """Create DataLoader with sensible defaults."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify dataset integrity."""
        results = {
            'total_files': 0,
            'missing_files': [],
            'corrupt_files': [],
            'class_counts': Counter(self.labels),
            'duplicate_hashes': [],
        }
        
        seen_hashes = {}
        for idx, path in enumerate(self.image_paths):
            results['total_files'] += 1
            if not path.exists():
                results['missing_files'].append(str(path))
                continue
            
            try:
                with Image.open(path) as img:
                    img.verify()
                # Check for duplicates
                with open(path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash in seen_hashes:
                    results['duplicate_hashes'].append((str(path), seen_hashes[file_hash]))
                else:
                    seen_hashes[file_hash] = str(path)
            except Exception as e:
                results['corrupt_files'].append((str(path), str(e)))
        
        return results


class ConcatDataset(BaseHistopathologyDataset):
    """Concatenate multiple datasets with unified label space."""
    
    def __init__(self, datasets: List[BaseHistopathologyDataset], **kwargs):
        # Use first dataset's root_dir for caching
        super().__init__(datasets[0].root_dir, **kwargs)
        
        # Merge class mappings
        all_classes = set()
        for d in datasets:
            all_classes.update(d.class_to_idx.keys())
        
        self.class_to_idx = {c: i for i, c in enumerate(sorted(all_classes))}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Merge data
        self.image_paths = []
        self.labels = []
        self.patient_ids = []
        
        for d in datasets:
            for path, label in zip(d.image_paths, d.labels):
                orig_class = d.idx_to_class[label]
                new_label = self.class_to_idx[orig_class]
                self.image_paths.append(path)
                self.labels.append(new_label)
                if d.patient_ids:
                    self.patient_ids.append(d.patient_ids[len(self.image_paths) - 1])
        
        self._create_splits()
        self._apply_split()
    
    def _setup_class_mapping(self):
        """Already set in __init__."""
        pass
    
    def _load_dataset(self):
        """Already loaded in __init__."""
        pass


def compute_dataset_stats(dataset: BaseHistopathologyDataset, sample_size: int = 1000) -> Dict:
    """Compute mean, std, and other statistics for normalization."""
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    
    means = []
    stds = []
    shapes = []
    
    for idx in indices:
        img, _ = dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if img.ndim == 3 and img.shape[0] == 3:  # CHW
            img = img.transpose(1, 2, 0)
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))
        shapes.append(img.shape[:2])
    
    return {
        'mean': np.mean(means, axis=0).tolist(),
        'std': np.mean(stds, axis=0).tolist(),
        'shapes': shapes,
        'shape_stats': {
            'min_h': min(s[0] for s in shapes),
            'max_h': max(s[0] for s in shapes),
            'mean_h': np.mean([s[0] for s in shapes]),
            'min_w': min(s[1] for s in shapes),
            'max_w': max(s[1] for s in shapes),
            'mean_w': np.mean([s[1] for s in shapes]),
        }
    }