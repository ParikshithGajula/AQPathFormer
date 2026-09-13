"""
Evaluation Script for AQPathFormer

Standalone evaluation of trained models with comprehensive metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_lc25000_dataloaders, LC25000Dataset, get_lc25000_transforms
from src.eval.metrics import (
    compute_all_metrics,
    count_parameters,
    count_flops,
    measure_inference_time,
    aggregate_metrics_across_seeds,
    format_metrics_table,
)
from src.models.baselines import create_model as create_baseline_model
from src.models.aqpathformer import create_aqpathformer


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cpu',
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        return checkpoint
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint, strict=strict)
        return {'model_state_dict': checkpoint}


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    num_classes: int = 5,
    class_names: Optional[list] = None,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Evaluate model on dataloader."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # Compute metrics
    metrics = compute_all_metrics(
        preds=all_preds,
        labels=all_labels,
        probs=all_probs,
        num_classes=num_classes,
        class_names=class_names,
    )
    
    # Add predictions if requested
    if return_predictions:
        metrics['predictions'] = all_preds.numpy()
        metrics['labels'] = all_labels.numpy()
        metrics['probabilities'] = all_probs.numpy()
    
    return metrics


def save_predictions_csv(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list,
    save_path: str,
):
    """Save predictions to CSV."""
    import pandas as pd
    
    df = pd.DataFrame({
        'true_label': labels,
        'true_class': [class_names[l] for l in labels],
        'pred_label': predictions,
        'pred_class': [class_names[p] for p in predictions],
    })
    
    # Add probability columns
    for i, name in enumerate(class_names):
        df[f'prob_{name}'] = probabilities[:, i]
    
    df['correct'] = df['true_label'] == df['pred_label']
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str,
):
    """Save confusion matrix visualization."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    device: str = 'cpu',
    dataset: str = 'lc25000',
    batch_size: int = 8,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Run complete evaluation pipeline."""
    
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if dataset == 'lc25000':
        _, val_loader, test_loader = create_lc25000_dataloaders(
            root_dir=config['dataset']['root_dir'],
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=config['preprocessing']['image_size'],
        )
        class_names = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
        num_classes = 5
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create model
    model_config = config['model']
    model_name = model_config.get('name', 'resnet50')
    
    if model_name in ['resnet50', 'vit', 'swin', 'convnext', 'efficientnet']:
        from src.models.baselines import create_model
        model = create_model(model_name, num_classes=num_classes, **model_config)
    else:
        model = create_aqpathformer(**model_config, num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, device=device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_metrics = evaluate_model(
        model, val_loader, device, num_classes, class_names, return_predictions=True
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(
        model, test_loader, device, num_classes, class_names, return_predictions=True
    )
    
    # Computational metrics
    print("Computing computational metrics...")
    comp_metrics = {
        'parameters': count_parameters(model),
        'flops': count_flops(model, (1, 3, config['preprocessing']['image_size'], config['preprocessing']['image_size'])),
        'inference_time': measure_inference_time(
            model, 
            (1, 3, config['preprocessing']['image_size'], config['preprocessing']['image_size']),
            device=device
        ),
    }
    
    # Combine all metrics
    all_metrics = {
        'config': config,
        'checkpoint': checkpoint_path,
        'device': device,
        'validation': val_metrics,
        'test': test_metrics,
        'computational': comp_metrics,
    }
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Metrics saved to {metrics_path}")
    
    # Save predictions CSV
    for split_name, split_metrics in [('val', val_metrics), ('test', test_metrics)]:
        if 'predictions' in split_metrics:
            csv_path = output_dir / f'predictions_{split_name}.csv'
            save_predictions_csv(
                split_metrics['predictions'],
                split_metrics['labels'],
                split_metrics['probabilities'],
                class_names,
                str(csv_path)
            )
        
        if 'confusion_matrix' in split_metrics:
            cm_path = output_dir / f'confusion_matrix_{split_name}.png'
            save_confusion_matrix(
                np.array(split_metrics['confusion_matrix']),
                class_names,
                str(cm_path)
            )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for split_name, split_metrics in [('Validation', val_metrics), ('Test', test_metrics)]:
        print(f"\n{split_name} Metrics:")
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc_ovr', 'mcc', 'balanced_accuracy']
        for k in key_metrics:
            if k in split_metrics:
                print(f"  {k}: {split_metrics[k]:.4f}")
    
    print(f"\nParameters: {comp_metrics['parameters']['total_params']:,}")
    print(f"FLOPs: {comp_metrics['flops'].get('total_gflops', 'N/A'):.2f} GFLOPs")
    print(f"Inference: {comp_metrics['inference_time']['mean_ms']:.2f} ms")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--dataset', type=str, default='lc25000', help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    
    args = parser.parse_args()
    
    run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()