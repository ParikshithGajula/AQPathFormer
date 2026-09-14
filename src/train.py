"""
Training Script for AQPathFormer

Main entry point for training with experiment tracking, checkpointing, and logging.
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import create_lc25000_dataloaders, LC25000Dataset
from src.eval.metrics import (
    compute_all_metrics,
    count_parameters,
    count_flops,
    measure_inference_time,
)
from src.models.baselines import create_model as create_baseline_model
from src.models.aqpathformer import create_aqpathformer


class ExperimentTracker:
    """Experiment tracking with local logging and optional wandb."""
    
    def __init__(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        output_dir: Path,
        use_wandb: bool = False,
        wandb_project: str = "aqpathformer",
        wandb_entity: Optional[str] = None,
    ):
        self.experiment_id = experiment_id
        self.config = config
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        
        # Create experiment directory
        self.exp_dir = output_dir / experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Save environment info
        self._save_environment()
        
        # Save git commit
        self._save_git_commit()
        
        # Initialize wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_id,
                    config=config,
                    dir=str(output_dir),
                )
                self.wandb = wandb
            except Exception as e:
                print(f"Warning: wandb initialization failed: {e}")
                self.use_wandb = False
        
        # Training log
        self.log_file = self.exp_dir / 'train.log'
        self.metrics_history = []
        
        # Best metrics tracking
        self.best_val_metric = -1.0
        self.best_epoch = -1
    
    def _save_environment(self):
        """Save Python environment info."""
        env_info = {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'torchvision_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'packages': {},
        }
        
        # Get key package versions
        key_packages = [
            'torch', 'torchvision', 'timm', 'pennylane', 'numpy',
            'pandas', 'scikit-learn', 'matplotlib', 'opencv-python', 'yaml'
        ]
        for pkg in key_packages:
            try:
                mod = __import__(pkg.replace('-', '_'))
                env_info['packages'][pkg] = getattr(mod, '__version__', 'unknown')
            except:
                env_info['packages'][pkg] = 'not installed'
        
        with open(self.exp_dir / 'environment.json', 'w') as f:
            json.dump(env_info, f, indent=2)
    
    def _save_git_commit(self):
        """Save current git commit hash."""
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                cwd=Path(__file__).parent.parent, 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            with open(self.exp_dir / 'git_commit.txt', 'w') as f:
                f.write(commit)
        except:
            with open(self.exp_dir / 'git_commit.txt', 'w') as f:
                f.write('unknown')
    
    def log(self, message: str, level: str = 'INFO'):
        """Log message to file and console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ''):
        """Log metrics to file and wandb."""
        # Add to history
        entry = {'step': step, 'prefix': prefix, **metrics}
        self.metrics_history.append(entry)
        
        # Log to file
        self.log(f"Step {step} {prefix}: {metrics}")
        
        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.wandb.log(wandb_metrics, step=step)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': self.config,
        }
        
        # Save latest
        torch.save(checkpoint, self.exp_dir / 'checkpoint_latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.exp_dir / 'checkpoint_best.pt')
            self.log(f"New best model saved at epoch {epoch}")
        
        # Save periodic
        if epoch % self.config.get('training', {}).get('save_interval', 5) == 0:
            torch.save(checkpoint, self.exp_dir / f'checkpoint_epoch{epoch}.pt')
    
    def finish(self):
        """Finish experiment tracking."""
        if self.use_wandb:
            try:
                self.wandb.finish()
            except:
                pass


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer from config."""
    opt_config = config.get('training', {}).get('optimizer', {})
    opt_name = opt_config.get('name', 'adamw').lower()
    lr = float(opt_config.get('lr', 1e-4))
    weight_decay = float(opt_config.get('weight_decay', 1e-4))
    
    if opt_name == 'adamw':
        betas = opt_config.get('betas', [0.9, 0.999])
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = float(opt_config.get('momentum', 0.9))
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from config."""
    sched_config = config.get('training', {}).get('scheduler', {})
    sched_name = sched_config.get('name', 'cosine_annealing_warmup').lower()
    
    if sched_name == 'cosine_annealing_warmup':
        warmup_epochs = int(sched_config.get('warmup_epochs', 2))
        total_epochs = int(config.get('training', {}).get('epochs', 20))
        min_lr = float(sched_config.get('min_lr', 1e-6))
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr / optimizer.param_groups[0]['lr'] + (1 - min_lr / optimizer.param_groups[0]['lr']) * 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif sched_name == 'cosine_annealing':
        total_epochs = int(config.get('training', {}).get('epochs', 20))
        min_lr = float(sched_config.get('min_lr', 1e-6))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
    
    elif sched_name == 'step':
        step_size = int(sched_config.get('step_size', 10))
        gamma = float(sched_config.get('gamma', 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif sched_name == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
    
    return None


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    tracker: ExperimentTracker,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    scaler: Optional[GradScaler] = None,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, labels)
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps
            loss.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Track predictions for metrics
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
        
        total_loss += loss.item() * grad_accum_steps
        
        if batch_idx % log_interval == 0:
            tracker.log(f"Epoch {epoch} Batch {batch_idx}/{len(dataloader)} Loss: {loss.item()*grad_accum_steps:.4f}")
    
    # Compute epoch metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    metrics = compute_all_metrics(all_preds, all_labels, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
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
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            
            total_loss += loss.item()
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    metrics = compute_all_metrics(all_preds, all_labels, all_probs)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def train(
    config_path: str,
    experiment_id: str,
    output_dir: str = 'experiments',
    device: str = 'cpu',
    resume: Optional[str] = None,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """Main training loop."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    output_dir = Path(output_dir)
    tracker = ExperimentTracker(experiment_id, config, output_dir, use_wandb)
    tracker.log(f"Starting experiment {experiment_id}")
    tracker.log(f"Config: {config_path}")
    tracker.log(f"Device: {device}")
    
    # Set seeds
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Data
    dataset_name = config.get('dataset', {}).get('name', 'lc25000')
    if dataset_name == 'lc25000':
        train_loader, val_loader, test_loader = create_lc25000_dataloaders(
            root_dir=config['dataset']['root_dir'],
            batch_size=config['training'].get('batch_size', 8),
            num_workers=config['dataset'].get('num_workers', 4),
            image_size=config['preprocessing'].get('image_size', 224),
        )
        num_classes = 5
        class_names = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    tracker.log(f"Dataset: {dataset_name}, Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Model
    model_config = config['model'].copy()
    model_name = model_config.get('name', 'resnet50')
    
    # Remove num_classes from model_config to avoid duplicate argument
    model_config.pop('num_classes', None)
    
    if model_name in ['resnet50', 'vit', 'swin', 'convnext', 'efficientnet']:
        model = create_baseline_model(model_name, num_classes=num_classes, **model_config)
    else:
        model = create_aqpathformer(**model_config, num_classes=num_classes)
    
    model.to(device)
    
    # Log model info
    param_info = count_parameters(model)
    tracker.log(f"Model: {model_name}, Parameters: {param_info['total_params']:,}")
    
    # Optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision
    use_amp = config.get('experiment', {}).get('mixed_precision', False) and device == 'cuda'
    scaler = GradScaler() if use_amp else None
    grad_accum_steps = config.get('experiment', {}).get('gradient_accumulation_steps', 1)
    
    # Resume
    start_epoch = 0
    best_val_metric = -1.0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_metric = checkpoint.get('metrics', {}).get('validation', {}).get('accuracy', -1.0)
        tracker.log(f"Resumed from epoch {start_epoch-1}, best val acc: {best_val_metric:.4f}")
    
    # Training loop
    epochs = config['training'].get('epochs', 20)
    eval_interval = config.get('experiment', {}).get('eval_interval', 1)
    early_stopping = config.get('training', {}).get('early_stopping', {})
    early_stop_patience = early_stopping.get('patience', 10)
    early_stop_min_delta = early_stopping.get('min_delta', 1e-4)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        tracker.log(f"\n{'='*50}")
        tracker.log(f"Epoch {epoch+1}/{epochs}")
        tracker.log(f"{'='*50}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            tracker, grad_accum_steps, use_amp, scaler,
            config.get('experiment', {}).get('log_interval', 10)
        )
        tracker.log_metrics(train_metrics, epoch, 'train')
        
        # Validate
        if (epoch + 1) % eval_interval == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            tracker.log_metrics(val_metrics, epoch, 'val')
            
            # Check for improvement
            val_acc = val_metrics.get('accuracy', 0.0)
            if val_acc > best_val_metric + early_stop_min_delta:
                best_val_metric = val_acc
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
                is_best = False
            
            tracker.log(f"Val Acc: {val_acc:.4f} (Best: {best_val_metric:.4f})")
            
            # Save checkpoint
            tracker.save_checkpoint(model, optimizer, scheduler, epoch, 
                                  {'train': train_metrics, 'val': val_metrics}, is_best)
            
            # Early stopping
            if early_stopping.get('enabled', True) and patience_counter >= early_stop_patience:
                tracker.log(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        # Step scheduler
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
    
    # Final evaluation on test set
    tracker.log("\nRunning final test evaluation...")
    test_metrics = validate(model, test_loader, criterion, device)
    tracker.log_metrics(test_metrics, epochs, 'test')
    
    # Save final metrics
    final_metrics = {
        'train': train_metrics,
        'val': val_metrics if 'val_metrics' in locals() else {},
        'test': test_metrics,
        'best_val_accuracy': best_val_metric,
        'total_epochs': epoch + 1,
    }
    
    with open(tracker.exp_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2, default=str)
    
    tracker.log("Training completed!")
    tracker.finish()
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='Train AQPathFormer or baseline models')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--experiment-id', type=str, help='Experiment ID (auto-generated if not provided)')
    parser.add_argument('--output-dir', type=str, default='experiments', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='aqpathformer', help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, help='Wandb entity')
    
    args = parser.parse_args()
    
    # Generate experiment ID if not provided
    if args.experiment_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config['model'].get('name', 'model')
        args.experiment_id = f"{model_name}_{timestamp}"
    
    train(
        config_path=args.config,
        experiment_id=args.experiment_id,
        output_dir=args.output_dir,
        device=args.device,
        resume=args.resume,
        use_wandb=args.wandb,
    )


if __name__ == '__main__':
    main()