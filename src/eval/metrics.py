"""
Evaluation Metrics for AQPathFormer

Comprehensive metrics for classification, computational efficiency, and quantum-specific measures.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import time
from contextlib import contextmanager

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, matthews_corrcoef,
        confusion_matrix, classification_report, balanced_accuracy_score,
        cohen_kappa_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MetricsCalculator:
    """Calculate all evaluation metrics for classification tasks."""
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        average: str = 'macro',
    ):
        """
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
            average: Averaging method for multi-class ('macro', 'micro', 'weighted')
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds: torch.Tensor, labels: torch.Tensor, probs: Optional[torch.Tensor] = None):
        """Accumulate predictions."""
        self.all_preds.append(preds.detach().cpu().numpy())
        self.all_labels.append(labels.detach().cpu().numpy())
        if probs is not None:
            self.all_probs.append(probs.detach().cpu().numpy())
    
    def compute(self) -> Dict[str, Any]:
        """Compute all metrics from accumulated predictions."""
        if not self.all_preds:
            return {}
        
        y_pred = np.concatenate(self.all_preds)
        y_true = np.concatenate(self.all_labels)
        y_prob = np.concatenate(self.all_probs) if self.all_probs else None
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class and averaged metrics
        metrics['precision'] = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=self.average, zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        
        # Specificity (per class)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        metrics['confusion_matrix'] = cm.tolist()
        metrics['specificity_per_class'] = self._compute_specificity(cm).tolist()
        
        # MCC
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Cohen's Kappa
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # AUC metrics (if probabilities available)
        if y_prob is not None:
            try:
                if self.num_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=self.average)
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average=self.average)
                    # PR-AUC for multi-class (average over classes)
                    pr_aucs = []
                    for i in range(self.num_classes):
                        y_true_binary = (y_true == i).astype(int)
                        pr_aucs.append(average_precision_score(y_true_binary, y_prob[:, i]))
                    metrics['pr_auc'] = np.mean(pr_aucs)
                    metrics['pr_auc_per_class'] = pr_aucs
            except Exception as e:
                metrics['auc_error'] = str(e)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            labels=list(range(self.num_classes)),
            output_dict=True, zero_division=0
        )
        
        return metrics
    
    def _compute_specificity(self, cm: np.ndarray) -> np.ndarray:
        """Compute specificity per class from confusion matrix."""
        specificity = []
        for i in range(self.num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity.append(spec)
        return np.array(specificity)


def compute_all_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    num_classes: int = None,
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """Convenience function to compute all metrics in one call."""
    if num_classes is None:
        num_classes = preds.max().item() + 1 if preds.numel() > 0 else 2
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    calc = MetricsCalculator(num_classes, class_names)
    calc.update(preds, labels, probs)
    return calc.compute()


# Computational Metrics

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total,
        'trainable_params': trainable,
        'non_trainable_params': total - trainable,
    }


def count_flops(model: torch.nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
    """Compute FLOPs using fvcore."""
    if not FVCORE_AVAILABLE:
        return {'flops': -1, 'error': 'fvcore not available'}
    
    try:
        model.eval()
        dummy_input = torch.randn(input_shape)
        flop_counter = FlopCountAnalysis(model, dummy_input)
        total_flops = flop_counter.total()
        by_module = flop_counter.by_module()
        return {
            'total_flops': total_flops,
            'total_gflops': total_flops / 1e9,
            'by_module': {k: v for k, v in by_module.items()},
        }
    except Exception as e:
        return {'flops': -1, 'error': str(e)}


def measure_inference_time(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = 'cpu',
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """Measure inference time statistics."""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'median_ms': np.median(times) * 1000,
        'throughput_fps': 1.0 / np.mean(times),
    }


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


# Quantum-Specific Metrics

def compute_circuit_metrics(
    circuit_fn,
    num_qubits: int,
    num_layers: int,
    num_params: int,
) -> Dict[str, Any]:
    """Compute quantum circuit metrics."""
    import pennylane as qml
    
    dev = qml.device('default.qubit', wires=num_qubits)
    
    @qml.qnode(dev)
    def circuit(params, x):
        circuit_fn(params, x)
        return qml.probs(wires=range(num_qubits))
    
    # Get circuit specs
    specs = qml.specs(circuit)
    
    return {
        'num_qubits': num_qubits,
        'num_layers': num_layers,
        'num_parameters': num_params,
        'circuit_depth': specs['depth'] if 'depth' in specs else num_layers,
        'num_gates': specs['num_operations'] if 'num_operations' in specs else -1,
        'gate_types': specs['gate_types'] if 'gate_types' in specs else {},
        'num_wires': specs['num_wires'] if 'num_wires' in specs else num_qubits,
    }


def compute_fidelity(
    statevector_ideal: np.ndarray,
    statevector_noisy: np.ndarray,
) -> float:
    """Compute fidelity between ideal and noisy statevectors."""
    # Fidelity = |<psi_ideal|psi_noisy>|^2
    overlap = np.vdot(statevector_ideal, statevector_noisy)
    fidelity = np.abs(overlap) ** 2
    return float(fidelity)


def compute_noise_robustness(
    acc_ideal: float,
    acc_noisy: float,
) -> Dict[str, float]:
    """Compute noise robustness metrics."""
    absolute_drop = acc_ideal - acc_noisy
    relative_drop = absolute_drop / acc_ideal if acc_ideal > 0 else 0.0
    robustness = 1.0 - relative_drop
    return {
        'accuracy_ideal': acc_ideal,
        'accuracy_noisy': acc_noisy,
        'absolute_drop': absolute_drop,
        'relative_drop': relative_drop,
        'robustness_score': robustness,
    }


# Aggregation utilities

def aggregate_metrics_across_seeds(
    metrics_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics across multiple seeds (mean ± std)."""
    if not metrics_list:
        return {}
    
    # Get all keys
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    aggregated = {}
    for key in all_keys:
        values = [m.get(key) for m in metrics_list if key in m and isinstance(m[key], (int, float))]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
            aggregated[f'{key}_values'] = values
    
    return aggregated


def format_metrics_table(
    metrics: Dict[str, Any],
    prefix: str = '',
) -> str:
    """Format metrics as a readable table string."""
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, (int, np.integer)):
            lines.append(f"{prefix}{key}: {value}")
        elif isinstance(value, list):
            if all(isinstance(v, (int, float)) for v in value):
                lines.append(f"{prefix}{key}: [{', '.join(f'{v:.4f}' for v in value)}]")
            else:
                lines.append(f"{prefix}{key}: {value}")
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_metrics_table(value, prefix + '  '))
        else:
            lines.append(f"{prefix}{key}: {value}")
    return '\n'.join(lines)