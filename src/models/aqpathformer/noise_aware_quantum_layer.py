"""
Noise-Aware Quantum Layer for AQPathFormer

Implements configurable noise injection during quantum circuit execution
for noise-aware training and robustness evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


class NoiseAwareQuantumLayer(nn.Module):
    """
    Noise-aware quantum layer that injects configurable noise during training.
    
    Supports:
    - Depolarizing noise (general gate errors)
    - Amplitude damping (T1 relaxation)
    - Phase damping (T2 dephasing)
    - Readout error
    - Training with noise for robustness
    """
    
    def __init__(
        self,
        base_circuit: callable,
        num_qubits: int = 4,
        noise_config: Optional[Dict[str, Any]] = None,
        noise_during_training: bool = True,
        noise_during_eval: bool = False,
    ):
        """
        Args:
            base_circuit: PennyLane circuit function (without noise)
            num_qubits: Number of qubits
            noise_config: Dict specifying noise models and rates
            noise_during_training: Inject noise during training
            noise_during_eval: Inject noise during evaluation
        """
        super().__init__()
        
        self.base_circuit = base_circuit
        self.num_qubits = num_qubits
        self.noise_during_training = noise_during_training
        self.noise_during_eval = noise_during_eval
        
        # Default noise config
        self.noise_config = noise_config or {
            'depolarizing': {'enabled': True, 'rate': 0.01},
            'amplitude_damping': {'enabled': True, 'rate': 0.01},
            'phase_damping': {'enabled': False, 'rate': 0.01},
            'readout_error': {'enabled': False, 'p0': 0.01, 'p1': 0.01},
        }
        
        # Create noisy device
        self.noisy_device = self._create_noisy_device()
        
        # Wrap circuit with noise
        self.noisy_circuit = self._create_noisy_circuit()
        
        # For tracking
        self.current_noise_level = 1.0  # Multiplier for noise rates
    
    def _create_noisy_device(self) -> qml.device:
        """Create PennyLane device with noise."""
        return qml.device(
            'default.mixed',  # Mixed state simulator for noise
            wires=self.num_qubits,
        )
    
    def _create_noisy_circuit(self) -> callable:
        """Create circuit with noise injection."""
        
        @qml.qnode(self.noisy_device)
        def noisy_circuit(inputs, weights):
            # Feature map
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            
            # Variational layers with noise
            for layer_idx in range(weights.shape[0]):
                # StronglyEntanglingLayers
                for wire in range(self.num_qubits):
                    qml.Rot(*weights[layer_idx, wire], wires=wire)
                    
                    # Depolarizing noise after single-qubit gates
                    if self.noise_config['depolarizing']['enabled']:
                        rate = self.noise_config['depolarizing']['rate'] * self.current_noise_level
                        qml.DepolarizingChannel(rate, wires=wire)
                    
                    # Amplitude damping
                    if self.noise_config['amplitude_damping']['enabled']:
                        rate = self.noise_config['amplitude_damping']['rate'] * self.current_noise_level
                        qml.AmplitudeDamping(rate, wires=wire)
                    
                    # Phase damping
                    if self.noise_config['phase_damping']['enabled']:
                        rate = self.noise_config['phase_damping']['rate'] * self.current_noise_level
                        qml.PhaseDamping(rate, wires=wire)
                
                # Entangling gates
                for wire in range(self.num_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % self.num_qubits])
                    
                    # Depolarizing on two-qubit gates
                    if self.noise_config['depolarizing']['enabled']:
                        rate = self.noise_config['depolarizing']['rate'] * self.current_noise_level
                        qml.DepolarizingChannel(rate, wires=[wire, (wire + 1) % self.num_qubits])
            
            # Readout error
            if self.noise_config['readout_error']['enabled']:
                p0 = self.noise_config['readout_error']['p0']
                p1 = self.noise_config['readout_error']['p1']
                # Simulate readout error by bit-flip probabilities
                for wire in range(self.num_qubits):
                    qml.BitFlip(p0, wires=wire)
            
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]
        
        return noisy_circuit
    
    def forward(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        apply_noise: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Execute quantum circuit with optional noise.
        
        Args:
            inputs: (batch, input_dim) feature inputs
            weights: (num_layers, num_qubits, 3) variational parameters
            apply_noise: Override noise setting (None = use training/eval mode)
        Returns:
            expectation values: (batch, num_qubits)
        """
        # Determine if noise should be applied
        if apply_noise is None:
            apply_noise = self.noise_during_training if self.training else self.noise_during_eval
        
        if apply_noise:
            # Use noisy circuit
            return self._execute_noisy(inputs, weights)
        else:
            # Use ideal circuit
            return self._execute_ideal(inputs, weights)
    
    def _execute_ideal(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Execute ideal (noiseless) circuit."""
        # Use base_circuit on default.qubit
        dev = qml.device('default.qubit', wires=self.num_qubits)
        
        @qml.qnode(dev)
        def ideal_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            
            for layer_idx in range(weights.shape[0]):
                for wire in range(self.num_qubits):
                    qml.Rot(*weights[layer_idx, wire], wires=wire)
                for wire in range(self.num_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % self.num_qubits])
            
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]
        
        # Execute batch
        batch_size = inputs.shape[0]
        results = []
        for i in range(batch_size):
            result = ideal_circuit(inputs[i], weights)
            results.append(result)
        
        return torch.tensor(results, dtype=inputs.dtype, device=inputs.device)
    
    def _execute_noisy(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Execute noisy circuit."""
        batch_size = inputs.shape[0]
        results = []
        
        for i in range(batch_size):
            result = self.noisy_circuit(inputs[i], weights)
            results.append(result)
        
        return torch.tensor(results, dtype=inputs.dtype, device=inputs.device)
    
    def set_noise_level(self, level: float):
        """Scale all noise rates by a factor (for noise sweeps)."""
        self.current_noise_level = level
    
    def get_noise_config(self) -> Dict[str, Any]:
        """Get effective noise configuration."""
        effective = {}
        for noise_type, config in self.noise_config.items():
            if config.get('enabled', False):
                effective[noise_type] = config.copy()
                if 'rate' in effective[noise_type]:
                    effective[noise_type]['effective_rate'] = (
                        effective[noise_type]['rate'] * self.current_noise_level
                    )
        return effective


class NoiseScheduler:
    """
    Schedule noise levels during training for noise-aware training.
    """
    
    def __init__(
        self,
        noise_layer: NoiseAwareQuantumLayer,
        schedule: str = 'linear',
        start_level: float = 0.0,
        end_level: float = 1.0,
        warmup_epochs: int = 5,
    ):
        """
        Args:
            noise_layer: The noise layer to schedule
            schedule: 'linear', 'cosine', 'step', 'constant'
            start_level: Initial noise level
            end_level: Final noise level
            warmup_epochs: Epochs before noise starts increasing
        """
        self.noise_layer = noise_layer
        self.schedule = schedule
        self.start_level = start_level
        self.end_level = end_level
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self, epoch: int):
        """Update noise level for epoch."""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            level = self.start_level
        else:
            progress = (epoch - self.warmup_epochs) / 100  # Assume 100 epochs max
            progress = min(1.0, progress)
            
            if self.schedule == 'linear':
                level = self.start_level + (self.end_level - self.start_level) * progress
            elif self.schedule == 'cosine':
                level = self.start_level + (self.end_level - self.start_level) * (
                    1 - np.cos(np.pi * progress) / 2
                )
            elif self.schedule == 'step':
                level = self.end_level if progress > 0.5 else self.start_level
            else:
                level = self.end_level
        
        self.noise_layer.set_noise_level(level)
        return level


def create_noise_aware_layer(
    base_circuit: callable,
    config: Dict[str, Any],
) -> NoiseAwareQuantumLayer:
    """Factory to create noise-aware layer from config."""
    return NoiseAwareQuantumLayer(
        base_circuit=base_circuit,
        num_qubits=config.get('num_qubits', 4),
        noise_config=config.get('noise_config'),
        noise_during_training=config.get('noise_during_training', True),
        noise_during_eval=config.get('noise_during_eval', False),
    )