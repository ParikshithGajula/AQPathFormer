"""
Quantum Patch Encoder for AQPathFormer

Hybrid classical-quantum patch encoding with configurable qubits, circuit depth, and feature maps.
Supports both quantum simulation and classical proxy modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Callable
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


class QuantumPatchEncoder(nn.Module):
    """
    Hybrid classical-quantum patch encoder.
    
    Pipeline:
    Classical CNN backbone → AdaptiveAvgPool → Linear(2^n) → Quantum Feature Map → Variational Circuit → Expectation Values
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_qubits: int = 4,
        circuit_depth: int = 2,
        feature_map: str = 'angle_embedding',
        ansatz: str = 'strongly_entangling_layers',
        classical_backbone: str = 'resnet18',
        projection_dim: Optional[int] = None,
        shots: Optional[int] = None,
        use_quantum: bool = True,
        device: str = 'cpu',
    ):
        """
        Args:
            in_channels: Input image channels
            num_qubits: Number of qubits (determines output dimension = 2^num_qubits)
            circuit_depth: Number of variational layers
            feature_map: 'angle_embedding', 'amplitude_embedding', 'iqp_embedding'
            ansatz: 'strongly_entangling_layers', 'basic_entangler_layers', 'simplified_two_design'
            classical_backbone: 'resnet18', 'resnet34', 'vit_small', 'identity'
            projection_dim: Output dimension (defaults to 2^num_qubits)
            shots: Number of shots for quantum device (None = exact statevector)
            use_quantum: If False, use classical proxy (PCA)
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.feature_map_type = feature_map
        self.ansatz_type = ansatz
        self.shots = shots
        self.use_quantum = use_quantum
        self.device = device
        
        # Output dimension is 2^num_qubits (expectation values per qubit)
        self.output_dim = projection_dim or (2 ** num_qubits)
        if self.output_dim != 2 ** num_qubits:
            print(f"Warning: projection_dim ({self.output_dim}) != 2^num_qubits ({2**num_qubits}). Using {self.output_dim}")
        
        # Classical backbone for feature extraction
        self.classical_backbone = self._create_backbone(classical_backbone, in_channels)
        backbone_out_dim = self._get_backbone_out_dim(classical_backbone)
        
        # Projection to quantum dimension
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out_dim, self.output_dim),
            nn.Tanh(),  # Bound outputs to [-1, 1] for angle embedding
        )
        
        # Quantum circuit
        if use_quantum:
            self.quantum_device = qml.device('default.qubit', wires=num_qubits, shots=shots)
            self.quantum_circuit = self._create_quantum_circuit()
            self.quantum_layer = qml.qnn.TorchLayer(self.quantum_circuit, weight_shapes=self._get_weight_shapes())
        else:
            # Classical proxy: learnable linear layer
            self.quantum_layer = nn.Linear(self.output_dim, self.output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _create_backbone(self, name: str, in_channels: int) -> nn.Module:
        """Create classical backbone."""
        import timm
        
        if name == 'identity':
            return nn.Identity()
        elif name == 'resnet18':
            model = timm.create_model('resnet18', pretrained=True, features_only=True, out_indices=(3,))
            if in_channels != 3:
                model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            return model
        elif name == 'resnet34':
            model = timm.create_model('resnet34', pretrained=True, features_only=True, out_indices=(3,))
            if in_channels != 3:
                model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            return model
        elif name == 'vit_small':
            model = timm.create_model('vit_small_patch16_224', pretrained=True, features_only=True, out_indices=(3,))
            return model
        else:
            return timm.create_model(name, pretrained=True, features_only=True, out_indices=(3,))
    
    def _get_backbone_out_dim(self, name: str) -> int:
        """Get output dimension of backbone."""
        dims = {
            'identity': 3 * 224 * 224,  # Will be flattened
            'resnet18': 256,  # layer3 output
            'resnet34': 256,  # layer3 output
            'vit_small': 384,
        }
        return dims.get(name, 512)
    
    def _create_quantum_circuit(self):
        """Create PennyLane quantum circuit as a QNode."""
        
        # Create device
        dev = qml.device('default.qubit', wires=self.num_qubits, shots=self.shots)
        
        @qml.qnode(dev, interface='torch')
        def circuit(inputs, weights):
            # inputs: (batch, output_dim) where output_dim = 2^num_qubits
            # weights: variational parameters
            
            # Feature map
            if self.feature_map_type == 'angle_embedding':
                qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            elif self.feature_map_type == 'amplitude_embedding':
                qml.AmplitudeEmbedding(inputs, wires=range(self.num_qubits), normalize=True)
            elif self.feature_map_type == 'iqp_embedding':
                qml.IQPEmbedding(inputs, wires=range(self.num_qubits), n_repeats=1)
            else:
                qml.AngleEmbedding(inputs, wires=range(self.num_qubits), rotation='Y')
            
            # Variational ansatz
            if self.ansatz_type == 'strongly_entangling_layers':
                qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            elif self.ansatz_type == 'basic_entangler_layers':
                qml.BasicEntanglerLayers(weights, wires=range(self.num_qubits))
            elif self.ansatz_type == 'simplified_two_design':
                qml.SimplifiedTwoDesign(weights, wires=range(self.num_qubits))
            else:
                qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            
            # Measurements: expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]
        
        return circuit
    
    def _get_weight_shapes(self) -> Dict[str, Tuple]:
        """Get weight shapes for TorchLayer."""
        if self.ansatz_type == 'strongly_entangling_layers':
            # (num_layers, num_qubits, 3)
            return {'weights': (self.circuit_depth, self.num_qubits, 3)}
        elif self.ansatz_type == 'basic_entangler_layers':
            # (num_layers, num_qubits)
            return {'weights': (self.circuit_depth, self.num_qubits)}
        elif self.ansatz_type == 'simplified_two_design':
            # (num_layers, num_qubits, 3)
            return {'weights': (self.circuit_depth, self.num_qubits, 3)}
        else:
            return {'weights': (self.circuit_depth, self.num_qubits, 3)}
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        if not self.use_quantum:
            nn.init.xavier_uniform_(self.quantum_layer.weight)
            nn.init.zeros_(self.quantum_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            Quantum features: (batch, output_dim)
        """
        # Classical feature extraction
        features = self.classical_backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]  # Take last feature map
        
        # Project to quantum dimension
        projected = self.projection(features)  # (batch, output_dim)
        
        # Quantum or classical processing
        if self.use_quantum:
            # PennyLane TorchLayer expects (batch, input_dim) -> (batch, output_dim)
            quantum_features = self.quantum_layer(projected)
        else:
            quantum_features = self.quantum_layer(projected)
        
        return quantum_features
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get circuit information for logging."""
        if not self.use_quantum:
            return {'mode': 'classical_proxy'}
        
        # Get circuit specs
        import pennylane as qml
        specs = qml.specs(self.quantum_circuit)
        
        return {
            'mode': 'quantum',
            'num_qubits': self.num_qubits,
            'circuit_depth': self.circuit_depth,
            'feature_map': self.feature_map_type,
            'ansatz': self.ansatz_type,
            'shots': self.shots,
            'circuit_depth': specs.get('depth', self.circuit_depth),
            'num_operations': specs.get('num_operations', -1),
            'gate_types': specs.get('gate_types', {}),
            'output_dim': self.output_dim,
        }


class ClassicalProxyEncoder(nn.Module):
    """Classical proxy for quantum encoder (PCA + Linear)."""
    
    def __init__(
        self,
        in_channels: int = 3,
        output_dim: int = 16,
        classical_backbone: str = 'resnet18',
    ):
        super().__init__()
        
        import timm
        self.backbone = timm.create_model(classical_backbone, pretrained=True, features_only=True, out_indices=(3,))
        
        # Get actual backbone output dimension
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 224, 224)
            feat = self.backbone(dummy)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            backbone_out = feat.shape[1]
        
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_out, output_dim),
            nn.Tanh(),
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return self.projection(features)
    
    def get_circuit_info(self) -> Dict[str, Any]:
        return {'mode': 'classical_proxy', 'output_dim': self.output_dim}


def create_quantum_patch_encoder(config: Dict[str, Any]) -> QuantumPatchEncoder:
    """Factory function to create quantum patch encoder from config."""
    return QuantumPatchEncoder(
        in_channels=config.get('in_channels', 3),
        num_qubits=config.get('num_qubits', 4),
        circuit_depth=config.get('circuit_depth', 2),
        feature_map=config.get('feature_map', 'angle_embedding'),
        ansatz=config.get('ansatz', 'strongly_entangling_layers'),
        classical_backbone=config.get('classical_backbone', 'resnet18'),
        projection_dim=config.get('projection_dim'),
        shots=config.get('shots'),
        use_quantum=config.get('use_quantum', True),
        device=config.get('device', 'cpu'),
    )


def create_classical_encoder(config: Dict[str, Any]) -> ClassicalProxyEncoder:
    """Factory function to create classical encoder from config."""
    return ClassicalProxyEncoder(
        in_channels=config.get('in_channels', 3),
        output_dim=config.get('output_dim', 16),
        classical_backbone=config.get('classical_backbone', 'resnet18'),
    )