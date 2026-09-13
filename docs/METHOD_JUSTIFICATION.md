# METHOD_JUSTIFICATION.md - Design Decision Justifications

## Overview
This document provides detailed scientific justifications for all major design decisions in AQPathFormer, linking each to literature, theoretical foundations, and practical constraints.

---

## 1. Architecture-Level Decisions

### 1.1 Hybrid Classical-Quantum Architecture
**Decision**: Classical backbone → Quantum encoder → Classical Transformer
**Justification**:
- **Theoretical**: Quantum advantage conjectured for specific feature mappings (Havlíček et al., Nature 2019); classical layers handle high-dimensional data efficiently
- **Practical**: NISQ devices/simulators limited to 4-10 qubits; classical CNN reduces 768×768×3 → 2^n dimensions
- **Literature**: All successful QViT implementations use this hybrid pattern (no pure quantum ViT exists for 224×224 images)
- **Alternative Rejected**: Pure quantum (infeasible), Quantum attention only (O(n²) circuits prohibitive)

### 1.2 Modular Component Design (8 Separate Modules)
**Decision**: Each AQPathFormer component as independent PyTorch module
**Justification**:
- **Scientific**: Enables controlled ablation studies (7 ablations specified)
- **Engineering**: Unit testing, debugging, gradient checking per component
- **Reproducibility**: Clear input/output contracts, configuration isolation
- **Literature**: Standard practice in complex model development (e.g., detectron2, timm)

### 1.3 Patch-Level Classification (Not Slide-Level)
**Decision**: Classify individual 224×224 patches, not whole slides
**Justification**:
- **Scope**: Semester project; slide-level requires MIL framework
- **Quantum Simulation**: Patch-level enables quantum circuit per patch (tractable)
- **Evaluation**: Direct comparison with patch-based baselines
- **Extensibility**: MIL head can be added later (separate module)

---

## 2. Adaptive Patch Generator

### 2.1 Attention-Based Patch Scoring
**Decision**: Learnable attention scores → Top-k selection via Gumbel-Softmax
**Justification**:
- **Theoretical**: Differentiable patch selection enables end-to-end training
- **Literature**: A-ViT, EViT, DynamicViT use similar token selection
- **Pathology Relevance**: Large background regions in WSIs; attention focuses on tissue
- **Alternatives Considered**:
  - Tissue segmentation (requires annotations, not available)
  - Random sampling (not adaptive)
  - Reinforcement learning (unstable, high variance)

### 2.2 Top-k with Straight-Through Estimator
**Decision**: Hard top-k selection with straight-through gradient estimator
**Justification**:
- **Computational**: Reduces patches from 196 → k (e.g., 48), saving quantum compute
- **Gradient Flow**: Straight-through estimator provides usable gradients
- **Literature**: Standard in dynamic ViT literature (EViT, A-ViT)
- **Ablation**: Compare k=196 (fixed) vs k=48 (adaptive) in EXP010 vs EXP011

### 2.3 Patch Scoring Network
**Decision**: Lightweight MLP on patch embeddings → scalar scores
**Justification**:
- **Efficiency**: Minimal overhead vs full attention
- **Input**: Patch embeddings from classical projection (before quantum)
- **Output**: Single score per patch → softmax → top-k

---

## 3. Quantum Patch Encoder

### 3.1 Classical Projection: CNN → 2^n Dimensions
**Decision**: ResNet18/ViT-Small → AdaptiveAvgPool(1) → Linear(2^n)
**Justification**:
- **Dimensionality**: 768×768×3 = 1.7M pixels → 16-64 quantum dimensions (2^4 to 2^6)
- **Feature Quality**: CNN extracts hierarchical features; random projection loses structure
- **Trainable**: Linear layer learns optimal projection for quantum encoding
- **Literature**: Standard in hybrid quantum-classical image classification (e.g., PennyLane tutorials, QViT papers)

### 3.2 Feature Map: AngleEmbedding
**Decision**: PennyLane `AngleEmbedding` (RX/RZ rotations per qubit)
**Justification**:
- **Standard**: Most common feature map for continuous data
- **Expressivity**: Maps [0, π] features to rotation angles; universal for single-qubit states
- **Gradient**: Analytic gradients via parameter-shift rule
- **Alternatives Considered**:
  - AmplitudeEmbedding (requires 2^n amplitudes, state preparation hard)
  - IQPEmbedding (more expressive but deeper circuits)

### 3.3 Variational Ansatz: StronglyEntanglingLayers
**Decision**: PennyLane `StronglyEntanglingLayers` with 2-4 layers
**Justification**:
- **Expressivity**: All-to-all entanglement; hardware-efficient
- **Trainability**: Shallow depth avoids barren plateaus (McClean et al., 2018)
- **Parameters**: 3 params/qubit/layer × 4-6 qubits × 2-4 layers = 72-288 params (manageable)
- **Literature**: Default in PennyLane; used in QML benchmarks

### 3.4 Measurement: Pauli-Z Expectation Values
**Decision**: `qml.expval(qml.PauliZ(w))` for each qubit → n-dimensional output
**Justification**:
- **Output Dimension**: n qubits → n classical features (matches classical projection dim)
- **Information**: Expectation values capture quantum state properties
- **Gradient**: Analytic via parameter-shift
- **Alternative**: Probabilities (2^n outcomes, too many for n=6)

### 3.5 Classical Proxy Mode (PCA)
**Decision**: Classical PCA projection for debugging without quantum simulation
**Justification**:
- **Speed**: PCA is O(d³) once vs O(2^n) per quantum forward pass
- **Debugging**: Verify gradient flow, tensor shapes, training pipeline
- **Ablation**: Classical encoder baseline (EXP020) uses this
- **Honesty**: Clearly labeled "classical proxy" - never reported as quantum results

---

## 4. Adaptive Quantum Multi-Head Attention

### 4.1 Software-Level Adaptive Measurement Allocation
**Decision**: Attention weights → measurement shot budget per patch (NOT dynamic circuits)
**Justification**:
- **Hardware Reality**: True dynamic circuit allocation requires quantum hardware with mid-circuit measurement
- **Simulation Feasibility**: PennyLane supports variable shots per circuit execution
- **Scientific Honesty**: Explicitly defined as "adaptive measurement allocation" in AQPATHFORMER_DESIGN.md
- **Mechanism**:
  1. Compute classical attention scores for each patch
  2. Normalize to shot budget (e.g., total 1000 shots distributed proportionally)
  3. Execute quantum encoder with patch-specific shots
  4. Weight expectation values by attention scores
- **Ablation**: EXP030 (uniform shots) vs EXP031 (adaptive shots)

### 4.2 Multi-Head Structure
**Decision**: h heads, each with independent quantum encoder (shared weights optional)
**Justification**:
- **Parallelism**: Each head processes different feature subspaces
- **Quantum**: Shared weights reduce parameters; independent allows head specialization
- **Configuration**: `num_heads=4`, `shared_quantum_weights=True` (default)

---

## 5. Multi-Scale Quantum Feature Fusion

### 5.1 Three-Scale Pyramid
**Decision**: Scales = {4×, 8×, 16×} downsampling from 224×224 → {56×56, 28×28, 14×14}
**Justification**:
- **Pathology Relevance**:
  - 4× (56×56): Tissue architecture (glands, ducts)
  - 8× (28×28): Cellular clusters, stroma patterns
  - 16× (14×14): Nuclear detail, texture
- **Literature**: Pathologists use 4×, 10×, 20×, 40×; our scales approximate
- **Computational**: 3 scales × quantum encoder = 3× cost (manageable with small batches)

### 5.2 Learnable Fusion Weights
**Decision**: Softmax-weighted sum of scale-specific expectation vectors
**Justification**:
- **Flexibility**: Model learns optimal scale combination per dataset/task
- **Interpretability**: Fusion weights reveal scale importance
- **Gradient**: End-to-end differentiable
- **Alternative**: Concatenation + Linear (more params, less interpretable)

### 5.3 Independent Quantum Encoders Per Scale
**Decision**: Separate quantum encoder per scale (not weight-shared)
**Justification**:
- **Feature Diversity**: Different scales need different quantum feature maps
- **Parameters**: 3 × ~100 params = 300 params (negligible)
- **Ablation**: Compare weight-shared vs independent in EXP041 variants

---

## 6. Cross-Cancer Representation Learning

### 6.1 Shared Encoder + Dataset-Specific Heads
**Decision**: Single AQPathFormer backbone → Multiple prediction heads
**Justification**:
- **Label Incompatibility**: LC25000 (5 classes) ≠ CRC-100K (9 classes) ≠ BreakHis (8 classes)
- **Transfer Learning**: Low-level features (texture, edges) transfer; high-level (tissue types) differ
- **Literature**: UNI, CONCH use shared encoder; multi-task learning standard
- **Parameter Efficiency**: One backbone vs N separate models

### 6.2 Leave-One-Dataset-Out Evaluation
**Decision**: Train on N-1 datasets, test on held-out dataset
**Justification**:
- **Domain Generalization**: Measures true cross-cancer transfer
- **No Label Alignment Needed**: Each dataset evaluated on its own labels
- **Statistical Rigor**: Standard domain generalization protocol

### 6.3 Domain Adaptation (Not Primary)
**Decision**: Not using DANN/CORAL as primary method
**Justification**:
- **Complexity**: Adds adversarial/discriminator components
- **Data Requirement**: Needs target domain data during training
- **Our Setting**: Leave-one-out evaluates zero-shot transfer (stronger claim)

---

## 7. Noise-Aware Quantum Layer

### 7.1 Noise Models: Depolarizing + Amplitude Damping
**Decision**: PennyLane `DepolarizingChannel` + `AmplitudeDampingChannel`
**Justification**:
- **Depolarizing**: General gate error model (Pauli X/Y/Z with equal probability)
- **Amplitude Damping**: T1 relaxation (energy loss, physically motivated)
- **Coverage**: Captures both coherent and incoherent errors
- **Parameters**: Single rate per channel (configurable for robustness curves)

### 7.2 Noise Injection During Training
**Decision**: Apply noise channels during forward pass in training (not just evaluation)
**Justification**:
- **Noise-Aware Training**: Model learns noise-robust representations
- **Regularization**: Noise acts as data augmentation in Hilbert space
- **Literature**: Emerging area (noise-aware QNN training)
- **Comparison**: EXP060 (no noise during train) vs EXP061 (noise during train)

### 7.3 Configurable Error Rates
**Decision**: Error rates as hyperparameters (default: 0.01)
**Justification**:
- **Robustness Curves**: Sweep rates 0.001 → 0.1 for analysis
- **NISQ Reality**: Current devices ~0.001-0.01 per gate
- **Ablation**: Measure accuracy vs noise rate

---

## 8. Hybrid Classical-Quantum Decoder

### 8.1 Classical Transformer Decoder
**Decision**: Standard Transformer decoder layers (no quantum)
**Justification**:
- **Quantum Advantage**: Hypothesized in encoding/attention, not decoding
- **Efficiency**: Decoder operates on fused features (already classical)
- **Flexibility**: Standard cross-attention, MLP, LayerNorm
- **Output**: Sequence of fused features → classification head

---

## 9. Multi-Cancer Prediction Head

### 9.1 Configurable MLP Head
**Decision**: Linear → LayerNorm → GELU → Dropout → Linear(num_classes)
**Justification**:
- **Flexibility**: `num_classes` set per dataset
- **Standard**: ViT classification head pattern
- **Regularization**: Dropout (0.1) prevents overfitting

### 9.2 Explicit Label Mapping
**Decision**: Dataset config specifies `class_to_idx` mapping
**Justification**:
- **Reproducibility**: Fixed class ordering across runs
- **Multi-Dataset**: Each dataset has own mapping
- **Inference**: Mapping saved with checkpoint

---

## 10. Training Configuration

### 10.1 Optimizer: AdamW
**Decision**: AdamW (lr=1e-4 for ViT, 1e-3 for CNN)
**Justification**:
- **Standard**: Default for ViT training (DeiT, BEiT, etc.)
- **Weight Decay**: 1e-4 prevents overfitting
- **Quantum Params**: Same optimizer for classical + quantum params

### 10.2 Scheduler: Cosine Annealing with Warmup
**Decision**: 2-epoch warmup → cosine decay to 1e-6
**Justification**:
- **Stability**: Warmup prevents early divergence (especially quantum params)
- **Convergence**: Cosine annealing standard for ViT
- **Epochs**: 20 initial, 50 final

### 10.3 Batch Size: 8 + Gradient Accumulation (×4 = effective 32)
**Decision**: Physical batch 8 (8GB RAM limit) → accumulate 4 steps
**Justification**:
- **Memory**: 224×224 × 8 patches × model fits in 8GB
- **Optimization**: Effective batch 32 matches literature
- **BatchNorm**: Use GroupNorm/LayerNorm (batch stats unreliable with accumulation)

### 10.4 Mixed Precision: Disabled (CPU)
**Decision**: No AMP on CPU
**Justification**: PyTorch CPU AMP limited benefit; adds complexity

### 10.5 Seeds: 5 Seeds for Statistical Validation
**Decision**: Primary seed 42; final experiments use [42, 123, 456, 789, 999]
**Justification**:
- **Variance Estimation**: Report mean ± std across seeds
- **Reproducibility**: Fixed seeds in config
- **Statistical Tests**: Enable t-test between methods

---

## 11. Evaluation Protocol

### 11.1 Metrics Selection
**Decision**: 12+ metrics (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, MCC, FLOPs, Params, Time)
**Justification**:
- **Medical Relevance**: Specificity, Sensitivity (Recall), MCC critical for medical
- **Imbalance Handling**: Macro-averaging, PR-AUC, MCC robust to imbalance
- **Computational**: FLOPs, params, time for efficiency analysis
- **Quantum**: Circuit depth, qubits, fidelity for quantum characterization

### 11.2 Statistical Validation
**Decision**: 5 seeds → mean ± std; paired t-test for comparisons
**Justification**:
- **Rigor**: Single seed insufficient for claims
- **Standard**: ML reproducibility standards (Pineau et al., 2021)
- **Reporting**: Confidence intervals where n≥5

### 11.3 Computational Measurement
**Decision**: `fvcore.nn.FlopCountAnalysis` + wall-clock timing
**Justification**:
- **Standard**: fvcore used in detectron2, fair comparison
- **CPU Timing**: `time.perf_counter()` averaged over 100 inferences

---

## 12. Quantum-Specific Justifications

### 12.1 Qubit Count: 4-6 (Configurable)
**Decision**: Default 4 qubits (16 dims), max 6 (64 dims) for ablations
**Justification**:
- **Simulation Cost**: O(2^n) memory, O(4^n) time
  - 4 qubits: 16 dims, ~1ms forward
  - 6 qubits: 64 dims, ~100ms forward
  - 8 qubits: 256 dims, ~10s forward (too slow)
- **Expressivity**: 16-64 dims sufficient for patch features (PCA baseline)
- **Ablation**: EXP021 variants with 4/6 qubits

### 12.2 Circuit Depth: 2-4 Layers
**Decision**: Default 2 layers, ablate up to 4
**Justification**:
- **Barren Plateaus**: Gradient variance decays exponentially with depth
- **Expressivity**: 2 layers sufficient for 4-6 qubits (empirical)
- **Training Time**: Linear in depth

### 12.3 Shots: 1000 (Configurable)
**Decision**: 1000 shots per circuit execution
**Justification**:
- **Statistical Error**: ~1/√1000 ≈ 3% standard error
- **Time**: 1000 shots × 196 patches × batch 8 = manageable
- **Adaptive**: Attention can redistribute shots (EXP031)

### 12.4 PennyLane `default.qubit` Simulator
**Decision**: Pure Python state-vector simulator
**Justification**:
- **No Dependencies**: No C++ compilation, works on any Python env
- **Exact**: State-vector simulation (no sampling noise in ideal mode)
- **Autograd**: Native PyTorch gradient support
- **Alternative**: `lightning.qubit` (C++, faster but compile issues on Windows)

---

## 13. Reproducibility Measures

### 13.1 Configuration-Driven Experiments
**Decision**: All hyperparameters in YAML configs
**Justification**:
- **Reproducibility**: Config + seed + code = exact reproduction
- **Version Control**: Configs in git
- **Experiment Registry**: Auto-generated from config

### 13.2 Environment Capture
**Decision**: `environment.json` with pip freeze, git commit, hardware info
**Justification**:
- **Exact Reproduction**: Dependency versions recorded
- **Audit Trail**: Git commit links code to experiment

### 13.3 Raw Output Preservation
**Decision**: Save predictions.csv, confusion_matrix.png, metrics.json, train.log
**Justification**:
- **Post-hoc Analysis**: Recompute metrics without retraining
- **Transparency**: Raw data available for audit

---

## 14. Limitations Acknowledged

| Limitation | Mitigation |
|------------|------------|
| CPU-only, 8GB RAM | Small batches, 224×224, 4-6 qubits |
| Quantum simulation slow | Classical proxy mode, circuit caching |
| No patient IDs in LC25000 | Fixed seed splits, document limitation |
| No quantum hardware access | Simulation only, honest labeling |
| Single cancer type in CRC/BreakHis | LC25000 has 2 cancers; leave-one-out eval |
| 20 epochs initial | Extend to 50 for final if converging |

---

## 15. Traceability to Mentor Specification

| Spec Item | Design Decision | Justification Reference |
|-----------|-----------------|------------------------|
| Adaptive Patch Generator | Attention-based top-k | Section 2 |
| Quantum Patch Encoder | CNN→2^n→AngleEmbedding→VQC | Section 3 |
| Adaptive Quantum Attention | Measurement shot allocation | Section 4 |
| Multi-Scale Quantum Fusion | 3-scale pyramid + learnable fusion | Section 5 |
| Cross-Cancer Learning | Shared encoder + dataset heads | Section 6 |
| Noise-Aware Layer | Depolarizing + AmplitudeDamping during train | Section 7 |
| Hybrid Decoder | Classical Transformer decoder | Section 8 |
| Multi-Cancer Head | Configurable MLP per dataset | Section 9 |

---

*Last Updated: 2026-09-13*
*Phase 2 - Method Justification Complete*