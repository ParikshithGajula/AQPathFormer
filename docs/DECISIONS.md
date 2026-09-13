# DECISIONS.md - Technical Decision Log

## Decision Format
**Decision**: What was chosen  
**Alternatives**: What else was considered  
**Rationale**: Why this choice was made  
**Consequences**: Expected impact (positive/negative)  
**Date**: When decided  
**Status**: ACTIVE / SUPERSEDED / DEFERRED  

---

## DEC-001: Quantum Framework Selection
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: Use PennyLane as primary quantum framework with `default.qubit` simulator

**Alternatives Considered**:
- Qiskit with Aer simulator
- Qiskit + PennyLane hybrid
- Custom quantum simulation

**Rationale**:
- PennyLane has native PyTorch integration (`qml.qnn.TorchLayer`)
- Lighter weight than Qiskit for CPU-only simulation
- `default.qubit` is pure Python, no compilation needed
- Built-in gradient computation via parameter-shift rule
- Active development and good documentation

**Consequences**:
- (+) Faster development iteration
- (+) Native PyTorch autograd compatibility
- (-) Less mature noise modeling than Qiskit Aer
- (-) No direct IBM Quantum hardware access (not needed for this project)

---

## DEC-002: Pilot Dataset Selection
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: LC25000 (Lung and Colon Cancer Histopathological Images) as primary pilot dataset

**Alternatives Considered**:
- CRC-100K (Colorectal Cancer, 100K images, 9 classes)
- BreakHis (Breast Cancer, requires registration)
- CAMELYON16/17 (requires challenge registration)
- PANDA (Prostate, very large)
- TCGA (requires dbGaP authorization)

**Rationale**:
- **Publicly accessible**: Direct download from Zenodo/Kaggle, no registration
- **Appropriate size**: ~25,000 images, manageable on 8GB RAM
- **Multi-class**: 5 classes (lung_aca, lung_n, lung_scc, colon_aca, colon_n)
- **Multi-cancer**: Lung + Colon cancer types (aligns with cross-cancer objective)
- **Standard resolution**: 768×768, can resize to 224×224
- **Well-documented**: Clear class structure, used in literature

**Consequences**:
- (+) Immediate start without access delays
- (+) Sufficient for all development phases
- (-) Smaller than CRC-100K (less data for training)
- (-) Only 2 cancer types (limited cross-cancer diversity)
- **Mitigation**: Add CRC-100K and BreakHis in Phase 11

---

## DEC-003: Image Resolution & Patch Size
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: 224×224 input resolution, 16×16 patches (196 patches/image)

**Alternatives Considered**:
- 256×256 with 16×16 patches (256 patches)
- 224×224 with 32×32 patches (49 patches)
- 512×512 with 32×32 patches (256 patches)

**Rationale**:
- 224×224 is ViT standard (ImageNet pretraining compatible)
- 196 patches fits in 8GB RAM with batch_size=8
- 16×16 patches balance local detail vs global context
- Compatible with timm pretrained ViT weights

**Consequences**:
- (+) Memory efficient for CPU training
- (+) Can use pretrained ViT weights from timm
- (-) May lose fine cellular detail at 224×224
- (-) Fixed patch size contradicts "adaptive" goal (addressed by Adaptive Patch Generator)

---

## DEC-004: Quantum Qubit Count
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: 4-6 qubits (configurable, default 4)

**Alternatives Considered**:
- 2-3 qubits (too few for meaningful representation)
- 8-10 qubits (2^10=1024 dims, too slow on CPU)
- Dynamic qubit allocation per patch

**Rationale**:
- 4 qubits = 16 dimensions, 6 qubits = 64 dimensions
- Classical projection: CNN features → 2^n dimensions
- PennyLane simulation scales exponentially: O(2^n) memory, O(4^n) time
- 4 qubits: ~1ms/forward pass; 6 qubits: ~100ms/forward pass (est.)
- Configurable for ablation studies

**Consequences**:
- (+) Feasible simulation time on CPU
- (+) Allows quantum advantage exploration
- (-) Severe dimensionality bottleneck (classical features >> quantum dims)
- **Mitigation**: Strong classical projection + feature map design

---

## DEC-005: Classical-to-Quantum Projection Strategy
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: CNN backbone → AdaptiveAvgPool → Linear(2^n) → Quantum Feature Map

**Alternatives Considered**:
- Direct patch flattening + PCA
- Learned linear projection only
- Autoencoder bottleneck

**Rationale**:
- CNN backbone (ResNet18/ViT-small) extracts meaningful visual features
- AdaptiveAvgPool makes it resolution-agnostic
- Linear layer learns optimal projection to 2^n dims
- Compatible with PennyLane's AngleEmbedding

**Consequences**:
- (+) End-to-end trainable classical-quantum pipeline
- (+) Classical features guide quantum encoding
- (-) Adds classical parameters (but necessary for feasibility)
- (-) Quantum portion only sees compressed representation

---

## DEC-006: Adaptive Patch Generator Mechanism
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: Attention-based patch scoring with top-k selection

**Alternatives Considered**:
- Tissue segmentation + region sampling
- Random patch sampling
- Uniform grid with learned weights
- Reinforcement learning for patch selection

**Rationale**:
- Learnable attention scores from patch embeddings
- Top-k differentiable via Gumbel-Softmax or straight-through estimator
- Computationally efficient (single forward pass)
- Interpretable: shows which patches model attends to

**Consequences**:
- (+) End-to-end trainable
- (+) Directly optimizes for classification
- (+) Visualizable patch importance maps
- (-) May ignore subtle but important regions initially
- (-) Straight-through estimator introduces gradient bias

---

## DEC-007: Adaptive Quantum Attention Definition
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: Patch-conditioned circuit depth/parameter allocation (software-level policy)

**Alternatives Considered**:
- Dynamic quantum circuit execution (hardware-level)
- Per-patch different ansatz structures
- Classical attention with quantum-inspired scoring

**Rationale**:
- True dynamic circuit allocation requires hardware support
- Software-level: compute attention weights, allocate more measurement shots to important patches
- Implement as: attention scores → measurement budget per patch → weighted expectation values
- Clearly documented as "adaptive measurement allocation" not "dynamic circuits"

**Consequences**:
- (+) Honest about simulation vs hardware capabilities
- (+) Implements the "adaptive" concept tractably
- (+) Measurable: shot allocation vs accuracy trade-off
- (-) Not true dynamic circuit allocation
- **Documentation**: Explicitly defined in `docs/AQPATHFORMER_DESIGN.md`

---

## DEC-008: Cross-Cancer Learning Strategy
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: Shared encoder + dataset-specific prediction heads

**Alternatives Considered**:
- Joint multi-dataset training (single head)
- Domain adaptation (DANN, CORAL)
- Leave-one-cancer-out evaluation
- Separate models per cancer type

**Rationale**:
- Label spaces differ across datasets (incompatible for joint training)
- Shared encoder learns universal histopathological features
- Dataset-specific heads handle different class cardinalities
- Enables leave-one-dataset-out evaluation naturally

**Consequences**:
- (+) Handles label incompatibility cleanly
- (+) Enables transfer learning evaluation
- (+) Parameter efficient (shared backbone)
- (-) Requires aligned preprocessing across datasets
- (-) Domain shift may limit shared encoder effectiveness

---

## DEC-009: Noise Model Selection
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: PennyLane built-in noise models: DepolarizingChannel, AmplitudeDampingChannel

**Alternatives Considered**:
- Qiskit Aer noise models (requires Qiskit)
- Custom noise channels
- No noise modeling (ideal only)

**Rationale**:
- PennyLane native, no additional dependencies
- Depolarizing: general gate error model
- AmplitudeDamping: T1 relaxation (energy loss)
- Configurable error rates for robustness curves

**Consequences**:
- (+) Integrated with PennyLane simulation
- (+) Represents realistic NISQ noise
- (-) Limited to single-qubit channels (no crosstalk)
- (-) No readout error modeling (can add separately)

---

## DEC-010: Experiment Tracking
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: Weights & Biases (wandb) + local JSON registry

**Alternatives Considered**:
- MLflow (local only)
- TensorBoard + custom logging
- Sacred/MLflow combo
- Plain CSV/JSON only

**Rationale**:
- wandb: excellent PyTorch integration, auto-logging, web UI
- Local JSON: backup, reproducibility, no internet required
- Dual approach: cloud for visualization, local for provenance

**Consequences**:
- (+) Rich experiment comparison UI
- (+) Automatic system/environment capture
- (+) Offline mode supported
- (-) Requires wandb account (free tier sufficient)
- (-) Local JSON duplication adds complexity

---

## DEC-011: Baseline Model Selection
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: Implement 5 core baselines from timm: ResNet50, ViT-B/16, Swin-T, ConvNeXt-T, EfficientNet-B0

**Alternatives Considered**:
- All 9 baselines from spec (UNI, CONCH, QCNN, Hybrid QViT)
- Custom implementations from scratch

**Rationale**:
- timm provides all 5 with pretrained weights
- UNI/CONCH require specific weights/licenses (investigate later)
- QCNN/Hybrid QViT are research models (implement in Phase 7-8)
- 5 strong baselines sufficient for meaningful comparison

**Consequences**:
- (+) Immediate implementation with pretrained weights
- (+) Standardized evaluation via timm
- (-) UNI/CONCH not in initial baseline set
- **Phase 6 Extension**: Add UNI/CONCH if weights accessible

---

## DEC-012: Training Configuration Defaults
**Date**: 2026-09-13  
**Status**: ACTIVE

**Decision**: 
- Batch size: 8 (adaptive via gradient accumulation)
- Learning rate: 1e-4 (ViT), 1e-3 (CNN)
- Optimizer: AdamW
- Scheduler: Cosine annealing with warmup
- Epochs: 20 (initial), 50 (final)
- Mixed precision: Disabled (CPU)
- Seed: 42 (primary), [42, 123, 456, 789, 999] (statistical)

**Alternatives Considered**:
- SGD with momentum
- Larger batches with gradient accumulation
- Different schedulers (step, plateau)

**Rationale**:
- AdamW standard for ViT training
- Cosine annealing with warmup best practice
- Batch size 8 fits 8GB RAM at 224×224
- Gradient accumulation simulates batch 32
- 5 seeds for statistical validation

**Consequences**:
- (+) Reproducible, standard configuration
- (+) Comparable to literature
- (-) Longer training due to small effective batch
- (-) CPU training will be slow (~hours/epoch)

---