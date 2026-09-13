# MENTOR_ALIGNMENT.md - Specification to Implementation Mapping

## Format
**Mentor Specification Item** → **Implementation Component** → **Source File** → **Experiment** → **Result** → **Status**

Status Values: NOT_STARTED | IN_PROGRESS | COMPLETE | PARTIAL | BLOCKED

---

## Framework Components (8 Core Modules)

### 1. Adaptive Patch Generator
- **Mentor Spec**: "Adaptive Patch Generator" - Do not use only fixed patch size; design adaptive mechanism selecting/weighting patches using image/tissue characteristics
- **Implementation**: Attention-based patch scoring with top-k selection via Gumbel-Softmax
- **Source File**: `src/models/aqpathformer/adaptive_patch_generator.py`
- **Experiment**: EXP010 (Fixed) vs EXP011 (Adaptive) - Ablation A
- **Result**: TBD
- **Status**: NOT_STARTED

### 2. Quantum Patch Encoder
- **Mentor Spec**: "Quantum Patch Encoder" - Trainable hybrid quantum/classical patch encoder with configurable qubits, depth, feature map, variational layers
- **Implementation**: CNN backbone → AdaptiveAvgPool → Linear(2^n) → AngleEmbedding → StronglyEntanglingLayers → Expectation values
- **Source File**: `src/models/aqpathformer/quantum_patch_encoder.py`
- **Experiment**: EXP020 (Classical) vs EXP021 (Quantum) - Ablation B
- **Result**: TBD
- **Status**: NOT_STARTED

### 3. Adaptive Quantum Multi-Head Attention
- **Mentor Spec**: "Adaptive Quantum Multi-Head Attention" - Attention computation depends on patch/context; quantum processing capacity varies
- **Implementation**: Patch-conditioned measurement shot allocation; attention weights determine quantum resource budget per patch
- **Source File**: `src/models/aqpathformer/adaptive_quantum_attention.py`
- **Experiment**: EXP030 (Standard) vs EXP031 (Adaptive Quantum) - Ablation C
- **Result**: TBD
- **Status**: NOT_STARTED

### 4. Multi-Scale Quantum Feature Fusion
- **Mentor Spec**: "Multi-Scale Quantum Feature Fusion" - Coarse tissue context, medium structure, fine cellular detail; fuse multi-scale representations
- **Implementation**: 3-scale pyramid (4×, 8×, 16× downsampling) → independent quantum encoders → learnable weighted fusion
- **Source File**: `src/models/aqpathformer/multiscale_quantum_fusion.py`
- **Experiment**: EXP040 (Single-scale) vs EXP041 (Multi-scale) - Ablation D
- **Result**: TBD
- **Status**: NOT_STARTED

### 5. Cross-Cancer Representation Learning Module
- **Mentor Spec**: "Cross-Cancer Representation Learning Module" - Generalize across cancer types; dataset/task abstraction
- **Implementation**: Shared encoder backbone + dataset-specific prediction heads; leave-one-dataset-out evaluation protocol
- **Source File**: `src/models/aqpathformer/cross_cancer_representation.py`
- **Experiment**: EXP050 (No cross-cancer) vs EXP051 (Cross-cancer) - Ablation E
- **Result**: TBD
- **Status**: NOT_STARTED

### 6. Noise-Aware Quantum Layer
- **Mentor Spec**: "Noise-Aware Quantum Layer" - Configurable noise-aware experiments; compare ideal vs noisy; measure robustness
- **Implementation**: PennyLane noise models (DepolarizingChannel, AmplitudeDamping) injected during training; configurable error rates
- **Source File**: `src/models/aqpathformer/noise_aware_quantum_layer.py`
- **Experiment**: EXP060 (Ideal) vs EXP061 (Noisy) - Ablation F
- **Result**: TBD
- **Status**: NOT_STARTED

### 7. Hybrid Classical–Quantum Decoder
- **Mentor Spec**: "Hybrid Classical–Quantum Decoder" - Input → patch gen → classical/quantum repr → attention → multi-scale fusion → decoder/head → prediction
- **Implementation**: Classical transformer decoder layers processing fused quantum-classical features
- **Source File**: `src/models/aqpathformer/hybrid_decoder.py`
- **Experiment**: Integrated in full AQPathFormer (EXP070+)
- **Result**: TBD
- **Status**: NOT_STARTED

### 8. Multi-Cancer Prediction Head
- **Mentor Spec**: "Multi-Cancer Prediction Head" - Configurable head supporting binary, multiclass, multi-dataset
- **Implementation**: Configurable MLP head with dataset-specific output dimensions; explicit label mapping
- **Source File**: `src/models/aqpathformer/multicancer_head.py`
- **Experiment**: Used in all experiments
- **Result**: TBD
- **Status**: NOT_STARTED

---

## Algorithmic Contributions (5 Key Innovations)

### 1. Adaptive Quantum Patch Encoding
- **Mentor Spec**: "Adaptive Quantum Patch Encoding" - Combines adaptive patch selection with quantum encoding
- **Implementation**: AdaptivePatchGenerator + QuantumPatchEncoder integration; patches selected then quantum encoded
- **Source Files**: `adaptive_patch_generator.py`, `quantum_patch_encoder.py`
- **Experiment**: EXP011, EXP021, full model
- **Result**: TBD
- **Status**: NOT_STARTED

### 2. Quantum Multi-Scale Attention
- **Mentor Spec**: "Quantum Multi-Scale Attention" - Quantum attention operating across multiple scales
- **Implementation**: AdaptiveQuantumAttention applied at each scale in MultiScaleQuantumFusion
- **Source Files**: `adaptive_quantum_attention.py`, `multiscale_quantum_fusion.py`
- **Experiment**: EXP031, EXP041
- **Result**: TBD
- **Status**: NOT_STARTED

### 3. Dynamic Circuit Allocation
- **Mentor Spec**: "Dynamic Circuit Allocation" - Quantum processing capacity/circuit allocation varies per representation
- **Implementation**: Software-level policy: attention scores → measurement shot allocation per patch (not hardware dynamic circuits)
- **Source File**: `adaptive_quantum_attention.py` (measurement allocation logic)
- **Experiment**: EXP031 (shot allocation vs uniform)
- **Result**: TBD
- **Status**: NOT_STARTED

### 4. Quantum Feature Fusion
- **Mentor Spec**: "Quantum Feature Fusion" - Fuse multi-scale quantum representations
- **Implementation**: Learnable fusion weights combining expectation values from multi-scale quantum encoders
- **Source File**: `multiscale_quantum_fusion.py` (fusion module)
- **Experiment**: EXP041
- **Result**: TBD
- **Status**: NOT_STARTED

### 5. Noise-Aware Vision Learning
- **Mentor Spec**: "Noise-Aware Vision Learning" - Training with noise awareness for robustness
- **Implementation**: NoiseAwareQuantumLayer injecting noise during forward pass; noise-robust training objective
- **Source File**: `noise_aware_quantum_layer.py`
- **Experiment**: EXP061
- **Result**: TBD
- **Status**: NOT_STARTED

---

## Datasets (7 Candidate Datasets)

### CAMELYON16
- **Mentor Spec**: Candidate dataset
- **Implementation**: `src/data/camelyon.py` (Camelyon16Dataset class)
- **Experiment**: Phase 11 (EXP080+)
- **Result**: TBD
- **Status**: NOT_STARTED (requires challenge registration)

### CAMELYON17
- **Mentor Spec**: Candidate dataset
- **Implementation**: `src/data/camelyon.py` (Camelyon17Dataset class)
- **Experiment**: Phase 11 (EXP080+)
- **Result**: TBD
- **Status**: NOT_STARTED (requires challenge registration)

### BreakHis
- **Mentor Spec**: Candidate dataset
- **Implementation**: `src/data/breakhis.py` (BreakHisDataset class)
- **Experiment**: Phase 11 (EXP080+)
- **Result**: TBD
- **Status**: NOT_STARTED (requires registration)

### PANDA
- **Mentor Spec**: Candidate dataset
- **Implementation**: `src/data/panda.py` (PandaDataset class)
- **Experiment**: Phase 11 (EXP080+)
- **Result**: TBD
- **Status**: NOT_STARTED (large, requires Kaggle)

### TCGA Histopathology
- **Mentor Spec**: Candidate dataset
- **Implementation**: `src/data/tcga.py` (TCGADataset class)
- **Experiment**: Phase 11 (EXP080+)
- **Result**: TBD
- **Status**: NOT_STARTED (requires dbGaP authorization)

### LC25000 (Pilot Dataset)
- **Mentor Spec**: Candidate dataset
- **Implementation**: `src/data/lc25000.py` (LC25000Dataset class) - PRIMARY PILOT
- **Experiment**: All Phase 3-10 experiments (EXP001-079)
- **Result**: TBD
- **Status**: IN_PROGRESS (Phase 2-3)

### CRC-100K
- **Mentor Spec**: Candidate dataset
- **Implementation**: `src/data/crc100k.py` (CRC100KDataset class)
- **Experiment**: Phase 11 (EXP080+)
- **Result**: TBD
- **Status**: NOT_STARTED (backup pilot)

---

## Baselines (9 Baseline Models)

### ResNet50
- **Mentor Spec**: Baseline
- **Implementation**: `src/models/baselines/resnet.py` (timm resnet50)
- **Experiment**: EXP001
- **Result**: TBD
- **Status**: NOT_STARTED

### Vision Transformer (ViT-B/16)
- **Mentor Spec**: Baseline
- **Implementation**: `src/models/baselines/vit.py` (timm vit_base_patch16_224)
- **Experiment**: EXP002
- **Result**: TBD
- **Status**: NOT_STARTED

### Swin Transformer (Swin-T)
- **Mentor Spec**: Baseline
- **Implementation**: `src/models/baselines/swin.py` (timm swin_tiny_patch4_window7_224)
- **Experiment**: EXP003
- **Result**: TBD
- **Status**: NOT_STARTED

### ConvNeXt (ConvNeXt-T)
- **Mentor Spec**: Baseline
- **Implementation**: `src/models/baselines/convnext.py` (timm convnext_tiny)
- **Experiment**: EXP004
- **Result**: TBD
- **Status**: NOT_STARTED

### EfficientNet (EfficientNet-B0)
- **Mentor Spec**: Baseline
- **Implementation**: `src/models/baselines/efficientnet.py` (timm efficientnet_b0)
- **Experiment**: EXP005
- **Result**: TBD
- **Status**: NOT_STARTED

### UNI (Foundation Model)
- **Mentor Spec**: Baseline (where technically justified)
- **Implementation**: `src/models/baselines/uni.py` (if weights accessible)
- **Experiment**: EXP006
- **Result**: TBD
- **Status**: NOT_STARTED (investigate access)

### CONCH (Foundation Model)
- **Mentor Spec**: Baseline (where technically justified)
- **Implementation**: `src/models/baselines/conch.py` (if weights accessible)
- **Experiment**: EXP007
- **Result**: TBD
- **Status**: NOT_STARTED (investigate access)

### QCNN
- **Mentor Spec**: Baseline
- **Implementation**: `src/models/baselines/qcnn.py` (custom quantum CNN)
- **Experiment**: EXP008
- **Result**: TBD
- **Status**: NOT_STARTED (Phase 7)

### Hybrid Quantum Vision Transformer
- **Mentor Spec**: Baseline
- **Implementation**: `src/models/baselines/hybrid_qvit.py` (classical ViT + quantum patch encoder)
- **Experiment**: EXP009
- **Result**: TBD
- **Status**: NOT_STARTED (Phase 7)

---

## Evaluation Metrics (12+ Metrics)

### Classification Metrics
- **Accuracy** → `src/eval/metrics.py` → `accuracy_score` → All experiments → TBD → NOT_STARTED
- **Precision** → `src/eval/metrics.py` → `precision_score` (macro) → All experiments → TBD → NOT_STARTED
- **Recall** → `src/eval/metrics.py` → `recall_score` (macro) → All experiments → TBD → NOT_STARTED
- **Specificity** → `src/eval/metrics.py` → custom (per-class) → All experiments → TBD → NOT_STARTED
- **F1-Score** → `src/eval/metrics.py` → `f1_score` (macro) → All experiments → TBD → NOT_STARTED
- **ROC-AUC** → `src/eval/metrics.py` → `roc_auc_score` (ovr) → All experiments → TBD → NOT_STARTED
- **PR-AUC** → `src/eval/metrics.py` → `average_precision_score` → All experiments → TBD → NOT_STARTED
- **MCC** → `src/eval/metrics.py` → `matthews_corrcoef` → All experiments → TBD → NOT_STARTED

### Computational Metrics
- **FLOPs** → `src/eval/metrics.py` → `fvcore.nn.FlopCountAnalysis` → All experiments → TBD → NOT_STARTED
- **Parameters** → `src/eval/metrics.py` → `sum(p.numel())` → All experiments → TBD → NOT_STARTED
- **Training Time** → `src/train.py` → wall-clock per epoch → All experiments → TBD → NOT_STARTED
- **Inference Time** → `src/eval/evaluate.py` → avg over test set → All experiments → TBD → NOT_STARTED

### Quantum-Specific Metrics
- **Circuit Depth** → `quantum_patch_encoder.py` → `qml.specs(circuit)()` → Quantum experiments → TBD → NOT_STARTED
- **Fidelity** → `noise_aware_quantum_layer.py` → state fidelity vs ideal → Ablation F → TBD → NOT_STARTED
- **Noise Robustness** → `noise_aware_quantum_layer.py` → accuracy drop (ideal→noisy) → Ablation F → TBD → NOT_STARTED

---

## Phase Mapping

| Phase | Mentor Spec Sections | Key Deliverables | Status |
|-------|---------------------|------------------|--------|
| 1 | 4, 34 | ENVIRONMENT.md, AGENTS.md, DECISIONS.md, ASSUMPTIONS.md, MENTOR_ALIGNMENT.md, requirements.txt, dirs | **COMPLETE** |
| 2 | 5 | LITERATURE_REVIEW.md, DATASET_REVIEW.md, DATASET_STATUS.md, METHOD_JUSTIFICATION.md | NOT_STARTED |
| 3 | 6, 7 | Base dataset class, LC25000 loader, preprocessing pipeline | NOT_STARTED |
| 4 | 7 | Preprocessing validation, figures/tables | NOT_STARTED |
| 5 | 18, 20 | Metrics module, experiment runner, registry | NOT_STARTED |
| 6 | 8 | 5+ classical baselines trained | NOT_STARTED |
| 7 | 11 | QuantumPatchEncoder with sim + proxy modes | NOT_STARTED |
| 8 | 9, 10, 12, 13, 14, 15, 16, 17 | All 8 AQPathFormer components unit tested | NOT_STARTED |
| 9 | 9, 16 | Full AQPathFormer end-to-end training | NOT_STARTED |
| 10 | 19 | 7 ablations completed | NOT_STARTED |
| 11 | 6, 14 | 2+ additional datasets | NOT_STARTED |
| 12 | 15 | Ideal vs noisy comparison | NOT_STARTED |
| 13 | 14 | Leave-one-out / domain adaptation | NOT_STARTED |
| 14 | 21, 22 | Statistical validation, resource tables | NOT_STARTED |
| 15 | 25, 26 | 6 reports + 14 figures | NOT_STARTED |

---

## Critical Specification Requirements Checklist

### Scientific Integrity (Section 2)
- [ ] No fabricated results (Rule A)
- [ ] Full experiment provenance (Rule B, C)
- [ ] Explicit assumptions (Rule D) → `ASSUMPTIONS.md`
- [ ] No cherry-picking (Rule E)
- [ ] Data leakage prevention (Rule F) → patient-level splits
- [ ] Reproducible seeds (Rule G)
- [ ] Raw outputs before summaries (Rule H)

### Autonomous Execution (Section 3)
- [ ] No repeated stops for guidance
- [ ] Intervention points documented
- [ ] Failure diagnosis & recovery process

### Final Output (Section 29)
- [ ] Complete source code
- [ ] Dataset loaders
- [ ] Preprocessing pipeline
- [ ] Baseline implementations
- [ ] AQPathFormer implementation
- [ ] Quantum simulation
- [ ] Noise-aware implementation
- [ ] Training/evaluation scripts
- [ ] Configuration files
- [ ] Tests
- [ ] Experiment logs
- [ ] Checkpoints
- [ ] Metrics
- [ ] CSV results
- [ ] Publication figures
- [ ] Ablation results
- [ ] Dataset documentation
- [ ] Literature review
- [ ] Technical report
- [ ] Reproducibility guide
- [ ] Mentor-alignment matrix

---

*Last Updated: 2026-09-13*  
*Next Update: After Phase 2 completion*