# ASSUMPTIONS.md - Research Assumptions Log

## Format
**Assumption**: What is assumed  
**Source**: Where it comes from (spec, literature, practical constraint)  
**Justification**: Why this is reasonable  
**Impact if Wrong**: Consequences if assumption fails  
**Validation Plan**: How to verify  

---

## ASM-001: Quantum Advantage Exists for Histopathology
**Assumption**: Quantum encoding provides representational advantage over classical encoding for histopathological images  
**Source**: Mentor specification (algorithmic contributions 1, 2)  
**Justification**: Quantum feature maps can access exponentially large Hilbert spaces; prior work shows quantum kernels can separate classes classical kernels cannot  
**Impact if Wrong**: AQPathFormer reduces to classical ViT with extra overhead; no scientific contribution  
**Validation Plan**: Ablation B (Classical vs Quantum Encoder) directly tests this; report honest results regardless of outcome

---

## ASM-002: 4-6 Qubits Sufficient for Meaningful Representation
**Assumption**: Compressing patch features to 16-64 dimensions (2^4 to 2^6) retains discriminative information  
**Source**: Practical constraint (CPU simulation), DEC-004  
**Justification**: Classical PCA to 64 dims often retains >95% variance; quantum feature map may amplify relevant features  
**Impact if Wrong**: Severe information bottleneck; quantum encoder cannot represent patch diversity  
**Validation Plan**: Compare classical PCA(64) vs Quantum(6 qubits) vs Classical(768) in ablation; monitor reconstruction fidelity

---

## ASM-003: Adaptive Patch Selection Improves Over Fixed Grid
**Assumption**: Learning to select/weight patches based on tissue content improves classification  
**Source**: Mentor specification (motivation: "limitations of fixed patch representations")  
**Justification**: Pathology slides have large background/non-informative regions; attention can focus on diagnostically relevant areas  
**Impact if Wrong**: Added complexity without benefit; fixed grid equally effective  
**Validation Plan**: Ablation A (Fixed vs Adaptive Patch) with patch count/accuracy/compute trade-off analysis

---

## ASM-004: Multi-Scale Fusion Provides Complementary Information
**Assumption**: Coarse (tissue architecture), medium (glandular structure), fine (cellular detail) scales capture complementary diagnostic signals  
**Source**: Mentor specification (component 4, algorithmic contribution 4), pathology literature  
**Justification**: Pathologists examine multiple magnifications; different cancer types manifest at different scales  
**Impact if Wrong**: Single-scale sufficient; multi-scale adds compute without gain  
**Validation Plan**: Ablation D (Single vs Multi-Scale) with scale-specific feature visualization

---

## ASM-005: Cross-Cancer Generalization via Shared Encoder
**Assumption**: Histopathological features (nuclei, tissue architecture, stroma) share low-level representations across cancer types  
**Source**: Mentor specification (component 5, motivation: "limited generalisation across cancer types")  
**Justification**: Foundation models (UNI, CONCH) demonstrate cross-cancer transfer; low-level texture/shape features are universal  
**Impact if Wrong**: Shared encoder learns conflicting features; dataset-specific heads cannot compensate  
**Validation Plan**: Leave-one-dataset-out evaluation; compare shared vs separate encoders

---

## ASM-006: Noise-Aware Training Improves Robustness
**Assumption**: Training with simulated quantum noise improves generalization to ideal/noisy conditions  
**Source**: Mentor specification (component 6, algorithmic contribution 5)  
**Justification**: Noise injection acts as regularization; robust quantum models needed for NISQ era  
**Impact if Wrong**: Noise training degrades ideal performance without improving noisy performance  
**Validation Plan**: Ablation F (Ideal vs Noisy) with multiple noise levels; measure robustness curves

---

## ASM-007: PennyLane Simulation Faithfully Represents Quantum Behavior
**Assumption**: `default.qubit` simulator accurately models ideal quantum circuit execution  
**Source**: PennyLane documentation, quantum computing literature  
**Justification**: State-vector simulation is exact for ideal circuits; noise models are standard approximations  
**Impact if Wrong**: Simulation artifacts mistaken for quantum effects  
**Validation Plan**: Verify against analytical results for small circuits; compare with Qiskit Aer for validation subset

---

## ASM-008: LC25000 Represents Histopathology Classification Challenges
**Assumption**: LC25000 (lung/colon, 5 classes, 25K images) captures key difficulties of multi-cancer histopathology  
**Source**: DEC-002, dataset characteristics  
**Justification**: Multi-cancer, multi-class, standard resolution, public benchmark  
**Impact if Wrong**: Results don't generalize to other datasets (CAMELYON, TCGA, PANDA)  
**Validation Plan**: Phase 11 multi-dataset evaluation; explicit domain shift analysis

---

## ASM-009: Patient-Level Splits Prevent Data Leakage
**Assumption**: LC25000 images from different patients are independent samples  
**Source**: Scientific integrity rules (Rule F), dataset documentation  
**Justification**: Standard practice in medical imaging; LC25000 metadata includes patient IDs  
**Impact if Wrong**: Overestimated performance due to patient-level correlation  
**Validation Plan**: Implement patient-level splits; compare with random splits; report both

---

## ASM-010: Classical Proxy (PCA) Faithfully Approximates Quantum Encoder
**Assumption**: PCA projection to 2^n dimensions approximates quantum encoder for debugging/ablation  
**Source**: DEC-005, practical need for classical fallback  
**Justification**: Both are linear projections; PCA finds optimal linear subspace; quantum adds non-linear feature map  
**Impact if Wrong**: Classical proxy results mislead quantum encoder development  
**Validation Plan**: Only use proxy for debugging (shape checks, gradient flow); never for final results

---

## ASM-011: Gradient Accumulation Equivalent to Larger Batch Size
**Assumption**: Gradient accumulation (accum_steps=4, batch=8 → effective batch=32) provides similar optimization dynamics  
**Source**: Standard practice, PyTorch documentation  
**Justification**: Mathematically equivalent for SGD/Adam; only difference is batch norm statistics  
**Impact if Wrong**: Different convergence behavior; batch norm statistics computed on micro-batches  
**Validation Plan**: Use GroupNorm/LayerNorm instead of BatchNorm; monitor training curves vs theoretical large-batch

---

## ASM-012: 20 Epochs Sufficient for Initial Experiments
**Assumption**: 20 epochs shows convergence trends; 50+ epochs for final benchmarking  
**Source**: Resource constraints (CPU, time), practical ML experience  
**Justification**: ViT typically converges in 100-300 epochs on ImageNet; smaller dataset converges faster  
**Impact if Wrong**: Underfitting; false negative on model capability  
**Validation Plan**: Monitor validation loss curves; extend to 50 epochs if still decreasing

---

## ASM-013: Single Random Seed (42) Sufficient for Development
**Assumption**: Seed 42 represents typical behavior for development/debugging  
**Source**: Common practice, resource constraints  
**Justification**: Development needs speed; statistical validation uses 5 seeds in final phase  
**Impact if Wrong**: Seed-specific artifacts mistaken for general behavior  
**Validation Plan**: Run 3 seeds for key experiments (baselines, full AQPathFormer); report variance

---

## ASM-014: No GPU Available Throughout Project
**Assumption**: Project runs entirely on CPU (8GB RAM)  
**Source**: ENVIRONMENT.md inspection, no nvidia-smi  
**Justification**: Current environment has no CUDA; cloud GPU not in scope  
**Impact if Wrong**: Could have run larger experiments if GPU became available  
**Validation Plan**: Code written device-agnostic (.to(device)); ready for GPU if available

---

## ASM-015: Dataset Licenses Permit Research Use
**Assumption**: LC25000, CRC-100K, BreakHis allow academic research use  
**Source**: Dataset documentation (Zenodo, Kaggle, Warwick)  
**Justification**: All cited as academic benchmarks; standard in literature  
**Impact if Wrong**: Legal issues; cannot publish results  
**Validation Plan**: Document license terms in DATASET_STATUS.md; verify before Phase 11 expansion

---

## ASM-016: Quantum Circuit Depth 2-4 Layers Sufficient
**Assumption**: 2-4 variational layers provide expressive power without barren plateaus  
**Source**: Quantum ML literature, PennyLane tutorials  
**Justification**: Shallow circuits avoid barren plateaus; 4 layers × 6 qubits = 72 parameters manageable  
**Impact if Wrong**: Underfitting (too shallow) or untrainable (too deep/barren plateaus)  
**Validation Plan**: Ablation on circuit depth; monitor gradient magnitudes

---

## ASM-017: Angle Embedding Appropriate for Patch Features
**Assumption**: PennyLane's AngleEmbedding (RX/RZ rotations) suitable for classical image features  
**Source**: Quantum ML literature, PennyLane defaults  
**Justification**: Standard for continuous-valued data; maps [0, π] features to rotation angles  
**Impact if Wrong**: Poor feature representation; alternative embeddings needed (amplitude, IQP)  
**Validation Plan**: Test AngleEmbedding vs AmplitudeEmbedding in encoder ablation

---

## ASM-018: StronglyEntanglingLayers Good Default Ansatz
**Assumption**: PennyLane's StronglyEntanglingLayers provides good expressivity/entanglement  
**Source**: PennyLane documentation, quantum ML best practices  
**Justification**: Hardware-efficient ansatz; all-to-all entanglement; trainable parameters  
**Impact if Wrong**: Insufficient entanglement or too many parameters for optimization  
**Validation Plan**: Compare with BasicEntanglerLayers, SimplifiedTwoDesign in ablations

---

## ASM-019: Evaluation Metrics Cover Clinical Relevance
**Assumption**: Accuracy, Precision, Recall, Specificity, F1, AUC, MCC sufficient for histopathology evaluation  
**Source**: Mentor specification (evaluation metrics list), medical ML literature  
**Justification**: Standard metrics for medical classification; MCC handles imbalance; AUC threshold-independent  
**Impact if Wrong**: Missing clinically relevant metrics (e.g., sensitivity at fixed specificity)  
**Validation Plan**: Add operating point metrics if clinical collaboration emerges

---

## ASM-020: Computational Analysis Metrics Are Measurable
**Assumption**: FLOPs, params, training/inference time, circuit depth measurable on CPU  
**Source**: Mentor specification, standard tools (fvcore, torchinfo)  
**Justification**: Well-defined metrics; fvcore supports CPU profiling  
**Impact if Wrong**: Incomplete resource comparison tables  
**Validation Plan**: Implement measurement utilities; validate against known model sizes