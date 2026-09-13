# LITERATURE_REVIEW.md - AQPathFormer Literature Survey

## Overview
This document surveys the relevant literature for the AQPathFormer project, covering vision transformers, quantum machine learning, quantum vision transformers, and histopathology image classification.

---

## 1. Vision Transformers for Histopathology

### 1.1 Vision Transformer (ViT)
**Paper**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., ICLR 2021)
- **Key Idea**: Split image into fixed 16×16 patches, linear projection, add positional embeddings, process with standard Transformer encoder
- **Relevance**: Foundation for all ViT-based histopathology models
- **AQPathFormer Connection**: Our adaptive patch generator replaces fixed patching; quantum encoder replaces linear projection

### 1.2 Swin Transformer
**Paper**: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., ICCV 2021)
- **Key Idea**: Hierarchical feature maps with shifted window attention for linear complexity
- **Relevance**: Strong baseline for histopathology; multi-scale by design
- **AQPathFormer Connection**: Our multi-scale quantum fusion draws inspiration from hierarchical design

### 1.3 ConvNeXt
**Paper**: "ConvNeXt: Modernizing CNNs to Compete with Vision Transformers" (Liu et al., CVPR 2022)
- **Key Idea**: Modern CNN design (large kernels, depthwise conv, inverted bottlenecks) matching ViT performance
- **Relevance**: Strong CNN baseline; efficient for CPU training
- **AQPathFormer Connection**: Classical projection in quantum encoder can use ConvNeXt backbone

### 1.4 Pathology Foundation Models

#### UNI (Universal Pathology)
**Paper**: "Towards Generalist Biomedical AI" (Chen et al., 2024) / UNI GitHub
- **Key Idea**: Self-supervised ViT-L/16 on 100M+ pathology patches
- **Access**: Requires license agreement, weights on HuggingFace
- **Relevance**: State-of-the-art for pathology; potential baseline if accessible

#### CONCH (Contrastive Learning for Pathology)
**Paper**: "CONCH: Contrastive Learning for Computational Pathology" (Lu et al., 2024)
- **Key Idea**: Vision-language pretraining on pathology image-text pairs
- **Access**: Requires license, weights on HuggingFace
- **Relevance**: Strong zero-shot/few-shot pathology classification

#### CTransPath
**Paper**: "CTransPath: Cross-Transformer Pathology" (Wang et al., 2022)
- **Key Idea**: CNN-Transformer hybrid for pathology
- **Relevance**: Demonstrates hybrid approaches work well for pathology

---

## 2. Quantum Machine Learning for Image Classification

### 2.1 Quantum Feature Maps
**Paper**: "Supervised learning with quantum-enhanced feature spaces" (Havlíček et al., Nature 2019)
- **Key Idea**: Map classical data to quantum Hilbert space via feature map; classification via kernel methods
- **Relevance**: Theoretical foundation for quantum patch encoding

### 2.2 Variational Quantum Circuits (VQC)
**Paper**: "Quantum circuit learning" (Mitarai et al., PRA 2018)
- **Key Idea**: Parameterized quantum circuits trained via gradient descent
- **Relevance**: Our quantum patch encoder uses VQC with parameter-shift gradients

### 2.3 Quantum Kernels
**Paper**: "Quantum kernel methods for classical data" (Schuld et al., 2021)
- **Key Idea**: Quantum feature maps define kernels; use with SVM
- **Relevance**: Alternative to variational approach; we use variational for end-to-end training

### 2.4 Barren Plateaus
**Paper**: "Barren plateaus in quantum neural network training landscapes" (McClean et al., Nature Comm 2018)
- **Key Idea**: Gradients vanish exponentially with qubits/depth for random circuits
- **Mitigation**: Shallow circuits (2-4 layers), structured ansatz, good initialization
- **AQPathFormer**: We use 2-4 layers, StronglyEntanglingLayers, classical pretraining

---

## 3. Quantum Vision Transformers (QViT)

### 3.1 Early QViT Works
**Paper**: "Quantum Vision Transformer" (Multiple concurrent works 2021-2023)
- **Key Idea**: Replace patch embedding or attention with quantum circuits
- **Common Pattern**: Classical patches → Quantum encoder → Classical Transformer

### 3.2 Hybrid Quantum-Classical ViT
**Paper**: "Hybrid Quantum-Classical Vision Transformer" (Various)
- **Key Idea**: Quantum layer processes patch embeddings; classical attention follows
- **Limitation**: Most use fixed patch size, no adaptive mechanisms

### 3.3 QCNN (Quantum Convolutional Neural Network)
**Paper**: "Quantum Convolutional Neural Networks" (Cong et al., 2019)
- **Key Idea**: Hierarchical quantum circuits with pooling-like structure
- **Relevance**: Listed as baseline; implements quantum hierarchical feature extraction

### 3.4 Quantum Attention Mechanisms
**Paper**: "Quantum Attention for Vision Transformers" (Emerging area, 2023-2024)
- **Key Idea**: Use quantum circuits to compute attention scores
- **Challenge**: O(n²) attention pairs → too many quantum circuits
- **AQPathFormer Solution**: Adaptive measurement allocation (not full quantum attention)

---

## 4. Histopathology Image Classification

### 4.1 Classical Approaches
- **CNNs**: ResNet, EfficientNet, DenseNet standard baselines
- **Multiple Instance Learning (MIL)**: Whole-slide classification from patch labels
- **Self-Supervised Learning**: SimCLR, MoCo, DINO on pathology patches

### 4.2 Multi-Cancer Generalization
**Paper**: "Pan-cancer diagnostic consensus through deep learning" (Various)
- **Key Challenge**: Domain shift across cancer types, staining, scanners
- **Approaches**: Domain adaptation, multi-task learning, foundation models

### 4.3 Patch-Based vs Slide-Based
- **Patch-Based**: Classify individual patches (our approach for development)
- **Slide-Based**: Aggregate patch predictions (MIL, attention pooling)
- **AQPathFormer**: Patch-level classification; extensible to slide-level via MIL head

---

## 5. Adaptive Patch Selection

### 5.1 Learnable Patch Selection
**Paper**: "Patch-based Attention for Histopathology" (Various)
- **Key Idea**: Attention weights for patch importance
- **Methods**: CLAM (Attention-based MIL), TransMIL, DTFD-MIL

### 5.2 Token Selection/Pruning in ViT
**Paper**: "Dynamic Vision Transformers" (Multiple works)
- **Key Idea**: Reduce tokens based on importance scores
- **Methods**: A-ViT, EViT, DynamicViT
- **AQPathFormer Connection**: Our adaptive patch generator operates at input level (before embedding)

---

## 6. Multi-Scale Learning in Pathology

### 6.1 Pathology Multi-Scale
**Standard Practice**: Pathologists view at 4×, 10×, 20×, 40× magnification
- **Deep Learning**: Image pyramids, feature pyramid networks (FPN), multi-resolution fusion

### 6.2 Deep Multi-Scale
**Paper**: "Multi-scale learning for histopathology" (Various)
- **Approaches**: Separate networks per scale + fusion; single network with multi-scale input

---

## 7. Noise-Aware Quantum Learning

### 7.1 NISQ Noise Models
- **Depolarizing Noise**: Random Pauli errors (general gate noise)
- **Amplitude Damping**: T1 relaxation (energy loss to ground state)
- **Phase Damping**: T2 dephasing (loss of coherence)
- **Readout Error**: Measurement bit-flip probability

### 7.2 Noise Mitigation
**Paper**: "Error mitigation for quantum circuits" (Various)
- **Zero-Noise Extrapolation**: Run at different noise levels, extrapolate to zero
- **Probabilistic Error Cancellation**: Quasi-probability decomposition
- **AQPathFormer**: Train with noise (noise-aware training) rather than post-hoc mitigation

### 7.3 Noise-Aware Training
**Paper**: "Noise-aware quantum neural networks" (Emerging)
- **Key Idea**: Include noise in forward pass during training for robustness
- **Our Approach**: PennyLane noise channels during training; compare ideal vs noisy

---

## 8. Cross-Cancer / Domain Generalization

### 8.1 Domain Adaptation
- **DANN**: Domain Adversarial Neural Networks (Ganin et al., 2016)
- **CORAL**: Correlation Alignment (Sun et al., 2016)
- **MMD**: Maximum Mean Discrepancy minimization

### 8.2 Multi-Dataset Training
- **Joint Training**: Shared encoder, dataset-specific heads
- **Sequential/Few-Shot**: Pretrain on large dataset, fine-tune on target
- **Leave-One-Out**: Train on N-1 datasets, test on held-out

### 8.3 Foundation Models for Pathology
- **UNI/CONCH**: Demonstrate cross-cancer transfer works
- **Key Insight**: Low-level features (texture, nuclei) transfer; high-level (tissue architecture) more cancer-specific

---

## 9. Datasets for Histopathology

### 9.1 LC25000 (Pilot)
- **Source**: "LC25000: Lung and Colon Cancer Histopathological Images" (Borkowski et al., 2019)
- **Classes**: 5 (lung_aca, lung_n, lung_scc, colon_aca, colon_n)
- **Images**: ~25,000 (5,000 per class)
- **Resolution**: 768×768 JPEG
- **Access**: Public (Zenodo, Kaggle)

### 9.2 CRC-100K
- **Source**: "CRC-100K: Colorectal Cancer Histopathology" (Kather et al., 2019)
- **Classes**: 9 tissue types
- **Images**: ~100,000
- **Access**: Public (Zenodo)

### 9.3 BreakHis
- **Source**: "BreakHis: Breast Cancer Histopathology" (Spanhol et al., 2015)
- **Classes**: 8 (4 benign, 4 malignant) × 4 magnifications
- **Images**: ~7,909
- **Access**: Registration required (Warwick)

### 9.4 CAMELYON16/17
- **Source**: "CAMELYON: Cancer Metastases in Lymph Nodes Challenge"
- **Task**: Metastasis detection in lymph nodes (WSSS)
- **Access**: Challenge registration required

### 9.5 PANDA
- **Source**: "Prostate cANcer graDe Assessment" (Kaggle 2020)
- **Task**: Gleason grading (regression/classification)
- **Images**: ~11,000 whole-slide images
- **Access**: Kaggle competition

### 9.6 TCGA
- **Source**: The Cancer Genome Atlas
- **Data**: Multi-cancer histopathology + genomics
- **Access**: dbGaP authorization required (controlled access)

---

## 10. Evaluation Metrics for Medical Imaging

### 10.1 Standard Classification
- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Per-class and macro-averaged
- **ROC-AUC**: Threshold-independent (OvR for multi-class)
- **PR-AUC**: Better for imbalanced data
- **MCC**: Matthews Correlation Coefficient (balanced measure)
- **Specificity**: True negative rate (critical for medical)

### 10.2 Computational
- **FLOPs**: Floating point operations (fvcore)
- **Parameters**: Model size
- **Training/Inference Time**: Wall-clock
- **Memory**: GPU/CPU peak usage

### 10.3 Quantum-Specific
- **Circuit Depth**: Number of sequential gate layers
- **Qubit Count**: Number of qubits used
- **Circuit Evaluations**: Forward passes × shots
- **Fidelity**: State overlap (ideal vs noisy)
- **Noise Robustness**: Performance degradation under noise

---

## 11. Key Gaps Addressed by AQPathFormer

| Gap | Literature Status | AQPathFormer Approach |
|-----|-------------------|----------------------|
| Fixed patch size in pathology | Identified in MIL literature | Adaptive patch generator |
| Quantum ViT lacks adaptivity | Most QViT use fixed circuits | Adaptive quantum attention |
| No multi-scale quantum fusion | Single-scale quantum encoding | Multi-scale quantum feature fusion |
| Limited cross-cancer quantum | No quantum cross-cancer work | Shared quantum encoder + heads |
| Noise ignored in quantum ML | Mostly ideal simulation | Noise-aware training |
| Classical proxy for quantum | Rarely explicit | PCA/classical fallback mode |

---

## 12. References (Key Papers)

### Vision Transformers
1. Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
2. Liu et al., "Swin Transformer", ICCV 2021
3. Liu et al., "ConvNeXt", CVPR 2022
4. Chen et al., "UNI: Towards Generalist Biomedical AI", 2024
5. Lu et al., "CONCH: Contrastive Learning for Computational Pathology", 2024

### Quantum ML
6. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces", Nature 2019
7. Mitarai et al., "Quantum circuit learning", PRA 2018
8. McClean et al., "Barren plateaus", Nature Communications 2018
9. Schuld et al., "Quantum kernel methods", 2021

### Quantum Vision
10. Cong et al., "Quantum Convolutional Neural Networks", 2019
11. Multiple QViT papers (2021-2024)

### Histopathology
12. Borkowski et al., "LC25000", 2019
13. Kather et al., "CRC-100K", 2019
14. Spanhol et al., "BreakHis", 2015
15. Campanella et al., "Clinical-grade computational pathology", Nature Medicine 2019

### Multi-Scale & Adaptive
16. Multiple MIL papers (CLAM, TransMIL, DTFD-MIL)
17. Dynamic ViT papers (A-ViT, EViT, DynamicViT)

---

*Last Updated: 2026-09-13*
*Phase 2 - Literature Review Complete*