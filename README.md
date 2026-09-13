# AQPathFormer: Adaptive Quantum Vision Transformer for Multi-Cancer Histopathological Intelligence

## Overview
This repository implements **AQPathFormer**, an Adaptive Quantum Vision Transformer designed for multi-cancer histopathological image classification. The project explores the integration of quantum computing principles with vision transformers for improved generalization across cancer types.

## Key Features
- **Adaptive Patch Generation**: Learnable patch selection based on tissue content
- **Quantum Patch Encoder**: Hybrid classical-quantum encoding with configurable qubits/circuits
- **Adaptive Quantum Attention**: Patch-conditioned quantum resource allocation
- **Multi-Scale Quantum Fusion**: Coarse-to-fine quantum feature integration
- **Cross-Cancer Learning**: Shared encoder with dataset-specific heads
- **Noise-Aware Training**: Simulated NISQ noise for robust quantum models

## Architecture
```
Input Image
    ↓
Adaptive Patch Generator → Patch Selection/Weighting
    ↓
Quantum Patch Encoder → Classical Projection → Quantum Circuit → Expectation Values
    ↓
Adaptive Quantum Multi-Head Attention
    ↓
Multi-Scale Quantum Feature Fusion
    ↓
Hybrid Classical-Quantum Decoder
    ↓
Multi-Cancer Prediction Head
```

## Installation

### Prerequisites
- Python 3.11+
- 8GB+ RAM (CPU-only training)
- No GPU required (designed for CPU)

### Setup
```bash
# Clone repository
git clone <repository-url>
cd AQPathFormer

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, timm, pennylane, cv2; print('All imports successful')"
```

## Project Structure
```
AQPathFormer/
├── AGENTS.md                    # Project coordination
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── configs/                     # Configuration files
│   ├── base.yaml               # Base configuration
│   ├── baselines/              # Baseline model configs
│   ├── aqpathformer/           # AQPathFormer configs
│   └── ablations/              # Ablation study configs
├── src/
│   ├── data/                   # Dataset loaders & preprocessing
│   │   ├── base_dataset.py     # Abstract base class
│   │   ├── lc25000.py          # LC25000 dataset (pilot)
│   │   ├── crc100k.py          # CRC-100K dataset
│   │   ├── breakhis.py         # BreakHis dataset
│   │   ├── camelyon.py         # CAMELYON datasets
│   │   └── preprocessing.py    # Preprocessing pipeline
│   ├── models/
│   │   ├── baselines/          # Classical baseline models
│   │   │   ├── resnet.py
│   │   │   ├── vit.py
│   │   │   ├── swin.py
│   │   │   ├── convnext.py
│   │   │   └── efficientnet.py
│   │   └── aqpathformer/       # AQPathFormer components
│   │       ├── adaptive_patch_generator.py
│   │       ├── quantum_patch_encoder.py
│   │       ├── adaptive_quantum_attention.py
│   │       ├── multiscale_quantum_fusion.py
│   │       ├── cross_cancer_representation.py
│   │       ├── noise_aware_quantum_layer.py
│   │       ├── hybrid_decoder.py
│   │       ├── multicancer_head.py
│   │       └── aqpathformer.py
│   ├── eval/
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── evaluate.py         # Evaluation script
│   ├── train.py                # Training entry point
│   └── utils/                  # Utility functions
├── experiments/                # Experiment outputs (auto-generated)
│   └── EXP###/
├── reports/                    # Generated reports & figures
│   ├── figures/
│   ├── tables/
│   ├── technical_report.md
│   ├── methodology.md
│   ├── experiment_report.md
│   ├── results.md
│   ├── limitations.md
│   └── reproducibility.md
├── docs/                       # Documentation
│   ├── ENVIRONMENT.md          # Environment inspection
│   ├── DECISIONS.md            # Technical decisions log
│   ├── ASSUMPTIONS.md          # Research assumptions
│   ├── MENTOR_ALIGNMENT.md     # Spec-to-implementation mapping
│   ├── LITERATURE_REVIEW.md
│   ├── DATASET_REVIEW.md
│   ├── DATASET_STATUS.md
│   ├── METHOD_JUSTIFICATION.md
│   └── AQPATHFORMER_DESIGN.md
└── tests/                      # Unit tests
```

## Quick Start

### 1. Prepare Pilot Dataset (LC25000)
```bash
# Download LC25000 from Zenodo/Kaggle
# Place in data/lc25000/
# Structure: data/lc25000/{lung_aca,lung_n,lung_scc,colon_aca,colon_n}/*.jpeg
```

### 2. Run Baseline Experiment
```bash
# Train ResNet50 baseline
python -m src.train --config configs/baselines/resnet50_lc25000.yaml
```

### 3. Run AQPathFormer Experiment
```bash
# Train full AQPathFormer (after components implemented)
python -m src.train --config configs/aqpathformer/full_lc25000.yaml
```

### 4. Evaluate Model
```bash
# Evaluate on test set
python -m src.eval.evaluate --checkpoint experiments/EXP001/checkpoint/best.pt --config configs/baselines/resnet50_lc25000.yaml
```

## Configuration
All experiments configured via YAML files in `configs/`. Key parameters:
- `model`: Model architecture
- `dataset`: Dataset name and path
- `preprocessing`: Image size, normalization, augmentation
- `training`: Batch size, epochs, learning rate, optimizer
- `quantum`: Qubits, circuit depth, feature map, noise settings

## Experiment Tracking
Experiments automatically logged to:
- **Local**: `experiments/EXP###/` (config, metrics, logs, checkpoints)
- **Weights & Biases**: If configured (set `WANDB_API_KEY`)

## Datasets
| Dataset | Status | Classes | Images | Access |
|---------|--------|---------|--------|--------|
| LC25000 | Pilot | 5 | ~25K | Public (Zenodo) |
| CRC-100K | Planned | 9 | ~100K | Public (Zenodo) |
| BreakHis | Planned | 8 | ~7K | Registration |
| CAMELYON16 | Planned | 2 | ~400 | Challenge |
| CAMELYON17 | Planned | 5 | ~1K | Challenge |
| PANDA | Planned | 6 | ~11K | Kaggle |
| TCGA | Planned | Multi | ~30K | dbGaP |

## Baselines
| Model | Source | Experiment |
|-------|--------|------------|
| ResNet50 | timm | EXP001 |
| ViT-B/16 | timm | EXP002 |
| Swin-T | timm | EXP003 |
| ConvNeXt-T | timm | EXP004 |
| EfficientNet-B0 | timm | EXP005 |
| UNI | Foundation | EXP006 |
| CONCH | Foundation | EXP007 |
| QCNN | Custom | EXP008 |
| Hybrid QViT | Custom | EXP009 |

## Ablation Studies
| ID | Ablation | Experiments |
|----|----------|-------------|
| A | Fixed vs Adaptive Patch | EXP010 vs EXP011 |
| B | Classical vs Quantum Encoder | EXP020 vs EXP021 |
| C | Standard vs Adaptive Quantum Attention | EXP030 vs EXP031 |
| D | Single vs Multi-Scale | EXP040 vs EXP041 |
| E | No Cross-Cancer vs Cross-Cancer | EXP050 vs EXP051 |
| F | Ideal vs Noisy Quantum | EXP060 vs EXP061 |
| G | Full vs Reduced AQPathFormer | EXP070 vs EXP071 |

## Hardware Requirements
- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB+ RAM, CPU with AVX2 support
- **Quantum Simulation**: Scales exponentially with qubits (4-6 recommended)

## Reproducibility
- All experiments use fixed seeds (default: 42)
- Complete environment captured in `experiments/EXP###/environment.json`
- Git commit hash recorded for each experiment
- Configuration files version-controlled

## Documentation
- `docs/ENVIRONMENT.md` - System specifications
- `docs/DECISIONS.md` - Technical decision log
- `docs/ASSUMPTIONS.md` - Research assumptions
- `docs/MENTOR_ALIGNMENT.md` - Specification mapping
- `docs/LITERATURE_REVIEW.md` - Literature survey (Phase 2)
- `docs/DATASET_REVIEW.md` - Dataset analysis (Phase 2)

## License
This project is for academic research purposes. Dataset licenses vary - see `docs/DATASET_STATUS.md`.

## Citation
If you use this work, please cite:
```bibtex
@misc{aqpathformer2026,
  title={AQPathFormer: An Adaptive Quantum Vision Transformer for Multi-Cancer Histopathological Intelligence},
  author={Autonomous Research Engineer},
  year={2026},
  note={Semester Research Internship}
}
```

## Contact
For questions about this implementation, refer to the project documentation in `docs/` or the experiment logs in `experiments/`.