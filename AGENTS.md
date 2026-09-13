# AGENTS.md - AQPathFormer Project Coordination

## Project Overview
**Project**: AQPathFormer: An Adaptive Quantum Vision Transformer for Multi-Cancer Histopathological Intelligence  
**Type**: Semester Research Internship  
**Status**: Phase 1 - Environment Setup Complete  
**Lead**: Autonomous Research Engineer (OpenCode)  

---

## Agent Roles & Responsibilities

### 1. Lead Research Engineer (Primary Agent)
- **Role**: End-to-end implementation, experimentation, documentation
- **Responsibilities**:
  - Architecture design & implementation
  - Dataset acquisition & preprocessing
  - Baseline model training & evaluation
  - AQPathFormer component development
  - Ablation studies & statistical validation
  - Report generation & reproducibility package
- **Decision Authority**: All technical decisions within specification bounds

### 2. Human Mentor (Oversight)
- **Role**: Strategic guidance, ambiguity resolution, approval gates
- **Intervention Points** (per Autonomous Execution Policy):
  1. Credential/login required for dataset access
  2. Manual dataset terms acceptance
  3. Hardware physically unavailable
  4. Irreversible external action needed
  5. Scientifically critical ambiguity unresolvable

---

## Communication Protocol

### Status Updates
- **Frequency**: After each phase completion
- **Format**: TODO list update + key metrics summary
- **Artifacts**: Experiment logs, figures, updated docs

### Decision Logging
- **Location**: `docs/DECISIONS.md`
- **Format**: Decision | Alternatives | Rationale | Consequences
- **Trigger**: Every significant technical choice

### Assumption Tracking
- **Location**: `docs/ASSUMPTIONS.md`
- **Format**: Assumption | Source | Justification | Impact if Wrong
- **Trigger**: Any unspecified detail from mentor document

### Mentor Alignment
- **Location**: `docs/MENTOR_ALIGNMENT.md`
- **Format**: Spec Item → Implementation → Experiment → Result → Status
- **Update**: After each component completion

---

## Development Workflow

### Phase Gates (from Master Prompt)
| Phase | Description | Exit Criteria |
|-------|-------------|---------------|
| 1 | Environment + Repository + AGENTS.md | All deps installed, dirs created, docs initialized |
| 2 | Literature + Dataset Study | LITERATURE_REVIEW.md, DATASET_REVIEW.md, DATASET_STATUS.md |
| 3 | Dataset Abstraction + Pilot | Base dataset class, pilot loader, preprocessing |
| 4 | Preprocessing + Visualization | Pipeline, validation scripts, figures/tables |
| 5 | Evaluation Framework | Metrics module, experiment runner, registry |
| 6 | Classical Baselines | 5+ baselines trained on pilot dataset |
| 7 | Quantum Patch Encoder | Hybrid encoder with sim + proxy modes |
| 8 | AQPathFormer Components | All 8 modules implemented & unit tested |
| 9 | Full AQPathFormer | Integrated model trains end-to-end |
| 10 | Ablations | 7 ablation experiments completed |
| 11 | Multi-Dataset | 2+ additional datasets integrated |
| 12 | Noise Analysis | Ideal vs noisy quantum comparison |
| 13 | Cross-Cancer | Leave-one-out or domain adaptation |
| 14 | Final Benchmarking | Statistical validation, resource tables |
| 15 | Report Package | All 6 report files + 14 figures |

### Quality Gates (Per Phase)
- [ ] Unit tests pass (`pytest tests/`)
- [ ] Code linting (ruff/black if configured)
- [ ] Documentation updated
- [ ] Git commit with descriptive message
- [ ] Experiment registry updated

---

## Experiment Management

### Naming Convention
- **Experiment ID**: `EXP###` (sequential, zero-padded)
- **Config**: `configs/{model_type}/{experiment_name}.yaml`
- **Output**: `experiments/EXP###/`

### Required Artifacts Per Experiment
```
experiments/EXP###/
├── config.yaml           # Full configuration
├── environment.json      # Python packages, git commit, hardware
├── git_commit.txt        # Git SHA
├── train.log             # Training stdout/stderr
├── metrics.json          # All computed metrics
├── predictions.csv       # Test set predictions
├── confusion_matrix.png  # Visualization
├── checkpoint/           # Model weights
└── summary.md            # Human-readable summary
```

### Baseline Experiments (Reserved IDs)
| ID Range | Purpose |
|----------|---------|
| EXP001-009 | Classical baselines (ResNet50, ViT, Swin, ConvNeXt, EfficientNet, UNI, CONCH, QCNN, Hybrid QViT) |
| EXP010-019 | Ablation A: Fixed vs Adaptive Patch |
| EXP020-029 | Ablation B: Classical vs Quantum Encoder |
| EXP030-039 | Ablation C: Standard vs Adaptive Quantum Attention |
| EXP040-049 | Ablation D: Single vs Multi-Scale |
| EXP050-059 | Ablation E: Cross-Cancer Learning |
| EXP060-069 | Ablation F: Ideal vs Noisy Quantum |
| EXP070-079 | Ablation G: Full vs Reduced AQPathFormer |
| EXP080+ | Multi-dataset, noise analysis, cross-cancer |

---

## Resource Constraints & Adaptations

### Hardware Limits (Documented in ENVIRONMENT.md)
- **Compute**: CPU-only (8GB RAM)
- **Batch Size**: 4-16 (adaptive)
- **Resolution**: 224×224 (ViT standard)
- **Quantum Qubits**: 4-6 (simulation feasibility)
- **Epochs**: 10-20 (initial), 50+ (final)

### Software Adaptations
- **Quantum Simulation**: PennyLane `default.qubit` (CPU)
- **Classical Proxy**: PCA projection for debugging
- **Mixed Precision**: Disabled (CPU)
- **Gradient Accumulation**: Enabled for effective batch size

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Quantum sim too slow | High | High | Classical proxy; limit qubits; circuit caching |
| Dataset access denied | Medium | High | Start with public datasets; document blockers |
| OOM errors | High | Medium | Gradient accumulation; smaller batches; resolution |
| No quantum advantage | Medium | Medium | Honest reporting; focus on architectural novelty |
| Timeline overrun | High | High | Strict phase deadlines; MVP first |

---

## Success Metrics (Minimum Viable)

1. **All 5 classical baselines** trained & evaluated on pilot dataset
2. **AQPathFormer components** unit-tested (100% pass)
3. **Full AQPathFormer** trains without crashing
4. **≥4/7 ablations** completed with real metrics
5. **Experiment registry** with full provenance for all runs
6. **Reproducible** from `requirements.txt` + configs

---

## Contact & Escalation

**Primary Channel**: This repository (commits, issues, docs)
**Escalation**: Direct mentor consultation at defined intervention points
**Documentation**: All decisions, assumptions, results in `docs/`

---

*Last Updated: 2026-09-13*  
*Phase 1 Complete - Proceeding to Phase 2*