# DATASET_STATUS.md - Dataset Availability & Experiment Tracking

## Overview
Tracking document for all 7 candidate datasets. Records availability, access status, download progress, and experiment execution status.

---

## Dataset Status Summary

| Dataset | Availability | Access Status | Download | Preprocessing | Loader Implemented | Experiments Run |
|---------|--------------|---------------|----------|---------------|-------------------|-----------------|
| LC25000 | ✅ Public | ✅ No barriers | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ None |
| CRC-100K | ✅ Public | ✅ No barriers | ⬜ Planned | ⬜ Planned | ⬜ Planned | ⬜ Planned (Ph 11) |
| BreakHis | ⚠️ Registration | ⬜ Not started | ⬜ Planned | ⬜ Planned | ⬜ Planned | ⬜ Planned (Ph 11) |
| CAMELYON16 | ⚠️ Challenge | ⬜ Not started | ❌ Not planned | ❌ Not planned | ❌ Not planned | ❌ Not planned |
| CAMELYON17 | ⚠️ Challenge | ⬜ Not started | ❌ Not planned | ❌ Not planned | ❌ Not planned | ❌ Not planned |
| PANDA | ⚠️ Kaggle | ⬜ Not started | ❌ Not planned | ❌ Not planned | ❌ Not planned | ❌ Not planned |
| TCGA | ❌ dbGaP | ❌ Not feasible | ❌ Not planned | ❌ Not planned | ❌ Not planned | ❌ Not planned |

**Legend**: ✅ Complete | ⬜ Pending/Planned | ⚠️ Barriers | ❌ Not pursued

---

## Detailed Dataset Records

### 1. LC25000 (PRIMARY PILOT)

#### Access Information
- **Source URL**: https://zenodo.org/records/3531430
- **Alternative**: https://www.kaggle.com/datasets/andrewmvd/lc25000
- **License**: CC BY 4.0 (allows commercial/research use with attribution)
- **Access Type**: Direct download, no registration
- **File Size**: ~2.5 GB (ZIP)
- **Download Time**: ~5-10 minutes (broadband)

#### Download Procedure
```bash
# Option 1: Zenodo (direct)
wget https://zenodo.org/records/3531430/files/LC25000.zip -O data/lc25000.zip
unzip data/lc25000.zip -d data/lc25000/

# Option 2: Kaggle (requires API token)
# kaggle datasets download -d andrewmvd/lc25000 -p data/
# unzip data/lc25000.zip -d data/lc25000/
```

#### Dataset Structure (Post-Download)
```
data/lc25000/
├── lung_aca/     # 5,000 images
├── lung_n/       # 5,000 images
├── lung_scc/     # 5,000 images
├── colon_aca/    # 5,000 images
└── colon_n/      # 5,000 images
```

#### Preprocessing Plan
- **Resize**: 768×768 → 224×224 (LANCZOS)
- **Normalization**: ImageNet mean/std (0.485, 0.456, 0.406 / 0.229, 0.224, 0.225)
- **Augmentation**: RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), RandomRotation(90), ColorJitter(0.1)
- **Split Strategy**: Random (stratified) 70/15/15 with fixed seed 42
- **Patient-Level Split**: NOT POSSIBLE (no patient IDs in base dataset)

#### Experiment Tracking
| Experiment | Model | Config | Status | Metrics | Notes |
|------------|-------|--------|--------|---------|-------|
| EXP001 | ResNet50 | resnet50_lc25000.yaml | ⬜ Pending | - | Baseline 1 |
| EXP002 | ViT-B/16 | vit_lc25000.yaml | ⬜ Pending | - | Baseline 2 |
| EXP003 | Swin-T | swin_lc25000.yaml | ⬜ Pending | - | Baseline 3 |
| EXP004 | ConvNeXt-T | convnext_lc25000.yaml | ⬜ Pending | - | Baseline 4 |
| EXP005 | EfficientNet-B0 | efficientnet_lc25000.yaml | ⬜ Pending | - | Baseline 5 |
| EXP010 | AQPathFormer (Fixed) | aqpathformer_fixed.yaml | ⬜ Planned | - | Ablation A |
| EXP011 | AQPathFormer (Adaptive) | aqpathformer_adaptive.yaml | ⬜ Planned | - | Ablation A |
| EXP020 | AQPathFormer (Classical Enc) | aqpathformer_classical_enc.yaml | ⬜ Planned | - | Ablation B |
| EXP021 | AQPathFormer (Quantum Enc) | aqpathformer_quantum_enc.yaml | ⬜ Planned | - | Ablation B |
| EXP030 | AQPathFormer (Std Attn) | aqpathformer_std_attn.yaml | ⬜ Planned | - | Ablation C |
| EXP031 | AQPathFormer (Adapt Q Attn) | aqpathformer_adapt_q_attn.yaml | ⬜ Planned | - | Ablation C |
| EXP040 | AQPathFormer (Single-Scale) | aqpathformer_single_scale.yaml | ⬜ Planned | - | Ablation D |
| EXP041 | AQPathFormer (Multi-Scale) | aqpathformer_multi_scale.yaml | ⬜ Planned | - | Ablation D |
| EXP050 | AQPathFormer (No Cross-Cancer) | aqpathformer_single.yaml | ⬜ Planned | - | Ablation E |
| EXP051 | AQPathFormer (Cross-Cancer) | aqpathformer_cross_cancer.yaml | ⬜ Planned | - | Ablation E |
| EXP060 | AQPathFormer (Ideal Quantum) | aqpathformer_ideal.yaml | ⬜ Planned | - | Ablation F |
| EXP061 | AQPathFormer (Noisy Quantum) | aqpathformer_noisy.yaml | ⬜ Planned | - | Ablation F |
| EXP070 | AQPathFormer (Reduced) | aqpathformer_reduced.yaml | ⬜ Planned | - | Ablation G |
| EXP071 | AQPathFormer (Full) | aqpathformer_full.yaml | ⬜ Planned | - | Ablation G |

#### Statistics to Compute
- [ ] Class distribution verification
- [ ] Image size statistics (min/max/mean/std)
- [ ] Color channel statistics
- [ ] Duplicate detection (perceptual hash)
- [ ] Split distribution verification

---

### 2. CRC-100K (SECONDARY - PHASE 11)

#### Access Information
- **Source URL**: https://zenodo.org/records/2530835
- **License**: CC BY 4.0
- **Access Type**: Direct download
- **File Size**: ~5 GB
- **Classes**: 9 tissue types (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM)

#### Download Procedure
```bash
wget https://zenodo.org/records/2530835/files/CRC-VAL-HE-7K.zip -O data/crc100k_val.zip
wget https://zenodo.org/records/2530835/files/NCT-CRC-HE-100K.zip -O data/crc100k_train.zip
unzip data/crc100k_train.zip -d data/crc100k/
unzip data/crc100k_val.zip -d data/crc100k/
```

#### Structure
```
data/crc100k/
├── NCT-CRC-HE-100K/     # Training ~100K
│   ├── ADI/
│   ├── BACK/
│   └── ...
└── CRC-VAL-HE-7K/       # Validation ~7K
    ├── ADI/
    └── ...
```

#### Experiment Tracking (Phase 11)
| Experiment | Purpose | Status |
|------------|---------|--------|
| EXP080 | CRC-100K baseline (ResNet50) | ⬜ Planned |
| EXP081 | Cross-cancer: LC25000→CRC | ⬜ Planned |
| EXP082 | Joint training LC+CRC | ⬜ Planned |

---

### 3. BreakHis (TERTIARY - PHASE 11)

#### Access Information
- **Source**: http://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- **License**: Academic/Research use only
- **Access Type**: Registration form → manual approval (1-3 days)
- **Contact**: vri@inf.ufpr.br
- **File Size**: ~2 GB
- **Classes**: 8 subtypes × 4 magnifications

#### Download Procedure
```bash
# After registration approval, download link provided
# Typically: wget <provided_link> -O data/breakhis.zip
unzip data/breakhis.zip -d data/breakhis/
```

#### Structure
```
data/breakhis/
├── benign/
│   ├── SOB/
│   │   ├── A/ (Adenosis)
│   │   ├── F/ (Fibroadenoma)
│   │   ├── PT/ (Phyllodes Tumor)
│   │   └── TA/ (Tubular Adenoma)
├── malignant/
│   ├── SOB/
│   │   ├── DC/ (Ductal Carcinoma)
│   │   ├── LC/ (Lobular Carcinoma)
│   │   ├── MC/ (Mucinous Carcinoma)
│   │   └── PC/ (Papillary Carcinoma)
```

#### Magnification Subdirectories
Each class folder contains: `40X/`, `100X/`, `200X/`, `400X/`

#### Experiment Tracking (Phase 11)
| Experiment | Purpose | Status |
|------------|---------|--------|
| EXP090 | Multi-scale baseline (4 mags) | ⬜ Planned |
| EXP091 | Cross-cancer: LC25000→BreakHis | ⬜ Planned |
| EXP092 | Leave-one-magnification-out | ⬜ Planned |

---

### 4. CAMELYON16 (NOT PURSUED)

#### Access Barrier
- **Requirement**: Grand Challenge account + challenge registration
- **Process**: Register on grand-challenge.org → Join CAMELYON16 → Accept terms → Download
- **Timeline**: Days to weeks
- **Data Type**: Whole-slide images (WSI), not patch classification

#### Reason for Exclusion
- Task mismatch: Metastasis detection + segmentation (not patch classification)
- Data format: WSI requiring patch extraction pipeline
- Access barrier: Challenge registration delays

---

### 5. CAMELYON17 (NOT PURSUED)

#### Access Barrier
- **Requirement**: Grand Challenge account + CAMELYON17 registration
- **Data Type**: WSI with multi-center domain shift
- **Task**: Domain generalization for metastasis detection

#### Reason for Exclusion
- Same as CAMELYON16: task mismatch, WSI format, access barriers

---

### 6. PANDA (NOT PURSUED)

#### Access Information
- **Source**: Kaggle PANDA Challenge
- **Requirement**: Kaggle account + competition rules acceptance
- **Data Type**: WSI (prostate biopsy)
- **Task**: Gleason grading (ordinal regression)

#### Reason for Exclusion
- Task mismatch: Grading/regression vs classification
- Data format: WSI
- 11K WSIs = massive computational requirement

---

### 7. TCGA (NOT FEASIBLE)

#### Access Barrier
- **Requirement**: dbGaP authorization (NIH)
- **Process**: Research project proposal → IRB approval → dbGaP application → Review (weeks-months)
- **Data Type**: Controlled-access (genomics + pathology)

#### Reason for Exclusion
- Access timeline: Months (exceeds semester)
- Computational requirement: HPC needed
- Overkill for classification benchmarking

---

## 8. Download & Preparation Checklist

### LC25000 (IMMEDIATE)
- [ ] Download from Zenodo
- [ ] Verify file integrity (checksum)
- [ ] Extract to `data/lc25000/`
- [ ] Verify class folders and counts
- [ ] Generate dataset statistics
- [ ] Create train/val/test splits (seed 42)
- [ ] Save split indices for reproducibility

### CRC-100K (PHASE 11)
- [ ] Download train + val sets
- [ ] Extract and verify
- [ ] Map 9 tissue types to compatible labels
- [ ] Create splits

### BreakHis (PHASE 11)
- [ ] Submit registration form
- [ ] Wait for approval
- [ ] Download
- [ ] Organize by magnification
- [ ] Create patient-level splits

---

## 9. Data Leakage Prevention

### LC25000 (No Patient IDs)
- **Risk**: Unknown if multiple patches from same patient
- **Mitigation**: 
  - Use fixed random seed for all splits
  - Report split methodology explicitly
  - Document limitation in assumptions

### CRC-100K (Patient IDs Available)
- **Source**: Metadata file with patient IDs
- **Strategy**: Patient-level stratified split

### BreakHis (Patient IDs Available)
- **Source**: Filename encodes patient ID
- **Strategy**: Patient-level stratified split

---

## 10. Licensing Compliance

| Dataset | License | Attribution Required | Commercial Use | Redistribution |
|---------|---------|---------------------|----------------|----------------|
| LC25000 | CC BY 4.0 | Yes | Yes | Yes |
| CRC-100K | CC BY 4.0 | Yes | Yes | Yes |
| BreakHis | Academic Only | Yes | No | No |
| CAMELYON | Challenge Terms | Yes | Restricted | No |
| PANDA | Kaggle Terms | Yes | Restricted | No |
| TCGA | dbGaP | Yes | No | No |

**Compliance**: All used datasets properly attributed in reports; no redistribution of raw data.

---

*Last Updated: 2026-09-13*
*Next Update: After LC25000 download and statistics generation*