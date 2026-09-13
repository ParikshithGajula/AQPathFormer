# DATASET_REVIEW.md - Comprehensive Dataset Analysis

## Overview
Analysis of all 7 candidate datasets from the AQPathFormer specification, evaluating suitability for pilot development and full benchmarking.

---

## 1. LC25000 - Lung and Colon Cancer Histopathological Images (SELECTED PILOT)

### Basic Information
- **Source**: "LC25000: Lung and Colon Cancer Histopathological Images" (Borkowski et al., 2019)
- **Repository**: Zenodo (DOI: 10.5281/zenodo.3531430), Kaggle
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Access**: **PUBLIC - No registration required**

### Dataset Characteristics
| Property | Value |
|----------|-------|
| **Total Images** | 25,000 |
| **Classes** | 5 |
| **Class Names** | lung_aca (lung adenocarcinoma), lung_n (lung benign), lung_scc (lung squamous cell carcinoma), colon_aca (colon adenocarcinoma), colon_n (colon benign) |
| **Images per Class** | 5,000 each (balanced) |
| **Resolution** | 768 × 768 pixels |
| **Format** | JPEG |
| **Color Space** | RGB |
| **Staining** | H&E (Hematoxylin & Eosin) |
| **Magnification** | 20× (estimated) |
| **Patient Metadata** | Not explicitly provided in base dataset |

### Class Distribution
```
lung_aca:  5,000 (20%)
lung_n:    5,000 (20%)
lung_scc:  5,000 (20%)
colon_aca: 5,000 (20%)
colon_n:   5,000 (20%)
Total:    25,000 (100%)
```
**Perfectly balanced** - ideal for initial development.

### Suitability for AQPathFormer
| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| **Accessibility** | 5 | Direct download, no barriers |
| **Size** | 4 | 25K images manageable on 8GB RAM |
| **Multi-Cancer** | 4 | 2 cancer types (lung + colon), 5 classes |
| **Resolution** | 4 | 768×768 → resize to 224×224 standard |
| **Balance** | 5 | Perfectly balanced classes |
| **Reproducibility** | 5 | Fixed dataset, public, versioned |
| **Relevance** | 5 | Histopathology, multi-cancer, classification |

**Overall: 4.7/5 - EXCELLENT PILOT DATASET**

### Download & Structure
```bash
# From Zenodo
wget https://zenodo.org/record/3531430/files/LC25000.zip
unzip LC25000.zip -d data/lc25000/

# Expected structure:
data/lc25000/
├── lung_aca/
│   ├── lung_aca_001.jpeg
│   └── ...
├── lung_n/
├── lung_scc/
├── colon_aca/
└── colon_n/
```

### Preprocessing Considerations
- Resize 768×768 → 224×224 (ViT standard)
- H&E stain normalization optional (Macenko/Reinhard)
- Patient-level splits not possible without metadata (use random with fixed seed)
- Augmentation: flip, rotation, color jitter

---

## 2. CRC-100K - Colorectal Cancer Histopathology (BACKUP PILOT)

### Basic Information
- **Source**: "CRC-100K: 100,000 Colorectal Cancer Histopathology Images" (Kather et al., 2019)
- **Repository**: Zenodo (DOI: 10.5281/zenodo.2530835)
- **License**: CC BY 4.0
- **Access**: **PUBLIC - No registration required**

### Dataset Characteristics
| Property | Value |
|----------|-------|
| **Total Images** | ~100,000 |
| **Classes** | 9 |
| **Class Names** | ADI (adipose), BACK (background), DEB (debris), LYM (lymphocytes), MUC (mucus), MUS (muscle), NORM (normal mucosa), STR (stroma), TUM (tumor) |
| **Resolution** | 150 × 150 (smaller patches) |
| **Format** | PNG |
| **Staining** | H&E |
| **Source** | 86 patients, 13 centers |

### Class Distribution (Approximate)
```
TUM:  ~20,000
NORM: ~15,000
STR:  ~12,000
LYM:  ~10,000
MUS:  ~10,000
ADI:  ~10,000
MUC:  ~8,000
DEB:  ~8,000
BACK: ~7,000
```
**Imbalanced** - requires weighted sampling/loss

### Suitability
| Criterion | Score | Notes |
|-----------|-------|-------|
| **Accessibility** | 5 | Public, no barriers |
| **Size** | 3 | 100K images challenging on 8GB RAM |
| **Multi-Cancer** | 2 | Single cancer type (colorectal), 9 tissue types |
| **Resolution** | 3 | 150×150 native (can resize up) |
| **Balance** | 2 | Highly imbalanced |
| **Patient Metadata** | 4 | 86 patients documented |

**Overall: 3.2/5 - GOOD FOR SCALING, NOT IDEAL PILOT**

### Use Case
- Phase 11: Scale to larger dataset
- Test class imbalance handling
- More tissue-type diversity within single cancer

---

## 3. BreakHis - Breast Cancer Histopathology

### Basic Information
- **Source**: "BreakHis: Breast Cancer Histopathological Image Classification" (Spanhol et al., 2015)
- **Repository**: Warwick University (requires registration)
- **License**: Academic/Research use only
- **Access**: **REGISTRATION REQUIRED** (manual approval)

### Dataset Characteristics
| Property | Value |
|----------|-------|
| **Total Images** | 7,909 |
| **Classes** | 8 (4 benign + 4 malignant) |
| **Benign** | Adenosis (A), Fibroadenoma (F), Phyllodes Tumor (PT), Tubular Adenoma (TA) |
| **Malignant** | Ductal Carcinoma (DC), Lobular Carcinoma (LC), Mucinous Carcinoma (MC), Papillary Carcinoma (PC) |
| **Magnifications** | 40×, 100×, 200×, 400× (multi-scale native!) |
| **Resolution** | 700 × 460 (varies) |
| **Format** | PNG |
| **Patients** | 82 patients |
| **Staining** | H&E |

### Class Distribution (Approximate)
| Class | 40× | 100× | 200× | 400× | Total |
|-------|-----|------|------|------|-------|
| A (Benign) | 112 | 112 | 112 | 112 | 448 |
| F (Benign) | 228 | 228 | 228 | 228 | 912 |
| PT (Benign) | 92 | 92 | 92 | 92 | 368 |
| TA (Benign) | 156 | 156 | 156 | 156 | 624 |
| DC (Malignant) | 444 | 444 | 444 | 444 | 1776 |
| LC (Malignant) | 244 | 244 | 244 | 244 | 976 |
| MC (Malignant) | 188 | 188 | 188 | 188 | 752 |
| PC (Malignant) | 148 | 148 | 148 | 148 | 592 |

### Suitability
| Criterion | Score | Notes |
|-----------|-------|-------|
| **Accessibility** | 2 | Registration + approval delay |
| **Size** | 4 | 7.9K manageable |
| **Multi-Cancer** | 2 | Single cancer (breast) |
| **Multi-Scale** | 5 | **Native 4 magnifications!** |
| **Patient Metadata** | 5 | 82 patients, patient IDs available |
| **Balance** | 3 | Moderate imbalance |

**Overall: 3.2/5 - EXCELLENT FOR MULTI-SCALE ABLATION, ACCESS BARRIER**

### Use Case
- Phase 11: Multi-scale ablation (native 40×/100×/200×/400×)
- Patient-level split validation
- Breast cancer addition to multi-cancer suite

---

## 4. CAMELYON16 - Lymph Node Metastasis Detection

### Basic Information
- **Source**: CAMELYON16 Challenge (Grand Challenge)
- **Repository**: Grand Challenge platform
- **License**: Challenge-specific (research use)
- **Access**: **CHALLENGE REGISTRATION REQUIRED**

### Dataset Characteristics
| Property | Value |
|----------|-------|
| **Task** | Metastasis detection (binary) + segmentation |
| **Training Slides** | 270 (160 normal, 110 tumor) |
| **Test Slides** | 129 |
| **Resolution** | Whole-slide images (WSI) ~100,000×100,000 |
| **Format** | TIFF (multi-resolution pyramid) |
| **Pixel Spacing** | 0.243 μm/pixel (40× equivalent) |
| **Staining** | H&E |
| **Centers** | 2 (Radboud, UMC Utrecht) |

### Suitability
| Criterion | Score | Notes |
|-----------|-------|-------|
| **Accessibility** | 1 | Challenge registration, approval process |
| **Task Type** | 2 | Detection/segmentation (not patch classification) |
| **Size** | 1 | WSIs too large for patch classification without MIL |
| **Multi-Cancer** | 1 | Single task (lymph node metastasis) |
| **Relevance** | 3 | Metastasis detection relevant but different task |

**Overall: 1.4/5 - NOT SUITABLE FOR PATCH CLASSIFICATION**

### Use Case
- Not recommended for AQPathFormer patch classification
- Would require MIL framework (different architecture)

---

## 5. CAMELYON17 - Multi-Center Lymph Node Metastasis

### Basic Information
- **Source**: CAMELYON17 Challenge
- **Repository**: Grand Challenge platform
- **Access**: **CHALLENGE REGISTRATION REQUIRED**

### Dataset Characteristics
| Property | Value |
|----------|-------|
| **Task** | Metastasis detection + domain generalization |
| **Centers** | 5 hospitals (different scanners/staining) |
| **Training** | 3 centers (~500 slides) |
| **Validation** | 1 center |
| **Test** | 1 center (unseen domain) |
| **Format** | WSI TIFF |

### Suitability
| Criterion | Score | Notes |
|-----------|-------|-------|
| **Accessibility** | 1 | Challenge registration |
| **Domain Shift** | 5 | **Explicit multi-center domain shift** |
| **Task Type** | 2 | Detection, not classification |
| **Multi-Cancer** | 1 | Single task |

**Overall: 1.5/5 - NOT SUITABLE FOR PATCH CLASSIFICATION**

### Use Case
- Domain adaptation research (if accessible)
- Not for core AQPathFormer development

---

## 6. PANDA - Prostate Cancer Grade Assessment

### Basic Information
- **Source**: PANDA Challenge (Kaggle 2020)
- **Repository**: Kaggle (requires account + competition rules acceptance)
- **License**: Kaggle competition terms
- **Access**: **KAGGLE ACCOUNT + TERMS ACCEPTANCE**

### Dataset Characteristics
| Property | Value |
|----------|-------|
| **Task** | Gleason grading (regression: 0-10) / Classification (ISUP 0-5) |
| **Images** | ~11,000 WSIs |
| **Resolution** | Whole-slide (multi-resolution) |
| **Format** | TIFF |
| **Staining** | H&E |
| **Labels** | Gleason score (primary + secondary pattern) |

### Suitability
| Criterion | Score | Notes |
|-----------|-------|-------|
| **Accessibility** | 3 | Kaggle account + terms |
| **Task Type** | 2 | Grading (ordinal regression) |
| **Size** | 2 | 11K WSIs = massive patch count |
| **Multi-Cancer** | 1 | Prostate only |

**Overall: 2.0/5 - TASK MISMATCH (GRADING vs CLASSIFICATION)**

### Use Case
- Ordinal regression extension (future work)
- Not for core classification experiments

---

## 7. TCGA Histopathology

### Basic Information
- **Source**: The Cancer Genome Atlas (NCI/GDC)
- **Repository**: GDC Data Portal
- **License**: TCGA Data Use Certification Required
- **Access**: **dbGaP AUTHORIZATION REQUIRED (Controlled Access)**

### Dataset Characteristics
| Property | Value |
|----------|-------|
| **Cancer Types** | 33+ cancer types |
| **Total Slides** | ~30,000+ WSIs |
| **Modalities** | Histopathology + Genomics + Clinical |
| **Format** | SVS (Aperio) / TIFF |
| **Access Timeline** | Weeks to months for approval |

### Suitability
| Criterion | Score | Notes |
|-----------|-------|-------|
| **Accessibility** | 1 | dbGaP authorization (months) |
| **Multi-Cancer** | 5 | **33+ cancer types** |
| **Size** | 1 | Massive, requires HPC |
| **Task Diversity** | 5 | Survival, subtype, mutation prediction |

**Overall: 1.8/5 - NOT FEASIBLE FOR SEMESTER PROJECT**

### Use Case
- Long-term research (PhD+ timescale)
- Not for semester internship

---

## 8. Comparative Summary

| Dataset | Access | Size | Classes | Cancer Types | Task | Pilot Suitability |
|---------|--------|------|---------|--------------|------|-------------------|
| **LC25000** | ★★★★★ Public | 25K | 5 | 2 (Lung, Colon) | Classification | **★★★★★ BEST** |
| **CRC-100K** | ★★★★★ Public | 100K | 9 | 1 (Colorectal) | Classification | ★★★☆☆ Good backup |
| **BreakHis** | ★★☆☆☆ Registration | 7.9K | 8 | 1 (Breast) | Classification | ★★★☆☆ Multi-scale |
| **CAMELYON16** | ★☆☆☆☆ Challenge | 270 WSI | 2 | 1 (Lymph) | Detection | ★☆☆☆☆ Wrong task |
| **CAMELYON17** | ★☆☆☆☆ Challenge | 500 WSI | 2 | 1 (Lymph) | Detection | ★☆☆☆☆ Wrong task |
| **PANDA** | ★★★☆☆ Kaggle | 11K WSI | 6 (ISUP) | 1 (Prostate) | Grading | ★★☆☆☆ Wrong task |
| **TCGA** | ★☆☆☆☆ dbGaP | 30K+ WSI | 33+ | 33+ | Multi-task | ★☆☆☆☆ Not feasible |

---

## 9. Recommended Dataset Strategy

### Phase 2-10 (Core Development): LC25000 Only
- **Reason**: Public, balanced, multi-cancer, manageable size
- **All experiments**: EXP001-EXP079 on LC25000

### Phase 11 (Multi-Dataset): Add CRC-100K + BreakHis
- **CRC-100K**: Larger scale, tissue-type diversity
- **BreakHis**: Native multi-scale, patient metadata
- **Cross-cancer evaluation**: Train on LC25000 → Test on CRC/BreakHis

### Phase 13 (Cross-Cancer): Leave-One-Dataset-Out
- Train on {LC25000, CRC-100K} → Test on BreakHis
- Train on {LC25000, BreakHis} → Test on CRC-100K
- Train on {CRC-100K, BreakHis} → Test on LC25000

### Not Pursued (This Project)
- CAMELYON16/17: Wrong task (detection/segmentation)
- PANDA: Wrong task (grading/regression)
- TCGA: Access timeline incompatible

---

## 10. Implementation Priority

| Priority | Dataset | Implementation File | Phase |
|----------|---------|---------------------|-------|
| 1 | LC25000 | `src/data/lc25000.py` | Phase 3 |
| 2 | CRC-100K | `src/data/crc100k.py` | Phase 11 |
| 3 | BreakHis | `src/data/breakhis.py` | Phase 11 |
| 4 | CAMELYON | `src/data/camelyon.py` | Not planned |
| 5 | PANDA | `src/data/panda.py` | Not planned |
| 6 | TCGA | `src/data/tcga.py` | Not planned |

---

*Last Updated: 2026-09-13*
*Phase 2 - Dataset Review Complete*