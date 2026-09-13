You are the lead autonomous research engineer responsible for implementing, experimentally validating, documenting, and packaging the semester research internship project:

“AQPathFormer: An Adaptive Quantum Vision Transformer for Multi-Cancer Histopathological Intelligence”

The authoritative project specification is the supplied document:
AQPathFormer.docx

Your job is to transform that research concept into a reproducible, scientifically defensible research implementation and experimental package.

IMPORTANT:
You are an autonomous research ENGINEER, not a text-generation assistant.
Do not merely create a theoretical implementation.
Inspect the environment, research relevant technical details, write the code, install/configure dependencies where possible, obtain datasets where legally and technically possible, preprocess data, run experiments, collect real results, debug failures, perform ablations, compare against baselines, generate figures/tables, and maintain complete documentation.

NEVER fabricate experimental results.
NEVER invent metrics.
NEVER claim an experiment was completed unless you actually executed it and recorded the result.
NEVER silently replace a failed experiment with a guessed result.
When an experiment cannot be completed because of missing hardware, inaccessible data, unavailable credentials, insufficient memory, or another external limitation, record the exact limitation and continue with the strongest valid alternative.

============================================================
1. AUTHORITATIVE RESEARCH OBJECTIVE
============================================================

The project specification describes AQPathFormer as an Adaptive Quantum Vision Transformer intended for multi-cancer histopathological intelligence.

The stated motivation includes:
- heterogeneous tissue morphology
- very large pathology images
- limited generalisation across cancer types
- limitations of fixed patch representations
- limitations of static attention
- limited multi-scale reasoning

The proposed framework contains:
1. Adaptive Patch Generator
2. Quantum Patch Encoder
3. Adaptive Quantum Multi-Head Attention
4. Multi-Scale Quantum Feature Fusion
5. Cross-Cancer Representation Learning Module
6. Noise-Aware Quantum Layer
7. Hybrid Classical–Quantum Decoder
8. Multi-Cancer Prediction Head

The stated algorithmic contributions are:
1. Adaptive Quantum Patch Encoding
2. Quantum Multi-Scale Attention
3. Dynamic Circuit Allocation
4. Quantum Feature Fusion
5. Noise-Aware Vision Learning

Candidate datasets listed in the specification:
- CAMELYON16
- CAMELYON17
- BreakHis
- PANDA
- TCGA Histopathology
- LC25000
- CRC-100K

Baselines listed in the specification:
- Vision Transformer
- Swin Transformer
- ConvNeXt
- ResNet50
- EfficientNet
- UNI
- CONCH
- QCNN
- Hybrid Quantum Vision Transformers

Evaluation metrics listed in the specification:
- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC-AUC
- PR-AUC
- MCC
- FLOPs
- Parameters
- Training Time
- Inference Time
- Circuit Depth
- Fidelity
- Noise Robustness

These items must guide the implementation.

============================================================
2. SCIENTIFIC INTEGRITY RULES
============================================================

These rules are mandatory.

A. NEVER fabricate results.

B. Every reported metric must have:
- dataset
- split
- random seed
- model version/commit
- configuration
- date/time
- experiment identifier
- actual output source

C. Maintain an experiment registry.

For every experiment store:
- experiment ID
- model
- dataset
- preprocessing configuration
- train/validation/test split
- hyperparameters
- random seed
- environment information
- training duration
- inference duration
- metrics
- checkpoint path
- logs
- git commit
- failure status if applicable

D. Make research assumptions explicit.

The mentor's document does NOT specify every implementation detail.
Therefore:
- identify missing specifications
- choose reasonable scientifically defensible defaults
- document every choice
- never present an assumption as if it came from the mentor

Create:
docs/ASSUMPTIONS.md

Difficult decisions must be recorded there.

E. Do not cherry-pick favorable results.
All successful, failed, partial, and aborted experiments must remain traceable.

F. Prevent data leakage.
Especially verify:
- patient-level separation where metadata allows it
- no duplicate images across splits
- no augmented version crossing train/test boundaries
- no test-set tuning

G. Use reproducible seeds wherever possible.

H. Save raw experimental outputs before generating summarized tables.

============================================================
3. AUTONOMOUS EXECUTION POLICY
============================================================

Operate autonomously.

Do NOT repeatedly stop and ask me what to do next.

Proceed through the research stages sequentially.

Only stop and request human intervention when:
1. a required credential/login cannot be obtained automatically;
2. a dataset requires manual acceptance of terms;
3. hardware is physically unavailable;
4. an irreversible external action requires my explicit approval;
5. a scientifically critical ambiguity cannot be resolved using the supplied specification or defensible research practice.

For everything else:
- investigate
- choose a defensible solution
- implement it
- test it
- document it
- continue

If a method fails:
- diagnose the failure
- fix it
- rerun
- record the failure
- continue

Do not abandon the project after the first implementation failure.

============================================================
4. FIRST PHASE — PROJECT RECONNAISSANCE
============================================================

Before writing the main implementation:

1. Inspect the operating system.
2. Inspect CPU, RAM, GPU, GPU VRAM and CUDA availability.
3. Inspect Python version.
4. Inspect installed ML/quantum libraries.
5. Determine whether the environment supports:
   - PyTorch
   - torchvision
   - transformers
   - timm
   - Qiskit
   - PennyLane
   - scikit-learn
   - pandas
   - matplotlib
   - seaborn
   - OpenCV/PIL
   - experiment tracking tools if useful
6. Determine whether the system can run:
   - CPU training
   - CUDA training
   - quantum simulation
   - remote quantum execution if available

Record the result in:
docs/ENVIRONMENT.md

Do not assume a GPU exists.

Adapt the experimental scale to available hardware.

============================================================
5. SECOND PHASE — RESEARCH/LITERATURE VERIFICATION
============================================================

Before implementing new architecture details, research current authoritative documentation and relevant peer-reviewed literature.

Research:
- Vision Transformers
- Swin Transformer
- ConvNeXt
- ResNet50
- EfficientNet
- pathology foundation models
- quantum machine learning for image classification
- quantum vision transformers
- QCNNs
- quantum attention mechanisms
- parameterized quantum circuits
- quantum feature maps
- quantum noise models
- histopathology image classification
- the listed datasets

Prefer:
- original papers
- official dataset documentation
- official framework documentation
- authoritative benchmark papers

Create:
docs/LITERATURE_REVIEW.md
docs/DATASET_REVIEW.md
docs/METHOD_JUSTIFICATION.md

For every important research decision provide citations.

Do not claim AQPathFormer itself already exists unless an actual source proves it.
Treat the supplied mentor document as the specification of the proposed research concept.

============================================================
6. DATASET STRATEGY
============================================================

The specification lists seven datasets.

Do NOT immediately assume that all seven must be downloaded.

Implement the project so datasets are plug-in modules.

Create a common dataset interface:

Dataset
  ├── metadata loading
  ├── validation
  ├── preprocessing
  ├── train/val/test split
  ├── class mapping
  ├── statistics
  └── dataloader

Initially identify the dataset that provides the best balance between:
- accessibility
- size
- compute requirements
- relevance to the stated research objective
- reproducibility
- compatibility with the proposed model

Start with a pilot dataset.

Then scale to additional datasets.

The final implementation must support all seven listed datasets whenever technically and legally possible, without requiring all seven to be processed simultaneously.

Create:
docs/DATASET_STATUS.md

For every dataset record:
- availability
- license/access restrictions
- download procedure
- image count
- class structure
- resolution characteristics
- metadata
- preprocessing
- split strategy
- whether the experiment was actually executed

If a dataset is inaccessible, document why.

============================================================
7. DATA PREPROCESSING
============================================================

Build robust pathology image preprocessing.

Implement configurable:
- resizing
- cropping
- normalization
- patch extraction
- augmentation
- stain-related preprocessing where justified
- multi-scale patch extraction

Avoid destructive preprocessing by default.

Make all preprocessing parameters configurable via YAML/JSON.

Do not hard-code dataset-specific constants into model code.

Create preprocessing validation scripts.

Generate:
- class-distribution plots
- sample image grids
- patch visualizations
- image-size statistics
- dataset summary tables

Save these under:
reports/figures/
reports/tables/

============================================================
8. BASELINE PIPELINE FIRST
============================================================

Before AQPathFormer, build a reliable baseline pipeline.

At minimum implement and evaluate:

1. ResNet50
2. Vision Transformer
3. Swin Transformer
4. ConvNeXt
5. EfficientNet

Where technically justified and available, investigate:
6. UNI
7. CONCH

Also investigate:
8. QCNN
9. Hybrid Quantum Vision Transformer

Do not implement baselines merely as placeholders.

Each baseline must be trainable/evaluable through the same experiment framework.

Use consistent:
- dataset split
- preprocessing
- metrics
- seed strategy
- evaluation code

This makes comparisons scientifically meaningful.

============================================================
9. AQPathFORMER ARCHITECTURE
============================================================

Implement AQPathFormer as modular components.

Create separate modules for:

models/aqpathformer/
    adaptive_patch_generator.py
    quantum_patch_encoder.py
    adaptive_quantum_attention.py
    multiscale_quantum_fusion.py
    cross_cancer_representation.py
    noise_aware_quantum_layer.py
    hybrid_decoder.py
    multicancer_head.py
    aqpathformer.py

Each component must have:
- clearly defined input/output dimensions
- unit tests
- documentation
- configurable parameters

============================================================
10. ADAPTIVE PATCH GENERATOR
============================================================

Do not use only a fixed patch size.

Design an adaptive mechanism that can select or weight patches using image/tissue characteristics.

Test at least one defensible adaptive strategy.

Compare against a fixed-patch version.

Perform an ablation:

Fixed Patch
vs
Adaptive Patch

Record:
- accuracy impact
- computational cost
- number of selected patches
- runtime impact

============================================================
11. QUANTUM PATCH ENCODER
============================================================

Implement a trainable hybrid quantum/classical patch encoder.

The quantum portion must have:
- configurable number of qubits
- configurable circuit depth
- configurable feature map
- configurable variational layers

Because classical images contain far more dimensions than practical near-term quantum circuits can currently handle directly, implement a classical dimensionality-reduction/projection stage before quantum encoding where necessary.

Do not pretend the complete pathology image is directly loaded into a large quantum computer.

Document exactly:
classical input
→ dimensionality reduction
→ quantum feature map
→ variational circuit
→ expectation/features
→ classical continuation

Support both:
- quantum simulation
- classical fallback/proxy mode for debugging

Label these modes clearly.

============================================================
12. ADAPTIVE QUANTUM MULTI-HEAD ATTENTION
============================================================

Implement the research concept of adaptive quantum attention.

Define a scientifically meaningful mechanism in which:
- attention computation depends on patch/context information
- quantum processing capacity/circuit allocation can vary according to the representation
- the mechanism remains computationally tractable

Do not use the phrase “dynamic circuit allocation” unless the actual implementation changes circuit execution/allocation or you clearly define it as a software-level allocation policy.

Create an explicit technical definition in:
docs/AQPATHFORMER_DESIGN.md

Include equations where appropriate.

============================================================
13. MULTI-SCALE QUANTUM FEATURE FUSION
============================================================

Implement multiple image/patch scales.

For example:
- coarse tissue context
- medium-scale structure
- fine cellular detail

Fuse multi-scale representations.

Compare:
single-scale
vs
multi-scale quantum fusion

Record ablation results.

============================================================
14. CROSS-CANCER REPRESENTATION LEARNING
============================================================

The project is intended to generalize across cancer types.

Create a dataset/task abstraction allowing models to train/evaluate across multiple cancer datasets.

Implement at least one scientifically defensible cross-cancer learning protocol.

Examples to investigate:
- joint multi-dataset training
- domain-shared encoder with cancer-specific heads
- domain adaptation
- leave-one-cancer-type-out evaluation
- cross-dataset transfer

Choose the most feasible valid approach based on actual dataset compatibility.

Do not combine datasets with incompatible labels without designing an explicit mapping/task definition.

============================================================
15. NOISE-AWARE QUANTUM LAYER
============================================================

Implement configurable noise-aware quantum experiments where simulation infrastructure permits.

Investigate representative noise effects supported by the selected quantum framework.

Compare:
- ideal simulation
- noisy simulation

Measure robustness.

Report:
- circuit depth
- noise setting
- fidelity or relevant quantum metric
- classification metrics

Never report noise robustness without actually running the noisy experiment.

============================================================
16. HYBRID CLASSICAL–QUANTUM DECODER
============================================================

Implement the architecture:

input
→ patch generation
→ classical/quantum representation
→ attention
→ multi-scale fusion
→ decoder/head
→ prediction

Keep classical and quantum components modular.

The model must be runnable entirely in simulation if actual quantum hardware is unavailable.

If hardware is available and appropriate:
- perform a small hardware validation experiment
- do not make hardware execution a dependency for the main benchmark

============================================================
17. MULTI-CANCER PREDICTION HEAD
============================================================

Implement a configurable prediction head.

Support:
- binary classification
- multiclass classification
- multi-dataset settings where valid

Ensure label mappings are explicit and reproducible.

============================================================
18. EXPERIMENT SYSTEM
============================================================

Create a central experiment runner.

Example:

python -m src.train --config configs/...

Every experiment must produce:

experiments/
    EXP001/
        config.yaml
        environment.json
        git_commit.txt
        train.log
        metrics.json
        predictions.csv
        confusion_matrix.png
        checkpoint/
        summary.md

Use unique experiment IDs.

Never overwrite previous experiment results.

============================================================
19. ABLATION STUDIES
============================================================

The final research package must include meaningful ablations.

At minimum:

A. Fixed patch vs adaptive patch

B. Classical encoder vs quantum patch encoder

C. Standard attention vs adaptive quantum attention

D. Single-scale vs multi-scale

E. No cross-cancer learning vs cross-cancer learning

F. Ideal quantum circuit vs noisy quantum circuit

G. Full AQPathFormer vs reduced AQPathFormer

Each ablation must be independently executable.

============================================================
20. EVALUATION
============================================================

Implement a single trusted evaluation module.

Compute where applicable:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC-AUC
- PR-AUC
- MCC

Also compute:

- FLOPs
- parameter count
- training time
- inference time

Quantum-specific measurements:

- circuit depth
- number of qubits
- number of circuit evaluations
- fidelity where meaningful
- noise robustness

Do not calculate a metric in an invalid configuration.

For example, ROC-AUC must be handled correctly for binary vs multiclass settings.

============================================================
21. STATISTICAL VALIDATION
============================================================

Where compute permits:

- run multiple seeds
- calculate mean and standard deviation
- report confidence intervals where appropriate
- test whether improvements are meaningful

Do not declare one model “better” solely because of a tiny difference.

Report uncertainty.

============================================================
22. COMPUTATIONAL EFFICIENCY
============================================================

Measure:

- parameter count
- FLOPs
- GPU memory where available
- training time
- inference time
- number of quantum operations
- circuit depth

Produce resource comparison tables.

Discuss accuracy-efficiency trade-offs.

============================================================
23. REPRODUCIBILITY
============================================================

Create:

requirements.txt or pyproject.toml
environment.yml if appropriate
Dockerfile if practical
.gitignore
README.md
AGENTS.md

Add:
- setup instructions
- dataset setup instructions
- training instructions
- evaluation instructions
- reproduction instructions
- expected directory structure
- troubleshooting

Every main experiment must be reproducible from configuration files.

============================================================
24. TESTING
============================================================

Create tests for:

- dataset loaders
- preprocessing
- patch generation
- tensor dimensions
- quantum encoding
- attention
- fusion
- decoder
- evaluation metrics
- checkpoint loading
- configuration parsing

Run tests before large experiments.

Do not spend hours training a model if basic tensor/unit tests are failing.

============================================================
25. VISUALIZATION
============================================================

Generate publication-quality figures:

1. AQPathFormer architecture diagram
2. pipeline diagram
3. dataset distribution
4. training curves
5. confusion matrices
6. ROC curves
7. PR curves
8. baseline comparison
9. ablation comparison
10. accuracy vs computational cost
11. circuit-depth/noise analysis
12. cross-cancer generalization
13. patch-selection visualizations
14. attention/feature visualizations where scientifically valid

Every plot must be generated from saved experimental data.

============================================================
26. RESEARCH REPORT PACKAGE
============================================================

Create:

reports/
    technical_report.md
    methodology.md
    experiment_report.md
    results.md
    limitations.md
    reproducibility.md

The final report should contain:

1. Abstract
2. Introduction
3. Background
4. Problem Statement
5. Research Gap
6. Objectives
7. Dataset Description
8. Preprocessing
9. Proposed AQPathFormer Architecture
10. Mathematical Formulation
11. Quantum Components
12. Adaptive Patch Encoding
13. Adaptive Quantum Attention
14. Multi-Scale Feature Fusion
15. Cross-Cancer Learning
16. Noise-Aware Learning
17. Experimental Setup
18. Baselines
19. Evaluation Metrics
20. Results
21. Ablation Studies
22. Computational Analysis
23. Discussion
24. Limitations
25. Future Work
26. Conclusion
27. References

Do not write final numerical results until they actually exist.

============================================================
27. MENTOR-ALIGNMENT DOCUMENT
============================================================

Create:

docs/MENTOR_ALIGNMENT.md

Make a table:

Mentor specification
→ implementation component
→ source file
→ experiment
→ result
→ status

For example:

Adaptive Patch Generator
→ models/aqpathformer/adaptive_patch_generator.py
→ EXP-XXX
→ ...
→ COMPLETE / PARTIAL / BLOCKED

Repeat for every item in the supplied specification.

============================================================
28. RESEARCH LIMITATIONS
============================================================

Be honest about limitations.

Possible limitations may include:
- quantum simulation cost
- available qubit count
- circuit depth
- noisy simulation cost
- limited dataset access
- compute resources
- dataset label incompatibility
- imbalance
- domain shift
- insufficient hardware access

Do not hide these limitations.

A scientifically honest limitation is better than a fabricated success.

============================================================
29. FINAL OUTPUT
============================================================

At the end, the repository must contain:

1. Complete source code
2. Dataset loaders
3. Preprocessing pipeline
4. Baseline implementations
5. AQPathFormer implementation
6. Quantum simulation implementation
7. Noise-aware implementation
8. Training scripts
9. Evaluation scripts
10. Configuration files
11. Tests
12. Experiment logs
13. Checkpoints where practical
14. Metrics
15. CSV result tables
16. Publication-quality figures
17. Ablation results
18. Dataset documentation
19. Literature review
20. Technical report
21. Reproducibility guide
22. Mentor-alignment matrix

============================================================
30. CRITICAL STOP CONDITIONS
============================================================

NEVER do any of these:

- fabricate metrics
- fabricate dataset statistics
- fabricate citations
- claim an unavailable dataset was downloaded
- claim a quantum hardware experiment was performed when it was not
- hide failed experiments
- silently use the test set during training
- silently change the research objective
- call a classical approximation a quantum experiment without disclosure
- copy a published model and falsely claim it is AQPathFormer

============================================================
31. DEVELOPMENT STRATEGY
============================================================

Use the following order:

PHASE 1
Environment + repository + AGENTS.md

PHASE 2
Literature + dataset study

PHASE 3
Dataset abstraction + pilot dataset

PHASE 4
Preprocessing + visualization

PHASE 5
Evaluation framework

PHASE 6
Classical baselines

PHASE 7
Quantum patch encoder

PHASE 8
AQPathFormer components

PHASE 9
Full AQPathFormer

PHASE 10
Ablations

PHASE 11
Multi-dataset experiments

PHASE 12
Noise analysis

PHASE 13
Cross-cancer generalization

PHASE 14
Final benchmarking

PHASE 15
Report + figures + reproducibility package

After every phase:
- run tests
- record progress
- update documentation
- commit working changes where appropriate

============================================================
32. AUTONOMOUS DECISION POLICY
============================================================

When there are multiple technically valid options:

1. Prefer reproducibility.
2. Prefer open/public datasets.
3. Prefer officially documented APIs.
4. Prefer stable libraries.
5. Prefer methods that can run on the available hardware.
6. Prefer simpler scientifically valid methods over unnecessarily complex ones.
7. Prefer an honest partial experiment over a fabricated “complete” result.

Keep a decision log:

docs/DECISIONS.md

For every important decision write:
- decision
- alternatives considered
- rationale
- consequences

============================================================
33. FINAL SELF-AUDIT
============================================================

Before declaring the project complete, independently audit:

DATA:
- Are datasets genuine?
- Are splits correct?
- Any leakage?

CODE:
- Do tests pass?
- Does training run from a clean environment?
- Are configs reproducible?

QUANTUM:
- Which parts are genuinely quantum?
- Which parts are classical?
- Is simulation clearly distinguished from hardware?

RESULTS:
- Are all numbers generated from actual experiments?
- Are failed experiments documented?
- Are comparisons fair?

RESEARCH:
- Are claims supported by evidence?
- Are assumptions clearly identified?
- Are limitations stated?

REPRODUCIBILITY:
- Can another student clone the repository and understand how the results were produced?

============================================================
34. EXECUTION INSTRUCTION
============================================================

Start NOW.

Do not merely explain what you plan to do.

Inspect the repository and environment first.

Create/update:
AGENTS.md
docs/DECISIONS.md
docs/ASSUMPTIONS.md
docs/ENVIRONMENT.md
docs/MENTOR_ALIGNMENT.md

Then proceed through the phases above.

Maintain a live TODO/progress system.

At every major milestone, leave the repository in a working state.

Your objective is to produce the strongest scientifically valid implementation possible from the mentor specification, not to create the appearance of completion.