# Credit Card Fraud Detection — Faithful Reproduction of IEEE Access (2022)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-22%2F22%20Passing-brightgreen.svg)]()

> **Paper Title**: *Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms*  
> **Authors**: Fawaz Khaled Alarfaj, Iqra Malik, Hikmat Ullah Khan, Naif Almusallam, Muhammad Ramzan, and Muzamil Ahmed  
> **Journal**: *IEEE Access*, Volume 10, 2022. Pages 39700–39715.  
> **DOI**: [10.1109/ACCESS.2022.3166891](https://doi.org/10.1109/ACCESS.2022.3166891)

---

## 1. Project Overview & Scientific Reproducibility

This repository contains a **complete, faithful, scientifically auditable reproduction** of the 2022 IEEE Access study on Credit Card Fraud Detection.

In accordance with strict reproduction guidelines:
- **No methodology substitution**: We do not replace the paper's techniques with modern tabular architectures (e.g. LightGBM/CatBoost/TabNet) simply because they are more popular.
- **No silent enhancements**: We do not introduce SMOTE, synthetic oversampling, or hyperparameter optimization that the original authors did not perform.
- **Strict featurization ordering**: Preprocessing scalers (`StandardScaler` on `Amount`) are fitted **strictly on the training split** to prevent data leakage.
- **Auditable reconstructions**: Where paper descriptions contain ambiguities (such as the "5-layer baseline CNN" which Table 9 documents as a 497-parameter MLP, or kernel dimensions written as `32 x 2`), all parameters are mathematically matched against paper tables and isolated in explicit YAML configuration files.

---

## 2. Benchmark Dataset & Provenance

The study evaluates the European credit card fraud benchmark commonly distributed as `creditcard.csv`:
- **Total Transactions**: 284,807
- **Legitimate Transactions (Class 0)**: 284,315 (99.827%)
- **Fraudulent Transactions (Class 1)**: 492 (0.173%)
- **Features**: 30 input features:
  - `Time`: Elapsed seconds since first transaction.
  - `V1`–`V28`: Principal components obtained via PCA (anonymized).
  - `Amount`: Transaction amount in Euros (standardized with `StandardScaler`).
- **Target**: `Class` (binary: 0 = Legitimate, 1 = Fraud).
- **Split Ratio**: 80% Train (227,845 records, 394 fraud) / 20% Test (56,962 records, 98 fraud), verified against Table 6 confusion matrix numbers ($56849 + 6 + 24 + 83 = 56962$).

> **Note on Paper Inconsistency**: Section III-A of the paper states the dataset covers transactions from "September 2018" or "October 2018". The true provenance is the ULB Machine Learning Group benchmark collected in **September 2013**. This discrepancy is documented in [`docs/reproduction_gap_analysis.md`](docs/reproduction_gap_analysis.md).

---

## 3. Repository Architecture

```text
credit-card-fraud-reproduction/
├── configs/                       # Centralized, auditable YAML configurations
│   ├── base.yaml                  # Global data paths, seeds, feature splits
│   ├── classical_ml.yaml          # DT, KNN, LR, SVM, RF, XGBoost, ELM settings
│   ├── cnn_baseline.yaml          # Baseline 497-parameter network (Table 9)
│   ├── cnn_10_layers.yaml         # 10-layer sequential model (Table 8)
│   ├── cnn_14_layers.yaml         # Proposed 14-layer CNN (Section IV-C-11-a)
│   ├── cnn_17_layers.yaml         # Proposed 17-layer CNN (Section IV-C-11-b)
│   ├── cnn_20_layers.yaml         # Proposed 20-layer CNN (Section IV-C-11-c)
│   └── balanced_data.yaml         # Balanced undersampling config (Section IV-C-9)
├── data/
│   ├── raw/                       # Raw creditcard.csv (not committed to VCS)
│   ├── processed/                 # Scaled and partitioned artifacts
│   └── external/                  # OCR and extracted table data
├── docs/                          # In-depth research documentation
│   ├── paper_methodology.md       # Extracted formulas (Eq 1-12) and architectures
│   ├── reproduction_gap_analysis.md# Methodological audit, ambiguities, assumptions
│   └── results_reference.md       # Authoritative published baseline numbers
├── reports/
│   ├── figures/                   # Publication-quality charts (PNG & PDF)
│   ├── results/                   # Machine-readable experiment outputs (CSV/JSON)
│   └── reproduction_report.md     # Auto-generated scientific reproduction report
├── src/
│   ├── data/                      # Loader, schema validator, scaler, undersampler
│   ├── features/                  # Feature ranking (RF Gini & ANOVA F-value)
│   ├── models/                    # Classical ML, ELM, and CNN architectures
│   ├── training/                  # Central experiment runners and callbacks
│   ├── evaluation/                # Confusion matrix, ROC, PR curves, comparisons
│   ├── visualization/             # Paper-accurate figures (Figures 5, 7, 8, 14, 17)
│   └── reporting/                 # Markdown & CSV reporting pipeline
├── scripts/                       # Executable command-line utilities
│   ├── validate_dataset.py        # CLI dataset validation & integrity check
│   ├── run_classical_experiments.py # Trains all 7 classical ML algorithms
│   ├── run_cnn_experiments.py     # Runs baseline, epoch, depth, & balanced CNNs
│   ├── run_all_experiments.py     # Runs end-to-end experimental reproduction
│   └── generate_report.py         # Compiles final reproduction report
├── tests/                         # Automated unit & integration tests
│   ├── test_data_validation.py    # Schema, row counts, type validation
│   ├── test_preprocessing.py      # Leak-free splitting and scaling
│   ├── test_feature_selection.py  # Ranking dimensions and scores
│   ├── test_model_shapes.py       # Input/output tensor dimensionality
│   ├── test_model_parameters.py   # Exact parameter count verification (497, 119k)
│   ├── test_metrics.py            # Precision, Recall, F1, AUC verification
│   └── test_experiment_tracking.py# Configuration loading & serialization
├── app.py                         # Multi-page Streamlit Research Dashboard
├── pyproject.toml                 # Modern Python packaging configuration
└── requirements.txt               # Pinned reproducible dependencies
```

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10 or 3.11 (tested on Python 3.11.9)
- macOS, Linux, or Windows (POSIX bash / zsh recommended)

### Step 1: Clone the Repository & Create Virtual Environment
```bash
git clone <repository_url>
cd credit-card-fraud-reproduction

# Create isolated Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Pinned Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Dataset Placement
Ensure `creditcard.csv` is located in `data/raw/creditcard.csv`.
If the dataset is elsewhere, provide the path to the validation command.

---

## 5. Command-Line Reproduction Workflow

All experiments can be executed sequentially or individually:

### 1. Validate Dataset Integrity
```bash
python scripts/validate_dataset.py --data-path data/raw/creditcard.csv
```
*Validates schema (Time, V1-V28, Amount, Class), row counts (284,807), class balance (492 fraud), duplicate records, null values, and outputs `reports/results/data_validation_report.json`.*

### 2. Run Automated Unit & Smoke Tests
```bash
pytest tests/ -v
```
*Executes all 22 test suites covering data validation, leak-free preprocessing, feature ranking, baseline parameter count (exact 497), CNN-10 parameter count (exact 119,457), model shapes, and toy metrics.*

### 3. Run Classical Machine Learning Experiments (Table 5)
```bash
python scripts/run_classical_experiments.py
```
*Trains and evaluates Decision Tree (entropy, depth=4), KNN (k=5), Logistic Regression (RandomizedSearchCV on C), Random Forest, SVM (RBF), XGBoost, and Extreme Learning Machine (ELM). Generates `reports/results/model_comparison.csv`, confusion matrices, and ROC/PR curves.*

### 4. Run Deep Learning & CNN Experiments (Tables 6, 7, 8, 9 & Section IV-C-9)
```bash
# Full reproduction (Baseline, Epochs 14/35, Balanced Undersampling, Depths 14/17/20)
python scripts/run_cnn_experiments.py --depth-epochs 20
```

### 5. Run Entire End-to-End Experimental Suite
```bash
python scripts/run_all_experiments.py --depth-epochs 20
```

### 6. Compile Final Markdown Reproduction Report
```bash
python scripts/generate_report.py
```
*Compiles all experiment metrics, published targets, and traceability matrices into `reports/reproduction_report.md`.*

---

## 6. Interactive Streamlit Research Dashboard

To explore the dataset, interactively run training experiments, inspect confusion matrices, and review paper-versus-reproduced findings:

```bash
streamlit run app.py
```

### Dashboard Capabilities:
1. **Overview**: Citation, research summary, dataset breakdown, and workflow flowchart.
2. **Dataset Explorer**: Class balance visualizations, Amount statistics, correlation matrices, and PCA feature distributions.
3. **Model Training**: Interactive UI allowing the selection and real-time execution of any classical model or CNN architecture with custom hyperparameters.
4. **Model Evaluation**: Side-by-side metric tables, interactive confusion matrices, and ROC/PR curves.
5. **CNN Deep Dive**: Architectural parameter summaries, epoch history curves (accuracy and loss), and depth comparisons (14 vs 17 vs 20 layers).
6. **Reproduction Report**: Complete audit comparing published numbers to reproduced figures with one-click report download.

---

## 7. Experimental Results & Verification

### Table 5: Classical Machine Learning Benchmark Comparison

| Model | Published Accuracy | Reproduced Accuracy | Published F1-Score | Reproduced F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 99.80% | 99.93% | 85.71% | 80.45% | 0.893 |
| **Random Forest** | 99.70% | 99.96% | 84.70% | 86.84% | 0.957 |
| **KNN (k=5)** | 99.60% | 99.95% | 85.00% | 85.87% | 0.923 |
| **Logistic Regression** | 99.50% | 99.92% | 84.00% | 72.84% | 0.970 |
| **SVM (RBF)** | 98.60% | 99.94% | 82.00% | 80.90% | 0.941 |
| **XGBoost** | 99.00% | 99.96% | 84.00% | 87.78% | 0.980 |
| **Extreme Learning Machine** | 96.00% | 99.83% | 79.00% | 0.00%* | 0.531 |

*\*Note on ELM: With unscaled raw PCA features and random weights, ELM collapses to majority-class prediction without regularization tuning. Full analysis in `docs/reproduction_gap_analysis.md`.*

### Table 6: CNN 10-Layer Sequential Epoch Comparison

| Epochs | Published Accuracy | Reproduced Accuracy | Published Precision | Reproduced Precision | Published Recall | Reproduced Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **14** | 99.80% | 99.94% | 56.80% | 86.96% | 81.63% | 81.63% |
| **35** | 99.90% | 99.94% | 93.20% | 85.11% | 81.63% | 81.63% |

### Table 7: Proposed CNN Depth Variations (14, 17, 20 Layers)

| Architecture | Conv Layers | Total Layers | Reproduced Train Acc | Reproduced Val Acc | Published Val Acc Target |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN 14-Layers** | 2 | 14 | 99.93% | 99.93% | 94.30% (balanced) |
| **CNN 17-Layers** | 3 | 17 | 99.93% | 99.93% | 94.70% (balanced) |
| **CNN 20-Layers** | 4 | 20 | 99.90% | 99.93% | 93.50% (balanced) |

### Section IV-C-9: Balanced Dataset Experiment (Undersampling)
- **Class Distribution**: 492 Legitimate + 492 Fraudulent transactions ($1:1$ ratio).
- **Published Validation Accuracy**: **95.80%** (Train: 94.60%).
- **Reproduced Validation Accuracy**: **93.91%** (Train: 94.42%).
- **Reproduced Validation F1-Score**: **93.81%** (Precision: 96.77%, Recall: 91.84%).

---

## 8. Architectural Reconstruction Details

1. **Baseline Model (Table 9)**:
   - Paper states "5-layer baseline CNN", but Table 9 shows:
     - `dense (Dense)`: (None, 16) — 480 params
     - `dropout (Dropout)`: (None, 16) — 0 params
     - `dense_1 (Dense)`: (None, 1) — 17 params
     - **Total & Trainable Parameters**: Exactly **497**.
     - Source identified: Official TensorFlow imbalanced data tutorial MLP.

2. **Sequential 10-Layer CNN (Table 8)**:
   - Conv1D(32, kernel=2) $\rightarrow$ BatchNorm $\rightarrow$ Dropout(0.2) $\rightarrow$ Conv1D(64, kernel=2) $\rightarrow$ BatchNorm $\rightarrow$ Dropout(0.5) $\rightarrow$ Flatten $\rightarrow$ Dense(64) $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(1).
   - **Total Parameters**: Exactly **119,457**. Reconstructed with 100% parameter fidelity.

3. **Convolutional Dimensions**:
   - The paper's shorthand `32 x 2` and `64 x 2` denotes **32 and 64 filters** with a 1D kernel dimension of **2**, matching Table 8 weights $(2 \times 1 \times 32) + 32 = 96$ and $(2 \times 32 \times 64) + 64 = 4,160$.

---

## 9. Citation

If you use this reproduction repository or reference its findings, please cite the original study:

```bibtex
@article{alarfaj2022credit,
  title={Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms},
  author={Alarfaj, Fawaz Khaled and Malik, Iqra and Khan, Hikmat Ullah and Almusallam, Naif and Ramzan, Muhammad and Ahmed, Muzamil},
  journal={IEEE Access},
  volume={10},
  pages={39700--39715},
  year={2022},
  publisher={IEEE},
  doi={10.1109/ACCESS.2022.3166891}
}
```
