# Results Reference: Published Numerical Benchmark Values

This document compiles all experimental results, metric values, parameter counts, and model comparisons published in:  
**“Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms”** (IEEE Access, 2022).

These values serve as the authoritative ground truth for all reproduction evaluations.

---

## 1. Machine Learning Classifiers (Table 5)

Evaluated on the European cardholder dataset (80/20 train/test split, `Amount` standardized):

| Model Name | Published Accuracy (%) | Published F1-Score (%) | Notes from Text |
| :--- | :--- | :--- | :--- |
| **Decision Tree** | 99.93 | 81.05 | `max_depth = 4`, `criterion = 'entropy'` |
| **K-Nearest Neighbors (KNN)** | 99.95 | 85.71 | `n_neighbors = 5` |
| **Logistic Regression** | 99.91 | 73.56 | `C` tuned via `RandomizedSearchCV` |
| **Support Vector Machine (SVM)** | 99.93 | 77.71 | Scikit-learn defaults |
| **Random Forest** | 99.92 | 77.27 | Bagging ensemble of trees |
| **XGBoost** | 99.94 | 84.49 | Serviceable defaults |
| **Extreme Learning Machine (ELM)** | *Not reported in Table 5* | *Not reported in Table 5* | Formally defined in Section III-B-i |

---

## 2. CNN Model: Epoch 35 vs Epoch 14 (Table 6)

10-Layer Sequential Architecture from Table 8 evaluated on the 56,962 test set:

| Evaluation Metric | Epoch Size 35 | Epoch Size 14 |
| :--- | :--- | :--- |
| **Loss** | 0.004 | 0.014 |
| **True Positives (TP)** | 83.0 | 87 |
| **False Positives (FP)** | 6.0 | 66 |
| **True Negatives (TN)** | 56,849.0 | 56,789 |
| **False Negatives (FN)** | 24.0 | 20 |
| **Accuracy** | 0.999 (99.9%) | 0.998 (99.8%) |
| **Precision** | 0.932 (93.2%) | 0.568 (56.8%) |
| **Recall** | 0.775 (77.5%) | 0.813 (81.3%) |
| **ROC-AUC** | 0.929 | 0.942 |
| **PRC (AUPRC)** | 0.816 | 0.741 |
| **Total Test Fraud Transactions** | 107 | 107 |

---

## 3. Deep Learning Architectures by Depth & Epochs (Table 7)

| Architecture / Experiment | Layer Count | Epoch Count | Training Accuracy (%) | Validation Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| **CNN Imbalanced** | 11 layers | 20 epochs | 93.2 | 91.3 |
| **CNN Imbalanced** | 13 layers | 50 epochs | 91.2 | 91.3 |
| **CNN Imbalanced** | 14 layers | 100 epochs | 96.3 (96.34 in text) | 94.3 |
| **CNN Imbalanced** | 17 layers | 100 epochs | 95.5 (95.53 in text) | 94.7 |
| **CNN Imbalanced** | 20 layers | 100 epochs | 94.9 (94.92 in text) | 93.5 |
| **CNN Balanced Dataset** | 14 layers | 100 epochs | 94.60 | 95.80 |
| **Baseline Architecture** | 5 layers (MLP) | 20 epochs | 90.7 | 88.3 |

---

## 4. CNN Sequential Architecture Layer Breakdown (Table 8)

| Layer Index | Layer Name & Type | Output Shape | Parameter Count |
| :--- | :--- | :--- | :--- |
| 1 | `conv1d (Conv1D)` | `(None, 29, 32)` | 96 |
| 2 | `batch_normalization` | `(None, 29, 32)` | 128 |
| 3 | `dropout` | `(None, 29, 32)` | 0 |
| 4 | `conv1d_1 (Conv1D)` | `(None, 28, 64)` | 4,160 |
| 5 | `batch_normalization_1` | `(None, 28, 64)` | 256 |
| 6 | `dropout_1` | `(None, 28, 64)` | 0 |
| 7 | `flatten (Flatten)` | `(None, 1792)` | 0 |
| 8 | `dense (Dense)` | `(None, 64)` | 114,752 |
| 9 | `dropout_2 (Dropout)` | `(None, 64)` | 0 |
| 10 | `dense_1 (Dense)` | `(None, 1)` | 65 |
| **Total Parameters** | | | **119,457** |
| **Trainable Parameters** | | | **119,265** |
| **Non-Trainable Parameters** | | | **192** |

---

## 5. Baseline Neural Network Layer Breakdown (Table 9)

| Layer Index | Layer Name & Type | Output Shape | Parameter Count |
| :--- | :--- | :--- | :--- |
| 1 | `dense (Dense)` | `(None, 16)` | 480 |
| 2 | `dropout (Dropout)` | `(None, 16)` | 0 |
| 3 | `dense_1 (Dense)` | `(None, 1)` | 17 |
| **Total Parameters** | | | **497** |
| **Trainable Parameters** | | | **497** |
| **Non-Trainable Parameters** | | | **0** |

---

## 6. Baseline Model: Training vs Validation Metrics (Table 10)

| Metric | Epoch 35 Training (%) | Epoch 35 Validation (%) | Epoch 14 Training (%) | Epoch 14 Validation (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | 93 | 42 | 91 | 89 |
| **Recall** | 90 | 85 | 80 | 68 |
| **ROC-AUC** | 98 | 97 | 94 | 95 |
| **PRC (AUPRC)** | 56 | 22 | 84 | 80 |
| **Accuracy** | 98 | 96 | 99 | 99 |

---

## 7. Comparative Performance Across Literature (Table 11)

| Model Name | Existing Literature Acc (%) | Paper Reproduced Acc (%) | Citations in Paper |
| :--- | :--- | :--- | :--- |
| **CNN** | 93.00, 94.6 | 96.34 | [32], [24] |
| **Baseline (BL)** | 83.00 | 99.72 | [22] |
| **Random Forest (RF)** | 97.55, 92.3 | 99.92 | [32], [24] |
| **SVM** | 97.43 | 99.93 | [32] |
| **KNN** | 93.27, 91.11 | 99.91 (99.95) | [33], [34] |
| **Decision Tree (DT)** | 97.08, 66.5 | 99.93 | [35], [24] |
| **Logistic Regression (LR)** | 97.18, 67.8 | 99.91 | [35], [24] |
