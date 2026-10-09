# Methodology Analysis: "Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms"

**Authors**: Fawaz Khaled Alarfaj, Iqra Malik, Hikmat Ullah Khan, Naif Almusallam, Muhammad Ramzan, Muzamil Ahmed  
**Journal**: IEEE Access, Volume 10, 2022, pp. 39700–39715  
**DOI**: [10.1109/ACCESS.2022.3166891](https://doi.org/10.1109/ACCESS.2022.3166891)

---

## 1. Executive Summary & Objective

The primary objective of the paper is to detect fraudulent credit card transactions by conducting an empirical comparative study between:
1. **Traditional machine learning algorithms**: Decision Tree, K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machine (SVM), Random Forest, and Extreme Gradient Boosting (XGBoost), alongside theoretical formulation of Extreme Learning Machine (ELM).
2. **Deep learning algorithms**: A 5-layer baseline neural network (based on TensorFlow/Keras imbalanced classification benchmark) and three proposed Convolutional Neural Network (CNN) architectures (14 layers, 17 layers, and 20 layers), as well as a 10-layer sequential model presented in Table 8.

The study investigates:
* Impact of feature standardization (specifically `StandardScaler` on `Amount`).
* Epoch variations (14, 20, 30, 35, 50, and 100 epochs).
* Depth variations (5 layers baseline, 10 layers, 11 layers, 13 layers, 14 layers, 17 layers, 20 layers).
* Class balancing via random undersampling of legitimate transactions over 100 epochs.

---

## 2. Dataset Description & Characteristics

### 2.1 Benchmark Provenance
* **Dataset**: European Credit Card Fraud Detection benchmark (ULB Machine Learning Group / Andrea Dal Pozzo et al.).
* **Observation Window**: Transactions made by European credit cardholders over a two-day period in September 2013 (the paper text mentions "September 2018" or "October 2018" due to author typo, but the cited dataset characteristics match the standard 2013 Kaggle/ULB dataset).
* **Record Count**: 284,807 transactions.
* **Class Breakdown**:
  * Legitimate (`Class = 0`): 284,315 transactions (99.828%).
  * Fraudulent (`Class = 1`): 492 transactions (0.172%).
* **Missing Values**: 0 null values across all 31 columns.
* **Duplicate Records**: 1,081 identical rows in the raw dataset (preserved as standard).

### 2.2 Feature Inventory
The dataset contains 31 attributes:
1. `Time`: Elapsed seconds between transaction and first transaction in dataset (numeric).
2. `V1` through `V28`: 28 continuous numerical principal components derived via PCA to protect cardholder confidentiality.
3. `Amount`: Transaction amount in Euros (numeric).
4. `Class`: Ground-truth binary target (`0` for legitimate, `1` for fraud).

*(Note: Table 3 in the paper lists generic credit card attributes such as Account Number, Card Number, Merchant Category Code, etc. for background domain context, whereas Table 4 and Section III-A-1-a specify the actual 31-column dataset used).*

---

## 3. Data Preprocessing & Pipeline

### 3.1 Feature Scaling
* As identified in Section IV-A-3 and Figure 7, `Amount` exhibits extreme variance and positive skew compared to the PCA features `V1`–`V28`.
* Preprocessing applies `StandardScaler` to `Amount`.
* To prevent data leakage, scalers must be fit strictly on the training partition and applied to validation/test partitions.
* Two dataset variants are examined:
  1. `StandardScaler` applied to `Amount`, with PCA features preserved as is.
  2. Full standardization of all predictors (`Time`, `Amount`, and `V1`–`V28`).

### 3.2 Train/Test Partitioning
* From Table 6 confusion matrix totals:
  $$\text{TN} + \text{FP} + \text{TP} + \text{FN} = 56,849 + 6 + 83 + 24 = 56,962$$
* $56,962 / 284,807 = 0.200002$ (exactly a 20% test split, 80% train split).
* Test fraud cases: 107 transactions (approx 21.7% of total 492 frauds).
* Stratified 80/20 train/test split maintains approximately 394/98 or 385/107 fraud distribution across splits.

### 3.3 Feature Selection
* In Section I and Section III-B-vi, the authors note:
  > "The top attributes from the prepared dataset are ranked using feature selection techniques."
  > "The parameter 'C' is tuned using RandomizedSearchCV() for the different datasets: the original, the standardized and the dataset with the most important features."
* However, the authors omit the exact algorithm name, threshold, and feature subset list in the publication.
* Reproducible implementation uses Random Forest Gini feature importance and SelectKBest (ANOVA F-statistic) as explicit, documented candidate rankers.

---

## 4. Traditional Machine Learning Models

### 4.1 Decision Tree Classifier
* **Library**: `sklearn.tree.DecisionTreeClassifier`
* **Explicit Hyperparameters**:
  * `max_depth = 4`
  * `criterion = "entropy"`
* **Reported Performance (Table 5)**:
  * Accuracy: 99.93%
  * F1-Score: 81.05%

### 4.2 K-Nearest Neighbors (KNN)
* **Library**: `sklearn.neighbors.KNeighborsClassifier`
* **Explicit Hyperparameters**:
  * `n_neighbors = 5`
* **Reported Performance (Table 5)**:
  * Accuracy: 99.95%
  * F1-Score: 85.71%

### 4.3 Logistic Regression
* **Library**: `sklearn.linear_model.LogisticRegression`
* **Explicit Hyperparameters**:
  * Tuning of inverse regularization parameter `C` using `RandomizedSearchCV`.
* **Reported Performance (Table 5)**:
  * Accuracy: 99.91%
  * F1-Score: 73.56%

### 4.4 Support Vector Machine (SVM)
* **Library**: `sklearn.svm.SVC`
* **Inferred Hyperparameters**:
  * Standard RBF kernel or linear kernel; defaults from scikit-learn (`C=1.0`, `gamma='scale'`).
* **Reported Performance (Table 5)**:
  * Accuracy: 99.93%
  * F1-Score: 77.71%

### 4.5 Random Forest (RF)
* **Library**: `sklearn.ensemble.RandomForestClassifier`
* **Inferred Hyperparameters**:
  * Standard bagging ensemble of decision trees; `n_estimators=100`, `criterion='entropy'` or `'gini'`.
* **Reported Performance (Table 5)**:
  * Accuracy: 99.92%
  * F1-Score: 77.27%

### 4.6 XGBoost (Extreme Gradient Boosting)
* **Library**: `xgboost.XGBClassifier`
* **Explicit Settings**: Serviceable default parameters via scikit-learn API.
* **Reported Performance (Table 5)**:
  * Accuracy: 99.94%
  * F1-Score: 84.49%

### 4.7 Extreme Learning Machine (ELM)
* **Formulation**: Single hidden layer feed-forward network with random hidden weights $W, b$ and closed-form analytic output weights calculation via Moore-Penrose pseudoinverse $\beta = H^\dagger T$.
* Formally detailed in Section III-B-i (Equations 1-2). Not included in Table 5 results.

---

## 5. Baseline Deep Learning Model

The baseline neural network corresponds to the benchmark architecture detailed in Section IV-C4 and Table 9:
* **Architecture**:
  1. Dense layer: 16 units, ReLU activation. With 29-30 input features, $29 \times 16 + 16 = 480$ parameters.
  2. Dropout layer: rate 0.5 (0 parameters).
  3. Dense output layer: 1 unit, Sigmoid activation, $16 \times 1 + 1 = 17$ parameters.
* **Parameter Verification (Table 9)**:
  * Total parameters: 497
  * Trainable parameters: 497
  * Non-trainable parameters: 0
* **Batch Size**: 2048 (designed for imbalanced batches).
* **Careful Bias Initialization**: Initial bias set to $\log(p / (1 - p)) \approx -6.35$ where $p = 492 / 284807$.
* **Epochs**: Evaluated at 14, 20, and 35 epochs.
* **Reported Performance (Table 7 & Table 10)**:
  * 20 Epochs: Training Acc 90.7%, Validation Acc 88.3%.
  * 14 Epochs: Training Acc 99%, Validation Acc 99%, Precision 91%/89%, Recall 80%/68%, AUC 94%/95%.
  * 35 Epochs: Training Acc 98%, Validation Acc 96%, Precision 93%/42%, Recall 90%/85%, AUC 98%/97%.

---

## 6. Proposed CNN Architectures

The tabular input vector (e.g. 29 or 30 features) is reshaped to a 1D sequence tensor of shape `(batch, steps, channels)` = `(batch, 30, 1)` or `(batch, 29, 1)`.

### 6.1 Proposed 14-Layer CNN
* **Structure**:
  1. `Conv1D(filters=32, kernel_size=2, activation='relu')`
  2. `BatchNormalization()`
  3. `Dropout(0.2)`
  4. `Conv1D(filters=64, kernel_size=2, activation='relu')`
  5. `BatchNormalization()`
  6. `Dropout(0.5)`
  7. `Flatten()`
  8. `Dense(units=100, activation='relu')`
  9. `Dropout(0.5)`
  10. `Dense(units=50, activation='relu')`
  11. `Dropout(0.25)` or `Dense(units=25, activation='relu')`
  12. `Dense(units=25, activation='relu')`
  13. `Dropout(0.25)`
  14. `Dense(units=1, activation='sigmoid')`
* **Reported Accuracy (Section IV-C-11-a & Table 7)**:
  * Imbalanced (100 Epochs): Training Acc 96.3% (96.34%), Validation Acc 94.3%.
  * Balanced (100 Epochs): Training Acc 94.60%, Validation Acc 95.80%.

### 6.2 Proposed 17-Layer CNN
* **Structure**:
  Adds an additional convolutional stage `[Conv1D(64, 2), BatchNorm, Dropout(0.25)]` prior to flattening:
  1. `Conv1D(32, 2, activation='relu')`
  2. `BatchNormalization()`
  3. `Dropout(0.2)`
  4. `Conv1D(64, 2, activation='relu')`
  5. `BatchNormalization()`
  6. `Dropout(0.5)`
  7. `Conv1D(64, 2, activation='relu')`
  8. `BatchNormalization()`
  9. `Dropout(0.25)`
  10. `Flatten()`
  11. `Dense(100, activation='relu')`
  12. `Dropout(0.5)`
  13. `Dense(50, activation='relu')`
  14. `Dropout(0.25)`
  15. `Dense(25, activation='relu')`
  16. `Dropout(0.25)`
  17. `Dense(1, activation='sigmoid')`
* **Reported Accuracy (Section IV-C-11-b & Table 7)**:
  * Imbalanced (100 Epochs): Training Acc 95.5% (95.53%), Validation Acc 94.7%.

### 6.3 Proposed 20-Layer CNN
* **Structure**:
  Adds a fourth convolutional stage:
  1. `Conv1D(32, 2, activation='relu')`
  2. `BatchNormalization()`
  3. `Dropout(0.2)`
  4. `Conv1D(64, 2, activation='relu')`
  5. `BatchNormalization()`
  6. `Dropout(0.5)`
  7. `Conv1D(64, 2, activation='relu')`
  8. `BatchNormalization()`
  9. `Dropout(0.5)`
  10. `Conv1D(64, 2, activation='relu')`
  11. `BatchNormalization()`
  12. `Dropout(0.25)`
  13. `Flatten()`
  14. `Dense(100, activation='relu')`
  15. `Dropout(0.5)`
  16. `Dense(50, activation='relu')`
  17. `Dropout(0.25)`
  18. `Dense(25, activation='relu')`
  19. `Dropout(0.25)`
  20. `Dense(1, activation='sigmoid')`
* **Reported Accuracy (Section IV-C-11-c & Table 7)**:
  * Imbalanced (100 Epochs): Training Acc 94.9% (94.92%), Validation Acc 93.5%.

### 6.4 The 10-Layer Sequential CNN (Table 8)
* Exact layer parameter summary from Table 8:
  * `conv1d (Conv1D)`: `(None, 29, 32)`, 96 params
  * `batch_normalization`: `(None, 29, 32)`, 128 params
  * `dropout`: `(None, 29, 32)`, 0 params
  * `conv1d_1 (Conv1D)`: `(None, 28, 64)`, 4160 params
  * `batch_normalization_1`: `(None, 28, 64)`, 256 params
  * `dropout_1`: `(None, 28, 64)`, 0 params
  * `flatten`: `(None, 1792)`, 0 params
  * `dense`: `(None, 64)`, 114752 params
  * `dropout_2`: `(None, 64)`, 0 params
  * `dense_1`: `(None, 1)`, 65 params
  * **Total Parameters**: 119,457
  * **Trainable Parameters**: 119,265
  * **Non-trainable Parameters**: 192
* Used for the 14-epoch and 35-epoch experiments in Table 6.

---

## 7. Training Configurations & Experiments

| Experiment Name | Model | Epochs | Batch Size | Data Variant | Reported Metric Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ML Models Comparison** | DT, KNN, LR, SVM, RF, XGB | N/A | N/A | Scaled Amount | Table 5 Acc & F1 |
| **CNN 10-Layer Epoch 14** | Table 8 CNN | 14 | 2048 / 32 | Imbalanced | Table 6: Acc 99.8%, Prec 56.8%, Rec 81.3% |
| **CNN 10-Layer Epoch 35** | Table 8 CNN | 35 | 2048 / 32 | Imbalanced | Table 6: Acc 99.9%, Prec 93.2%, Rec 77.5% |
| **CNN 11-Layer Epoch 20** | 11-Layer CNN | 20 | 2048 | Imbalanced | Table 7: Train 93.2%, Val 91.3% |
| **CNN 13-Layer Epoch 50** | 13-Layer CNN | 50 | 2048 | Imbalanced | Table 7: Train 91.2%, Val 91.3% |
| **CNN 14-Layer Epoch 100** | Proposed 14-Layer | 100 | 2048 | Imbalanced | Table 7: Train 96.3%, Val 94.3% |
| **CNN 17-Layer Epoch 100** | Proposed 17-Layer | 100 | 2048 | Imbalanced | Table 7: Train 95.5%, Val 94.7% |
| **CNN 20-Layer Epoch 100** | Proposed 20-Layer | 100 | 2048 | Imbalanced | Table 7: Train 94.9%, Val 93.5% |
| **Balanced CNN 14-Layer** | Proposed 14-Layer | 100 | 32 / 64 | Undersampled (1:1) | Table 7: Train 94.60%, Val 95.80% |
| **Baseline NN Epoch 14** | Baseline 497-param | 14 | 2048 | Imbalanced | Table 10: Train 99%, Val 99% |
| **Baseline NN Epoch 20** | Baseline 497-param | 20 | 2048 | Imbalanced | Table 7: Train 90.7%, Val 88.3% |
| **Baseline NN Epoch 35** | Baseline 497-param | 35 | 2048 | Imbalanced | Table 10: Train 98%, Val 96% |

---

## 8. Mathematical Evaluation Equations

As extracted from Section III-D:
1. **Accuracy (Equation 9)**:
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
2. **Precision (Equation 10)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
3. **Recall / Sensitivity (Equation 12)**:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
4. **F1-Score / F-Measure (Equation 11)**:
   $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Categorical Cross-Entropy (Equation 8 as printed)**:
   $$\text{CCE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(y_i) + (1 - y_i) \log(1 - y_i) \right]$$
   *(Note: The paper labels Equation 8 as CCE but writes the binary cross-entropy formulation, matching the binary classification task).*
6. **Receiver Operating Characteristic - Area Under Curve (ROC-AUC)** and **Precision-Recall Area (PRC/AUPRC)**.
