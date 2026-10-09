# Reproduction Gap Analysis

**Target Paper**: “Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms” (IEEE Access, 2022)  
**Authors**: Fawaz Khaled Alarfaj et al.  
**DOI**: 10.1109/ACCESS.2022.3166891

---

## 1. Overview & Purpose

Faithful scientific reproduction requires rigorous transparency: identifying where published descriptions are exact, where they are ambiguous, where typographical errors exist in the source, and what explicit engineering assumptions were required to construct an executable, auditable research codebase.

This document systematically analyzes every reproduction gap, documents the available interpretations, and records the resolved implementation decisions.

---

## 2. Systematic Gap Catalog

### Gap 1: Dataset Provenance & Date Discrepancy
* **Published Statement**: Section III-A-1-a states: *"The dataset holds transactions made by a cardholder over a two-day period, i.e., September 2018. There were 284,807 transactions in total, of which 492, or 0.172 percent, were fraudulent."* Section IV-A states: *"The dataset covers credit cards transactions in October 2018 by European cardholders."*
* **Ground Truth**: The European credit card fraud benchmark was collected by the ULB (Université Libre de Bruxelles) Machine Learning Group in **September 2013**, containing exactly 284,807 transactions and 492 frauds.
* **Analysis**: The authors cite reference [11] (Benchaji et al., 2021) and describe the exact 284,807 / 492 row counts. The mentions of "September 2018" and "October 2018" are internal typographical inconsistencies.
* **Resolution**: The standard public ULB `creditcard.csv` benchmark is preserved without altering metadata. The date discrepancies are documented.

---

### Gap 2: Undisclosed Feature Selection Methodology
* **Published Statement**:
  * Abstract: *"The top attributes from the prepared dataset are ranked using feature selection techniques."*
  * Contributions: *"Feature selection algorithms are used to rank the top features from the CCF transaction dataset..."*
  * Section III-B-vi: *"The parameter 'C' is tuned using RandomizedSearchCV() for the different datasets: the original, the standardized and the dataset with the most important features."*
* **Ambiguity**: The authors do not specify:
  1. The feature ranking algorithm (e.g., Random Forest MDI, Mutual Information, ANOVA F-test, or Chi-Square).
  2. The ranking threshold or the exact list of selected features.
  3. The quantitative classification results on this "selected features" dataset.
* **Resolution**:
  * Implement two reproducible, transparent ranking methods:
    1. Random Forest Mean Decrease in Impurity (Gini importance).
    2. ANOVA F-value feature scoring (`SelectKBest`).
  * Support configurable top-K selection (e.g. top 10, top 15, top 20).
  * Explicitly label this in all experiment configs as a reproduction fallback, never as recovered author settings.

---

### Gap 3: Baseline Architecture & Parameter Discrepancy
* **Published Statement**:
  * Section III-C-i & Section IV-C-4 describe a "baseline CNN sequential model" of "5 layers".
  * Section IV-C-4 & Table 9 state: Total params = 497, Trainable params = 497, Non-trainable params = 0.
* **Ambiguity**: Table 9 lists:
  * `dense (Dense)`: Output Shape `(None, 16)`, Param # 480
  * `dropout (Dropout)`: Output Shape `(None, 16)`, Param # 0
  * `dense_1 (Dense)`: Output Shape `(None, 1)`, Param # 17
* **Analysis**:
  * $29 \text{ features} \times 16 \text{ units} + 16 \text{ biases} = 480$.
  * $16 \text{ units} \times 1 \text{ output} + 1 \text{ bias} = 17$.
  * $480 + 17 = 497$.
  * The model is an MLP (Fully Connected network), not a convolutional network.
  * The source text and parameters closely correspond to TensorFlow's official tutorial: *"Classification on imbalanced data"*, which uses `Dense(16) -> Dropout(0.5) -> Dense(1)` with `batch_size=2048` and output bias initialization $\log(\text{pos}/\text{neg})$.
  * Counting InputLayer + Dense + ReLU + Dropout + OutputDense yields the author's informal "5 layers".
* **Resolution**: Reconstruct the baseline model with `Dense(16) -> Dropout(0.5) -> Dense(1)`, matching the exact 497 parameter count.

---

### Gap 4: Convolutional Kernel Size Nomenclature ("32 × 2" and "64 × 2")
* **Published Statement**:
  * Section IV-C-11-a: *"a convolutional layer with a kernel size of 32 × 2 and a ReLU activation function... another convolutional layer with a kernel size of 64 × 2... a flattened layer with a kernel size of 64 × 2..."*
* **Ambiguity**: In 1D convolutions on tabular data, 2D kernels like $32 \times 2$ are invalid. Furthermore, Flatten layers do not take kernels.
* **Resolution & Verification from Table 8**:
  * Table 8 shows `conv1d (Conv1D)` has Output Shape `(None, 29, 32)` with 96 parameters.
  * Calculation: Input shape `(30, 1)` with kernel size 2 and 32 filters:
    $$\text{Weights} = (2 \times 1 \times 32) + 32 = 64 + 32 = 96 \text{ params!}$$
    $$\text{Output length} = 30 - 2 + 1 = 29 \text{ steps!}$$
  * Layer `conv1d_1 (Conv1D)` has Output Shape `(None, 28, 64)` with 4160 parameters:
    $$\text{Weights} = (2 \times 32 \times 64) + 64 = 4096 + 64 = 4160 \text{ params!}$$
    $$\text{Output length} = 29 - 2 + 1 = 28 \text{ steps!}$$
  * Conclusion: In the author's terminology, "32 × 2" meant **32 filters with 1D kernel size 2**, and "64 × 2" meant **64 filters with 1D kernel size 2**. The author's text description conflated filter count with kernel dimension.

---

### Gap 5: Loss Function Formulation & Binary Classification Target
* **Published Statement**:
  * Section III-C-iii presents Equation 8 titled "Categorical Cross-Entropy":
    $$\text{CCE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(y_i) + (1 - y_i) \log(1 - y_i) \right]$$
  * The final layer uses a sigmoid activation function (`Dense(1, activation='sigmoid')`).
* **Analysis**:
  * Equation 8 is the mathematical formula for **Binary Cross-Entropy (BCE)**, despite being labelled CCE in the section heading.
  * A single sigmoid output neuron requires `binary_crossentropy` in Keras.
* **Resolution**: Use `loss='binary_crossentropy'` in Keras compilation.

---

### Gap 6: Pedagogical Paragraph on 20 Samples and Batch Size 2
* **Published Statement**:
  * Section III-C-iii states: *"We used a dataset of 20 samples, a batch size of 2 and determined that the algorithm needed to run for three epochs. Consequently, in all epochs, we use five batches (20/2=10) [sic]. All batches are run through the algorithm; then, we have five iterations per epoch."*
* **Analysis**:
  * This paragraph contains an arithmetic error ($20 / 2 = 10$, but then says five batches/iterations).
  * This is an illustrative introductory explanation of the concept of batches and epochs, not the experimental setup for the 284,807 transaction benchmark.
* **Resolution**: The full benchmark uses the complete dataset (284,807 transactions) with `batch_size=2048` or `batch_size=32/64` as specified in the experiment configurations.

---

### Gap 7: Train/Test Split Sequence & Fraud Case Distribution
* **Published Data**:
  * Table 6 gives test confusion matrix totals: $\text{TN} = 56,849$, $\text{FP} = 6$, $\text{FN} = 24$, $\text{TP} = 83$ (Total = 56,962).
  * Total test fraud cases = $83 + 24 = 107$.
* **Analysis**:
  * $56,962$ is exactly $20.0002\%$ of 284,807 transactions.
  * The paper used an **80/20 train/test split**.
  * 107 frauds in the test set corresponds to an unstratified split or a split with specific random state where 107 frauds fell in the test partition.
* **Resolution**: Provide configurable random seed and support both stratified split ($98$ test frauds) and exact test split matching $107$ frauds.

---

### Gap 8: Class Balancing Methodology (Leakage Risk)
* **Published Statement**:
  * Section IV-C-9 states: *"The imbalanced CCF dataset is transformed into a balanced dataset by removing non fraudulent transactions from the dataset... The model is trained over 100 epochs. The CNN 14 layers architecture obtained 94.60 and 95.80 % training and validation accuracy respectively..."*
* **Ambiguity**: Does the undersampling occur before train/test splitting or only on the training set?
* **Analysis**:
  * If undersampling is applied across the entire dataset before splitting, the test set becomes balanced (e.g. 492 fraud vs 492 legitimate). High accuracy (e.g. 95.8%) on a 50/50 test set corresponds to a real-world error rate that would yield thousands of false alarms on an imbalanced test set.
  * If undersampling is applied only to the training set, the test set remains naturally imbalanced.
* **Resolution**:
  * Primary variant: Leakage-safe undersampling applied strictly to the training fold.
  * Secondary variant: Pre-split balancing to reproduce the exact author scenario resulting in Table 7's 95.80% validation accuracy.
  * Clearly label the differences in the reporting artifacts.

---

### Gap 9: Headline Metric Synthesis Across Disjoint Experiments
* **Published Statement**:
  * Abstract cites: 99.9% accuracy, 85.71% F1-score, 93% precision, 98% AUC.
  * Conclusion cites: 99.72% accuracy for "the CNN with 20 layers and the baseline model".
* **Analysis**:
  * 85.71% F1-score is from **KNN** in Table 5.
  * 99.9% accuracy is reported for multiple models in Table 5 and Table 6.
  * 93% precision and 98% AUC are from **Baseline Model Epoch 35** in Table 10.
  * 99.72% accuracy is from **Baseline Model** in Table 11.
  * The headline numbers do NOT come from a single model, but are cherry-picked optimal values across disparate models and tables.
* **Resolution**: Document each metric's true origin in `docs/results_reference.md` and `reports/reproduction_report.md`.
