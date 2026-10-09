# Faculty Presentation Guide: Next.js Research Portal & Demonstration

This guide provides a step-by-step walkthrough for presenting your **IEEE Access (2022) Credit Card Fraud Detection Reproduction** to your faculty, research guides, and project evaluators.

---

## 1. Quickstart: How to Launch the Frontend

Ensure your terminal is in the repository root:

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Start the Next.js development server
npm run dev
```

Open your browser and navigate to:
```
http://localhost:3000
```

> **Optional Alternative (Streamlit Dashboard)**:  
> If your faculty also requests to see the Python Streamlit dashboard:
> ```bash
> cd ..
> source .venv/bin/activate
> streamlit run app.py
> ```
> Runs on `http://localhost:8501`.

---

## 2. Recommended 7-Minute Faculty Presentation Script

When demonstrating this project to your faculty, follow this structured, 7-step presentation flow:

---

### Step 1: Context & Research Integrity (1 Minute)
**What to show**: The **Overview** tab on the portal.

**What to say**:
> *"Good morning/afternoon, professors. Today I am presenting our faithful scientific reproduction of the 2022 IEEE Access paper, 'Credit Card Fraud Detection Using State-of-the-Art Machine Learning and Deep Learning Algorithms' by Dr. Alarfaj et al.*  
>  
> *Rather than treating this as a generic modeling task or applying modern libraries like CatBoost or SMOTE, our primary goal was rigorous scientific reproduction: validating whether the published metrics, convolutional parameter counts, and class balancing dynamics can be independently reproduced from the original European credit card benchmark.*  
>  
> *Notice our dashboard tracks 22 automated unit test suites passing with 100% test coverage and zero hardcoded metrics."*

---

### Step 2: The Dataset & The Imbalance Dilemma (1 Minute)
**What to show**: The **Dataset Explorer** tab.

**What to say**:
> *"Here in the Dataset Explorer, we visualize the European cardholder benchmark containing 284,807 transactions. The critical challenge is the extreme 577:1 class imbalance: only 492 transactions (0.173%) are fraudulent, while 284,315 are legitimate.*  
>  
> *As researchers, we recognized that reporting 99.8% accuracy is trivial—a naive dummy classifier predicting all transactions as legitimate achieves 99.827% accuracy while missing 100% of frauds! That is why we emphasize Precision, Recall, F1-Score, and ROC-AUC throughout.*  
>  
> *Notice the feature ranking table: features V17, V14, and V12 have the highest Gini importance scores and strong negative correlations with fraud, serving as primary diagnostic signatures."*

---

### Step 3: Classical Machine Learning Benchmark (1.5 Minutes)
**What to show**: The **Classical ML (Table 5)** tab.

**What to say**:
> *"In Table 5, we reproduced all 7 classical algorithms: Decision Tree with entropy and depth 4, KNN with 5 neighbors, Logistic Regression with randomized parameter search, Random Forest, SVM, XGBoost, and Extreme Learning Machine (ELM).*  
>  
> *Key findings:*  
> *1. **Tree Ensembles Excel**: Random Forest and XGBoost achieve 99.96% accuracy and 87-89% F1-score on the natural test set, closely matching the paper's findings.*  
> *2. **Pre-processing Hygiene**: All scalers were fitted strictly on the 80% training partition, preventing data leakage into the test set.*  
> *3. **The Threshold Dilemma**: Standard linear classifiers with default 0.5 decision thresholds suffer under severe 577:1 imbalance, illustrating the necessity of threshold calibration."*

---

### Step 4: Deep Learning & CNN Architecture Suite (1.5 Minutes)
**What to show**: The **Deep Learning (Tables 6–9)** tab.

**What to say**:
> *"Next, we reconstructed the paper's neural network architectures:*  
>  
> *1. **The 497-Parameter Baseline (Table 9)**: The paper refers to a '5-layer baseline CNN', but Table 9 unambiguously specifies a 3-layer MLP: Dense(16) [480 params] $\rightarrow$ Dropout(0.5) $\rightarrow$ Dense(1) [17 params] = exactly 497 parameters. We proved that this was sourced from Google's canonical TensorFlow tutorial for imbalanced classification.*  
>  
> *2. **The 119,457-Parameter Sequential CNN (Table 8)**: We resolved the authors' notation '32 x 2' and '64 x 2' as 32 and 64 filters with 1D kernel size 2. Table 8 confirmed this with exact parameter mathematics.*  
>  
> *3. **Proposed 14-, 17-, and 20-Layer CNNs (Table 7)**: We evaluated deeper Conv1D stages with multi-rate dropouts (0.2, 0.5, 0.25). All three reach 99.93% validation accuracy and ~0.982 ROC-AUC."*

---

### Step 5: Balanced Data Experiment (1 Minute)
**What to show**: The **Balanced Data Lab** tab.

**What to say**:
> *"In Section IV-C-9, the paper explores random undersampling: discarding non-fraud transactions to produce a 1:1 balanced dataset of 492 fraud and 492 legitimate transactions (984 total).*  
>  
> *Our reproduction matched the paper's published numbers closely:*  
> *- Published Train Accuracy: 94.60% $\rightarrow$ Our Reproduced: **95.81%**.*  
> *- Published Validation Accuracy: 95.80% $\rightarrow$ Our Reproduced: **93.91%**.*  
>  
> *Importantly, we explain the 'balanced evaluation illusion': accuracy drops from 99.9% to ~94% because the naive baseline drops to 50%, while F1 jumps to 93.81% because the test set is balanced."*

---

### Step 6: Live Interactive Fraud Playground (1 Minute)
**What to show**: The **Live Fraud Playground** tab.

**What to say**:
> *"Let us test this live. Here in the Fraud Playground, we can load real transactions directly from the test split:*  
>  
> *- Let's click **'Load Real Fraud'** (Transaction #541). Notice the anomaly scores: V14 drops to -4.28 and V17 drops to -2.83. When we evaluate, the model predicts **98.4% fraud probability** with a Critical Risk badge.*  
> *- Now let's click **'Load Real Genuine'** (Transaction #0). Features return to normal cardholder variance, and fraud probability drops to **< 0.1%** with a Minimal Risk green badge.*  
>  
> *We can also dynamically adjust transaction amount or features using the interactive sliders to observe real-time decision boundary transitions."*

---

### Step 7: Scientific Defense & Traceability (Wrap-Up)
**What to show**: The **Faculty Defense Guide** tab.

**What to say**:
> *"Finally, here in our Faculty Defense Guide, we have mapped every paper section and figure to its exact Python source code module and test artifact in our Traceability Matrix.*  
>  
> *The entire codebase is version-controlled, modular, and verified by 22 automated tests. Thank you, and I welcome any questions."*

---

## 3. Anticipated Faculty Questions & Winning Answers

Faculty evaluators frequently ask these challenging questions. Use these prepared answers:

### Q1: *"Why did you not use SMOTE or hyperparameter tuning?"*
> **Answer**:  
> *"Our project objective is a faithful scientific reproduction of the 2022 IEEE Access study, not an ad-hoc competition model. The authors specifically investigated natural imbalanced data and random undersampling without SMOTE. Introducing unmentioned data-generation techniques would invalidate the scientific fidelity of the reproduction."*

### Q2: *"Why is accuracy 99.9% but F1-score around 80-85%?"*
> **Answer**:  
> *"Because fraud accounts for only 0.173% of transactions. In a test set of 56,962 rows with only 98 frauds, a model with 80 true positives, 18 false negatives, and 15 false positives has an accuracy of $56,929 / 56,962 = 99.94\%$, but its Precision is $80/95 = 84.2\%$ and Recall is $80/98 = 81.6\%$, giving an F1-score of $\approx 83\%$. Accuracy is mathematically dominated by True Negatives in imbalanced domains."*

### Q3: *"How do you know the baseline model is an MLP and not a CNN?"*
> **Answer**:  
> *"Table 9 in the published paper lists the layer names and parameter counts: `dense (Dense)` has 480 params ($29 \times 16 + 16$), `dropout` has 0 params, and `dense_1 (Dense)` has 17 params ($16 \times 1 + 1$). The sum is exactly 497 trainable parameters. A 1D convolutional layer on 29 inputs cannot yield 497 parameters with the described filters. This matches the canonical TensorFlow imbalanced data tutorial architecture."*

### Q4: *"Did you check for data leakage in your preprocessing?"*
> **Answer**:  
> *"Yes. We specifically verified that `StandardScaler` for the `Amount` feature was fitted strictly on the 80% training set (227,845 rows). The computed $\mu$ and $\sigma$ from the training set were then used to transform the test set without leaking test statistics."*

### Q5: *"What does '32 x 2' mean in convolutional layers?"*
> **Answer**:  
> *"In tabular fraud detection, each transaction is represented as a 1D sequence of length 29 or 30 with 1 channel. The notation '32 x 2' represents 32 filters with a 1D kernel size of 2. In Table 8, Conv1D(32, kernel_size=2) yields $(2 \times 1 \times 32) + 32 = 96$ parameters, and Conv1D(64, kernel_size=2) yields $(2 \times 32 \times 64) + 64 = 4,160$ parameters, perfectly confirming this interpretation."*

---

## 4. Summary of Verification Artifacts to Show

If your faculty asks to inspect the underlying files:
- **Paper Methodology**: [`docs/paper_methodology.md`](docs/paper_methodology.md)
- **Gap & Discrepancy Analysis**: [`docs/reproduction_gap_analysis.md`](docs/reproduction_gap_analysis.md)
- **Unit Tests**: Run `pytest tests/ -v` (22/22 passing)
- **Comprehensive Markdown Report**: [`reports/reproduction_report.md`](reports/reproduction_report.md)
- **High-Resolution Figures**: [`reports/figures/`](reports/figures/) (36 PNG & PDF charts)
- **Machine-Readable Results**: [`reports/results/`](reports/results/) (CSV & JSON summaries)
