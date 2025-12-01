# Data Mining Project: Notebook Summary and Report Structure

---

## NOTEBOOK SUMMARIES

---

### PHASE 1: Dataset Overview
**File:** `phase1_dataset_overview.ipynb`

**Purpose:** Initial exploration of the raw dataset to understand its structure, size, and basic characteristics.

**Input:**
- `Electronic_sales_Sep2023-Sep2024.csv` (raw dataset)

**Output:**
- No files saved (exploration only)

**Key Tasks:**
- Load and verify dataset
- Check dataset dimensions (20,000 rows x 16 columns)
- Identify column names and data types
- Check for missing values
- Preview sample records

**Key Findings for Report:**
- Dataset contains 20,000 e-commerce transactions
- 16 features: Customer ID, Age, Gender, Loyalty Member, Product Type, SKU, Rating, Order Status, Payment Method, Total Price, Unit Price, Quantity, Purchase Date, Shipping Type, Add-ons Purchased, Add-on Total
- Missing values: Gender (1), Add-ons Purchased (4,868)
- Date range: September 2023 - September 2024

---

### PHASE 2: Descriptive Statistics
**File:** `phase2_descriptive_statistics_v2.ipynb`

**Purpose:** Statistical analysis of all features and critical TARGET VARIABLE ANALYSIS to understand feature-target relationships.

**Input:**
- `Electronic_sales_Sep2023-Sep2024.csv`

**Outputs:**

| Type | Files |
|------|-------|
| CSV | phase2_summary.csv, chi_square_results.csv, ttest_results.csv, target_correlations.csv |
| PNG | numerical_distributions.png, numerical_boxplots.png, categorical_distributions.png, correlation_matrix.png, target_distribution.png, target_vs_categorical.png, target_vs_numerical.png, target_correlations.png |

**Key Tasks:**
- Compute descriptive statistics (mean, median, std, min, max)
- Analyze distribution shapes (skewness, kurtosis)
- Visualize numerical and categorical distributions
- Detect outliers using IQR method
- Correlation analysis between features
- TARGET VS FEATURES ANALYSIS (Chi-square, T-tests, correlations)

**Key Findings for Report:**
- Target distribution: 67.2% Completed, 32.8% Cancelled
- Outliers found in Total Price (1.9%) and Add-on Total (1.2%)
- Outliers verified as VALID bulk purchases (not errors)
- CRITICAL: No significant relationship between target and ANY feature
- Chi-square tests: 4/5 features not significant
- T-tests: 0/6 numerical features significant
- All correlations < 0.02 (negligible)

---

### PHASE 3: Data Preprocessing
**File:** `phase3_data_preprocessing.ipynb`

**Purpose:** Clean and transform data for machine learning models.

**Input:**
- `Electronic_sales_Sep2023-Sep2024.csv`

**Outputs:**

| Type | Files |
|------|-------|
| CSV | preprocessed_dataset.csv (20,000 records), data_train.csv (16,000), data_test.csv (4,000), feature_list.csv (29 features), scaler_params.csv |

**Key Tasks (8 Steps):**
1. Step 3.1: Handle missing values (Gender: mode imputation, Add-ons: "None")
2. Step 3.2: Fix inconsistencies (Paypal -> PayPal)
3. Step 3.3: Feature engineering (6 new features: Purchase_Month, Purchase_DayOfWeek, Is_Weekend, Has_Addon, Addon_Count, Is_Repeat_Customer)
4. Step 3.4: Handle outliers (capping at 1st-99th percentile for model stability)
5. Step 3.5: Encode categorical variables (binary + one-hot encoding)
6. Step 3.6: Feature scaling (StandardScaler on 9 continuous features)
7. Step 3.7: Class imbalance decision (use class_weight='balanced')
8. Step 3.8: Train-test split (80/20 stratified)

**Key Findings for Report:**
- Final feature count: 29 (from 16 original)
- Outlier treatment: Capping applied for model stability (+7.82% F1 improvement for Decision Tree)
- Business justification documented: outliers are valid bulk purchases

---

### PHASE 4: Feature Selection
**File:** `phase4_feature_selection.ipynb`

**Purpose:** Identify the most important features for prediction using multiple methods.

**Input:**
- `Electronic_sales_Sep2023-Sep2024.csv`

**Outputs:**

| Type | Files |
|------|-------|
| CSV | feature_correlations.csv, chi_square_results.csv, feature_importance.csv, selected_features_all.csv, selected_features_top15.csv, selected_features_top10.csv |
| PNG | feature_correlations_target.png, chi_square_pvalues.png, feature_importance_rf.png, cumulative_importance.png |

**Key Tasks:**
1. Correlation analysis with target
2. Chi-square tests for categorical features
3. Random Forest feature importance
4. Create feature sets: All 29, Top 15, Top 10

**Key Findings for Report:**
- Top 5 features by RF importance: Age (14.6%), Add-on Total (12.7%), Purchase_Month (10.1%), Total Price (9.9%), Quantity (7.6%)
- All correlations with target < 0.02 (negligible)
- Chi-square tests confirm no significant relationships
- Decision: Use all 29 features + create Top 15 and Top 10 subsets for experiments

---

### PHASE 5: Model Building
**File:** `phase5_model_building.ipynb`

**Purpose:** Build, evaluate, and compare classification models.

**Input:**
- `Electronic_sales_Sep2023-Sep2024.csv`

**Outputs:**

| Type | Files |
|------|-------|
| CSV | experiment_results.csv (27 experiments), best_model_summary.csv, final_model_predictions.csv |
| PNG | performance_by_scaler.png, performance_by_features.png, performance_by_model.png, f1_score_heatmap.png, confusion_matrix_best_model.png, roc_curve_best_model.png, feature_importance_best_model.png |

**Key Tasks:**
1. Design 27 experiments (3 scalers x 3 feature sets x 3 models)
2. Run 5-fold stratified cross-validation
3. Compare results by scaler, feature set, and model
4. Evaluate best model on test set
5. Generate confusion matrix, ROC curve, feature importance

**Experiment Configuration:**
- Scalers: StandardScaler, MinMaxScaler, None
- Feature Sets: All 29, Top 15, Top 10
- Models: Logistic Regression, Decision Tree, Random Forest

**Key Findings for Report:**
- Best configuration: MinMaxScaler + Top 15 + Random Forest
- Best F1-Score: 74.40%
- Best Accuracy: 61.42%
- ROC-AUC: ~49-50% (near random)
- CRITICAL: ROC-AUC ~50% confirms target cannot be predicted from features

---

## FINAL REPORT STRUCTURE

---

### Title Page
- Project Title: "Online Shopping Behavior Prediction Using Data Mining Techniques"
- Course Name
- Team Members and Contributions Table
- Date

---

### Table of Contents

---

### 1. Introduction (1 page)
- Background on e-commerce order prediction
- Project objectives
- Dataset description (source, size, timeframe)

---

### 2. Dataset Overview (1-2 pages)
**Use outputs from Phase 1:**
- Dataset dimensions: 20,000 records, 16 features
- Feature descriptions table
- Data types summary
- Missing values summary

---

### 3. Descriptive Statistics and EDA (3-4 pages)
**Use outputs from Phase 2:**

3.1 Numerical Features Analysis
- Include: `numerical_distributions.png`
- Include: `numerical_boxplots.png`
- Statistics table (mean, median, std, min, max)
- Skewness and kurtosis interpretation

3.2 Categorical Features Analysis
- Include: `categorical_distributions.png`
- Frequency tables

3.3 Correlation Analysis
- Include: `correlation_matrix.png`
- Key correlations: Total Price vs Unit Price (0.67), Total Price vs Quantity (0.65)

3.4 Target Variable Analysis (CRITICAL SECTION)
- Include: `target_distribution.png`
- Include: `target_vs_categorical.png`
- Include: `target_vs_numerical.png`
- Include: `target_correlations.png`
- Chi-square test results table
- T-test results table
- Finding: No significant feature-target relationships

---

### 4. Outlier Analysis (1-2 pages)
**Use outputs from Phase 2 and Phase 3:**

4.1 Outlier Detection
- IQR method results
- Outliers identified: Total Price (1.9%), Add-on Total (1.2%)

4.2 Noise vs Valid Outlier Investigation
- Formula verification: Total Price = Unit Price x Quantity
- Business justification: bulk purchases, not errors
- Decision: Cap for model stability, not error correction
- Impact: +7.82% F1 improvement for Decision Tree

---

### 5. Data Preprocessing (2-3 pages)
**Use outputs from Phase 3:**

5.1 Missing Value Treatment
- Gender: mode imputation (1 record)
- Add-ons Purchased: replaced with "None" (4,868 records)

5.2 Data Inconsistencies
- Payment Method: "Paypal" -> "PayPal"

5.3 Feature Engineering
- 6 new features created:
  - Purchase_Month, Purchase_DayOfWeek, Is_Weekend
  - Has_Addon, Addon_Count, Is_Repeat_Customer

5.4 Encoding
- Binary encoding: Gender, Loyalty Member, Order Status
- One-hot encoding: Product Type, Payment Method, Shipping Type
- Final feature count: 29

5.5 Feature Scaling
- StandardScaler applied to 9 continuous features

5.6 Train-Test Split
- 80% training (16,000), 20% testing (4,000)
- Stratified to maintain class distribution

---

### 6. Feature Selection (2 pages)
**Use outputs from Phase 4:**

6.1 Correlation Analysis
- Include: `feature_correlations_target.png`
- All correlations < 0.02

6.2 Chi-Square Tests
- Include: `chi_square_pvalues.png`
- Results: 4/5 features not significant

6.3 Random Forest Feature Importance
- Include: `feature_importance_rf.png`
- Include: `cumulative_importance.png`
- Top 10 features table

6.4 Feature Set Selection
- All 29 features
- Top 15 features (54.8% cumulative importance)
- Top 10 features (75.8% cumulative importance)

---

### 7. Model Building and Evaluation (3-4 pages)
**Use outputs from Phase 5:**

7.1 Experimental Design
- 27 experiments: 3 scalers x 3 feature sets x 3 models
- 5-fold stratified cross-validation
- Metrics: Accuracy, F1-Score, ROC-AUC

7.2 Results by Scaler
- Include: `performance_by_scaler.png`
- Finding: Minimal difference between scalers

7.3 Results by Feature Set
- Include: `performance_by_features.png`
- Finding: Top 15 slightly better than All 29

7.4 Results by Model
- Include: `performance_by_model.png`
- Finding: Random Forest significantly better

7.5 All Experiments Summary
- Include: `f1_score_heatmap.png`
- Full results table (27 experiments)

7.6 Best Model Evaluation
- Configuration: MinMaxScaler + Top 15 + Random Forest
- Include: `confusion_matrix_best_model.png`
- Include: `roc_curve_best_model.png`
- Include: `feature_importance_best_model.png`

Performance metrics:
| Metric | Value |
|--------|-------|
| Accuracy | 61.65% |
| Precision | 66.91% |
| Recall | 84.85% |
| F1-Score | 74.82% |
| ROC-AUC | 48.76% |

---

### 8. Discussion and Findings (1-2 pages)

8.1 Model Performance Interpretation
- ROC-AUC ~50% indicates near-random classification
- High F1 (74%) driven by class imbalance, not learning
- Model predicts "Completed" too often

8.2 Critical Finding
- Target variable shows no relationship with available features
- All statistical tests confirm this (chi-square, t-tests, correlations)
- Order Status appears randomly distributed or depends on external factors

8.3 Possible Explanations
- Cancellations depend on factors not in dataset:
  - Inventory availability
  - Payment processing issues
  - Customer service interactions
  - Delivery problems
  - Fraud detection

8.4 Business Implications
- Current features insufficient for cancellation prediction
- Need additional data: customer service logs, payment status, inventory data

---

### 9. Conclusion (1 page)
- Summary of methodology
- Key findings
- Limitations
- Recommendations for future work

---

### 10. References
- Python libraries used
- Dataset source
- Relevant papers/resources

---

### Appendix
- Full experiment results table
- Code snippets (key sections)
- Additional figures

---

## FIGURES CHECKLIST FOR REPORT

| Phase | Figure | Report Section |
|-------|--------|----------------|
| 2 | numerical_distributions.png | 3.1 |
| 2 | numerical_boxplots.png | 3.1 |
| 2 | categorical_distributions.png | 3.2 |
| 2 | correlation_matrix.png | 3.3 |
| 2 | target_distribution.png | 3.4 |
| 2 | target_vs_categorical.png | 3.4 |
| 2 | target_vs_numerical.png | 3.4 |
| 2 | target_correlations.png | 3.4 |
| 4 | feature_correlations_target.png | 6.1 |
| 4 | chi_square_pvalues.png | 6.2 |
| 4 | feature_importance_rf.png | 6.3 |
| 4 | cumulative_importance.png | 6.3 |
| 5 | performance_by_scaler.png | 7.2 |
| 5 | performance_by_features.png | 7.3 |
| 5 | performance_by_model.png | 7.4 |
| 5 | f1_score_heatmap.png | 7.5 |
| 5 | confusion_matrix_best_model.png | 7.6 |
| 5 | roc_curve_best_model.png | 7.6 |
| 5 | feature_importance_best_model.png | 7.6 |

**Total: 19 figures**

---

## CSV FILES FOR TABLES

| Phase | File | Use in Report |
|-------|------|---------------|
| 2 | phase2_summary.csv | Section 3 statistics |
| 2 | chi_square_results.csv | Section 3.4 |
| 2 | ttest_results.csv | Section 3.4 |
| 2 | target_correlations.csv | Section 3.4 |
| 4 | feature_importance.csv | Section 6.3 |
| 5 | experiment_results.csv | Section 7.5 |
| 5 | best_model_summary.csv | Section 7.6 |

