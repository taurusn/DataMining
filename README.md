# 📊 Electronics Sales Data Mining Project

## 📘 Description
This project analyzes sales transaction data for an electronics company (Sept 2023 – Sept 2024) to identify customer behavior patterns, product performance, and purchase prediction using advanced data mining techniques.

## 📁 Project Structure
```
DataMining/
├── data/
│   ├── raw/                    # Original CSV files (never modified)
│   ├── interim/                # Intermediate data (after cleaning)
│   └── processed/              # Final datasets ready for modeling
├── notebooks/
│   ├── 01_exploration.ipynb    # Exploratory Data Analysis (EDA)
│   ├── 02_preprocessing.ipynb  # Data cleaning, encoding, transformation
│   ├── 03_modeling.ipynb       # Training ML models
│   ├── 04_evaluation.ipynb     # Model evaluation, metrics, visualizations
│   └── 05_reporting.ipynb      # Insight summary & plots
├── src/
│   ├── data_loader.py          # Handles reading/writing data files
│   ├── preprocess.py           # Cleans, encodes, and scales data
│   ├── feature_engineering.py  # Creates new derived features
│   ├── modeling.py             # Model training and saving
│   ├── evaluation.py           # Accuracy, precision, recall, etc.
│   └── utils.py                # Helper functions (logging, paths, etc.)
├── outputs/
│   ├── figures/                # Visualizations (histograms, heatmaps, etc.)
│   ├── models/                 # Saved ML models (.pkl files)
│   ├── metrics/                # Evaluation reports (.csv / .json)
│   └── reports/                # Final PDF or summaries
├── tests/                      # Unit tests for code validation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions
└── README.md                   # This file
```

## 🧠 Project Goals
- Clean and preprocess transactional data for analysis
- Perform comprehensive exploratory data analysis (EDA)
- Analyze patterns in customer demographics, product categories, and purchasing behavior
- Build and compare multiple machine learning models for purchase prediction
- Evaluate model performance using comprehensive metrics
- Generate actionable business insights and visualizations

## 🎯 Machine Learning Objectives
- **Classification Task**: Predict customer purchase behavior (buy/no-buy)
- **Models to Compare**: Logistic Regression, Decision Tree, Random Forest
- **Key Features**: Product price, quantity, category, customer demographics, temporal patterns
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/taurusn/DataMining.git
cd DataMining
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebooks
```bash
jupyter notebook
```

### 5. Data Setup
- Place your electronics sales CSV file in `data/raw/`
- Follow the notebooks in order: 01 → 02 → 03 → 04 → 05

## 📊 Dataset Information
- **Source**: Kaggle – Electronic Sales (Sep 2023 – Sep 2024)
- **Size**: ~46,000 sales records (one year of transactions)
- **Features**:
  - **Numeric**: product price, quantity sold, total sales amount
  - **Categorical**: product category, region, sales channel, month
- **Target Variable**: Revenue (True/False → purchase made or not)
- **Link**: [Customer purchase behavior - Electronic Sales Data - Kaggle](https://www.kaggle.com/datasets/cameronseamons/electronic-sales-sep2023-sep2024)

## 🔬 Methodology Overview

### 1. Data Preprocessing (`02_preprocessing.ipynb`)
- Handle missing values and data quality issues
- Encode categorical attributes (product category, region, sales channel)
- Normalize continuous variables (price, quantity, sales amount)
- Feature engineering and transformation

### 2. Exploratory Data Analysis (`01_exploration.ipynb`)
- Statistical descriptions and data profiling
- Visualizations: histograms, boxplots, correlation heatmaps
- Identify patterns and relationships in purchase behavior
- Temporal analysis of sales trends

### 3. Outlier Analysis
- Apply IQR method to detect extreme values in sales data
- Analyze unusual purchasing patterns
- Decision framework for handling outliers

### 4. Class Imbalance Handling
- Address imbalanced purchase/non-purchase classes
- Implement SMOTE (Synthetic Minority Oversampling Technique)
- Compare sampling strategies

### 5. Model Development (`03_modeling.ipynb`)
- **Logistic Regression**: Baseline linear classifier
- **Decision Tree**: Interpretable rule-based model
- **Random Forest**: Ensemble method for improved accuracy
- Hyperparameter tuning using GridSearchCV

### 6. Model Evaluation (`04_evaluation.ipynb`)
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves, feature importance
- Cross-validation and performance comparison
- Model selection and final recommendations

## 🚀 Expected Deliverables
- ✅ Clean, well-structured dataset ready for machine learning
- 📊 Comprehensive EDA report with key insights about customer behavior
- 🤖 Trained ML models with optimized hyperparameters
- 📈 Detailed performance evaluation and model comparison
- 📋 Business recommendations based on data-driven insights
- 🎯 Best-performing model for purchase prediction

## 🛠️ Technologies Used
- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and evaluation
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebooks**: Interactive development environment
- **XGBoost**: Advanced gradient boosting (optional)
- **Imbalanced-learn**: Handling class imbalance

## 📚 References
- [Kaggle Dataset: Customer purchase behavior - Electronic Sales Data](https://www.kaggle.com/datasets/cameronseamons/electronic-sales-sep2023-sep2024)
- Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques*
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Imbalanced-learn (SMOTE)](https://imbalanced-learn.org)
- [Pandas Documentation](https://pandas.pydata.org/)

## 📞 Contact & Contributors
- **Team**: Data Mining Project Team
- **Course**: Data Mining (25.26)
- **Institution**: IAU

---
*This project follows industry best practices for data science workflows and reproducible research.*