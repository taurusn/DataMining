# Online Shopping Behavior Prediction using Data Mining Techniques

## Table Of Contents
- [Introduction](#1-introduction)
- [Problem Statement](#2-problem-statement)
- [Dataset](#3-dataset)
- [Methodology](#4-methodology)
  - [Data Preprocessing](#41-data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#42-exploratory-data-analysis-eda)
  - [Outlier Analysis](#43-outlier-analysis)
  - [Handling Imbalance](#44-handling-imbalance)
  - [Model Development](#45-model-development)
  - [Model Evaluation](#46-model-evaluation)
- [Expected Outcomes](#5-expected-outcomes)
- [References](#6-references)

## 1. Introduction
Online shopping continues to grow, with massive amounts of transaction data generated daily. E-commerce platforms want to know which customers are likely to complete a purchase and which ones may abandon their carts. Predicting this behavior using recent sales data can help companies improve marketing strategies, manage inventory, and increase sales.

## 2. Problem Statement
The goal is to build a model that predicts whether a customer will purchase a product or not. By analyzing transactional and behavioral features (such as product category, quantity, price, and date of purchase), the project will identify the most important factors influencing buying decisions and select the best classification approach.

## 3. Dataset
- **Source**: Kaggle – Electronic Sales (Sep 2023 – Sep 2024)
- **Size**: ~46,000 sales records (one year of transactions)
- **Features**:
  - **Numeric**: product price, quantity sold, total sales amount
  - **Categorical**: product category, region, sales channel, month
- **Target Variable**: Revenue (True/False → purchase made or not)
- **Link**: [Customer purchase behavior - Electronic Sales Data - Kaggle](https://www.kaggle.com/datasets/cameronseamons/electronic-sales-sep2023-sep2024)

## 4. Methodology

### 4.1 Data Preprocessing
- Handle missing values
- Encode categorical attributes (month, visitor type)
- Normalize continuous variables (time spent, number of pages)

### 4.2 Exploratory Data Analysis (EDA)
- Statistical descriptions (mean, median, frequency)
- Visuals: histograms, boxplots, heatmaps
- Identify correlations between features and purchase behavior

### 4.3 Outlier Analysis
- Apply IQR to detect extreme browsing sessions (very long or very short visits)
- Decide whether to remove or adjust these records

### 4.4 Handling Imbalance
- Purchases are much fewer than non-purchases
- Use SMOTE to create balance between the two classes

### 4.5 Model Development
- Train the following classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Apply parameter tuning to improve accuracy

### 4.6 Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Tools**: Confusion matrix, ROC curves
- Compare models and select the most effective one

## 5. Expected Outcomes
- A well-prepared dataset ready for classification
- Insights into which browsing features most strongly predict purchases
- Performance comparison across three models
- Identification of the most reliable classifier for predicting online buying behavior

## 6. References
- [Kaggle Dataset: Customer purchase behavior - Electronic Sales Data](https://www.kaggle.com/datasets/cameronseamons/electronic-sales-sep2023-sep2024)
- Han, J., Kamber, M., & Pei, J. (2011). Data Mining: Concepts and Techniques
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Imbalanced-learn (SMOTE)](https://imbalanced-learn.org)