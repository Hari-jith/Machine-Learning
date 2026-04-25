# 💳 Loan Default Prediction using Logistic Regression

## 📌 Project Overview

This project builds a machine learning model to predict whether a borrower will **repay a loan or default**. The goal is to assist financial institutions in minimizing risk by identifying high-risk applicants.

The project follows a complete **end-to-end data science pipeline**, including data cleaning, exploratory data analysis (EDA), feature engineering, handling class imbalance, model training, and optimization.

---

## 🎯 Objective

* Predict loan repayment status:

  * **1 → Loan Paid Back**
  * **0 → Default**
* Improve detection of risky borrowers while maintaining overall model performance.

---

## 📂 Dataset

Dataset Source: Kaggle

Dataset: https://www.kaggle.com/datasets/nabihazahid/loan-prediction-dataset-2025

The dataset contains financial and behavioral attributes of borrowers, including:

* Age
* Annual Income
* Credit Score
* Debt-to-Income Ratio
* Loan Term
* Number of Open Accounts
* Delinquency History
* Public Records
* Number of Delinquencies

---

## 🧠 Workflow

### 1. Data Cleaning

* Removed duplicate records
* Verified absence of missing values
* Ensured correct data types

---

### 2. Exploratory Data Analysis (EDA)

* Target distribution analysis
* Feature distribution visualization
* Correlation heatmap (numerical features)
* Identified skewness in financial variables
* Detected multicollinearity

---

### 3. Feature Engineering

* Applied **log transformation** on skewed features
* Removed highly correlated features using VIF analysis
* Selected meaningful financial indicators

---

### 4. Handling Multicollinearity

* Used Variance Inflation Factor (VIF)
* Dropped redundant features like:

  * `total_credit_limit`
  * `loan_amount`
* Improved model stability and interpretability

---

### 5. Handling Class Imbalance

* Applied **SMOTE + Tomek Links**

  * SMOTE → synthetic minority samples
  * Tomek → removes noisy boundaries
* Improved detection of defaulters

---

### 6. Model Building

* Algorithm: **Logistic Regression**
* Used pipeline for:

  * Scaling
  * Model training
* Applied **regularization (L2)**

---

### 7. Hyperparameter Tuning

* Used **RandomizedSearchCV**
* Reduced computational cost using:

  * `cv = 3`
  * Limited parameter search space

---

### 8. Threshold Optimization

* Default threshold (0.5) was suboptimal
* Tuned threshold to **0.35**
* Improved minority class (defaulters) detection

---

## 📊 Final Model Performance

### Confusion Matrix

```
[[ 518  282]
 [ 255 2945]]
```

### Metrics

| Metric              | Value     |
| ------------------- | --------- |
| Accuracy            | **86.6%** |
| ROC-AUC             | **0.885** |
| Precision (Class 0) | **0.67**  |
| Recall (Class 0)    | **0.65**  |
| Precision (Class 1) | **0.91**  |
| Recall (Class 1)    | **0.92**  |

---

## 📈 Key Insights

* The model demonstrates **strong classification capability (ROC-AUC ≈ 0.88)**
* Class imbalance handling significantly improved detection of defaulters
* Threshold tuning helped balance precision and recall
* Financial indicators like credit score and debt ratio strongly influence predictions

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-learn

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/Hari-jith/Loan-Prediction.git
cd Loan-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Key Learnings

* Importance of handling class imbalance in real-world datasets
* Trade-off between model interpretability and performance
* Role of threshold tuning in classification problems
* Practical optimization of machine learning pipelines

---

## 👨‍💻 Author

Harijith M. M
