# 🌦️ Weather Classification using Machine Learning

## 📌 Problem Statement
Accurately classifying weather conditions based on environmental parameters is important for applications like climate monitoring, agriculture planning, and forecasting systems.

This project builds a **multi-class classification model** to predict weather types using structured meteorological data.

---

## 📊 Dataset Overview

Dataset: weather_classification_data.csv

Source: Kaggle

Dataset Link: https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/data

The dataset contains both numerical and categorical features:

**Numerical Features:**
- Temperature  
- Humidity  
- Wind Speed  
- Precipitation  
- Atmospheric Pressure  
- UV Index  
- Visibility (km)  

**Categorical Features:**
- Cloud Cover  
- Season  
- Location  

**Target Variable:**
- Weather Type (multi-class)

---

## 🔍 Exploratory Data Analysis (EDA)

Key observations:

- Target classes are reasonably balanced  
- Significant variation observed in:
  - Temperature
  - Visibility
  - Atmospheric Pressure  
- Correlation heatmap revealed moderate relationships between some features  
- Outliers detected in:
  - Wind Speed  
  - Atmospheric Pressure  
  - Visibility  

### ✔ Action Taken:
- Applied **IQR-based outlier capping** instead of removal to retain data integrity  

---

## ⚙️ Data Preprocessing

- Standardized column names and categorical values  
- Applied **One-Hot Encoding** for categorical variables  
- Used different preprocessing strategies:
  - **StandardScaler** for distance-based models (KNN, SVM, Logistic Regression)  
  - No scaling for tree-based models (Decision Tree, Random Forest, XGBoost)  

✔ Implemented preprocessing pipelines using `ColumnTransformer` to avoid data leakage  

---

## 🧠 Feature Engineering Strategy

- Retained all features after validating importance through EDA  
- Avoided unnecessary feature removal to preserve information  
- Used **Stratified Train-Test Split** to maintain class balance  

---

## 🤖 Models Implemented

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- XGBoost  

---

## 🔧 Hyperparameter Tuning

- Used **GridSearchCV** with **Stratified K-Fold (5 folds)**  
- Optimized based on:
  - Accuracy  
  - Weighted F1 Score  

---

## 📈 Model Performance

| Model               | Accuracy | F1 Score |
|--------------------|---------|---------|
| Logistic Regression | 0.87 | 0.87 |
| KNN                | 0.89 | 0.89 |
| SVM                | 0.90 | 0.90 |
| Decision Tree      | 0.90 | 0.90 |
| Random Forest      | 0.91 | 0.91 |
| **XGBoost**        | **0.91** | **0.91** |

---

## 🏆 Best Model: XGBoost

- Accuracy: **91.47%**  
- F1 Score: **0.9149**  

### ✔ Why XGBoost performed best:
- Captures non-linear relationships effectively  
- Robust to outliers and feature interactions  
- Works well with mixed feature types  

---

## 📊 Model Evaluation

- Evaluated using:
  - Accuracy  
  - Weighted F1 Score  
  - Confusion Matrix  

✔ Achieved strong class-wise prediction performance with minimal misclassification  

---

## 🚀 Key Takeaways

- Proper preprocessing and pipeline design improved model performance significantly  
- Tree-based models outperformed linear models due to non-linear patterns in data  
- Outlier handling via capping preserved useful data while reducing noise  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  

---

## 📌 Future Improvements

- Feature importance analysis (XGBoost / SHAP)  
- Model deployment using Flask or FastAPI  
- Real-time weather classification system  

---

## 👤 Author

**Harijith M. M**
