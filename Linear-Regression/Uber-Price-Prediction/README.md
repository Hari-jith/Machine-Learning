# 🚖 Uber Fare Prediction using Advanced Machine Learning

## 📌 Overview

This project aims to predict Uber ride fares using historical trip data by leveraging advanced feature engineering, spatial analysis, and ensemble learning techniques. The model captures complex non-linear relationships between trip characteristics and pricing, achieving high predictive performance.

---

## 📊 Dataset

* Source: Uber Fare Dataset (Kaggle)
* Link: https://www.kaggle.com/datasets/kushsheth/uber-ride-price-prediction/data 
* Total Records: ~200,000
* Features:

  * Pickup & dropoff coordinates (latitude, longitude)
  * Pickup timestamp
  * Passenger count
  * Fare amount (target variable)

---

## ⚙️ Data Preprocessing

* Removed missing and duplicate entries
* Filtered invalid fare values (≤ 0)
* Eliminated erroneous geographic coordinates (0 values and out-of-bound ranges)
* Restricted passenger count to realistic limits (1–6)
* Converted timestamp to datetime format
* Removed extreme outliers using 99th percentile filtering on distance

---

## 🧠 Feature Engineering

### ⏱️ Temporal Features

* Hour of day
* Day of week
* Month, Year
* Weekend indicator
* Peak hour indicator (7–10 AM, 4–8 PM)
* Night ride indicator (midnight–4 AM)

### 🌍 Spatial Features

* **Haversine distance** (actual geographic distance)
* **Manhattan distance** (grid-based urban distance approximation)

### 📍 Location Intelligence

* Applied **K-Means clustering** on:

  * Pickup coordinates
  * Dropoff coordinates
* Captures location-based pricing zones

### 🔗 Feature Interactions

* Distance × Hour interaction to model time-dependent fare variation

### 🎯 Target Transformation

* Applied **log transformation** to reduce skewness and stabilize variance

---

## 🤖 Models Implemented

* Linear Regression
* Ridge Regression
* Lasso Regression
* XGBoost Regressor (final model)

---

## 🔍 Model Performance

| Model             | Test R²    | Test RMSE  | Test MAE   |
| ----------------- | ---------- | ---------- | ---------- |
| Linear Regression | 0.7018     | 0.2723     | 0.1951     |
| Ridge Regression  | 0.7018     | 0.2723     | 0.1951     |
| Lasso Regression  | 0.6973     | 0.2743     | 0.1983     |
| XGBoost (Tuned)   | **0.8007** | **0.2226** | **0.1458** |

### 📈 Cross-Validation

* Best CV R²: **0.8030**
* Confirms strong generalization ability

---

## ⚙️ Hyperparameter Tuning

Optimized using **RandomizedSearchCV**:

* n_estimators = 800
* max_depth = 7
* learning_rate = 0.1
* subsample = 1.0
* colsample_bytree = 0.8
* reg_alpha = 1
* reg_lambda = 1

---

## 📊 Model Interpretation

### 🔹 Feature Importance (XGBoost)

Top contributors:

1. Distance (Haversine)
2. Manhattan distance
3. Distance-hour interaction
4. Hour of day
5. Month & day patterns

### 🔹 SHAP Analysis

* **Distance features dominate pricing**
* Longer trips → strong positive contribution to fare
* Time-based effects (hour, peak) influence pricing moderately
* Location clusters capture zone-based fare variations
* Passenger count has minimal impact

---

## 🧪 Evaluation Metrics

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score
* Cross-validation R²

---

## 📉 Residual Analysis

* Residuals approximately normally distributed
* No strong bias patterns observed
* Indicates good model fit and error balance

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP (Explainability)
* Matplotlib, Seaborn

---

## 🚀 Key Insights

* Uber fare pricing is highly **non-linear**
* Distance is the dominant factor but not sufficient alone
* Temporal and spatial features significantly improve predictions
* Tree-based ensemble models outperform linear models by a large margin
* Feature interactions play a critical role in capturing real-world pricing behavior

---

## 📎 Conclusion

This project demonstrates a complete machine learning pipeline, including data cleaning, advanced feature engineering, model optimization, and interpretability. The final model achieves strong predictive performance with an R² of **0.80**, making it a robust solution for fare estimation tasks.

---


> 📌 *Final performance metrics can be found in the notebook.*
