# 🚖 Uber Ride Price Prediction

## 📌 Project Overview
This project focuses on predicting Uber ride fares using machine learning techniques. The model is trained on trip-level data including pickup and dropoff coordinates, timestamps, and passenger information to estimate fare prices accurately.

The objective is to build a robust regression model that captures the underlying patterns in ride pricing and improves prediction accuracy through effective data preprocessing and feature engineering.

---

## 📊 Dataset
- **Source:** Kaggle  
- **Dataset Name:** Uber Ride Price Prediction  
- **Link:** https://www.kaggle.com/datasets/kushsheth/uber-ride-price-prediction/data  

### 🧾 Features in Dataset:
- `pickup_datetime` – Timestamp of ride start  
- `pickup_latitude`, `pickup_longitude` – Pickup location  
- `dropoff_latitude`, `dropoff_longitude` – Dropoff location  
- `passenger_count` – Number of passengers  
- `fare_amount` – Target variable (ride cost)  

---

## 🧹 Data Preprocessing

The dataset contains noise and inconsistencies typical of real-world data. The following preprocessing steps were applied:

- Removed missing/null values  
- Filtered out invalid fares (≤ 0)  
- Removed unrealistic passenger counts  
- Eliminated incorrect geographic coordinates  
- Removed extreme outliers to stabilize model training  

---

## ⚙️ Feature Engineering

To improve model performance, several new features were derived:

### ⏱ Time-Based Features
- Hour of the day  
- Day of the week  
- Month and year  
- Weekend indicator  

### 📍 Spatial Features
- Distance between pickup and dropoff locations (Haversine distance)

### 🚦 Contextual Features
- Rush hour indicator  

These features help the model capture temporal and spatial dependencies in pricing.

---

## 📈 Exploratory Data Analysis (EDA)

Key insights derived through EDA:

- Fare distribution is **right-skewed**, requiring transformation  
- Strong relationship between **distance and fare**  
- Peak-hour rides tend to have higher fares  
- Presence of significant outliers in fare values  

Visualizations used:
- Distribution plots  
- Scatter plots (Distance vs Fare)  
- Box plots (Time vs Fare)  
- Correlation heatmap  

---

## 🤖 Models Implemented

The following regression models were trained and evaluated:

- **Linear Regression**  
- **Ridge Regression**  
- **Lasso Regression**  
- **XGBoost Regressor**  

---

## ⚙️ Hyperparameter Tuning

To optimize model performance, hyperparameter tuning was performed.

- Techniques used:
  - Randomized Search
- Parameters tuned (for XGBoost):
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
  - `subsample`
  - `colsample_bytree`

The tuned model did not show significant diffference with the base model

---

## 📊 Model Evaluation

Models were evaluated using:

- **RMSE (Root Mean Squared Error)** – Measures prediction error  
- **R² Score** – Measures explained variance  

---

## 🏆 Results

- Linear models provided baseline performance but struggled with non-linear relationships  
- Regularization (Ridge/Lasso) improved generalization slightly  
- **XGBoost achieved the best performance**, effectively capturing complex patterns  

> 📌 *Final performance metrics can be found in the notebook.*
