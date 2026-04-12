# 🏡 Real Estate Price Prediction using Machine Learning

## 📌 Project Overview

This project focuses on building a machine learning regression model to predict real estate property prices based on various structural, location, and amenity-related features. The workflow follows a systematic approach including data exploration, feature engineering, model building, and evaluation.

---

## 🎯 Objective

To develop an accurate regression model that can estimate property prices using key features such as size, number of rooms, age of the house, and additional amenities.

---

## 📊 Dataset

Dataset Used: https://www.kaggle.com/datasets/denkuznetz/housing-prices-regression?select=real_estate_dataset.csv

The dataset contains numerical and binary features representing different aspects of a property:

### 🔹 Features

* **Square_Feet** – Total area of the property
* **Num_Bedrooms** – Number of bedrooms
* **Num_Bathrooms** – Number of bathrooms
* **Num_Floors** – Number of floors
* **Year_Built** – Year the property was constructed
* **House_Age** – Derived feature (Current Year - Year_Built)
* **Garage_Size** – Size of garage
* **Location_Score** – Score representing location quality
* **Distance_to_Center** – Distance from city center
* **Has_Pool** – Binary (0/1) indicating pool availability
* **Has_Garden** – Binary (0/1) indicating garden availability

### 🎯 Target Variable

* **Price** – Property price (continuous variable)

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

* ✔ Distribution analysis of the target variable (Price)
* ✔ Correlation heatmap to identify relationships between features
* ✔ Scatter plots for continuous variables
* ✔ Boxplots for categorical/discrete features
* ✔ Outlier detection using IQR method

### 📌 Key Observations

* Price distribution is approximately normal with slight skew
* Square footage and number of bedrooms strongly influence price
* Amenities like pool and garden increase average property value
* Minimal outliers detected, ensuring data quality

---

## ⚙️ Feature Engineering

* Created a new feature:

  * **House_Age = Current Year - Year_Built**
* Removed redundant or less useful representations where necessary
* Ensured all features are numeric and model-compatible

---

## 🤖 Model Building

Three models were developed to evaluate performance improvements:

---

### 🔹 1. Baseline Model

**Features Used:**

* Square_Feet
* Num_Bedrooms
* Year_Built

**Performance:**

* RMSE: 54,580
* R² Score: 0.80

---

### 🔹 2. Selected Features Model

**Features Used:**

* Square_Feet
* Num_Bedrooms
* House_Age
* Num_Bathrooms
* Num_Floors
* Has_Pool
* Has_Garden

**Performance:**

* RMSE: 31,004
* R² Score: 0.93

---

### 🔹 3. Final Model (All Features)

**Features Used:**

* All available features

**Performance:**

* RMSE: 21,020
* R² Score: 0.97

---

## 📈 Model Evaluation

The models were evaluated using:

* **RMSE (Root Mean Squared Error)** – Measures prediction error
* **R² Score** – Measures explained variance
* **Residual Analysis** – Checks model assumptions

---

## 🧠 Key Insights

* Square footage and number of bedrooms are the most influential predictors
* Feature engineering (House_Age) improved interpretability
* Adding more relevant features significantly improved model performance
* Even features with low individual correlation contributed when combined
* Final model achieved high accuracy with R² ≈ 0.97

---

## ⚠️ Limitations

* Dataset may be synthetic or highly structured (due to very high R²)
* Real-world data may introduce more noise and variability
* Model assumes mostly linear relationships

---

## 🚀 Future Improvements

* Apply advanced models:

  * Random Forest Regressor
  * Gradient Boosting / XGBoost
* Perform hyperparameter tuning
* Deploy model using FastAPI or Flask
* Integrate frontend using React

---

## 🧑‍💻 Author

**Harijith M**

---

## 📌 Conclusion

This project demonstrates a complete machine learning pipeline for regression problems, from data analysis to model evaluation. The results highlight the importance of feature selection and engineering in improving predictive performance. The final model provides a strong foundation for further enhancements and real-world deployment.

---
