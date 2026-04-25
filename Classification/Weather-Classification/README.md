# 🌦️ Weather Classification using Machine Learning

## 📌 Overview

This project focuses on building a **multi-class classification system** to predict weather conditions based on atmospheric features. The pipeline includes **data cleaning, exploratory data analysis (EDA), preprocessing, model training, hyperparameter tuning, and evaluation**.

The objective is to compare multiple machine learning algorithms and select the best-performing model based on **accuracy and F1-score**.

---

## 📂 Dataset

Dataset: weather_classification_data.csv

Source: Kaggle

Dataset Link: https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/data

* Contains meteorological features such as:

  * Temperature
  * Humidity
  * Wind Speed
  * Precipitation
  * Atmospheric Pressure
  * UV Index
  * Visibility

* **Target Variable:**

  * `weather_type` (categorical)

---

## ⚙️ Project Workflow

### 1. Data Inspection

* Checked dataset shape, structure, and data types
* Identified missing and duplicate values
* Analyzed class distribution

### 2. Data Cleaning

* Standardized column names (lowercase, removed special characters)
* Cleaned categorical values (trim + lowercase)
* Renamed inconsistent columns

### 3. Exploratory Data Analysis (EDA)

* Target class distribution visualization
* Feature distribution using histograms
* Boxplots to analyze feature vs target relationships
* Correlation heatmap for numerical features
* Categorical vs target analysis
* Outlier detection using **Z-score**

### 4. Data Preprocessing

* Outlier handling using **IQR capping**
* Feature encoding:

  * Label Encoding (target)
  * OneHot Encoding (categorical features)
* Feature scaling using **StandardScaler**
* Train-test split (80-20) with stratification

---

## 🤖 Models Implemented

The following models were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* XGBoost

Each model is implemented using a **Pipeline**, ensuring:

* Consistent preprocessing
* No data leakage

---

## 🔍 Hyperparameter Tuning

* Used **GridSearchCV** with:

  * Stratified K-Fold Cross Validation (k=5)
* Optimized key parameters for each model

---

## 📊 Evaluation Metrics

* Accuracy
* F1 Score
* Confusion Matrix

---

## 📈 Results

* Compared all models using:

  * Bar plots (Accuracy vs F1 Score)
  * Line plots for performance trends
* Selected the **best model based on F1 Score**

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost

---

## 🚀 Installation

```bash
git clone https://github.com/Hari-jith/Weather-Classification.git
cd Weather-Classification
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the notebook:

```bash
jupyter notebook weather_classification.ipynb
```

---

## 📌 Key Highlights

* End-to-end ML pipeline
* Proper use of **Pipeline + GridSearchCV**
* Handles both numerical and categorical data
* Prevents data leakage
* Model comparison with clear visualization

---

## 👤 Author

**Harijith M. M**
