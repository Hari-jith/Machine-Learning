# 🩺 Liver Disease Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a patient has liver disease based on various biochemical parameters. Multiple classification algorithms were implemented and compared to identify the most effective model.

The final solution uses **Logistic Regression with threshold tuning**, deployed through an interactive **Streamlit web application**.

---

## 🎯 Objectives

* Perform **Exploratory Data Analysis (EDA)** on liver patient data
* Handle **class imbalance** and skewed features
* Train and compare multiple classification models
* Optimize model performance using **threshold tuning**
* Deploy the final model using **Streamlit**

---

## 📂 Dataset

Dataset Used: https://www.kaggle.com/datasets/shauryasrivastava01/liver-patient-dataset
The dataset contains medical attributes such as:

* Age
* Gender
* Total Bilirubin (TB)
* Direct Bilirubin (DB)
* Alkaline Phosphotase
* SGPT
* SGOT
* Total Proteins (TP)
* Albumin (ALB)
* Albumin/Globulin Ratio (A/G Ratio)

### 🎯 Target Variable

* **1 → Liver Disease**
* **0 → No Liver Disease**

---

## 🔍 Exploratory Data Analysis (EDA)

Key observations:

* The dataset is **imbalanced (~72% liver disease cases)**
* Several features (TB, DB, SGPT, SGOT) show **right-skewed distributions**
* Presence of **extreme values**, which are medically significant
* Certain features show noticeable variation between diseased and non-diseased patients

### ⚙️ Preprocessing Steps

* Categorical encoding (Gender, Target)
* Log transformation for skewed features
* Feature scaling using **StandardScaler**
* Handling imbalance using **SMOTE**

---

## 🤖 Models Used

The following models were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Naive Bayes
* Decision Tree

---

## 📊 Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### ⚠️ Key Consideration

Since this is a **medical classification problem**, **Recall (Sensitivity)** for the disease class was prioritized to minimize false negatives.

---

## 🏆 Final Model Selection

**Logistic Regression** was selected as the best-performing model due to:

* Balanced performance across metrics
* Higher stability compared to other models

### 🔧 Threshold Tuning

The decision threshold was adjusted from **0.5 → 0.4** to improve recall.

#### ✅ Result:

* Recall improved significantly (**~0.67 → ~0.80**)
* Reduced false negatives
* Slight decrease in precision (acceptable in medical context)

---

## 🚀 Streamlit Web Application

An interactive web app was built using Streamlit to allow real-time predictions.

### Features:

* User-friendly input interface
* Real-time prediction
* Probability score display
* Clean and responsive UI

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd Liver_Disease_Prediction
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```
streamlit run app.py
```

---

## 📈 Sample Output

* Displays prediction:

  * **High Risk of Liver Disease**
  * **No Liver Disease Detected**
* Shows probability score and confidence

---

## 🧠 Key Learnings

* Importance of **EDA in understanding medical data**
* Handling **imbalanced datasets**
* Effect of **feature scaling on model performance**
* Trade-off between **precision and recall**
* Practical use of **threshold tuning**
* Building and deploying ML models with **Streamlit**

---

## 📌 Conclusion

This project demonstrates a complete machine learning pipeline—from data preprocessing and model training to deployment. The use of threshold tuning significantly improved the model's ability to detect liver disease, making it more suitable for real-world medical applications.

---

## 👨‍💻 Author

**Harijith M. M**
