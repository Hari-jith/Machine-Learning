import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_churn_model():
    print("Loading data...")
    df = pd.read_csv('churn.csv')
    
    # Preprocessing
    print("Preprocessing data...")
    # Drop CustomerID
    df = df.drop('customerID', axis=1)
    
    # TotalCharges has empty strings " " for some new customers. Replace with NaN then 0 (or median)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Train test split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluation
    print("Evaluating model...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Save model and encoders
    print("Saving model and encoders...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
        
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
        
    print("Done!")

if __name__ == "__main__":
    train_churn_model()
