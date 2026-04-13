import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Churn Prediction System", layout="wide")

# Load model and encoders
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

try:
    model, encoders = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run train_model.py first.")
    st.stop()

# Title and description
st.title("🛡️ Customer Churn Prediction System")
st.markdown("""
This application uses a Random Forest Classifier to predict the likelihood of a customer churning.
Please enter the customer details below to get a prediction.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    st.subheader("Service Details")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    st.subheader("Service Details (Cont.)")
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    st.subheader("Billing & Contract")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=50.0)

# Prediction Logic
if st.button("Predict Churn", type="primary"):
    # Map 'Yes'/'No' for SeniorCitizen back to 0/1 if needed by training data logic, 
    # BUT wait, the CSV had 0/1 for SeniorCitizen. 
    # Let's check how other cols were.
    # We rely on the implicit knowledge that 'No' usually maps to 0 and 'Yes' to 1 in simple binary, 
    # but more importantly we must match the *exact strings* passed to the LabelEncoder during training.
    
    # Construct input dictionary
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Transformations
    # 1. Label Encoding
    # We need to be careful: LabelEncoder fits on the training data. 
    # If the app input has a value exact same as training, it works.
    
    encoded_input = input_df.copy()
    
    try:
        for col, le in encoders.items():
            if col in encoded_input.columns:
                # Handle unseen labels carefully or assume inputs match drop-down options which match unique values in CSV
                # The drop-downs above were hardcoded based on common knowledge of this dataset.
                # Ideally, we should populate dropdowns from the encoders classes.
                encoded_input[col] = le.transform(encoded_input[col])
        
        # Predict
        prediction = model.predict(encoded_input)
        probability = model.predict_proba(encoded_input)
        
        st.divider()
        st.subheader("Prediction Result")
        
        churn_prob = probability[0][1]
        
        if prediction[0] == 1: # Churn is "Yes" usually encoded as 1 if "Yes" was in the column
            # Wait, 'Churn' column was 'Yes'/'No'. LabelEncoder maps 'No'->0, 'Yes'->1 alphabetically.
            st.error(f"⚠️ High Risk of Churn ({churn_prob:.1%})")
            st.write("This customer is likely to discontinue the service.")
        else:
            st.success(f"✅ Low Risk of Churn ({churn_prob:.1%})")
            st.write("This customer is likely to stay.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Debug info - Encoded Input:", encoded_input)

st.divider()
st.caption("Built with Streamlit & Scikit-learn")
