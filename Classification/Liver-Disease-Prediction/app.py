import streamlit as st
import pandas as pd
import joblib

# Load files (your setup)
model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Page config
st.set_page_config(
    page_title="Liver Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# Title
st.markdown("<h1 style='text-align: center;'>🩺 Liver Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar inputs
st.sidebar.header("📝 Patient Details")

age = st.sidebar.slider("Age", 1, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tb = st.sidebar.number_input("Total Bilirubin", min_value=0.0)
db = st.sidebar.number_input("Direct Bilirubin", min_value=0.0)
alkphos = st.sidebar.number_input("Alkaline Phosphotase", min_value=0.0)
sgpt = st.sidebar.number_input("SGPT", min_value=0.0)
sgot = st.sidebar.number_input("SGOT", min_value=0.0)
tp = st.sidebar.number_input("Total Proteins", min_value=0.0)
alb = st.sidebar.number_input("Albumin", min_value=0.0)
ag_ratio = st.sidebar.number_input("A/G Ratio", min_value=0.0)

# Encode gender
gender = 1 if gender == "Male" else 0

# Create input dataframe (IMPORTANT: use your saved column order)
input_data = pd.DataFrame([[
    age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio
]], columns=columns)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Summary")
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.subheader("🚀 Prediction")

    if st.button("Predict", use_container_width=True):

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict probability
        prob = model.predict_proba(input_scaled)[0][1]

        # Apply your tuned threshold
        prediction = 1 if prob > 0.4 else 0

        # Output
        if prediction == 1:
            st.error("⚠️ High Risk of Liver Disease")
        else:
            st.success("✅ No Liver Disease Detected")

        # Probability display
        st.metric("Disease Probability", f"{prob*100:.2f}%")

        # Progress bar
        st.progress(float(prob))

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>ML Project - Logistic Regression</p>",
    unsafe_allow_html=True
)