import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🩺", layout="centered")

st.title("🩺 Stroke Risk Prediction System")
st.write("Patient details ඇතුළත් කර Stroke Risk එක ගණනය කරගන්න.")

# Streamlit Cloud එකේ Path එක නිවැරදිව ලබාගැනීම
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'stroke_rf_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Assets load කිරීම
try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model/scaler files: {e}")
    st.stop()

st.subheader("📋 Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (වයස)", 1, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Hypertension (High BP)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    ever_married = st.selectbox("Ever Married?", ["Yes", "No"])

with col2:
    avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 105.5)
    bmi = st.number_input("BMI Index", 10.0, 60.0, 28.1)
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

if st.button("🔍 Predict Stroke Risk", use_container_width=True):
    try:
        # Model එකට අගයයන් සකස් කිරීම ('id' එක එකතු කර ඇත)
        patient_data = {
            'id': 0,
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        input_df = pd.DataFrame([patient_data])
        input_encoded = pd.get_dummies(input_df)

        # Model එක fit කරද්දී තිබුණු exact feature names ටික
        feature_cols = [
            'id', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
            'gender_Male', 'ever_married_Yes',
            'work_type_Govt_job', 'work_type_Private', 'work_type_Self-employed',
            'Residence_type_Urban', 
            'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
        ]

        # Missing columns වලට 0 දමා Align කිරීම
        input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
        
        # Scaling කිරීම සහ Prediction ලබාගැනීම
        input_scaled = scaler.transform(input_encoded)
        probability = model.predict_proba(input_scaled)[0][1] * 100

        # Results පෙන්වීම
        st.divider()
        st.subheader("📊 Assessment Result")
        st.progress(int(probability))
        st.write(f"**Calculated Risk Probability:** `{probability:.2f}%`")

        if probability >= 50:
            st.error("⚠️ **High Risk of Stroke!** (ඉහළ අවදානමක් ඇත)")
        else:
            st.success("✅ **Low Risk of Stroke.** (අඩු අවදානමක් ඇත)")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
