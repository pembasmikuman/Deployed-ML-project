import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# --- 1. Load Model & Scaler ---
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('student_gpa_model.keras')
    scaler = joblib.load('minmax_scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# --- 2. App Layout ---
st.title("🎓 Student GPA Predictor")
st.write("Enter student details below to predict their GPA.")

# Define the exact features names in the correct order (Must match training!)
feature_names = [
    'Ethnicity', 'ParentalEducation', 'Gender', 'StudyTimeWeekly',
    'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular',
    'Sports', 'Music'
]

# --- 3. Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        ethnicity = st.selectbox("Ethnicity", options=[0, 1, 2, 3],
                                 format_func=lambda x: {0:"Caucasian", 1:"African American", 2:"Asian", 3:"Other"}[x])
        p_edu = st.selectbox("Parental Education", options=[0, 1, 2, 3, 4],
                             format_func=lambda x: {0:"None", 1:"High School", 2:"Some College", 3:"Bachelor's", 4:"Higher"}[x])
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        study_time = st.slider("Weekly Study Time (Hours)", 0.0, 20.0, 5.0)
        absences = st.number_input("Absences", min_value=0, max_value=30, value=0)

    with col2:
        tutoring = st.radio("Tutoring?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        p_support = st.selectbox("Parental Support", options=[0, 1, 2, 3, 4],
                                 format_func=lambda x: {0:"None", 1:"Low", 2:"Moderate", 3:"High", 4:"Very High"}[x])
        extra = st.radio("Extracurriculars?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        sports = st.radio("Sports?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        music = st.radio("Music?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    submitted = st.form_submit_button("Predict GPA")

# --- 4. Prediction Logic ---
if submitted:
    # A. Create DataFrame (Same format as Notebook)
    input_data = pd.DataFrame([[
        ethnicity, p_edu, gender, study_time, absences,
        tutoring, p_support, extra, sports, music
    ]], columns=feature_names)

    # B. Scale the data
    scaled_data = scaler.transform(input_data)

    # C. Reshape for CNN-BiGRU (Batch, Features, 1)
    reshaped_data = scaled_data.reshape(1, scaled_data.shape[1], 1)

    # D. Predict
    prediction = model.predict(reshaped_data)
    gpa = float(prediction[0][0])

    # E. Clamp result (GPA usually 0.0 - 4.0)
    gpa = max(0.0, min(4.0, gpa))

    # Output
    st.success(f"Predicted GPA: **{gpa:.2f}**")
