# -*- coding: utf-8 -*-
"""Streamlit App for Stress & Grade Prediction (Rule-based Grade + ML Stress)"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =======================
# CONFIGURATION
# =======================
STRESS_MODEL_FILE = "xgb_stress_model.json"
ARTIFACTS_FILE = "stress_grade_artifacts.pkl"
GRADE_ENCODER_FILE = "grade_label_encoder.pkl"
FEATURE_NAMES_STRESS_FILE = "feature_names_stress.pkl"
STRESS_SCALER_FILE = "scaler_stress.pkl"

# =======================
# LOAD MODELS AND ARTIFACTS
# =======================
@st.cache_resource
def load_artifacts():
    required_files = [
        STRESS_MODEL_FILE,
        ARTIFACTS_FILE,
        GRADE_ENCODER_FILE,
        FEATURE_NAMES_STRESS_FILE,
        STRESS_SCALER_FILE
    ]
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"❌ Missing required file: {f}")
            st.stop()

    # Load artifacts
    artifacts = joblib.load(ARTIFACTS_FILE)
    encoders = artifacts["encoders"]
    le_target = artifacts["le_target"]
    le_grade = joblib.load(GRADE_ENCODER_FILE)

    # Load model
    stress_model = xgb.XGBClassifier()
    stress_model.load_model(STRESS_MODEL_FILE)

    # Load scaler & feature names
    scaler_stress = joblib.load(STRESS_SCALER_FILE)
    feature_names_stress = joblib.load(FEATURE_NAMES_STRESS_FILE)

    return stress_model, scaler_stress, encoders, le_grade, le_target, feature_names_stress


# =======================
# PREPROCESS INPUT FUNCTION
# =======================
def preprocess_input(input_df, feature_names, scaler, encoders):
    df = input_df.copy()

    categorical_cols = [
        "Gender", "Department", "Extracurricular_Activities",
        "Internet_Access_at_Home", "Family_Income_Level",
        "Parent_Education_Level"
    ]

    for col in categorical_cols:
        if col in df.columns:
            try:
                le = encoders[col]
                df[col] = le.transform([df[col].iloc[0]])[0]
            except Exception:
                st.error(f"Unknown category '{df[col].iloc[0]}' in '{col}'. Please check inputs.")
                st.stop()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    final_df = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in df.columns:
        if col in final_df.columns:
            final_df[col] = df[col].values[0]

    scaled = scaler.transform(final_df)
    return scaled


# =======================
# GRADE CALCULATION FUNCTION
# =======================
def assign_grade(score):
    """Rule-based grade assignment"""
    if score >= 85:
        return "A"
    elif score >= 78:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    elif score >= 50:
        return "E"
    else:
        return "F"


# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title="🎓 Stress & Grade Predictor", layout="centered")
st.title("🎓 EduStress Analyzer")
st.write("Predict a student's **Grade** and **Stress Level** based on details.")
st.markdown("---")

try:
    stress_model, scaler_stress, encoders, le_grade, le_target, feature_names_stress = load_artifacts()
except Exception as e:
    st.error("⚠️ Could not load models or artifacts.")
    st.exception(e)
    st.stop()

with st.form("prediction_form"):
    st.header("🧾 Enter Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 17, 30, 20)
        department = st.selectbox("Department", ["CS", "Engineering", "Business", "Mathematics", "Psychology", "History", "Physics"])
        study_hours = st.slider("Study Hours per Week", 0.0, 50.0, 15.0, 0.5)
        sleep_hours = st.slider("Sleep Hours per Night", 3.0, 10.0, 7.0, 0.1)

    with col2:
        attendance = st.slider("Attendance (%)", 50.0, 100.0, 90.0, 0.1)
        midterm = st.slider("Midterm Score", 0.0, 100.0, 75.0, 0.1)
        final = st.slider("Final Score", 0.0, 100.0, 80.0, 0.1)
        total = st.slider("Total Score", 0.0, 100.0, 78.0, 0.1)

    with col3:
        assignments = st.slider("Assignments Avg", 0.0, 100.0, 85.0, 0.1)
        quizzes = st.slider("Quizzes Avg", 0.0, 100.0, 70.0, 0.1)
        participation = st.slider("Participation Score", 0.0, 100.0, 90.0, 0.1)
        projects = st.slider("Projects Score", 0.0, 100.0, 88.0, 0.1)
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
        parent_edu = st.selectbox("Parent Education Level", ["High School", "Associate's Degree", "Bachelor's", "Master's", "PhD", "Unknown/Not Disclosed"])
        income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("🔍 Predict Grade & Stress")

# =======================
# PREDICTION LOGIC
# =======================
if submitted:
    input_data = {
        "Gender": [gender],
        "Age": [age],
        "Department": [department],
        "Attendance (%)": [attendance],
        "Midterm_Score": [midterm],
        "Final_Score": [final],
        "Assignments_Avg": [assignments],
        "Quizzes_Avg": [quizzes],
        "Participation_Score": [participation],
        "Projects_Score": [projects],
        "Total_Score": [total],
        "Study_Hours_per_Week": [study_hours],
        "Extracurricular_Activities": [extracurricular],
        "Internet_Access_at_Home": [internet],
        "Parent_Education_Level": [parent_edu],
        "Family_Income_Level": [income],
        "Sleep_Hours_per_Night": [sleep_hours],
    }

    input_df = pd.DataFrame(input_data)

    # --- 1️⃣ Predict Grade using rules ---
    predicted_grade_alpha = assign_grade(total)
    grade_encoded = le_grade.transform([predicted_grade_alpha])[0]
    input_df["Predicted_Grade"] = grade_encoded

    # --- 2️⃣ Predict Stress ---
    processed_stress = preprocess_input(input_df, feature_names_stress, scaler_stress, encoders)
    pred_stress_encoded = stress_model.predict(processed_stress)
    predicted_stress = le_target.inverse_transform(pred_stress_encoded)[0]

    # --- Display Results ---
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    st.success(f"🎓 **Predicted Grade:** {predicted_grade_alpha}")

    if predicted_stress == "High":
        st.error(f"🧠 **Predicted Stress Level:** {predicted_stress} 😥")
    elif predicted_stress == "Medium":
        st.warning(f"🧠 **Predicted Stress Level:** {predicted_stress} 😐")
    else:
        st.success(f"🧠 **Predicted Stress Level:** {predicted_stress} 😊")

    st.markdown(f"""
    **Interpretation:**
    - Grade **{predicted_grade_alpha}** based on your total score ({total}).
    - Stress Level **{predicted_stress}** predicted from academic and lifestyle factors.
    """)
