# -*- coding: utf-8 -*-
"""Rule-Based Grade + ML Stress Prediction (Final Version)"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ===========================
# Load Dataset
# ===========================
file_path = "Modified_Student_Performance_Dataset.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop(["Student_ID", "First_Name", "Last_Name", "Email"], axis=1)

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# ===========================
#  Rule-Based Grade Assignment
# ===========================
def assign_grade(score):
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

# If "Total_Score" column doesn't exist, calculate it
if "Total_Score" not in df.columns:
    score_cols = [c for c in df.columns if "Score" in c or "score" in c]
    df["Total_Score"] = df[score_cols].mean(axis=1)

# Assign grades according to the given rule
df["Predicted_Grade"] = df["Total_Score"].apply(assign_grade)
print(" Rule-based grades assigned successfully.")
print(df[["Total_Score", "Predicted_Grade"]].head())

grade_categories = ["A", "B", "C", "D", "E", "F"]
le_grade = LabelEncoder()
le_grade.fit(grade_categories)

# Encode the grades
df["Predicted_Grade_Encoded"] = le_grade.transform(df["Predicted_Grade"])
print(" Grade classes fixed:", list(le_grade.classes_))

# ===========================
#  Categorize Stress Level
# ===========================
def categorize_stress(x):
    if x <= 3:
        return "Low"
    elif 4 <= x <= 6:
        return "Medium"
    else:
        return "High"

df["Stress_Category"] = df["Stress_Level (1-10)"].apply(categorize_stress)

# Encode target variable
le_target = LabelEncoder()
y_stress = le_target.fit_transform(df["Stress_Category"])

# ===========================
#  Prepare Features
# ===========================
# Keep all useful predictors, including Predicted_Grade_Encoded
X_stress = df.drop(["Grade", "Stress_Level (1-10)", "Stress_Category"], axis=1, errors="ignore")

# Encode categorical features
categorical_cols = X_stress.select_dtypes(include=["object"]).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_stress[col] = le.fit_transform(X_stress[col])
    encoders[col] = le

# ===========================
#  Normalize Data
# ===========================
scaler_stress = StandardScaler()
X_stress_scaled = scaler_stress.fit_transform(X_stress)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_stress_scaled, y_stress, test_size=0.2, random_state=42, stratify=y_stress
)

# ===========================
#  Train XGBoost Stress Model
# ===========================
xgb_stress = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
xgb_stress.fit(X_train, y_train)

# Evaluate model
y_pred = xgb_stress.predict(X_test)
print("\n✅ Stress Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report (Stress):\n", classification_report(y_test, y_pred, target_names=le_target.classes_))

# ===========================
#  Save All Artifacts
# ===========================
xgb_stress.save_model("xgb_stress_model.json")
joblib.dump(scaler_stress, "scaler_stress.pkl")
joblib.dump(list(X_stress.columns), "feature_names_stress.pkl")
joblib.dump(encoders, "feature_encoders.pkl")
joblib.dump(le_target, "stress_label_encoder.pkl")
joblib.dump(le_grade, "grade_label_encoder.pkl")

# Save all artifacts in one file (for Streamlit load)
artifacts = {
    "encoders": encoders,
    "le_target": le_target,
    "le_grade": le_grade
}
joblib.dump(artifacts, "stress_grade_artifacts.pkl")

print("\n All models and artifacts saved successfully!")
